"""Global motion model (constant inter-frame homography) + multi-object tracker.

All geometry is in SOURCE pixels (3840x2160). The drone flies a straight line at
constant speed and height, so the ground-plane homography between consecutive
source frames is (to ~1 px) one constant matrix H per flight. We estimate it
online by registering consecutive camera views (ORB + RANSAC), pooling all
correspondences and solving  q = H^k p  in least squares, where k is the frame
gap between the two views. Tracks are then propagated with H every frame, so we
keep answering for objects long after the camera has looked away.
"""
import math
import time

import cv2
import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment

from common import H as FH, W as FW, LEVEL_SCALE, NC, VIEW_W, VIEW_H


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #

def warp_pts(Hm, pts):
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    ph = np.hstack([pts, np.ones((len(pts), 1))]) @ Hm.T
    return ph[:, :2] / ph[:, 2:3]


def warp_box(Hm, box):
    """Warp an axis-aligned box: centre through H, size through local Jacobian.
    (Taking min/max of warped corners would inflate boxes under tiny rotations.)"""
    x1, y1, x2, y2 = box
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    p = warp_pts(Hm, [[cx, cy], [cx - w / 2, cy], [cx + w / 2, cy], [cx, cy - h / 2], [cx, cy + h / 2]])
    c = p[0]
    nw = np.linalg.norm(p[2] - p[1])
    nh = np.linalg.norm(p[4] - p[3])
    return np.array([c[0] - nw / 2, c[1] - nh / 2, c[0] + nw / 2, c[1] + nh / 2])


def mpow(Hm, k):
    if k == 0:
        return np.eye(3)
    if k > 0:
        return np.linalg.matrix_power(Hm, k)
    return np.linalg.matrix_power(np.linalg.inv(Hm), -k)


def iou_matrix(a, b):
    a = np.asarray(a, np.float64).reshape(-1, 4)
    b = np.asarray(b, np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def rect_inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


FRAME_RECT = (0.0, 0.0, float(FW), float(FH))


# --------------------------------------------------------------------------- #
# motion model
# --------------------------------------------------------------------------- #

class MotionModel:
    """Estimates the constant source-frame homography H (frame t -> t+1)."""

    def __init__(self, max_groups=40, max_pts_per_group=120):
        self.H = np.eye(3)
        self.n_fits = 0
        self.groups = []  # (p, q, k)
        self.history = []  # recent views: dict(frame, level, region, gray, kp_src, des)
        self.max_groups = max_groups
        self.max_pts = max_pts_per_group
        self.orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=6, fastThreshold=10)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last_info = ""

    @property
    def valid(self):
        return self.n_fits > 0

    def _features(self, gray, level, region, work_scale):
        """ORB on the view resampled to `work_scale` source px per pixel."""
        s = LEVEL_SCALE[level]
        img = gray
        if work_scale != s:
            f = s / work_scale
            img = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        kps, des = self.orb.detectAndCompute(img, None)
        if des is None or len(kps) < 8:
            return None, None
        pts = np.array([k.pt for k in kps], np.float64) * work_scale + np.array(region[:2], np.float64)
        return pts, des

    def observe(self, frame, img_bgr, level, region):
        """Register the new view against recent views, refit H. Returns True if H changed."""
        t0 = time.perf_counter()
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        cur = dict(frame=frame, level=level, region=tuple(float(v) for v in region), gray=gray, feats={})
        changed = False
        self.n_obs = getattr(self, "n_obs", 0) + 1
        if self.n_fits >= 12 and self.n_obs % 3 != 0:
            # H has converged: refine only every third frame (latency)
            self.history.append(cur)
            self.history = self.history[-6:]
            return False
        # try the most recent views first; at most 2 registrations per frame
        tried = 0
        for prev in reversed(self.history):
            if tried >= (2 if self.n_fits < 3 else 1):
                break
            k = frame - prev["frame"]
            if k <= 0 or k > 6 or abs(prev["level"] - level) > 1:
                continue
            # predicted overlap of prev region at current frame
            pr = warp_box(mpow(self.H, k), prev["region"]) if self.valid else np.array(prev["region"])
            if self.valid:
                inter = rect_inter(pr, cur["region"])
                small = min((pr[2] - pr[0]) * (pr[3] - pr[1]), (cur["region"][2] - cur["region"][0]) * (cur["region"][3] - cur["region"][1]))
                if inter < 0.12 * small:
                    continue
            tried += 1
            if self._register(prev, cur, k):
                changed = True
                break
        self.history.append(cur)
        self.history = self.history[-6:]
        self.last_info += f" reg {1000 * (time.perf_counter() - t0):.0f}ms"
        return changed

    def _get_feats(self, view, work_scale):
        f = view["feats"].get(work_scale)
        if f is None:
            f = self._features(view["gray"], view["level"], view["region"], work_scale)
            view["feats"][work_scale] = f
        return f

    def _register(self, prev, cur, k):
        ws = max(LEVEL_SCALE[prev["level"]], LEVEL_SCALE[cur["level"]])
        p_pts, p_des = self._get_feats(prev, ws)
        c_pts, c_des = self._get_feats(cur, ws)
        if p_des is None or c_des is None:
            return False
        matches = self.bf.knnMatch(p_des, c_des, k=2)
        good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
        if len(good) < 12:
            return False
        p = p_pts[[m.queryIdx for m in good]]
        q = c_pts[[m.trainIdx for m in good]]
        if self.valid:
            # gate with the prediction (generous: estimate may still be rough)
            pred = warp_pts(mpow(self.H, k), p)
            ok = np.linalg.norm(pred - q, axis=1) < max(40.0, 12.0 * ws) + 10 * k * (self.n_fits < 3)
            p, q = p[ok], q[ok]
            if len(p) < 12:
                return False
        Hk, inl = cv2.findHomography(p, q, cv2.RANSAC, 1.5 * ws, maxIters=500)
        if Hk is None:
            return False
        inl = inl.ravel().astype(bool)
        if inl.sum() < 12:
            return False
        p, q = p[inl], q[inl]
        # plausibility: per-frame motion should be modest
        disp = np.median(np.linalg.norm(q - p, axis=1)) / k
        if disp > 400:
            return False
        if len(p) > self.max_pts:
            idx = np.random.default_rng(0).choice(len(p), self.max_pts, replace=False)
            p, q = p[idx], q[idx]
        self.groups.append((p, q, k))
        self.groups = self.groups[-self.max_groups:]
        self._fit(Hk if (not self.valid and k == 1) else None)
        return True

    def _fit(self, init=None):
        H0 = init if init is not None else self.H
        H0 = H0 / H0[2, 2]
        x0 = np.array([H0[0, 0] - 1, H0[0, 1], H0[0, 2], H0[1, 0], H0[1, 1] - 1, H0[1, 2], H0[2, 0] * 1e4, H0[2, 1] * 1e4])

        def mat(x):
            return np.array([[1 + x[0], x[1], x[2]], [x[3], 1 + x[4], x[5]], [x[6] * 1e-4, x[7] * 1e-4, 1.0]])

        byk = {}
        for p, q, k in self.groups:
            byk.setdefault(k, []).append((p, q))
        byk = {k: (np.vstack([a for a, _ in v]), np.vstack([b for _, b in v])) for k, v in byk.items()}

        def resid(x):
            Hm = mat(x)
            return np.concatenate([((warp_pts(np.linalg.matrix_power(Hm, k), p) - q) / k).ravel()
                                   for k, (p, q) in byk.items()])

        try:
            r = least_squares(resid, x0, loss="huber", f_scale=2.0, max_nfev=30)
            Hn = mat(r.x)
            if np.all(np.isfinite(Hn)):
                self.H = Hn
                self.n_fits += 1
        except Exception:
            pass

    def entry_edge_flow(self):
        """Mean per-frame displacement of the frame centre (source px)."""
        c = np.array([[FW / 2, FH / 2]])
        return (warp_pts(self.H, c) - c)[0]


# --------------------------------------------------------------------------- #
# tracker
# --------------------------------------------------------------------------- #

LEVEL_W = {0: 0.55, 1: 0.9, 2: 1.0}  # how much we trust a detection from each level


class Track:
    _next = 1

    def __init__(self, box, cls_probs, conf, level, frame):
        self.id = Track._next
        Track._next += 1
        self.box = np.asarray(box, np.float64)
        self.cls_ev = np.zeros(NC)
        self.exist_miss = 1.0  # prod(1 - c_i)
        self.hits = 0
        self.miss = 0.0
        self.best_level = level
        self.last_seen = frame
        self.first_seen = frame
        self.partial = False
        self.vres = np.zeros(2)  # residual velocity vs. H prediction (px/frame)
        self.max_conf_l2 = 0.0  # best raw confidence seen at L2 (native resolution)
        self.add(box, cls_probs, conf, level, frame, first=True)

    def add(self, box, cls_probs, conf, level, frame, first=False, box_weight=None):
        w = LEVEL_W[level]
        if not first:
            if box_weight is None:
                box_weight = 0.75 if level >= self.best_level else 0.3
            if box_weight > 0 and level >= 1 and frame > self.last_seen:
                dt = frame - self.last_seen
                box = np.asarray(box, np.float64)
                innov = (box[:2] + box[2:]) / 2 - (self.box[:2] + self.box[2:]) / 2
                self.vres = np.clip(0.5 * self.vres + 0.5 * innov / dt * min(1.0, dt / 3.0), -6, 6)
            self.box = box_weight * np.asarray(box) + (1 - box_weight) * self.box
        self.cls_ev += w * conf * np.asarray(cls_probs)
        self.exist_miss *= (1 - min(0.95, w * conf))
        self.hits += 1
        self.miss = max(0.0, self.miss - 1.0)
        self.best_level = max(self.best_level, level)
        self.last_seen = frame
        if level == 2:
            self.max_conf_l2 = max(self.max_conf_l2, conf)

    # SAFER: precision-first reporting. A track is CONFIRMED once it was detected
    # in >= 2 views, or once at native resolution (L2) with a confident detection.
    # Unconfirmed tracks are still reported, but ranked below every confirmed one
    # (x UNCONF_W): under COCO AP a low-ranked extra box can only add recall, never
    # push a true positive down the ranking.
    CONFIRM_HITS = 2
    L2_CONFIRM_CONF = 0.5
    UNCONF_W = 0.15

    @property
    def confirmed(self):
        return self.hits >= self.CONFIRM_HITS or self.max_conf_l2 >= self.L2_CONFIRM_CONF

    def score(self):
        """Whole-history score: noisy-OR of level-weighted confidences x hit ratio,
        where looks at L1/L2 that missed it (self.miss, level-weighted) count against."""
        p = 1 - self.exist_miss
        ratio = (self.hits + 0.5) / (self.hits + 0.5 + 1.5 * self.miss)
        return p * ratio

    def report_score(self):
        return self.score() * (1.0 if self.confirmed else self.UNCONF_W)

    def class_dist(self):
        s = self.cls_ev.sum()
        return self.cls_ev / s if s > 0 else np.full(NC, 1.0 / NC)


class Tracker:
    def __init__(self, new_thr=0.25, match_iou=0.15):
        self.tracks = []
        self.frame = None
        # DRONE_NEW_THR overrides the track-birth threshold (fable session: the fine-tuned detector's confidences
        # are sharper than v2's, so the 0.25 tuned for v2 + verifier may not be its optimum)
        self.new_thr = float(__import__('os').environ.get('DRONE_NEW_THR', new_thr))
        self.match_iou = match_iou

    def predict(self, frame, Hm):
        if self.frame is None:
            self.frame = frame
            return
        k = frame - self.frame
        if k > 0:
            Hk = mpow(Hm, k)
            for t in self.tracks:
                t.box = warp_box(Hk, t.box)
                t.box[[0, 2]] += t.vres[0] * k
                t.box[[1, 3]] += t.vres[1] * k
        self.frame = frame
        # drop tracks that left the frame
        keep = []
        for t in self.tracks:
            b = t.box
            area = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
            if rect_inter(b, FRAME_RECT) / area > 0.03 and (b[2] - b[0]) < 1500:
                keep.append(t)
        self.tracks = keep

    def update(self, dets, level, region, frame):
        """dets: list of (box_src[4], cls_probs[NC], conf, truncated_bool)."""
        region = tuple(float(v) for v in region)
        tb = np.array([t.box for t in self.tracks]).reshape(-1, 4)
        db = np.array([d[0] for d in dets]).reshape(-1, 4)
        matched_t, matched_d = set(), set()
        if len(tb) and len(db):
            iou = iou_matrix(tb, db)
            # centre-distance fallback for small boxes
            tc = (tb[:, :2] + tb[:, 2:]) / 2
            dc = (db[:, :2] + db[:, 2:]) / 2
            dist = np.linalg.norm(tc[:, None] - dc[None], axis=2)
            size = np.maximum(np.maximum(tb[:, 2] - tb[:, 0], tb[:, 3] - tb[:, 1])[:, None], 16.0)
            near = dist < 0.6 * size + 4 * LEVEL_SCALE[level]
            cost = 1 - iou
            cost[~((iou > self.match_iou) | near)] = 9
            # small class agreement bonus
            for j, d in enumerate(dets):
                for i, t in enumerate(self.tracks):
                    if cost[i, j] < 9:
                        cost[i, j] -= 0.2 * float(np.dot(t.class_dist(), d[1]))
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] < 9:
                    box, probs, conf, trunc = dets[j]
                    t = self.tracks[i]
                    if trunc:
                        t.add(t.box, probs, conf * 0.5, level, frame, box_weight=0.0)
                    else:
                        t.add(box, probs, conf, level, frame)
                    matched_t.add(i)
                    matched_d.add(j)
        # new tracks
        for j, (box, probs, conf, trunc) in enumerate(dets):
            if j in matched_d or conf < self.new_thr or trunc:
                continue
            # avoid spawning on top of an existing track
            if len(tb):
                if iou_matrix(tb, [box]).max() > 0.3:
                    continue
            self.tracks.append(Track(box, probs, conf, level, frame))
        # misses for tracks that should have been visible
        s = LEVEL_SCALE[level]
        margin = 3 * s
        for i, t in enumerate(self.tracks[:len(tb)]):
            if i in matched_t:
                continue
            b = t.box
            inside = b[0] >= region[0] + margin and b[1] >= region[1] + margin and b[2] <= region[2] - margin and b[3] <= region[3] - margin
            size_view = min(b[2] - b[0], b[3] - b[1]) / s
            if inside and size_view >= 5:
                t.miss += LEVEL_W[level] * (1.0 if size_view >= 10 else 0.5)
        # remove dead tracks
        self.tracks = [t for t in self.tracks if not (t.miss >= 2.5 and t.miss > 0.5 * t.hits + 1)]
        self._merge()

    def _merge(self):
        if len(self.tracks) < 2:
            return
        tb = np.array([t.box for t in self.tracks])
        iou = iou_matrix(tb, tb)
        np.fill_diagonal(iou, 0)
        dead = set()
        order = sorted(range(len(self.tracks)), key=lambda i: -self.tracks[i].score())
        for i in order:
            if i in dead:
                continue
            for j in np.where(iou[i] > 0.45)[0]:
                if j in dead or j == i:
                    continue
                a, b = self.tracks[i], self.tracks[j]
                a.cls_ev += b.cls_ev
                a.exist_miss *= b.exist_miss
                a.hits += b.hits
                a.max_conf_l2 = max(a.max_conf_l2, b.max_conf_l2)
                dead.add(j)
        self.tracks = [t for i, t in enumerate(self.tracks) if i not in dead]

    def outputs(self, topk=3, min_conf=0.003):
        """List of (box_src, cls_idx, conf). conf = report_score (confirmation-tiered)."""
        out = []
        for t in self.tracks:
            b = np.array([max(0.0, t.box[0]), max(0.0, t.box[1]), min(float(FW), t.box[2]), min(float(FH), t.box[3])])
            if b[2] - b[0] < 2 or b[3] - b[1] < 2:
                continue
            p = t.report_score()
            cd = t.class_dist()
            order = np.argsort(-cd)[:topk]
            for rank, c in enumerate(order):
                conf = p * cd[c] * (1.0 if rank == 0 else 0.7)
                if conf >= min_conf:
                    out.append((b, int(c), float(conf)))
        return out
