"""Camera policy: greedy information-gain planner on a coverage map.

We keep, for a coarse grid over the source frame, q = how well the ground now
under each cell has already been inspected (0 = never; QUALITY[level] after a
look). Every frame the map is warped with the estimated motion H, so ground
entering the frame arrives with q = 0, and q slowly decays (re-looks help find
misses and refine boxes). The value of looking at a cell is

    (QUALITY[level] - q)+  *  remaining frames the cell stays in view

(an object found there is reported for the rest of its life). Candidates are
all legal centres (from camera_constraints) on a grid; we pick the best view
for the NEXT frame (map advanced by one step). Everything returned is legal by
construction: integer centres inside the level bounds, within the maximum
centre distance, level from allowed_resolution_levels.
"""
import math

import cv2
import numpy as np

from common import H as FH, W as FW
from tracker import mpow, warp_pts

CELL = 60
GW, GH = FW // CELL, FH // CELL  # 64 x 36
QUALITY = {0: 0.3, 1: 0.85, 2: 1.0}
# DRONE_Q0/Q1/Q2 override the per-level coverage quality (fable session): a lower Q1 makes an L1 look leave
# more value on the table, so the greedy planner spends more frames on L2 re-looks of L1-seen ground
# (objects 8-15 px at L1 are 15-30 px at L2; production makes ~0 L2 looks at the small ground classes).
for _L in (0, 1, 2):
    _v = __import__("os").environ.get(f"DRONE_Q{_L}")
    if _v:
        QUALITY[_L] = float(_v)
REGION = {0: (3840, 2160), 1: (1920, 1080), 2: (960, 540)}
DECAY = 0.96
MAX_LIFE = 30
DENSITY = 0.006
VERIFY_W = float(__import__("os").environ.get("DRONE_VERIFY_W", "0.1"))
ALLOWED_FROM = {0: (0, 1), 1: (0, 1, 2), 2: (1, 2)}
MAXD = {0: 2203.0, 1: 1102.0, 2: 551.0}  # prior P(object centre in a 60px cell)


class CameraPlanner:
    def __init__(self):
        self.q = np.zeros((GH, GW), np.float32)
        self.frame = None
        self.last_level = 0
        self.l0_age = 0  # frames since last full view
        self.S = np.diag([1.0 / CELL, 1.0 / CELL, 1.0])
        self.Si = np.linalg.inv(self.S)
        self._life = None
        self._life_H = None

    # ---- map maintenance ---------------------------------------------------
    def _warp(self, q, Hk):
        Hg = self.S @ Hk @ self.Si
        # cell centres convention: grid index i <-> source (i+0.5)*CELL; the
        # half-cell offset is ignored (sub-cell).
        return cv2.warpPerspective(q, Hg, (GW, GH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def advance(self, frame, Hm):
        if self.frame is not None and frame > self.frame:
            k = frame - self.frame
            self.q = self._warp(self.q, mpow(Hm, k)) * (DECAY ** k)
        self.frame = frame

    def observe(self, level, region):
        x1, y1, x2, y2 = region
        gx1, gy1 = int(math.ceil(x1 / CELL)), int(math.ceil(y1 / CELL))
        gx2, gy2 = int(x2 // CELL), int(y2 // CELL)
        sub = self.q[gy1:gy2, gx1:gx2]
        np.maximum(sub, QUALITY[level], out=sub)
        self.last_level = level
        self.l0_age = 0 if level == 0 else self.l0_age + 1

    def life(self, Hm):
        """Frames each cell stays inside the frame (cap MAX_LIFE)."""
        if self._life is not None and self._life_H is not None and np.allclose(self._life_H, Hm, atol=1e-9):
            return self._life
        ys, xs = np.mgrid[0:GH, 0:GW]
        pts = np.stack([(xs.ravel() + 0.5) * CELL, (ys.ravel() + 0.5) * CELL], 1)
        life = np.full(len(pts), MAX_LIFE, np.float32)
        alive = np.ones(len(pts), bool)
        p = pts.copy()
        for k in range(1, MAX_LIFE + 1):
            p = warp_pts(Hm, p)
            out = (p[:, 0] < 0) | (p[:, 0] > FW) | (p[:, 1] < 0) | (p[:, 1] > FH)
            newly = out & alive
            life[newly] = k
            alive &= ~out
        self._life = life.reshape(GH, GW)
        self._life_H = Hm.copy()
        return self._life

    # ---- planning ------------------------------------------------------------
    def track_bonus(self, tracks, frame, Hm, motion_valid):
        """Per-level grid of value for re-observing tracks whose propagated box
        may have drifted (older + smaller = riskier), weighted by remaining life."""
        life = self.life(Hm) if motion_valid else np.full((GH, GW), MAX_LIFE / 2, np.float32)
        out = {L: np.zeros((GH, GW), np.float32) for L in (0, 1, 2)}
        for t in tracks:
            b = t.box
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            gx, gy = int(cx // CELL), int(cy // CELL)
            if not (0 <= gx < GW and 0 <= gy < GH):
                continue
            size = max(4.0, min(b[2] - b[0], b[3] - b[1]))
            age = frame + 1 - t.last_seen
            drift = 0.8 * age
            fail = float(np.clip((drift - 0.12 * size) / (0.25 * size), 0, 1))
            cls_unsure = 1.0 - float(t.class_dist().max())
            v = (0.6 * fail + 0.5 * cls_unsure) * life[gy, gx] * min(1.0, t.score() + 0.2)
            for L, s in ((0, 4), (1, 2), (2, 1)):
                if size / s >= 8:
                    out[L][gy, gx] += v * (0.7 if L == 0 else 1.0)
            # SAFER verification: an unconfirmed candidate is worth a closer look
            # (L2, or L1 if it was only seen at L0) to confirm or kill it.
            if not t.confirmed:
                ver = VERIFY_W * life[gy, gx] * min(1.0, t.score() + 0.1)
                out[2][gy, gx] += ver
                if t.best_level == 0:
                    out[1][gy, gx] += 0.6 * ver
        return out

    def plan(self, request, Hm, motion_valid, track_bonus=None, alt=None):
        """Return (level, cx, cy) for the next frame, guaranteed legal.

        alt = (level, cx, cy) of our last requested view. When responses are slow
        the evaluator renders the next frame before our previous command lands,
        so the camera may already be at `alt` when this command is judged; we then
        only pick commands that are legal from BOTH positions."""
        if alt is not None and tuple(alt) != (request.view.resolution_level, request.view.center_x, request.view.center_y):
            r = self._plan(request, Hm, motion_valid, track_bonus, alt)
            if r is not None:
                return r
        return self._plan(request, Hm, motion_valid, track_bonus, None)

    def _plan(self, request, Hm, motion_valid, track_bonus=None, alt=None):
        view = request.view
        cons = request.camera_constraints
        cur_level, ccx, ccy = view.resolution_level, view.center_x, view.center_y
        allowed = list(cons.allowed_resolution_levels)
        max_d = float(cons.maximum_center_delta)

        q_next = self._warp(self.q, Hm) * DECAY if motion_valid else self.q * DECAY
        life = self.life(Hm) if motion_valid else np.full((GH, GW), MAX_LIFE / 2, np.float32)
        best = None
        for L in allowed:
            if alt is not None and L not in ALLOWED_FROM[alt[0]]:
                continue
            b = cons.bounds_for_level(L)
            if b is None:
                continue
            val = np.clip(QUALITY[L] - q_next, 0, None) * life * DENSITY
            if track_bonus is not None:
                val = val + track_bonus.get(L, 0)
            ii = cv2.integral(val.astype(np.float64))
            rw, rh = REGION[L]
            if L == 0:
                cands = [(1920, 1080)] if (b.minimum_center_x <= 1920 <= b.maximum_center_x) else []
            else:
                xs = np.unique(np.clip(np.arange(b.minimum_center_x, b.maximum_center_x + 1, CELL // 2), b.minimum_center_x, b.maximum_center_x))
                ys = np.unique(np.clip(np.arange(b.minimum_center_y, b.maximum_center_y + 1, CELL // 2), b.minimum_center_y, b.maximum_center_y))
                xs = np.unique(np.append(xs, [b.maximum_center_x, min(max(ccx, b.minimum_center_x), b.maximum_center_x)]))
                ys = np.unique(np.append(ys, [b.maximum_center_y, min(max(ccy, b.minimum_center_y), b.maximum_center_y)]))
                gx, gy = np.meshgrid(xs, ys)
                d = np.hypot(gx - ccx, gy - ccy)
                ok = d <= max_d - 1e-6
                if alt is not None:
                    ok &= np.hypot(gx - alt[1], gy - alt[2]) <= MAXD[alt[0]] - 1e-6
                cands = list(zip(gx[ok].tolist(), gy[ok].tolist()))
            if not cands:
                continue
            cxy = np.array(cands, np.float64)
            x1 = cxy[:, 0] - rw // 2
            y1 = cxy[:, 1] - rh // 2
            gx1 = np.clip(np.round(x1 / CELL), 0, GW).astype(int)
            gy1 = np.clip(np.round(y1 / CELL), 0, GH).astype(int)
            gx2 = np.clip(np.round((x1 + rw) / CELL), 0, GW).astype(int)
            gy2 = np.clip(np.round((y1 + rh) / CELL), 0, GH).astype(int)
            sc = ii[gy2, gx2] - ii[gy1, gx2] - ii[gy2, gx1] + ii[gy1, gx1]
            # tie-break: prefer small moves
            sc = sc - 1e-6 * np.hypot(cxy[:, 0] - ccx, cxy[:, 1] - ccy)
            j = int(np.argmax(sc))
            if best is None or sc[j] > best[0] + 1e-9:
                best = (float(sc[j]), L, int(cxy[j, 0]), int(cxy[j, 1]))
        if best is None:
            return None
        return best[1], best[2], best[3]
