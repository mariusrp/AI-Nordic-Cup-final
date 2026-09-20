"""Detectors: Ultralytics YOLO on the 960x540 view, or a GT-based simulator.

Every detector returns a list of dicts in VIEW pixels:
    {"box": (x1, y1, x2, y2), "probs": np.ndarray[NC], "conf": float}
where overlapping boxes of different classes have already been merged into one
detection carrying a class distribution (the tracker accumulates it).
"""
import json
import os

import numpy as np

from common import CLASSES, CLS_INDEX, DRONE_UPSTREAM, LEVEL_SCALE, NC, VIEW_H, VIEW_W


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def merge_classes(raw, iou_thr=0.55):
    """raw: list of (box, cls, conf) sorted any way -> merged detections."""
    raw = sorted(raw, key=lambda r: -r[2])
    out = []
    for box, c, conf in raw:
        for d in out:
            if _iou(d["box"], box) > iou_thr:
                d["probs"][c] = max(d["probs"][c], conf)
                break
        else:
            p = np.zeros(NC)
            p[c] = conf
            out.append({"box": tuple(float(v) for v in box), "probs": p, "conf": conf})
    for d in out:
        s = d["probs"].sum()
        d["conf"] = float(min(1.0, d["probs"].max() + 0.5 * (s - d["probs"].max())))
        d["probs"] = d["probs"] / s
    return out


class YoloDetector:
    def __init__(self, weights, imgsz=1280, conf=0.05, device=None, half=True, max_det=150):
        self.max_det = max_det
        from ultralytics import YOLO
        self.model = YOLO(weights, task="detect")
        self.imgsz = imgsz
        self.conf = conf
        try:
            import torch
            cuda = torch.cuda.is_available()
        except Exception:
            cuda = False
        self.device = device if device is not None else (0 if cuda else "cpu")
        self.half = half and cuda and str(weights).endswith(".pt")
        # map model class names to our indices (robust to ordering)
        names = self.model.names
        self.cmap = {int(k): CLS_INDEX.get(v, -1) for k, v in names.items()}
        self.cmap_arr = np.array([self.cmap.get(i, -1) for i in range(max(self.cmap) + 1)])
        # fast path: raw torch model + GPU resize + GPU NMS (the Results/predictor path costs
        # 50-250 ms of CPU on the throttled pod)
        self.fast = False
        if os.environ.get("DRONE_FAST", "1") == "1" and self.device != "cpu":
            try:
                import torch
                from ultralytics.utils.nms import non_max_suppression
                self._nms = non_max_suppression
                self.torch = torch
                self.net = self.model.model.to(f"cuda:{self.device}" if isinstance(self.device, int) else self.device).eval()
                self.net = self.net.half() if self.half else self.net.float()
                self.fast = True
            except Exception:
                self.fast = False
        self.warmup()

    def warmup(self, n=3):
        img = np.zeros((VIEW_H, VIEW_W, 3), np.uint8)
        for _ in range(n):
            self(img, 1)

    def _fast(self, img_bgr):
        torch = self.torch
        h, w = img_bgr.shape[:2]
        sc = self.imgsz / max(h, w)
        nw, nh = int(round(w * sc)), int(round(h * sc))
        ph = (32 - nh % 32) % 32
        pw = (32 - nw % 32) % 32
        dev = next(self.net.parameters()).device
        with torch.inference_mode():
            x = torch.from_numpy(img_bgr).to(dev, non_blocking=True)
            x = x[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float()
            x = torch.nn.functional.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
            x = torch.nn.functional.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), value=114.0) / 255.0
            x = x.half() if self.half else x
            pred = self.net(x)
            pred = pred[0] if isinstance(pred, (list, tuple)) else pred
            out = self._nms(pred.float(), self.conf, 0.6, max_det=self.max_det)[0].cpu().numpy()
        if len(out) == 0:
            return []
        xyxy = out[:, :4].copy()
        xyxy[:, [0, 2]] -= pw // 2
        xyxy[:, [1, 3]] -= ph // 2
        xyxy /= sc
        cls = self.cmap_arr[out[:, 5].astype(int)]
        raw = [(xyxy[i], int(cls[i]), float(out[i, 4])) for i in range(len(out)) if cls[i] >= 0]
        return merge_classes(raw)

    def __call__(self, img_bgr, level):
        if self.fast:
            try:
                return self._fast(img_bgr)
            except Exception:
                self.fast = False
        r = self.model.predict(img_bgr, imgsz=self.imgsz, conf=self.conf, iou=0.6, device=self.device,
                               half=self.half, verbose=False, max_det=300, rect=True)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return []
        xyxy = b.xyxy.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int)
        cf = b.conf.cpu().numpy()
        raw = [(xyxy[i], self.cmap.get(int(cls[i]), -1), float(cf[i])) for i in range(len(cf))]
        raw = [r_ for r_ in raw if r_[1] >= 0]
        return merge_classes(raw)


class OracleDetector:
    """Simulated detector from local GT (for tuning tracker + camera only).

    P(detect) grows with the object's short side in VIEW pixels; box noise and
    class confusion shrink with size. Deterministic per (frame, object, level).
    """

    def __init__(self, scene_dir=None, p50=7.0, noise=0.6, fp_rate=0.3):
        scene_dir = scene_dir or os.environ.get("DRONE_ORACLE_SCENE") or os.path.join(DRONE_UPSTREAM, "src", "helsinki")
        self.gt = {}
        adir = os.path.join(scene_dir, "annotations")
        for f in os.listdir(adir):
            d = json.load(open(os.path.join(adir, f)))
            self.gt[d["frame"]] = d["annotations"]
        self.p50, self.noise, self.fp_rate = p50, noise, fp_rate
        self.fp_conf = float(os.environ.get("DRONE_ORACLE_FPCONF", "0.4"))  # max FP confidence
        self.frame = 0
        self.region = (0, 0, 3840, 2160)

    def set_context(self, frame, region):
        self.frame, self.region = frame, region

    def __call__(self, img_bgr, level):
        s = LEVEL_SCALE[level]
        rx, ry = self.region[0], self.region[1]
        rng = np.random.default_rng(self.frame * 7919 + level)
        out = []
        for a in self.gt.get(self.frame, []):
            x1, y1, x2, y2 = a["bbox"]
            vx1, vy1 = (max(x1, rx) - rx) / s, (max(y1, ry) - ry) / s
            vx2, vy2 = (min(x2, self.region[2]) - rx) / s, (min(y2, self.region[3]) - ry) / s
            if vx2 - vx1 < 1 or vy2 - vy1 < 1:
                continue
            short = min(vx2 - vx1, vy2 - vy1)
            p = 1 / (1 + np.exp(-(short - self.p50) / 1.5))
            if rng.random() > p:
                continue
            n = self.noise * (1 + 4 / max(short, 1))
            box = (vx1 + rng.normal(0, n), vy1 + rng.normal(0, n), vx2 + rng.normal(0, n), vy2 + rng.normal(0, n))
            probs = np.zeros(NC)
            c = CLS_INDEX[a["object_id"]]
            conf_ok = 0.3 + 0.65 * p
            if rng.random() < 0.3 / max(1.0, short / 6):
                c2 = int(rng.integers(NC))
                probs[c2] = 0.6
                probs[c] = 0.4
            else:
                probs[c] = 1.0
            out.append({"box": box, "probs": probs / probs.sum(), "conf": float(conf_ok * rng.uniform(0.8, 1.0))})
        # false positives
        for _ in range(rng.poisson(self.fp_rate)):
            x, y = rng.uniform(0, VIEW_W - 20), rng.uniform(0, VIEW_H - 20)
            w = rng.uniform(6, 30)
            probs = np.zeros(NC)
            probs[rng.integers(NC)] = 1
            out.append({"box": (x, y, x + w, y + w), "probs": probs, "conf": float(rng.uniform(0.05, self.fp_conf))})
        return out


class RoutedDetector:
    """Two YOLOs with complementary classes (g2-2): detections whose argmax class is in `route` come from `b`,
    all other classes from `a`. (valcity: v2 keeps hangar/small_tower/tank/ta-ta, r11 finds small_plane/helicopter/jet_plane.)"""

    def __init__(self, a, b, route):
        self.a, self.b, self.route = a, b, set(route)

    def __call__(self, img, level):
        da = [d for d in self.a(img, level) if int(np.argmax(d["probs"])) not in self.route]
        db = [d for d in self.b(img, level) if int(np.argmax(d["probs"])) in self.route]
        return da + db


class UnionDetector:
    """Union of two detectors (fable session): the boxes of `a` and `b` are pooled and merged like one detector's raw
    output (merge_classes: overlapping boxes of different classes become one detection with a class distribution,
    same-class overlaps keep the higher confidence). Used to hedge the flown-city fine-tune with the production
    routed stack: production's honest recall is kept, the fine-tune's is added. Costs one more YOLO pass."""

    def __init__(self, a, b):
        self.a, self.b = a, b

    def __call__(self, img, level):
        raw = []
        for d in self.a(img, level) + self.b(img, level):
            raw.append((np.asarray(d["box"], dtype=float), int(np.argmax(d["probs"])), float(d["conf"])))
        return merge_classes(raw) if raw else []


def build_detector(weights=None, weights2=None):
    """Production stack from env; `weights`/`weights2` override DRONE_WEIGHTS/DRONE_WEIGHTS2 (split-frame B arm).
    weights2="" disables routing for that arm."""
    kind = os.environ.get("DRONE_DETECTOR", "yolo")
    if kind == "oracle":
        return OracleDetector()
    if kind == "none":
        return None
    weights = weights or os.environ.get("DRONE_WEIGHTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "best.pt"))
    weights2 = os.environ.get("DRONE_WEIGHTS2") if weights2 is None else weights2
    imgsz = int(os.environ.get("DRONE_IMGSZ", "1280"))
    det = YoloDetector(weights, imgsz=imgsz)
    if weights2:  # g2-2 class-routed second detector (e.g. r11 for the plane classes)
        route = os.environ.get("DRONE_ROUTE2", "helicopter,jet_plane,small_plane").split(",")
        det = RoutedDetector(det, YoloDetector(weights2, imgsz=imgsz), [CLS_INDEX[c] for c in route])
    if os.environ.get("DRONE_UNION"):  # hedge: pool the routed stack with another detector's boxes (fine-tune)
        det = UnionDetector(det, YoloDetector(os.environ["DRONE_UNION"], imgsz=imgsz))
    if os.environ.get("DRONE_VERIFY"):  # I4: DINOv2 prototype verifier on L1/L2 boxes (proto_verify.py)
        import proto_verify
        det = proto_verify.from_env(det)
    return det
