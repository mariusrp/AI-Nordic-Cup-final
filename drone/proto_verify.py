"""I4 foundation-feature verifier: DINOv2-S/14 prototypes re-class and re-score YOLO proposals.

Every L1/L2 detection box (YOLO at conf 0.05) is cropped with context (square, side = CTX x max(w, h),
at least MIN_SIDE view px), resized to 112 px (8x8 tokens) and embedded by a frozen DINOv2-S/14
(CLS token + mean patch token, both L2-normalised). Per class the score is the mean of the top-k cosine
similarities to that class's prototype embeddings; the background score is the top-k mean over a
background bank. p = softmax([class scores, bg score] / T).
    drop the box if p_bg > DROP;   new conf = conf^a * p_cls^(1-a);   class dist = p over prototype classes.
Classes without prototypes (hangar, medium_plane: no fully visible cut-out) keep the YOLO class and are
only subject to the background test. L0 boxes (5-40 source px -> 1-10 view px) pass through unchanged.

Bank build (python proto_verify.py build --out bank.pt [--frames 0-9] [--seen-runs dirA dirB]):
  prototypes: each cut-out of the given Helsinki frames, 8 rotations x 2 flips x 2 levels (L1, L2),
     scale jitter, colour jitter, pasted at its native source scale on a random patch of an inpainted
     Helsinki background (same frames) or a LoveDA tile (x2 upscaled, ~0.15 m/px), rendered like the
     evaluator (INTER_AREA /2 at L1), cropped around the placed box.
  background: v2 detections (conf >= 0.05) on LoveDA views (tiles never used for evaluation), v2 detections
     on the given Helsinki frames' L1/L2 tiles that do not touch a GT box, random object-free Helsinki
     patches; optionally (--seen-runs) v2 detections on recorded validation-city runs (same-scene leak:
     only as a separate, optimistic variant).
Serving: DRONE_VERIFY=/path/bank.pt [DRONE_VERIFY_A=0.5 DRONE_VERIFY_T=0.05 DRONE_VERIFY_DROP=0.5]
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, CLS_INDEX, DRONE_UPSTREAM, LEVEL_SCALE, NC  # noqa: E402

MODEL = "vit_small_patch14_dinov2.lvd142m"
SIZE = 112
CTX = 1.5
MIN_SIDE = 16
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255
SCENE = os.path.join(DRONE_UPSTREAM, "src", "helsinki")
REG = {0: (3840, 2160), 1: (1920, 1080), 2: (960, 540)}


def crop_box(img, box, ctx=CTX, min_side=MIN_SIDE):
    x1, y1, x2, y2 = box
    s = max(ctx * max(x2 - x1, y2 - y1), min_side)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a, b = int(round(cx - s / 2)), int(round(cy - s / 2))
    n = max(2, int(round(s)))
    H, W = img.shape[:2]
    pl, pt, pr, pb = max(0, -a), max(0, -b), max(0, a + n - W), max(0, b + n - H)
    c = img[max(0, b):min(H, b + n), max(0, a):min(W, a + n)]
    if pl or pt or pr or pb:
        c = cv2.copyMakeBorder(c, pt, pb, pl, pr, cv2.BORDER_REFLECT_101 if min(c.shape[:2]) > 1 else cv2.BORDER_REPLICATE)
    return c


class Embedder:
    def __init__(self, device="cuda"):
        import timm
        import torch
        self.torch = torch
        self.dev = device
        self.m = timm.create_model(MODEL, pretrained=True, img_size=SIZE).to(device).half().eval()
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
        self.embed([np.zeros((20, 20, 3), np.uint8)] * 4)

    def embed(self, crops, bs=256):
        torch = self.torch
        out = []
        for i in range(0, len(crops), bs):
            x = np.stack([cv2.resize(c, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC) for c in crops[i:i + bs]])
            with torch.inference_mode():
                t = torch.from_numpy(x).to(self.dev)[..., [2, 1, 0]].permute(0, 3, 1, 2).float()
                t = ((t - self.mean) / self.std).half()
                tok = self.m.forward_features(t)
                cls = torch.nn.functional.normalize(tok[:, 0].float(), dim=1)
                pm = torch.nn.functional.normalize(tok[:, 1 + getattr(self.m, "num_reg_tokens", 0):].float().mean(1), dim=1)
                out.append(torch.nn.functional.normalize(torch.cat([cls, pm], 1), dim=1))
        return torch.cat(out) if out else torch.zeros((0, 768), device=self.dev)


class Verifier:
    def __init__(self, bank, emb=None, a=0.5, T=0.03, drop=0.5, k=3, levels=(1, 2), use_bg=True, bg_key="bg", mode="keep"):
        import torch
        self.torch = torch
        self.emb = emb or Embedder()
        b = torch.load(bank, map_location=self.emb.dev) if isinstance(bank, str) else bank
        self.protos = b["protos"].to(self.emb.dev).float()
        self.pcls = b["proto_cls"].to(self.emb.dev).long()
        self.bg = b[bg_key].to(self.emb.dev).float()
        self.has = torch.zeros(NC, dtype=torch.bool, device=self.emb.dev)
        self.has[self.pcls.unique()] = True
        self.a, self.T, self.drop, self.k, self.levels, self.use_bg = a, T, drop, k, set(levels), use_bg
        self.mode = mode
        self.ms = []

    def probs(self, feats):
        """feats [N, D] -> p [N, NC + 1] (last = background)."""
        torch = self.torch
        S = feats @ self.protos.T  # [N, P]
        cs = torch.full((len(feats), NC), -1e4, device=feats.device)
        for c in self.pcls.unique().tolist():
            s = S[:, self.pcls == c]
            cs[:, c] = s.topk(min(self.k, s.shape[1]), dim=1).values.mean(1)
        if self.use_bg and len(self.bg):
            bs = (feats @ self.bg.T).topk(min(self.k, len(self.bg)), dim=1).values.mean(1, keepdim=True)
        else:
            bs = torch.full((len(feats), 1), -1e4, device=feats.device)
        return torch.softmax(torch.cat([cs, bs], 1) / self.T, 1)

    def __call__(self, img, level, dets):
        if level not in self.levels or not dets:
            return dets
        t0 = time.perf_counter()
        feats = self.emb.embed([crop_box(img, d["box"]) for d in dets])
        P = self.probs(feats).cpu().numpy()
        has = self.has.cpu().numpy()
        out = []
        for d, p in zip(dets, P):
            yc = int(np.argmax(d["probs"]))
            if p[-1] > self.drop:
                continue
            if not has[yc] and d["probs"][yc] >= 0.5:  # class without prototypes: background test only
                out.append(d)
                continue
            if self.mode == "keep":  # keep YOLO's class, re-score by the prototype probability of that class
                out.append({"box": d["box"], "probs": d["probs"], "conf": float(d["conf"] ** self.a * p[yc] ** (1 - self.a))})
                continue
            pc = p[:NC] * has
            s = pc.sum()
            if s <= 0:
                out.append(d)
                continue
            c = int(np.argmax(pc))
            oh = np.zeros(NC); oh[c] = 1.0  # class = argmax over prototypes; scorers use conf * probs[c]
            out.append({"box": d["box"], "probs": oh, "conf": float(d["conf"] ** self.a * p[c] ** (1 - self.a))})
        self.ms.append(1000 * (time.perf_counter() - t0))
        return out


class VerifiedDetector:
    def __init__(self, base, verifier):
        self.base, self.v = base, verifier

    def __call__(self, img, level):
        return self.v(img, level, self.base(img, level))


def from_env(base):
    bank = os.environ.get("DRONE_VERIFY")
    if not bank:
        return base
    v = Verifier(bank, a=float(os.environ.get("DRONE_VERIFY_A", "0.5")), T=float(os.environ.get("DRONE_VERIFY_T", "0.03")),
                 drop=float(os.environ.get("DRONE_VERIFY_DROP", "0.5")), bg_key=os.environ.get("DRONE_VERIFY_BG", "bg"),
                 mode=os.environ.get("DRONE_VERIFY_MODE", "keep"))
    return VerifiedDetector(base, v)


# --------------------------------------------------------------------------- bank build
def helsinki_tiles(level):
    rw, rh = REG[level]
    n = 3840 // rw
    return [(x * rw, y * rh, (x + 1) * rw, (y + 1) * rh) for y in range(n) for x in range(n)]


def render(src, level):
    return cv2.resize(src, (960, 540), interpolation=cv2.INTER_AREA) if level == 1 else src


def loveda_region(path, level, rng):
    """A LoveDA tile upscaled x~2 (0.3 m/px -> ~0.15 m/px source scale), random region of the level's size."""
    t = cv2.imread(path)
    f = float(rng.uniform(1.8, 2.2))
    rw, rh = REG[level]
    f = max(f, rw / t.shape[1] + 1e-3, rh / t.shape[0] + 1e-3)
    t = cv2.resize(t, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    y = int(rng.integers(0, t.shape[0] - rh + 1)); x = int(rng.integers(0, t.shape[1] - rw + 1))
    return np.ascontiguousarray(t[y:y + rh, x:x + rw])


def _parse_frames(s):
    a, b = s.split("-")
    return set(range(int(a), int(b) + 1))


def build(a):
    import torch
    import make_synth as MS
    from detector import YoloDetector
    rng = np.random.default_rng(a.seed)
    frames = _parse_frames(a.frames)
    emb = Embedder()
    meta = [m for m in json.load(open(os.path.join(a.src, "cutouts.json"))) if m["frame"] in frames]
    bgs = [p for p in sorted(glob.glob(os.path.join(a.src, "bg", "*.jpg")))
           if json.load(open(p[:-4] + ".json"))["frame"] in frames]
    bg_imgs = [cv2.imread(p) for p in bgs]
    loveda = sorted(glob.glob(os.path.join(a.loveda, "*.png")))
    print("cutouts", len(meta), "helsinki bg", len(bgs), "loveda", len(loveda), flush=True)
    # ---- prototypes
    crops, cls = [], []
    for m in meta:
        rgba = cv2.imread(os.path.join(a.src, "cutouts", m["name"] + ".png"), cv2.IMREAD_UNCHANGED)
        for level in (1, 2):
            for r in range(8):
                for fl in (0, 1):
                    img = rgba[:, ::-1] if fl else rgba
                    box = m["box"]
                    if fl:
                        w = rgba.shape[1]; box = (w - box[2], box[1], w - box[0], box[3])
                    img = np.ascontiguousarray(img)
                    h, w = img.shape[:2]
                    ang = r * 45 + float(rng.uniform(-10, 10))
                    sc = float(rng.uniform(0.85, 1.15))
                    diag = int(math.ceil(math.hypot(w, h) * sc)) + 2
                    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
                    M[0, 2] += diag / 2 - w / 2; M[1, 2] += diag / 2 - h / 2
                    o = cv2.warpAffine(img, M, (diag, diag), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
                    bm = np.zeros((h, w), np.uint8)
                    bm[int(box[1]):int(math.ceil(box[3])), int(box[0]):int(math.ceil(box[2]))] = 255
                    bm &= (img[..., 3] > 127).astype(np.uint8) * 255
                    rb = cv2.warpAffine(bm, M, (diag, diag), flags=cv2.INTER_NEAREST)
                    ys, xs = np.where(rb > 0)
                    if len(xs) < 3:
                        continue
                    o = np.dstack([MS.jitter_color(o[..., :3], rng, 0.7), o[..., 3]])
                    # native-scale background patch around the object (4x its size, >= 1 view)
                    ps = max(4 * diag, 96 * LEVEL_SCALE[level])
                    ps = min(ps, 1080)
                    if rng.random() < 0.5 and bg_imgs:
                        B = bg_imgs[int(rng.integers(len(bg_imgs)))]
                        y0 = int(rng.integers(0, B.shape[0] - ps)); x0 = int(rng.integers(0, B.shape[1] - ps))
                        canvas = B[y0:y0 + ps, x0:x0 + ps].copy()
                    else:
                        t = loveda_region(loveda[int(rng.integers(len(loveda)))], 1, rng)
                        canvas = t[:ps, :ps].copy()
                    px, py = (ps - diag) // 2 + int(rng.integers(-diag // 4 - 1, diag // 4 + 1)), (ps - diag) // 2 + int(rng.integers(-diag // 4 - 1, diag // 4 + 1))
                    MS.paste(canvas, o, px, py)
                    s = LEVEL_SCALE[level]
                    if s > 1:
                        canvas = cv2.resize(canvas, (ps // s, ps // s), interpolation=cv2.INTER_AREA)
                    vb = [(px + xs.min()) / s, (py + ys.min()) / s, (px + xs.max() + 1) / s, (py + ys.max() + 1) / s]
                    crops.append(crop_box(canvas, vb)); cls.append(CLS_INDEX[m["cls"]])
    protos = emb.embed(crops).cpu()
    print("protos", len(crops), {CLASSES[c]: cls.count(c) for c in sorted(set(cls))}, flush=True)
    if a.sheet:
        cv2.imwrite(a.sheet, cv2.vconcat([cv2.hconcat([cv2.resize(c, (64, 64)) for c in crops[i:i + 32]])
                                          for i in range(0, min(len(crops), 32 * 16), 32) if len(crops[i:i + 32]) == 32]))
    # ---- background bank
    det = YoloDetector(a.weights, conf=0.05)
    bgc = {"loveda": [], "helsinki_fp": [], "helsinki_rand": [], "seen": []}
    for p in loveda[:a.n_loveda]:
        for level in (1, 2):
            v = render(loveda_region(p, level, rng), level)
            for d in det(v, level):
                bgc["loveda"].append(crop_box(v, d["box"]))
    scene = os.path.join(DRONE_UPSTREAM, "src", "helsinki")
    for f in sorted(frames):
        img = cv2.imread(os.path.join(scene, "images", f"frame_{f:06d}.png"))
        gt = [x["bbox"] for x in json.load(open(os.path.join(scene, "annotations", f"frame_{f:06d}.json")))["annotations"]]
        for level in (1, 2):
            s = LEVEL_SCALE[level]
            for t in helsinki_tiles(level):
                v = render(img[t[1]:t[3], t[0]:t[2]], level)
                g = [((b[0] - t[0]) / s, (b[1] - t[1]) / s, (b[2] - t[0]) / s, (b[3] - t[1]) / s) for b in gt]

                def near(bx, pad=4):
                    return any(bx[0] < q[2] + pad and q[0] < bx[2] + pad and bx[1] < q[3] + pad and q[1] < bx[3] + pad for q in g)
                for d in det(v, level):
                    if not near(d["box"]):
                        bgc["helsinki_fp"].append(crop_box(v, d["box"]))
                for _ in range(6):
                    w = float(rng.uniform(6, 40)); x = float(rng.uniform(0, 960 - w)); y = float(rng.uniform(0, 540 - w))
                    bx = (x, y, x + w, y + w * float(rng.uniform(0.6, 1.6)))
                    if not near(bx, 8):
                        bgc["helsinki_rand"].append(crop_box(v, bx))
    for run in a.seen_runs or []:
        for p in sorted(glob.glob(os.path.join(run, "*_L[12]_*.png"))):
            level = int(os.path.basename(p).split("_L")[1][0])
            v = cv2.imread(p)
            for d in det(v, level):
                if d["conf"] >= 0.1:
                    bgc["seen"].append(crop_box(v, d["box"]))
    out = {"protos": protos.half(), "proto_cls": torch.tensor(cls), "frames": sorted(frames), "model": MODEL}
    for k, v in bgc.items():
        if len(v) > a.max_bg:
            v = [v[i] for i in rng.choice(len(v), a.max_bg, replace=False)]
        out["bg_" + k] = emb.embed(v).cpu().half() if v else torch.zeros((0, 768)).half()
        print("bg", k, len(v), flush=True)
    out["bg"] = torch.cat([out["bg_" + k] for k in ("loveda", "helsinki_fp", "helsinki_rand")])
    out["bg_with_seen"] = torch.cat([out["bg"], out["bg_seen"]])
    torch.save(out, a.out)
    print("saved", a.out, "bg", len(out["bg"]), "bg+seen", len(out["bg_with_seen"]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--src", default="/workspace/g1b/synth_src_sam")
    b.add_argument("--loveda", default="/workspace/g1b/loveda_mine")
    b.add_argument("--weights", default="/workspace/drone_weights/v2.pt")
    b.add_argument("--frames", default="0-19")
    b.add_argument("--seen-runs", nargs="*")
    b.add_argument("--n-loveda", type=int, default=300)
    b.add_argument("--max-bg", type=int, default=2500)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--sheet", default=None)
    a = ap.parse_args()
    build(a)
