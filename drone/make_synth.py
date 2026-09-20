"""Synthetic training data for the drone-flyby detector.

Steps
  prep : cut every fully-visible annotated object out of the 4K Helsinki frames
         (GrabCut mask, ellipse fallback), and build object-free backgrounds by
         inpainting the annotated boxes.
  gen  : compose 4K canvases (background + pasted objects, rotated / flipped /
         rescaled / colour-jittered, optionally keeping the real objects), then
         render 960x540 camera views at resolution levels 0/1/2 exactly like the
         evaluator does (crop + INTER_AREA downscale by 4/2/1) and write a YOLO
         dataset.

Split: frames in HOLDOUT_FRAMES are never used for training cut-outs or
backgrounds; they only feed the `val` split (still the same 16 instances, so
the val number is optimistic; the platform validation is the real test).

  python make_synth.py prep --out data/synth_src
  python make_synth.py gen  --src data/synth_src --out data/yolo --canvases 1500 --workers 8
  # optional: extra background canvases (e.g. recorded validation views)
  python make_synth.py gen ... --extra-bg /workspace/drone_bg
"""
import argparse
import glob
import json
import math
import os
import random
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, CLS_INDEX, DRONE_UPSTREAM, H as FH, W as FW  # noqa: E402

SCENE = os.path.join(DRONE_UPSTREAM, "src", "helsinki")
HOLDOUT_FRAMES = {4, 12, 20}
LEVEL_REGION = {0: (3840, 2160), 1: (1920, 1080), 2: (960, 540)}


def load_scene():
    frames = []
    for f in sorted(glob.glob(os.path.join(SCENE, "annotations", "*.json"))):
        d = json.load(open(f))
        frames.append((d["frame"], os.path.join(SCENE, "images", f"frame_{d['frame']:06d}.png"), d["annotations"]))
    return frames


# --------------------------------------------------------------------------- #
# prep
# --------------------------------------------------------------------------- #

def object_mask(crop, box_in_crop):
    """Soft foreground mask for an object inside `crop` (box given in crop px)."""
    h, w = crop.shape[:2]
    x1, y1, x2, y2 = box_in_crop
    bw, bh = x2 - x1, y2 - y1
    try:
        m = np.zeros((h, w), np.uint8)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(crop, m, (x1, y1, max(1, bw), max(1, bh)), bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        fg = np.where((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        frac = fg[y1:y2, x1:x2].mean() / 255
        if not (0.04 < frac < 0.97):
            raise ValueError
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        # keep largest components that overlap the box
        n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
        keep = np.zeros_like(fg)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= 0.01 * bw * bh:
                keep[lab == i] = 255
        fg = keep
        ys, xs = np.where(fg[y1:y2, x1:x2] > 0)
        if len(xs) == 0 or (xs.max() - xs.min() + 1) < 0.6 * bw or (ys.max() - ys.min() + 1) < 0.6 * bh:
            raise ValueError  # mask does not span the annotated box
        mode = "grabcut"
    except Exception:
        fg = np.zeros((h, w), np.uint8)
        cv2.ellipse(fg, (int((x1 + x2) / 2), int((y1 + y2) / 2)), (max(1, int(bw * 0.55)), max(1, int(bh * 0.55))), 0, 0, 360, 255, -1)
        mode = "ellipse"
    # soften / slightly grow so object edges carry a bit of their own ground
    fg = cv2.dilate(fg, np.ones((3, 3), np.uint8))
    soft = cv2.GaussianBlur(fg, (5, 5), 0)
    return soft, mode


def prep(out):
    os.makedirs(os.path.join(out, "cutouts"), exist_ok=True)
    os.makedirs(os.path.join(out, "bg"), exist_ok=True)
    meta = []
    for frame, img_path, anns in load_scene():
        img = cv2.imread(img_path)
        split = "val" if frame in HOLDOUT_FRAMES else "train"
        mask_all = np.zeros(img.shape[:2], np.uint8)
        for k, a in enumerate(anns):
            x1, y1, x2, y2 = a["bbox"]
            bw, bh = x2 - x1, y2 - y1
            pad = int(max(6, 0.25 * max(bw, bh)))
            cv2.rectangle(mask_all, (max(0, x1 - pad), max(0, y1 - pad)), (min(FW - 1, x2 + pad), min(FH - 1, y2 + pad)), 255, -1)
            if x1 <= 1 or y1 <= 1 or x2 >= FW - 1 or y2 >= FH - 1:
                continue  # truncated at frame edge
            cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
            cx2, cy2 = min(FW, x2 + pad), min(FH, y2 + pad)
            crop = img[cy1:cy2, cx1:cx2].copy()
            box = (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)
            m, mode = object_mask(crop, box)
            name = f"{split}_{frame:03d}_{k:02d}_{a['object_id']}"
            cv2.imwrite(os.path.join(out, "cutouts", name + ".png"), np.dstack([crop, m]))
            meta.append(dict(name=name, split=split, frame=frame, cls=a["object_id"], box=box, mode=mode))
        # inpainted background (downscale for inpaint speed, then paste back only the holes)
        small = cv2.resize(img, (FW // 2, FH // 2), interpolation=cv2.INTER_AREA)
        msmall = cv2.resize(mask_all, (FW // 2, FH // 2), interpolation=cv2.INTER_NEAREST)
        inp = cv2.inpaint(small, msmall, 7, cv2.INPAINT_TELEA)
        up = cv2.resize(inp, (FW, FH), interpolation=cv2.INTER_LINEAR)
        mm = (cv2.GaussianBlur(mask_all, (9, 9), 0).astype(np.float32) / 255)[..., None]
        bg = (img * (1 - mm) + up * mm).astype(np.uint8)
        cv2.imwrite(os.path.join(out, "bg", f"{split}_{frame:03d}.jpg"), bg, [cv2.IMWRITE_JPEG_QUALITY, 95])
        json.dump(dict(frame=frame, split=split, anns=anns), open(os.path.join(out, "bg", f"{split}_{frame:03d}.json"), "w"))
        print("frame", frame, split, len(anns), flush=True)
    json.dump(meta, open(os.path.join(out, "cutouts.json"), "w"), indent=0)
    modes = [m["mode"] for m in meta]
    print("cutouts", len(meta), {m: modes.count(m) for m in set(modes)})


# --------------------------------------------------------------------------- #
# gen
# --------------------------------------------------------------------------- #

def jitter_color(img, rng, strength=1.0):
    img = img.astype(np.float32)
    hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8) * strength) % 180
    hsv[..., 1] *= rng.uniform(1 - 0.35 * strength, 1 + 0.35 * strength)
    hsv[..., 2] *= rng.uniform(1 - 0.3 * strength, 1 + 0.25 * strength)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    out = out * rng.uniform(0.9, 1.1) + rng.uniform(-12, 12) * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def transform_cutout(rgba, box, rng, scale):
    """Random flip / rotation / scale. Returns (rgba, tight box in new px)."""
    img = rgba
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]
    if rng.random() < 0.5:
        img = img[:, ::-1]
        x1, x2 = w - x2, w - x1
    if rng.random() < 0.6:
        # exact: multiples of 90 degrees, box stays tight
        k = rng.integers(4)
        for _ in range(k):
            h, w = img.shape[:2]
            img = np.rot90(img)  # counter-clockwise: (x, y) -> (y, w - x)
            x1, y1, x2, y2 = y1, w - x2, y2, w - x1
        img = np.ascontiguousarray(img)
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            x1, y1, x2, y2 = (v * scale for v in (x1, y1, x2, y2))
        return img, (x1, y1, x2, y2), False
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    ang = rng.uniform(0, 360)
    diag = int(math.ceil(math.hypot(w, h))) + 2
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, scale)
    M[0, 2] += diag * scale / 2 - w / 2
    M[1, 2] += diag * scale / 2 - h / 2
    size = int(math.ceil(diag * scale))
    out = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    # tight box = bbox of the rotated mask restricted to the rotated original box
    boxmask = np.zeros((h, w), np.uint8)
    boxmask[int(y1):int(math.ceil(y2)), int(x1):int(math.ceil(x2))] = 255
    boxmask &= (img[..., 3] > 127).astype(np.uint8) * 255
    rb = cv2.warpAffine(boxmask, M, (size, size), flags=cv2.INTER_NEAREST)
    ys, xs = np.where(rb > 0)
    if len(xs) < 3:
        return None, None, True
    return out, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), True


def paste(canvas, rgba, x, y):
    h, w = rgba.shape[:2]
    H_, W_ = canvas.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W_, x + w), min(H_, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    src = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    a = src[..., 3:4].astype(np.float32) / 255
    dst = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = (src[..., :3] * a + dst * (1 - a)).astype(np.uint8)


class Gen:
    def __init__(self, src, split, extra_bg=None, generic_bg=None, p_generic=0.5):
        self.generic = sorted(glob.glob(os.path.join(generic_bg, "*.png")) + glob.glob(os.path.join(generic_bg, "*.jpg"))) if generic_bg else []
        self.p_generic = p_generic
        self.src = src
        self.split = split
        meta = json.load(open(os.path.join(src, "cutouts.json")))
        self.cut = {}
        for m in meta:
            if m["split"] != split:
                continue
            rgba = cv2.imread(os.path.join(src, "cutouts", m["name"] + ".png"), cv2.IMREAD_UNCHANGED)
            self.cut.setdefault(m["cls"], []).append((rgba, tuple(m["box"])))
        self.bgs = sorted(glob.glob(os.path.join(src, "bg", f"{split}_*.jpg")))
        self.extra = []
        if extra_bg:
            for d in extra_bg:
                self.extra += sorted(glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png")))
        self.classes = [c for c in CLASSES if c in self.cut]

    def canvas(self, rng):
        """Returns (canvas BGR at source resolution, labels list, native_scale)."""
        labels = []
        native = 1
        if self.generic and rng.random() < self.p_generic:
            # generic aerial tiles (LoveDA 0.3 m/px): upscale ~2x to our ~0.14 m/px, 2x2 mosaic, crop to 4K
            f = float(rng.uniform(1.6, 2.4))
            ts = int(1024 * f)
            big = np.zeros((2 * ts, 2 * ts, 3), np.uint8)
            for i in range(2):
                for j in range(2):
                    t = cv2.imread(self.generic[rng.integers(len(self.generic))])
                    if t is None:
                        continue
                    t = cv2.resize(t, (ts, ts), interpolation=cv2.INTER_CUBIC)
                    if rng.random() < 0.5:
                        t = t[:, ::-1]
                    t = np.rot90(t, int(rng.integers(4)))
                    big[i * ts:(i + 1) * ts, j * ts:(j + 1) * ts] = t
            y0 = int(rng.integers(0, max(1, 2 * ts - FH)))
            x0 = int(rng.integers(0, max(1, 2 * ts - FW)))
            img = np.ascontiguousarray(big[y0:y0 + FH, x0:x0 + FW])
            if img.shape[0] < FH or img.shape[1] < FW:
                img = cv2.resize(img, (FW, FH))
            img = jitter_color(img, rng, strength=rng.uniform(0.3, 1.0))
            return img, labels, native
        if self.extra and rng.random() < 0.35:
            p = self.extra[rng.integers(len(self.extra))]
            img = cv2.imread(p)
            # file name convention from recorder: *_L{level}_*  -> native scale 4/2/1
            native = 4 if "_L0_" in p else 2 if "_L1_" in p else 1
            if native > 1:
                img = cv2.resize(img, None, fx=native, fy=native, interpolation=cv2.INTER_CUBIC)
        else:
            p = self.bgs[rng.integers(len(self.bgs))]
            keep_real = rng.random() < 0.5
            if keep_real:
                img = cv2.imread(os.path.join(SCENE, "images", "frame_%06d.png" % int(os.path.basename(p)[len(self.split) + 1:][:3])))
                for a in json.load(open(p[:-4] + ".json"))["anns"]:
                    labels.append((CLS_INDEX[a["object_id"]], tuple(float(v) for v in a["bbox"])))
            else:
                img = cv2.imread(p)
            # whole-canvas flips (+ labels)
            if rng.random() < 0.5:
                img = img[:, ::-1]
                labels = [(c, (FW - b[2], b[1], FW - b[0], b[3])) for c, b in labels]
            if rng.random() < 0.5:
                img = img[::-1]
                labels = [(c, (b[0], FH - b[3], b[2], FH - b[1])) for c, b in labels]
            img = np.ascontiguousarray(img)
        img = jitter_color(img, rng, strength=rng.uniform(0.3, 1.0))
        return img, labels, native

    def compose(self, rng):
        img, labels, native = self.canvas(rng)
        Hc, Wc = img.shape[:2]
        n = int(rng.integers(4, 22))
        placed = [b for _, b in labels]
        for _ in range(n):
            cls = self.classes[rng.integers(len(self.classes))]
            rgba, box = self.cut[cls][rng.integers(len(self.cut[cls]))]
            scale = float(np.exp(rng.uniform(np.log(0.7), np.log(1.4))))
            obj, b, _ = transform_cutout(rgba, box, rng, scale)
            if obj is None:
                continue
            # per-object colour jitter (lighting differences), alpha kept
            obj = np.dstack([jitter_color(obj[..., :3], rng, 0.4), obj[..., 3]])
            oh, ow = obj.shape[:2]
            for _try in range(10):
                x = int(rng.integers(-ow // 3, Wc - 2 * ow // 3))
                y = int(rng.integers(-oh // 3, Hc - 2 * oh // 3))
                gb = (x + b[0], y + b[1], x + b[2], y + b[3])
                if all(not (gb[0] < q[2] + 8 and q[0] < gb[2] + 8 and gb[1] < q[3] + 8 and q[1] < gb[3] + 8) for q in placed):
                    break
            else:
                continue
            paste(img, obj, x, y)
            placed.append(gb)
            labels.append((CLS_INDEX[cls], gb))
        # decoys: unlabeled background patches pasted with the same soft masks, so
        # "a blended disc of other texture" is never a shortcut for "object"
        for _ in range(int(rng.integers(3, 12))):
            sz = int(rng.integers(24, 180))
            sx, sy = int(rng.integers(0, Wc - sz)), int(rng.integers(0, Hc - sz))
            patch = jitter_color(img[sy:sy + sz, sx:sx + sz], rng, 0.4)
            m = np.zeros((sz, sz), np.uint8)
            if rng.random() < 0.6:
                cv2.ellipse(m, (sz // 2, sz // 2), (int(sz * rng.uniform(0.25, 0.5)), int(sz * rng.uniform(0.25, 0.5))), float(rng.uniform(0, 180)), 0, 360, 255, -1)
            else:
                a, b = int(sz * rng.uniform(0.1, 0.3)), int(sz * rng.uniform(0.1, 0.3))
                m[a:sz - a, b:sz - b] = 255
            m = cv2.GaussianBlur(cv2.dilate(m, np.ones((3, 3), np.uint8)), (5, 5), 0)
            x, y = int(rng.integers(0, Wc - sz)), int(rng.integers(0, Hc - sz))
            if any(x < q[2] and q[0] < x + sz and y < q[3] and q[1] < y + sz for q in placed):
                continue
            paste(img, np.dstack([patch, m]), x, y)
        # clip labels to canvas
        out = []
        for c, b in labels:
            cb = (max(0, b[0]), max(0, b[1]), min(Wc, b[2]), min(Hc, b[3]))
            full = (b[2] - b[0]) * (b[3] - b[1])
            if cb[2] - cb[0] > 1 and cb[3] - cb[1] > 1:
                out.append((c, cb, ((cb[2] - cb[0]) * (cb[3] - cb[1])) / max(1, full)))
        return img, out, native

    def views(self, rng, img, labels, native, n_views):
        Hc, Wc = img.shape[:2]
        res = []
        levels = [L for L in (0, 1, 2) if (4 >> L) >= native and LEVEL_REGION[L][0] <= Wc and LEVEL_REGION[L][1] <= Hc]
        if not levels:
            return res
        probs = np.array([{0: 0.3, 1: 0.4, 2: 0.3}[L] for L in levels])
        probs /= probs.sum()
        for _ in range(n_views):
            L = int(rng.choice(levels, p=probs))
            rw, rh = LEVEL_REGION[L]
            if labels and rng.random() < 0.75:
                c, b, _ = labels[rng.integers(len(labels))]
                cx = (b[0] + b[2]) / 2 + rng.uniform(-0.45, 0.45) * rw
                cy = (b[1] + b[3]) / 2 + rng.uniform(-0.45, 0.45) * rh
            else:
                cx, cy = rng.uniform(0, Wc), rng.uniform(0, Hc)
            x0 = int(np.clip(cx - rw / 2, 0, Wc - rw))
            y0 = int(np.clip(cy - rh / 2, 0, Hc - rh))
            crop = img[y0:y0 + rh, x0:x0 + rw]
            if L < 2:
                crop = cv2.resize(crop, (960, 540), interpolation=cv2.INTER_AREA)
            s = rw / 960.0
            yl = []
            for c, b, vis0 in labels:
                vb = (max(b[0], x0), max(b[1], y0), min(b[2], x0 + rw), min(b[3], y0 + rh))
                if vb[2] - vb[0] <= 0 or vb[3] - vb[1] <= 0:
                    continue
                area = (b[2] - b[0]) * (b[3] - b[1])
                vis = (vb[2] - vb[0]) * (vb[3] - vb[1]) / max(1, area)
                pw, ph = (vb[2] - vb[0]) / s, (vb[3] - vb[1]) / s
                if vis < 0.3 or pw < 2 or ph < 2:
                    continue
                yl.append((c, ((vb[0] + vb[2]) / 2 - x0) / rw, ((vb[1] + vb[3]) / 2 - y0) / rh, (vb[2] - vb[0]) / rw, (vb[3] - vb[1]) / rh))
            res.append((L, crop, yl))
        return res


_G = {}


def _work(args):
    src, split, extra, seed, n_views, out, generic = args
    key = (split, bool(extra))
    if key not in _G:
        _G[key] = Gen(src, split, extra, generic)
    g = _G[key]
    rng = np.random.default_rng(seed)
    img, labels, native = g.compose(rng)
    names = []
    for i, (L, crop, yl) in enumerate(g.views(rng, img, labels, native, n_views)):
        name = f"s{seed:06d}_{i}_L{L}"
        # JPEG q95 for size; the real views are PNG (lossless)
        cv2.imwrite(os.path.join(out, "images", split, name + ".jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(os.path.join(out, "labels", split, name + ".txt"), "w") as f:
            for c, x, y, w, h in yl:
                f.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        names.append(name)
    return len(names)


def gen(src, out, canvases, val_canvases, n_views, workers, extra_bg, seed0, generic=None):
    for split in ("train", "val"):
        os.makedirs(os.path.join(out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out, "labels", split), exist_ok=True)
    jobs = [(src, "train", extra_bg, seed0 + i, n_views, out, generic) for i in range(canvases)]
    jobs += [(src, "val", None, seed0 + 500000 + i, n_views, out, generic) for i in range(val_canvases)]
    total = 0
    with Pool(workers) as pool:
        for i, n in enumerate(pool.imap_unordered(_work, jobs, chunksize=4)):
            total += n
            if i % 100 == 0:
                print(f"{i}/{len(jobs)} canvases, {total} views", flush=True)
    with open(os.path.join(out, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(out)}\ntrain: images/train\nval: images/val\nnames:\n")
        for i, c in enumerate(CLASSES):
            f.write(f"  {i}: {c}\n")
    print("done", total)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "gen"])
    ap.add_argument("--src", default="data/synth_src")
    ap.add_argument("--out", default=None)
    ap.add_argument("--canvases", type=int, default=1500)
    ap.add_argument("--val-canvases", type=int, default=100)
    ap.add_argument("--views", type=int, default=6)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--extra-bg", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generic-bg", default=None, help="dir of generic aerial tiles (fetch_bg.py)")
    a = ap.parse_args()
    if a.cmd == "prep":
        prep(a.out or a.src)
    else:
        gen(a.src, a.out or "data/yolo", a.canvases, a.val_canvases, a.views, a.workers, a.extra_bg, a.seed, a.generic_bg)
