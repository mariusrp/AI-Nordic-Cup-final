"""FINALS: a HARDER unseen-city test flight in upstream scene format (for bb3/eval_scene.py).

Like drone/make_scene.py, but with the knobs the final needs: altitude (object scale + background ground
sample), object density, unlabelled decoy clutter (false-positive bait) and JPEG output (the pod's overlay
filesystem is full, so scenes live in /dev/shm).

Backgrounds MUST come from LoveDA tiles that were never used for training (fetch_bg.py with start >= 2562;
tiles 0..699 are the training pool, 2522..2561 were used by the old syn1 scene).
Objects prefer the "val" cut-out split = Helsinki holdout frames 4/12/20, whose pixels no checkpoint has
ever seen in any form.

    python make_scene2.py --bg /dev/shm/loveda_new --out /dev/shm/bench/a1 \
        --frames 30 --objects 32 --alt 1.0 --decoys 60 --seed 11 --jpg
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common  # noqa: F401,E402
from common import CLASSES, W as FW, H as FH  # noqa: E402
from make_synth import Gen, transform_cutout, paste, jitter_color  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg", required=True, help="dir of background tiles never used in training")
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=os.path.join(os.path.dirname(HERE), "data", "synth_src"))
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--objects", type=int, default=32)
    ap.add_argument("--decoys", type=int, default=60, help="unlabelled blended background patches (FP bait)")
    ap.add_argument("--alt", type=float, default=1.0, help="altitude: objects scale 1/alt, ground sample x alt")
    ap.add_argument("--dy", type=float, default=70)
    ap.add_argument("--dx", type=float, default=3)
    ap.add_argument("--tilepx", type=int, default=2048, help="render size of a 1024 px background tile at alt 1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jpg", action="store_true")
    ap.add_argument("--rot", type=float, default=0.0, help="extra whole-strip rotation in degrees")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    tiles = sorted(glob.glob(os.path.join(a.bg, "*.png")) + glob.glob(os.path.join(a.bg, "*.jpg")))
    if not tiles:
        raise SystemExit("no background tiles in " + a.bg)
    SH = int(FH + a.dy * a.frames + 400)
    SW = FW + int(abs(a.dx) * a.frames) + 400
    ts = max(512, int(a.tilepx / a.alt))          # higher altitude -> more ground per pixel -> smaller tiles
    strip = np.zeros((SH, SW, 3), np.uint8)
    for y in range(0, SH, ts):
        for x in range(0, SW, ts):
            t = cv2.imread(tiles[rng.integers(len(tiles))])
            if t is None:
                continue
            t = cv2.resize(t, (ts, ts), interpolation=cv2.INTER_CUBIC)
            t = np.rot90(t, int(rng.integers(4)))
            if rng.random() < 0.5:
                t = t[:, ::-1]
            t = jitter_color(np.ascontiguousarray(t), rng, strength=float(rng.uniform(0.3, 1.0)))
            h, w = min(ts, SH - y), min(ts, SW - x)
            strip[y:y + h, x:x + w] = t[:h, :w]

    gv, gt_ = Gen(a.src, "val"), Gen(a.src, "train")
    placed = []
    for k in range(a.objects):
        cls = CLASSES[k % len(CLASSES)] if k < len(CLASSES) else CLASSES[rng.integers(len(CLASSES))]
        pool = gv.cut.get(cls) or gt_.cut[cls]
        rgba, box = pool[rng.integers(len(pool))]
        s = float(np.exp(rng.uniform(np.log(0.8), np.log(1.25)))) / a.alt
        obj, b, _ = transform_cutout(rgba, box, rng, s)
        if obj is None:
            continue
        obj = np.dstack([jitter_color(obj[..., :3], rng, 0.3), obj[..., 3]])
        oh, ow = obj.shape[:2]
        for _ in range(60):
            x, y = int(rng.integers(0, SW - ow)), int(rng.integers(0, SH - oh))
            gb = (x + b[0], y + b[1], x + b[2], y + b[3])
            if all(not (gb[0] < q[2] + 40 and q[0] < gb[2] + 40 and gb[1] < q[3] + 40 and q[1] < gb[3] + 40)
                   for _, q in placed):
                break
        else:
            continue
        paste(strip, obj, x, y)
        placed.append((cls, gb))

    # decoys: unlabelled patches of the strip's own texture, blended with the same soft masks as a cut-out,
    # so "a blended blob that is not the ground" is never a shortcut for "object".
    for _ in range(a.decoys):
        sz = int(rng.integers(24, 200) / a.alt)
        sz = max(16, min(sz, min(SH, SW) // 4))
        sx, sy = int(rng.integers(0, SW - sz)), int(rng.integers(0, SH - sz))
        patch = jitter_color(strip[sy:sy + sz, sx:sx + sz].copy(), rng, 0.5)
        m = np.zeros((sz, sz), np.uint8)
        if rng.random() < 0.6:
            cv2.ellipse(m, (sz // 2, sz // 2), (int(sz * rng.uniform(0.25, 0.5)), int(sz * rng.uniform(0.25, 0.5))),
                        float(rng.uniform(0, 180)), 0, 360, 255, -1)
        else:
            u, v = int(sz * rng.uniform(0.1, 0.3)), int(sz * rng.uniform(0.1, 0.3))
            m[u:sz - u, v:sz - v] = 255
        m = cv2.GaussianBlur(cv2.dilate(m, np.ones((3, 3), np.uint8)), (5, 5), 0)
        x, y = int(rng.integers(0, SW - sz)), int(rng.integers(0, SH - sz))
        if any(x < q[2] + 8 and q[0] < x + sz + 8 and y < q[3] + 8 and q[1] < y + sz + 8 for _, q in placed):
            continue
        paste(strip, np.dstack([patch, m]), x, y)

    ext = "jpg" if a.jpg else "png"
    wparam = [cv2.IMWRITE_JPEG_QUALITY, 95] if a.jpg else [cv2.IMWRITE_PNG_COMPRESSION, 1]
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "annotations"), exist_ok=True)
    x0base = 200 + (int(abs(a.dx) * a.frames) if a.dx < 0 else 0)
    for f in range(a.frames):
        oy = int(round(SH - FH - a.dy * f)) - 1
        ox = int(round(x0base - a.dx * f))
        cv2.imwrite(os.path.join(a.out, "images", f"frame_{f:06d}.{ext}"), strip[oy:oy + FH, ox:ox + FW], wparam)
        anns = []
        for cls, (x1, y1, x2, y2) in placed:
            bx = [x1 - ox, y1 - oy, x2 - ox, y2 - oy]
            cb = [max(0, bx[0]), max(0, bx[1]), min(FW, bx[2]), min(FH, bx[3])]
            if cb[2] <= cb[0] or cb[3] <= cb[1]:
                continue
            if (cb[2] - cb[0]) * (cb[3] - cb[1]) < 0.4 * (bx[2] - bx[0]) * (bx[3] - bx[1]):
                continue
            anns.append({"object_id": cls, "bbox": [int(round(v)) for v in cb]})
        json.dump({"frame": f, "annotations": anns},
                  open(os.path.join(a.out, "annotations", f"frame_{f:06d}.json"), "w"))
    tot = {}
    for c, _ in placed:
        tot[c] = tot.get(c, 0) + 1
    json.dump({"capture": {"num_frames": a.frames}, "total_objects": len(placed), "object_totals": tot,
               "alt": a.alt, "decoys": a.decoys, "seed": a.seed, "bg": a.bg},
              open(os.path.join(a.out, "run_metadata.json"), "w"))
    print("scene", a.out, "frames", a.frames, "objects", len(placed), "alt", a.alt, "decoys", a.decoys)


if __name__ == "__main__":
    main()
