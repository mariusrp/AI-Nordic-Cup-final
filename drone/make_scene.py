"""Synthetic test flight in upstream scene format (for local_evaluator.py --scene NAME).

Background: LoveDA tiles NOT used for training (fetch_bg.py ... 1500), upscaled ~2x and tiled into a
tall strip; the camera slides DY px/frame (+DX drift). Objects: Helsinki cut-outs, preferring the
held-out frames' cut-outs, randomly rotated/scaled, placed along the strip. GT = pasted boxes,
clipped to the frame, kept when >= 40% visible.

    python make_scene.py --bg /root/dd/loveda_scene --out /root/dd/scenes/syn1 --frames 60 --objects 30 --seed 1
    ln -s /root/dd/scenes/syn1 /workspace/upstream/drone-flyby/src/syn1
"""
import argparse, glob, json, os
import cv2, numpy as np
import common  # noqa
from common import CLASSES, W as FW, H as FH
from make_synth import Gen, transform_cutout, paste, jitter_color


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "synth_src"))
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--objects", type=int, default=30)
    ap.add_argument("--dy", type=float, default=70)
    ap.add_argument("--dx", type=float, default=3)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    tiles = sorted(glob.glob(os.path.join(a.bg, "*.png")))
    SH = int(FH + a.dy * a.frames + 400)
    SW = FW + int(abs(a.dx) * a.frames) + 400
    ts = 2048
    strip = np.zeros((SH, SW, 3), np.uint8)
    for y in range(0, SH, ts):
        for x in range(0, SW, ts):
            t = cv2.resize(cv2.imread(tiles[rng.integers(len(tiles))]), (ts, ts), interpolation=cv2.INTER_CUBIC)
            t = np.rot90(t, int(rng.integers(4)))
            h, w = min(ts, SH - y), min(ts, SW - x)
            strip[y:y + h, x:x + w] = t[:h, :w]
    gv, gt_ = Gen(a.src, "val"), Gen(a.src, "train")
    placed = []
    for k in range(a.objects):
        cls = CLASSES[k % len(CLASSES)] if k < len(CLASSES) else CLASSES[rng.integers(len(CLASSES))]
        pool = gv.cut.get(cls) or gt_.cut[cls]
        rgba, box = pool[rng.integers(len(pool))]
        obj, b, _ = transform_cutout(rgba, box, rng, float(np.exp(rng.uniform(np.log(0.8), np.log(1.25)))))
        if obj is None:
            continue
        obj = np.dstack([jitter_color(obj[..., :3], rng, 0.3), obj[..., 3]])
        oh, ow = obj.shape[:2]
        for _ in range(50):
            x, y = int(rng.integers(0, SW - ow)), int(rng.integers(0, SH - oh))
            gb = (x + b[0], y + b[1], x + b[2], y + b[3])
            if all(not (gb[0] < q[2] + 40 and q[0] < gb[2] + 40 and gb[1] < q[3] + 40 and q[1] < gb[3] + 40) for _, q in placed):
                break
        else:
            continue
        paste(strip, obj, x, y)
        placed.append((cls, gb))
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "annotations"), exist_ok=True)
    x0base = 200 + (int(abs(a.dx) * a.frames) if a.dx < 0 else 0)
    for f in range(a.frames):
        oy = int(round(SH - FH - a.dy * f)) - 1
        ox = int(round(x0base - a.dx * f))
        cv2.imwrite(os.path.join(a.out, "images", f"frame_{f:06d}.png"), strip[oy:oy + FH, ox:ox + FW], [cv2.IMWRITE_PNG_COMPRESSION, 1])
        anns = []
        for cls, (x1, y1, x2, y2) in placed:
            bx = [x1 - ox, y1 - oy, x2 - ox, y2 - oy]
            cb = [max(0, bx[0]), max(0, bx[1]), min(FW, bx[2]), min(FH, bx[3])]
            if cb[2] <= cb[0] or cb[3] <= cb[1]:
                continue
            if (cb[2] - cb[0]) * (cb[3] - cb[1]) < 0.4 * (bx[2] - bx[0]) * (bx[3] - bx[1]):
                continue
            anns.append({"object_id": cls, "bbox": [int(round(v)) for v in cb]})
        json.dump({"frame": f, "annotations": anns}, open(os.path.join(a.out, "annotations", f"frame_{f:06d}.json"), "w"))
    tot = {}
    for c, _ in placed:
        tot[c] = tot.get(c, 0) + 1
    json.dump({"capture": {"num_frames": a.frames}, "total_objects": len(placed), "object_totals": tot},
              open(os.path.join(a.out, "run_metadata.json"), "w"))
    print("scene", a.out, "objects", len(placed))


if __name__ == "__main__":
    main()
