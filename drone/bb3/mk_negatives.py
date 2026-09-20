"""FINALS: pure-negative training views from aerial tiles of terrain no checkpoint has seen.

The discriminator that separates checkpoints on an unseen city is false positives per frame, not mAP@0.5
(which saturates). Background images with an EMPTY label file are the most direct signal against them:
"this terrain contains none of the 16 objects". Tiles must come from a pool that is disjoint from the
benchmark pool (see bb3/make_scene2.py).

Each tile pair is mosaicked into a 4K canvas and rendered at the three evaluator levels
(L0 crop 3840x2160 /4, L1 1920x1080 /2, L2 960x540 native), which is how views reach the detector.

    python mk_negatives.py /dev/shm/bg_train /dev/shm/neg 3
"""
import glob
import os
import sys

import cv2
import numpy as np

SRC, OUT = sys.argv[1], sys.argv[2]
PER = int(sys.argv[3]) if len(sys.argv) > 3 else 3
FW, FH = 3840, 2160
rng = np.random.default_rng(7)
tiles = sorted(glob.glob(os.path.join(SRC, "*.png")) + glob.glob(os.path.join(SRC, "*.jpg")))
os.makedirs(f"{OUT}/images/train", exist_ok=True)
os.makedirs(f"{OUT}/labels/train", exist_ok=True)
n = 0
for i, p in enumerate(tiles):
    f = float(rng.uniform(1.6, 2.4))            # LoveDA 0.3 m/px -> our ~0.14 m/px
    ts = int(1024 * f)
    big = np.zeros((2 * ts, 2 * ts, 3), np.uint8)
    for a in range(2):
        for b in range(2):
            t = cv2.imread(tiles[rng.integers(len(tiles))] if (a or b) else p)
            if t is None:
                continue
            t = cv2.resize(t, (ts, ts), interpolation=cv2.INTER_CUBIC)
            if rng.random() < 0.5:
                t = t[:, ::-1]
            big[a * ts:(a + 1) * ts, b * ts:(b + 1) * ts] = np.rot90(t, int(rng.integers(4)))
    y0 = int(rng.integers(0, max(1, 2 * ts - FH)))
    x0 = int(rng.integers(0, max(1, 2 * ts - FW)))
    canvas = np.ascontiguousarray(big[y0:y0 + FH, x0:x0 + FW])
    if canvas.shape[0] < FH or canvas.shape[1] < FW:
        canvas = cv2.resize(canvas, (FW, FH))
    for k in range(PER):
        L = k % 3
        rw, rh = {0: (3840, 2160), 1: (1920, 1080), 2: (960, 540)}[L]
        cx = int(rng.integers(0, FW - rw + 1))
        cy = int(rng.integers(0, FH - rh + 1))
        v = canvas[cy:cy + rh, cx:cx + rw]
        if L < 2:
            v = cv2.resize(v, (960, 540), interpolation=cv2.INTER_AREA)
        name = f"neg_{i:04d}_L{L}_{k}"
        cv2.imwrite(f"{OUT}/images/train/{name}.jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 92])
        open(f"{OUT}/labels/train/{name}.txt", "w").close()
        n += 1
print("negatives", n, "from", len(tiles), "tiles ->", OUT)
