"""Zoom sheet for canonical points (x*, t*) found on the mosaic (mosaic.py): for each point take up to K native level-2
survey views that contain it (most central first, distinct frames), crop +-R px at native resolution, upscale by Z, draw
10-px ticks on the border (source px) and optionally the table boxes of that frame. One row per point.
Usage: python zoom_pts.py H.npy OUT.jpg VIEWDIR[,..] "x,t[,label];x,t[,label];..." [--k 3] [--r 50] [--z 3] [--table T.json] [--perrow 1]
Prints, per tile, frame / view / point position in source px of that frame (for manual boxes)."""
import glob
import json
import os
import re
import sys

import cv2
import numpy as np
from scipy.linalg import expm, logm

H = np.load(sys.argv[1]); OUT = sys.argv[2]; DIRS = sys.argv[3].split(','); PTS = [p.split(',') for p in sys.argv[4].split(';') if p]
argv = sys.argv[5:]


def opt(n, d):
    return argv[argv.index(n) + 1] if n in argv else d


K = int(opt('--k', 3)); R = int(opt('--r', 50)); Z = float(opt('--z', 3)); TB = json.load(open(opt('--table', ''))) if opt('--table', '') else None
L = np.real(logm(H)); YC = 1080.0
views = []
for d in DIRS:
    for p in glob.glob(f'{d}/*_L2_*.*'):
        f, l, cx, cy = map(int, re.match(r'(\d+)_L(\d)_(\d+)_(\d+)\.', os.path.basename(p)).groups())
        views.append((f, cx, cy, p))
rows = []
for pt in PTS:
    x, t = float(pt[0]), float(pt[1]); lab = pt[2] if len(pt) > 2 else ''
    cand = []
    for f, cx, cy, p in views:
        if abs(f - t) > 6: continue
        q = expm((f - t) * L) @ [x, YC, 1]; u, v = q[0] / q[2], q[1] / q[2]
        du, dv = u - (cx - 480), v - (cy - 270)
        if R * 0.5 <= du <= 960 - R * 0.5 and R * 0.5 <= dv <= 540 - R * 0.5:
            cand.append((max(abs(du - 480) / 480, abs(dv - 270) / 270), f, cx, cy, p, u, v))
    cand.sort(); got = []; fs = set()
    for c in cand:
        if c[1] in fs: continue
        got.append(c); fs.add(c[1])
        if len(got) >= K: break
    tiles = []
    for sc, f, cx, cy, p, u, v in sorted(got, key=lambda c: c[1]):
        img = cv2.imread(p); rx, ry = cx - 480, cy - 270
        pad = cv2.copyMakeBorder(img, R, R, R, R, cv2.BORDER_CONSTANT)
        iu, iv = int(round(u - rx)), int(round(v - ry))
        crop = pad[iv:iv + 2 * R, iu:iu + 2 * R].copy()
        big = cv2.resize(crop, None, fx=Z, fy=Z, interpolation=cv2.INTER_CUBIC)
        if TB is not None:
            for c, b, s in TB.get(str(f), []):
                if s < 0.05: continue
                bx = [b[0] * 3840 - (u - R), b[1] * 2160 - (v - R), b[2] * 3840 - (u - R), b[3] * 2160 - (v - R)]
                if bx[2] < 0 or bx[3] < 0 or bx[0] > 2 * R or bx[1] > 2 * R: continue
                cv2.rectangle(big, (int(bx[0] * Z), int(bx[1] * Z)), (int(bx[2] * Z), int(bx[3] * Z)), (0, 255, 0) if s >= 0.25 else (0, 220, 255), 1)
                cv2.putText(big, f'{c[:6]}{s:.2f}', (int(bx[0] * Z), max(10, int(bx[1] * Z) - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        for k in range(0, 2 * R + 1, 10):
            L2 = 8 if k % 50 == 0 else 4
            cv2.line(big, (int(k * Z), 0), (int(k * Z), L2), (255, 255, 255), 1); cv2.line(big, (0, int(k * Z)), (L2, int(k * Z)), (255, 255, 255), 1)
        txt = f'{lab} f{f} ({u:.0f},{v:.0f})'
        cv2.putText(big, txt, (3, int(2 * R * Z) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
        cv2.putText(big, txt, (3, int(2 * R * Z) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        tiles.append(big); print(lab, 'f', f, 'view', os.path.basename(p), 'src', round(u, 1), round(v, 1), 'crop x0', round(u - R, 1), 'y0', round(v - R, 1))
    S = int(2 * R * Z)
    while len(tiles) < K: tiles.append(np.zeros((S, S, 3), np.uint8))
    rows.append(np.hstack(tiles))
PR = int(opt('--perrow', 1))
while len(rows) % PR: rows.append(np.zeros_like(rows[0]))
rows = [np.hstack(rows[i:i + PR]) for i in range(0, len(rows), PR)]
for r in rows: r[:, ::max(1, r.shape[1] // PR)][:, 1:] = 255  # separators between points
cv2.imwrite(OUT, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92]); print('wrote', OUT)
