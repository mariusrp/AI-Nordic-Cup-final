"""Measure an object's REAL image trajectory across the recorded views (template NCC), to check / correct the flight-
homography propagation of a manual cluster. Tall objects (towers, launchers) do not follow the ground-plane H: the
propagated box drifts by >100 px within 6 frames, so the manual box is only right at its own frame.
Given a box in one frame, the template (that crop, native L2) is matched in every other view that contains the
H-predicted position (search window +-WIN px, template rescaled by the H scale between the frames).
Prints per frame: measured centre, H-predicted centre, residual; then the least-squares linear residual
(a, b per axis, as fuse.py's cluster 'resid') and the box of the best-scoring frame.
Usage: python track_ncc.py H.npy VIEWDIR[,..] FRAME x1,y1,x2,y2 [--win 45] [--range 20] [--min-ncc 0.5]
"""
import glob
import os
import re
import sys

import cv2
import numpy as np
from scipy.linalg import expm, logm

H = np.load(sys.argv[1]); DIRS = sys.argv[2].split(','); F0 = int(sys.argv[3]); BOX = [float(v) for v in sys.argv[4].split(',')]
argv = sys.argv[5:]


def opt(n, d):
    return argv[argv.index(n) + 1] if n in argv else d


WIN = int(opt('--win', 45)); RNG = int(opt('--range', 20)); MINN = float(opt('--min-ncc', 0.5))
L = np.real(logm(H))
views = []
for d in DIRS:
    for p in glob.glob(f'{d}/*.*'):
        m = re.match(r'(\d+)_L(\d)_(\d+)_(\d+)\.', os.path.basename(p))
        if not m: continue
        f, l, cx, cy = map(int, m.groups()); views.append((f, l, cx, cy, p, {0: 4, 1: 2, 2: 1}[l]))
V = {}
for f, l, cx, cy, p, sc in views: V.setdefault(f, []).append((sc, cx, cy, p))


def crop_at(f, cx, cy, w, h, pad=0):
    """native-resolution crop of the source rect (cx,cy,w,h) from the best view of frame f (finest, most central)."""
    best = None
    for sc, vx, vy, p in sorted(V.get(f, []), key=lambda v: v[0]):
        rx, ry = vx - 480 * sc, vy - 270 * sc
        x1, y1 = (cx - w / 2 - pad - rx) / sc, (cy - h / 2 - pad - ry) / sc
        x2, y2 = (cx + w / 2 + pad - rx) / sc, (cy + h / 2 + pad - ry) / sc
        if x1 < 0 or y1 < 0 or x2 > 960 * 1.0 or y2 > 540 * 1.0: continue
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        patch = cv2.getRectSubPix(img.astype(np.float32), (int(round(x2 - x1)), int(round(y2 - y1))), ((x1 + x2) / 2, (y1 + y2) / 2))
        if sc > 1: patch = cv2.resize(patch, (int(round((x2 - x1) * sc)), int(round((y2 - y1) * sc))), interpolation=cv2.INTER_CUBIC)
        best = (patch, sc, p); break
    return best


cx0, cy0 = (BOX[0] + BOX[2]) / 2, (BOX[1] + BOX[3]) / 2; w0, h0 = BOX[2] - BOX[0], BOX[3] - BOX[1]
t0 = crop_at(F0, cx0, cy0, w0, h0)
if t0 is None: raise SystemExit('no view contains the template box in frame %d' % F0)
tpl = t0[0]
print('template frame', F0, 'from', os.path.basename(t0[2]), 'size', tpl.shape)
rows = []
for f in range(F0 - RNG, F0 + RNG + 1):
    if f == F0 or f not in V: continue
    M = expm((f - F0) * L)
    q = M @ [cx0, cy0, 1]; px, py = q[0] / q[2], q[1] / q[2]
    s = float(np.linalg.norm((M @ [cx0 + w0 / 2, cy0, 1])[:2] / (M @ [cx0 + w0 / 2, cy0, 1])[2] - (M @ [cx0 - w0 / 2, cy0, 1])[:2] / (M @ [cx0 - w0 / 2, cy0, 1])[2]) / w0)
    got = crop_at(f, px, py, w0 * s, h0 * s, pad=WIN)
    if got is None: continue
    img = got[0]
    T = cv2.resize(tpl, (max(4, int(round(w0 * s))), max(4, int(round(h0 * s)))), interpolation=cv2.INTER_CUBIC)
    if img.shape[0] <= T.shape[0] or img.shape[1] <= T.shape[1]: continue
    res = cv2.matchTemplate(img, T, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(res)
    dx = loc[0] - (img.shape[1] - T.shape[1]) / 2; dy = loc[1] - (img.shape[0] - T.shape[0]) / 2
    rows.append((f, mx, px + dx, py + dy, px, py, dx, dy, s))
    print(f'f{f:3d} ncc {mx:.2f} meas ({px + dx:7.1f},{py + dy:7.1f}) pred ({px:7.1f},{py:7.1f}) resid ({dx:6.1f},{dy:6.1f}) scale {s:.3f}')
good = [r for r in rows if r[1] >= MINN]
if len(good) >= 3:
    dt = np.array([r[0] - F0 for r in good], float)
    for k, name in ((6, 'dx'), (7, 'dy')):
        v = np.array([r[k] for r in good])
        A = np.stack([np.ones(len(dt)), dt], 1)
        sol = np.linalg.lstsq(A, v, rcond=None)[0]
        print(f'{name}: a={sol[0]:.2f} b={sol[1]:.2f} px/frame (n={len(good)}, rms={np.sqrt(np.mean((v - A @ sol) ** 2)):.1f})')
