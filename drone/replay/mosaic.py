"""Canonical ground mosaic of the deterministic validation flight from the SURVEY runs' native level-2 views (table v3 hunt).
Canvas coordinates = fuse.py's canonical coordinates: column = x* (source px where the object crosses the centre line
y=1080), row = VC * (TMAX - t*) (t* = fractional frame of that crossing; later objects on top, like the camera image).
Every canvas pixel is filled from the view in which that ground point is most central (z-buffer on the normalised
Chebyshev distance to the view centre), mapped with the fractional flight homography H^(f - t*) = expm((f - t*) logm H).
Scale is uniform (native L2 = source px at the centre line), so every object appears once, at native resolution.
Cluster overlay (fuse.py clusters json): support >= 0.25 green, 0.08-0.25 yellow (class abbrev + support), eye-demoted
(judged, < 0.08) magenta corner ticks, manual clusters cyan.
Usage: python mosaic.py H.npy CLUSTERS.json OUT_PREFIX VIEWDIR[,VIEWDIR...] [--seg 12] [--tile 1280] [--ov 40] [--tmin 0] [--tmax 254]
       [--levels 2|1|01] [--nooverlay] [--canvas out.npy]  -> OUT_PREFIX_t<start>_<col>.jpg tiles (1280 x ~860) + OUT_PREFIX_index.json
"""
import glob
import json
import os
import re
import sys

import cv2
import numpy as np
from scipy.linalg import expm, logm

H = np.load(sys.argv[1]); CL = json.load(open(sys.argv[2])) if sys.argv[2] != '-' else []; OUT = sys.argv[3]
DIRS = sys.argv[4].split(','); argv = sys.argv[5:]


def opt(n, d):
    return argv[argv.index(n) + 1] if n in argv else d


SEG = float(opt('--seg', 12)); TILE = int(opt('--tile', 1280)); OV = int(opt('--ov', 40))
TMIN = float(opt('--tmin', 0)); TMAX = float(opt('--tmax', 254)); YC = 1080.0
L = np.real(logm(H))
VC = float(np.linalg.norm((H @ [1920, YC, 1])[:2] / (H @ [1920, YC, 1])[2] - [1920, YC]))
NR = int(np.ceil((TMAX - TMIN) * VC)) + 1
canvas = np.zeros((NR, 3840, 3), np.uint8); zb = np.full((NR, 3840), 9.0, np.float32)
views = []
LEVELS = [int(c) for c in opt('--levels', '2')]
for d in DIRS:
    for p in glob.glob(f'{d}/*.*'):
        m = re.match(r'(\d+)_L(\d)_(\d+)_(\d+)\.', os.path.basename(p))
        if not m: continue
        f, l, cx, cy = map(int, m.groups())
        if l in LEVELS: views.append((f, cx, cy, p, {0: 4, 1: 2, 2: 1}[l]))
views.sort()
print('views', len(views), 'VC', round(VC, 2), 'canvas', canvas.shape, flush=True)
_mk = {}


def Mk(k):
    k = round(k, 4)
    if k not in _mk: _mk[k] = expm(k * L)
    return _mk[k]


for vi, (f, cx, cy, p, sc) in enumerate(views):
    img = cv2.imread(p)
    rx, ry = cx - 480 * sc, cy - 270 * sc
    ts = np.arange(max(TMIN, f - 5.5 * sc), min(TMAX, f + 5.5 * sc), 1.0 / VC)
    if not len(ts): continue
    rows = np.round((TMAX - ts) * VC).astype(int); okr = (rows >= 0) & (rows < NR); ts, rows = ts[okr], rows[okr]
    x0, x1 = max(0, rx - 60), min(3840, rx + 960 * sc + 60)
    xs = np.arange(x0, x1, dtype=np.float64)
    Ms = np.stack([Mk(f - t) for t in ts])  # (R,3,3)
    num_u = Ms[:, 0, 0:1] * xs[None] + (Ms[:, 0, 1] * YC + Ms[:, 0, 2])[:, None]
    num_v = Ms[:, 1, 0:1] * xs[None] + (Ms[:, 1, 1] * YC + Ms[:, 1, 2])[:, None]
    den = Ms[:, 2, 0:1] * xs[None] + (Ms[:, 2, 1] * YC + Ms[:, 2, 2])[:, None]
    u = ((num_u / den - rx) / sc).astype(np.float32); v = ((num_v / den - ry) / sc).astype(np.float32)
    valid = (u >= 0) & (u <= 959) & (v >= 0) & (v <= 539)
    if not valid.any(): continue
    score = (np.maximum(np.abs(u - 480) / 480, np.abs(v - 270) / 270) + (sc - 1) * 2).astype(np.float32)
    score[~valid] = 9
    patch = cv2.remap(img, u, v, cv2.INTER_CUBIC if sc > 1 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    sub_z = zb[rows, x0:x1]; better = score < sub_z
    sub_c = canvas[rows, x0:x1]; sub_c[better] = patch[better]; sub_z[better] = score[better]
    canvas[rows, x0:x1] = sub_c; zb[rows, x0:x1] = sub_z
    if vi % 100 == 0: print('view', vi, f, cx, flush=True)
if opt('--canvas', ''): np.save(opt('--canvas', ''), canvas)
raw = canvas.copy()
AB = dict(small_launcher='sLa', medium_launcher='mLa', large_launcher='LLa', small_plane='sPl', medium_plane='mPl', jet_plane='jet',
          small_tower='sTw', large_tower='LTw', mine_roller='mRo', spacecraft='spc', helicopter='hel', hangar='hng', jammer='jam',
          tank='tnk', condor='cnd', **{'ta-ta': 'tat'})
if '--nooverlay' not in argv:
    for c in CL:
        if c.get('drop'): continue
        s = c['support']; judged = c.get('judged')
        if s < 0.08 and not judged: continue
        X, Y = c['x_star'], (TMAX - c['t_star']) * VC; w, h = c['w_star'], c['h_star']
        x1, y1, x2, y2 = int(X - w / 2) - 3, int(Y - h / 2) - 3, int(X + w / 2) + 3, int(Y + h / 2) + 3
        if c.get('manual'): col = (255, 255, 0)
        elif s >= 0.25: col = (0, 255, 0)
        elif s >= 0.08: col = (0, 220, 255)
        else: col = (255, 0, 255)
        if s < 0.08:
            for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
                cv2.line(canvas, (px, py), (px + 5 * dx, py), col, 1); cv2.line(canvas, (px, py), (px, py + 5 * dy), col, 1)
            continue
        cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 1)
        cv2.putText(canvas, f'{AB.get(c["cls"], c["cls"])}{s:.2f}', (x1, max(9, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.33, col, 1)
index = []
t = TMIN
while t < TMAX:
    ya, yb = int((TMAX - min(TMAX, t + SEG)) * VC) - OV, int((TMAX - t) * VC) + OV
    ya, yb = max(0, ya), min(NR, yb)
    for ci, xa in enumerate(range(0, 3840, TILE)):
        xa0, xb0 = max(0, xa - OV), min(3840, xa + TILE + OV)
        tile = canvas[ya:yb, xa0:xb0].copy()
        for gx in range(0, 3840, 100):
            if xa0 <= gx < xb0: cv2.line(tile, (gx - xa0, 0), (gx - xa0, 4), (255, 255, 255), 1)
        for tt in np.arange(np.ceil(t), t + SEG + 0.01):
            gy = int((TMAX - tt) * VC) - ya
            if 0 <= gy < tile.shape[0]:
                cv2.line(tile, (0, gy), (6, gy), (255, 255, 255), 1); cv2.putText(tile, f'{int(tt)}', (8, gy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        lab = f't{t:.0f}-{t + SEG:.0f} x{xa0}-{xb0}'
        cv2.putText(tile, lab, (40, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3); cv2.putText(tile, lab, (40, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        fn = f'{OUT}_t{int(t):03d}_{ci}.jpg'
        cv2.imwrite(fn, tile, [cv2.IMWRITE_JPEG_QUALITY, 92])
        index.append(dict(file=fn, t0=t, t1=t + SEG, xa=xa0, xb=xb0, ya=ya, yb=yb))
    t += SEG
json.dump(dict(VC=VC, TMAX=TMAX, tiles=index), open(f'{OUT}_index.json', 'w'), indent=0)
if opt('--raw', ''): cv2.imwrite(opt('--raw', ''), raw, [cv2.IMWRITE_JPEG_QUALITY, 95])
print('MOSAIC_DONE tiles', len(index))
