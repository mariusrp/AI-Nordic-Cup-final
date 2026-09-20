"""Per-frame flight step check: for every consecutive frame pair of the recorded validation flight, measure the real
ground motion between the two frames (template NCC on native level-2 views) and compare it with the flight homography H.
A pair whose measured motion is ~0 is a STALLED frame (the flight image did not advance), which shifts every later
frame by one H step relative to a table built on a uniform model.
Output: json {"steps": {frame: k}, "resid": {...}} with k = 1 (normal) or 0 (stall) per step f -> f+1, plus the raw
residuals. Usage: python frame_steps.py H.npy VIEWDIR[,..] OUT.json [--f0 4] [--f1 249] [--r 70] [--search 130]
"""
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

H = np.load(sys.argv[1]); DIRS = sys.argv[2].split(','); OUT = sys.argv[3]; argv = sys.argv[4:]


def opt(n, d):
    return argv[argv.index(n) + 1] if n in argv else d


F0 = int(opt('--f0', 4)); F1 = int(opt('--f1', 249)); R = int(opt('--r', 70)); SE = int(opt('--search', 130))
V = {}
for d in DIRS:
    for p in glob.glob(f'{d}/*.*'):
        m = re.match(r'(\d+)_L(\d)_(\d+)_(\d+)\.', os.path.basename(p))
        if not m: continue
        f, l, cx, cy = map(int, m.groups())
        if l != 2: continue
        V.setdefault(f, []).append((cx, cy, p))
_img = {}


def img_of(p):
    if p not in _img:
        if len(_img) > 60: _img.clear()
        _img[p] = cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    return _img[p]


def patch(f, x, y, r):
    for cx, cy, p in V.get(f, []):
        u, v = x - (cx - 480), y - (cy - 270)
        if u - r < 0 or v - r < 0 or u + r > 959 or v + r > 539: continue
        return cv2.getRectSubPix(img_of(p), (2 * r, 2 * r), (u, v)), p
    return None, None


def pred(x, y):
    q = H @ [x, y, 1]; return q[0] / q[2], q[1] / q[2]


res = {}
for f in range(F0, F1):
    vals = []
    for (x, y) in ((500, 950), (1500, 950), (2500, 950), (3300, 950), (1000, 1000), (2000, 1000), (3000, 1000)):
        t, pt = patch(f, x, y, R)
        if t is None: continue
        px, py = pred(x, y)
        s, ps = patch(f + 1, px, py, SE)
        if s is None: continue
        m = cv2.matchTemplate(s, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(m)
        dx = loc[0] - (SE - R); dy = loc[1] - (SE - R)
        if mx >= 0.7: vals.append((dx, dy, mx, x, y))
    if vals:
        dy = float(np.median([v[1] for v in vals])); dx = float(np.median([v[0] for v in vals]))
        res[f] = dict(dx=dx, dy=dy, n=len(vals), ncc=round(float(np.median([v[2] for v in vals])), 2))
        print(f'{f}->{f + 1} dx {dx:6.1f} dy {dy:6.1f} n{len(vals)} ncc {res[f]["ncc"]}', flush=True)
    else:
        print(f'{f}->{f + 1} no overlap', flush=True)
steps = {}
for f, r in res.items():
    y0 = 975.0
    full = pred(1900, y0)[1] - y0  # H step at the sample band
    k = 1 if abs(r['dy']) < full * 0.4 else (0 if abs(r['dy'] + full) < full * 0.4 else -1)
    steps[f] = k
json.dump(dict(steps=steps, resid=res), open(OUT, 'w'), indent=0)
print('FRAME_STEPS_DONE pairs', len(res), 'stalls', sorted(f for f, k in steps.items() if k == 0), 'weird', sorted(f for f, k in steps.items() if k == -1))
