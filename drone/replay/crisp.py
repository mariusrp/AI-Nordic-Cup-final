"""Crispness of fuse.py clusters in the native level-2 views (drone_seen_full survey runs + any recorded L2 view):
the placed 3D models are rendered sharp, the photogrammetry terrain (roofs, bushes, rocks, boats) is blurry, so the
fine-scale Laplacian energy inside the propagated box separates real objects from terrain clutter. Per cluster: median
over up to K L2 views of lap = mean |Laplacian 3x3| (grey, native px) inside the box, ring = same in the 1-box-wide ring
around it, and grad = mean Sobel magnitude inside.
Usage: python crisp.py H.npy clusters.json OUT.json [--minsupport 0.01] [--k 4]"""
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

H = np.load(sys.argv[1]); CL = json.load(open(sys.argv[2])); OUT = sys.argv[3]; argv = sys.argv[4:]
MINS = float(argv[argv.index('--minsupport') + 1]) if '--minsupport' in argv else 0.01
K = int(argv[argv.index('--k') + 1]) if '--k' in argv else 4
Hi = np.linalg.inv(H); _pw = {}


def mpow(k):
    if k not in _pw: _pw[k] = np.linalg.matrix_power(H, k) if k >= 0 else np.linalg.matrix_power(Hi, -k)
    return _pw[k]


def warp_box(M, box):
    x1, y1, x2, y2 = box; cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    p = np.array([[cx, cy], [cx - w / 2, cy], [cx + w / 2, cy], [cx, cy - h / 2], [cx, cy + h / 2]])
    ph = np.hstack([p, np.ones((5, 1))]) @ M.T; p = ph[:, :2] / ph[:, 2:3]
    nw = np.linalg.norm(p[2] - p[1]); nh = np.linalg.norm(p[4] - p[3])
    return np.array([p[0][0] - nw / 2, p[0][1] - nh / 2, p[0][0] + nw / 2, p[0][1] + nh / 2])


byf = {}
seen = set()
for root in ('/workspace/drone_seen_full', '/workspace/drone_seen', '/workspace/.holdout/drone_seen_heldout'):
    for p in glob.glob(f'{root}/*/*_L2_*.png'):
        b = os.path.basename(p)
        if b in seen or 'local' in p: continue
        seen.add(b); f, l, cx, cy = map(int, re.match(r'(\d+)_L(\d)_(\d+)_(\d+)\.png', b).groups())
        byf.setdefault(f, []).append((cx - 480, cy - 270, p))
print('L2 views', len(seen), flush=True)
cache = {}


def lap_of(p):
    if p not in cache:
        if len(cache) > 400: cache.clear()
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        cache[p] = (np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=1)), cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)))
    return cache[p]


out = {}
for c in CL:
    if c.get('drop') or c['support'] < MINS: continue
    a, bb = np.array(c['resid'][0]), np.array(c['resid'][1]); res = []
    for t in range(max(1, c['tmin'] - 3), min(249, c['tmax'] + 3) + 1):
        dt = t - c['tref']; sh = a + bb * dt
        b = warp_box(mpow(dt), np.array(c['box_ref']) + np.array([sh[0], sh[1], sh[0], sh[1]]))
        w, h = b[2] - b[0], b[3] - b[1]
        for rx, ry, p in byf.get(t, []):
            x1, y1, x2, y2 = b[0] - rx, b[1] - ry, b[2] - rx, b[3] - ry
            if x1 - w < 0 or y1 - h < 0 or x2 + w > 960 or y2 + h > 540: continue
            res.append((abs(y1 + y2 - 540), p, x1, y1, x2, y2, w, h))
    if not res: continue
    res.sort()
    vals = []
    for _, p, x1, y1, x2, y2, w, h in res[:K]:
        L, G = lap_of(p)
        i1, j1, i2, j2 = int(y1), int(x1), int(np.ceil(y2)), int(np.ceil(x2))
        inner = L[i1:i2, j1:j2]; gin = G[i1:i2, j1:j2]
        R = L[int(y1 - h):int(np.ceil(y2 + h)), int(x1 - w):int(np.ceil(x2 + w))]
        ring = (R.sum() - inner.sum()) / max(1, R.size - inner.size)
        vals.append((float(inner.mean()), float(ring), float(gin.mean())))
    v = np.median(np.array(vals), 0)
    out[c['id']] = dict(lap=round(v[0], 2), ring=round(v[1], 2), grad=round(v[2], 2), n=len(vals), t=c['t_star'], x=c['x_star'], cls=c['cls'], s=round(c['support'], 3))
json.dump(out, open(OUT, 'w'))
print('CRISP_DONE', len(out))
