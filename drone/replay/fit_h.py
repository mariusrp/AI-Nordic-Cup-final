"""Offline refit of the constant frame-to-frame homography H of the validation flight from ALL unique recorded views
(the online tracker estimates differ by ~1e-3 in the scale terms between recordings; a 1e-3 scale error is ~20 px after
15 frames of propagation, i.e. a miss at IoU 0.5 for a 25 px object). SIFT features per view (source coords via the
view region), pairs with frame gap 1..K and predicted overlap, RANSAC per pair, then one Huber least-squares fit of
q = H^k p over all inlier correspondences (same parametrisation as tracker.MotionModel._fit), init = H0.
Usage: python fit_h.py H0.npy OUT_PREFIX [--k 6] [--feat 1500]   -> OUT_PREFIX.npy, OUT_PREFIX.json (residual stats)
"""
import glob
import json
import os
import re
import sys
import time

import cv2
import numpy as np
from scipy.optimize import least_squares

H0 = np.load(sys.argv[1]); OUT = sys.argv[2]
K = int(sys.argv[sys.argv.index('--k') + 1]) if '--k' in sys.argv else 6
NF = int(sys.argv[sys.argv.index('--feat') + 1]) if '--feat' in sys.argv else 1500
MAXC = int(sys.argv[sys.argv.index('--maxc') + 1]) if '--maxc' in sys.argv else 60  # correspondences kept per pair (least-squares size)
SEEN = ['/workspace/drone_seen', '/workspace/.holdout/drone_seen_heldout']
SC = {0: 4, 1: 2, 2: 1}
views = {}
for root in SEEN:
    for p in sorted(glob.glob(f'{root}/*/*.png')):
        b = os.path.basename(p); m = re.match(r'(\d+)_L(\d)_(\d+)_(\d+)\.png', b)
        if not m or b in views or 'local' in p: continue
        f, l, cx, cy = map(int, m.groups())
        if l == 0 or not 1 <= f <= 249: continue
        s = SC[l]
        views[b] = dict(file=p, frame=f, level=l, s=s, region=np.array([cx - 480 * s, cy - 270 * s, cx + 480 * s, cy + 270 * s], float))
keys = sorted(views, key=lambda b: (views[b]['frame'], b))
print('views', len(keys), flush=True)
sift = cv2.SIFT_create(nfeatures=NF)
t0 = time.time()
for b in keys:
    v = views[b]; img = cv2.imread(v['file'], cv2.IMREAD_GRAYSCALE)
    kp, des = sift.detectAndCompute(img, None)
    v['pts'] = (np.array([k.pt for k in kp], np.float64) * v['s'] + v['region'][:2]) if kp else np.zeros((0, 2))
    v['des'] = des
print(f'sift {time.time() - t0:.0f}s', flush=True)


def warp(Hm, pts):
    ph = np.hstack([pts, np.ones((len(pts), 1))]) @ Hm.T
    return ph[:, :2] / ph[:, 2:3]


def mpow(Hm, k):
    return np.linalg.matrix_power(Hm, k) if k >= 0 else np.linalg.matrix_power(np.linalg.inv(Hm), -k)


def inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


byf = {}
for b in keys: byf.setdefault(views[b]['frame'], []).append(b)
bf = cv2.BFMatcher(cv2.NORM_L2)
groups = []; t0 = time.time(); npairs = 0
for b in keys:
    va = views[b]
    if va['des'] is None: continue
    for k in range(1, K + 1):
        for c in byf.get(va['frame'] + k, []):
            vb = views[c]
            if vb['des'] is None: continue
            pr = warp(mpow(H0, k), va['region'].reshape(2, 2)).ravel()
            small = min(np.prod(va['region'][2:] - va['region'][:2]), np.prod(vb['region'][2:] - vb['region'][:2]))
            if inter(pr, vb['region']) < 0.2 * small: continue
            m = bf.knnMatch(va['des'], vb['des'], k=2)
            good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
            if len(good) < 15: continue
            p = va['pts'][[g.queryIdx for g in good]]; q = vb['pts'][[g.trainIdx for g in good]]
            pred = warp(mpow(H0, k), p); ok = np.linalg.norm(pred - q, axis=1) < 60 + 8 * k
            p, q = p[ok], q[ok]
            if len(p) < 15: continue
            ws = max(va['s'], vb['s'])
            Hk, inl = cv2.findHomography(p, q, cv2.RANSAC, 2.0 * ws, maxIters=1000)
            if Hk is None: continue
            inl = inl.ravel().astype(bool)
            if inl.sum() < 15: continue
            p, q = p[inl], q[inl]
            if len(p) > MAXC:
                sel = np.random.default_rng(npairs).choice(len(p), MAXC, replace=False); p, q = p[sel], q[sel]
            groups.append((p, q, k, va['frame'], ws)); npairs += 1
print(f'pairs {npairs} corr {sum(len(g[0]) for g in groups)} {time.time() - t0:.0f}s', flush=True)


def mat(x):
    return np.array([[1 + x[0], x[1], x[2]], [x[3], 1 + x[4], x[5]], [x[6] * 1e-4, x[7] * 1e-4, 1.0]])


def x_of(Hm):
    Hm = Hm / Hm[2, 2]
    return np.array([Hm[0, 0] - 1, Hm[0, 1], Hm[0, 2], Hm[1, 0], Hm[1, 1] - 1, Hm[1, 2], Hm[2, 0] * 1e4, Hm[2, 1] * 1e4])


byk = {}
for p, q, k, f, ws in groups: byk.setdefault(k, []).append((p, q))
byk = {k: (np.vstack([a for a, _ in v]), np.vstack([b for _, b in v])) for k, v in byk.items()}


def resid(x):
    Hm = mat(x)
    return np.concatenate([((warp(np.linalg.matrix_power(Hm, k), p) - q) / k).ravel() for k, (p, q) in byk.items()])


r = least_squares(resid, x_of(H0), loss='huber', f_scale=2.0, max_nfev=200)
Hf = mat(r.x)
np.save(OUT + '.npy', Hf)
stats = {}
for name, Hm in (('H0', H0), ('fit', Hf)):
    per_k = {}
    for k, (p, q) in byk.items():
        e = np.linalg.norm(warp(np.linalg.matrix_power(Hm, k), p) - q, axis=1)
        per_k[k] = dict(n=int(len(e)), med=float(np.median(e)), p90=float(np.percentile(e, 90)))
    blocks = {}
    for p, q, k, f, ws in groups:
        e = np.linalg.norm(warp(mpow(Hm, k), p) - q, axis=1) / k
        blocks.setdefault(f // 50 * 50, []).append(float(np.median(e)))
    stats[name] = dict(per_k=per_k, per_block_median_px_per_frame={b: float(np.median(v)) for b, v in sorted(blocks.items())})
print('H0 ', np.array2string(H0, precision=8)); print('fit', np.array2string(Hf, precision=8))
print(json.dumps(stats, indent=1))
# per-frame drift check: fit a separate H on pairs starting in each 50-frame block
for b0 in range(0, 250, 50):
    sub = {}
    for p, q, k, f, ws in groups:
        if b0 <= f < b0 + 50: sub.setdefault(k, []).append((p, q))
    if not sub: continue
    sb = {k: (np.vstack([a for a, _ in v]), np.vstack([b for _, b in v])) for k, v in sub.items()}
    rr = least_squares(lambda x: np.concatenate([((warp(np.linalg.matrix_power(mat(x), k), p) - q) / k).ravel() for k, (p, q) in sb.items()]),
                       r.x, loss='huber', f_scale=2.0, max_nfev=100)
    Hb = mat(rr.x); c = np.array([[1920.0, 1080.0]])
    print(f'block {b0}: a11 {Hb[0,0]:.6f} a22 {Hb[1,1]:.6f} tx {Hb[0,2]:.2f} ty {Hb[1,2]:.2f} centre flow {warp(Hb, c)[0] - c[0]} vs global {warp(Hf, c)[0] - c[0]}')
json.dump(dict(H=Hf.tolist(), stats=stats, pairs=npairs), open(OUT + '.json', 'w'), indent=1)
print('FIT_DONE', flush=True)
