"""Crispness hunt (table v3): dense fine-scale sharpness map of every native level-2 survey view -> blob candidates ->
canonical (x*, t*) (centre-line crossing, same frame as fuse.py) -> merged over views -> ranked list of crisp blobs that
no confident / eye-judged cluster explains. The placed 3D models render sharp, photogrammetry terrain is blurry.
Per view: L = |Laplacian| of grey; inner = box mean of L over WxW (W = 9 and 17); ring = box mean over 4Wx4W;
ratio = inner / (ring + 1). Peaks of ratio*sqrt(inner) (max filter 15 px) with ratio >= RMIN and inner >= IMIN.
Usage: python crisp_hunt.py H.npy CLUSTERS.json OUT.json VIEWDIR[,..] [--rmin 1.9] [--imin 5] [--known 0.2]
Output: list of {x, t, n (views), r (median ratio), i (median inner), w (window), near (nearest cluster: cls, support, dist)}"""
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

H = np.load(sys.argv[1]); CL = json.load(open(sys.argv[2])); OUT = sys.argv[3]; DIRS = sys.argv[4].split(','); argv = sys.argv[5:]


def opt(n, d):
    return argv[argv.index(n) + 1] if n in argv else d


RMIN = float(opt('--rmin', 1.9)); IMIN = float(opt('--imin', 5)); KNOWN = float(opt('--known', 0.2)); YC = 1080.0
Hi = np.linalg.inv(H)
VC = float(np.linalg.norm((H @ [1920, YC, 1])[:2] - [1920, YC]))


def canon(pts, f):
    """pts (n,2) in frame f -> (t*, x*) by stepping H / H^-1 until y crosses YC (linear interp within the step)."""
    out = np.full((len(pts), 2), np.nan)
    for M, sgn, act in ((H, 1, pts[:, 1] < YC), (Hi, -1, pts[:, 1] >= YC)):
        idx = np.where(act)[0]; P = pts[idx].copy(); k = 0
        while len(idx) and k < 20:
            Q = np.hstack([P, np.ones((len(P), 1))]) @ M.T; Q = Q[:, :2] / Q[:, 2:3]
            cr = (Q[:, 1] >= YC) if sgn > 0 else (Q[:, 1] <= YC)
            j = np.where(cr)[0]
            if len(j):
                fr = (YC - P[j, 1]) / (Q[j, 1] - P[j, 1])
                out[idx[j], 0] = f + sgn * (k + fr); out[idx[j], 1] = P[j, 0] + fr * (Q[j, 0] - P[j, 0])
            idx = idx[~cr]; P = Q[~cr]; k += 1
    return out


cands = []
views = []
for d in DIRS:
    for p in glob.glob(f'{d}/*_L2_*.*'):
        f, l, cx, cy = map(int, re.match(r'(\d+)_L(\d)_(\d+)_(\d+)\.', os.path.basename(p)).groups()); views.append((f, cx, cy, p))
views.sort()
for vi, (f, cx, cy, p) in enumerate(views):
    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    L = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=1))
    for W in (9, 17):
        inner = cv2.blur(L, (W, W)); big = cv2.blur(L, (4 * W, 4 * W))
        ring = (big * 16 - inner) / 15.0
        ratio = inner / (ring + 1.0)
        sc = ratio * np.sqrt(np.maximum(inner, 0))
        mx = cv2.dilate(sc, np.ones((15, 15), np.uint8))
        pk = (sc >= mx) & (ratio >= RMIN) & (inner >= IMIN)
        pk[:W, :] = pk[-W:, :] = False; pk[:, :W] = pk[:, -W:] = False
        vs, us = np.nonzero(pk)
        if not len(us): continue
        src = np.stack([us + cx - 480.0, vs + cy - 270.0], 1)
        cn = canon(src, f)
        for k in range(len(us)):
            if np.isfinite(cn[k, 0]): cands.append((cn[k, 1], cn[k, 0], float(ratio[vs[k], us[k]]), float(inner[vs[k], us[k]]), W, f))
    if vi % 100 == 0: print('view', vi, len(cands), flush=True)
cands = np.array(cands)
print('raw peaks', len(cands), flush=True)
# merge in canonical space (x*, VC t*) radius 14 px
order = np.argsort(-cands[:, 2] * np.sqrt(cands[:, 3]))
cx_, cy_, mem = [], [], []
for i in order:
    x, y = cands[i, 0], VC * cands[i, 1]
    if cx_:
        d = np.hypot(np.array(cx_) - x, np.array(cy_) - y); j = int(np.argmin(d))
        if d[j] < 14: mem[j].append(i); continue
    cx_.append(x); cy_.append(y); mem.append([i])
KC = [c for c in CL if not c.get('drop')]
KX = np.array([c['x_star'] for c in KC]); KT = np.array([c['t_star'] for c in KC]); KS = np.array([c['support'] for c in KC])
KJ = np.array([bool(c.get('judged')) for c in KC]); KR = np.array([max(12.0, 0.5 * np.hypot(c['w_star'], c['h_star'])) for c in KC])
res = []
for m in mem:
    m = np.array(m); fr = set(cands[m, 5].astype(int))
    x, t = float(np.median(cands[m, 0])), float(np.median(cands[m, 1]))
    d = np.hypot(KX - x, VC * (KT - t))
    rel = (KS >= KNOWN) | KJ
    near = None
    if rel.any():
        dd = np.where(rel, d - KR, 1e9); j = int(np.argmin(dd))
        near = dict(cls=KC[j]['cls'], s=round(float(KS[j]), 3), judged=bool(KJ[j]), d=round(float(d[j]), 1), covered=bool(dd[j] <= 6), t=round(KC[j]['t_star'], 1), x=round(KC[j]['x_star']))
    res.append(dict(x=round(x, 1), t=round(t, 2), n=len(fr), r=round(float(np.median(cands[m, 2])), 2), i=round(float(np.median(cands[m, 3])), 1),
                    rmax=round(float(cands[m, 2].max()), 2), w=int(np.median(cands[m, 4])), near=near))
res.sort(key=lambda r: -(r['r'] * np.sqrt(r['i']) * min(r['n'], 4)))
json.dump(res, open(OUT, 'w'), indent=0)
print('CRISP_HUNT_DONE blobs', len(res), 'uncovered', sum(1 for r in res if not (r['near'] and r['near']['covered'])))
