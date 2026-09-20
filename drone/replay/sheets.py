"""Crop sheets for fuse.py clusters (eyeball the class): for each cluster id take up to K recorded views (finest level
first, spread over the object's life) that contain its propagated box, crop with margin at native view resolution,
upscale to TILE px, draw the box, label with cluster id / vote / view level+frame / size. Rows of K tiles, ROWS rows per
sheet, sheets OUT_PREFIX_<n>.jpg.  Select ids explicitly or with --split (top vote < THR) / --late (tref >= 151)
/ --minsupport S.   Usage: python sheets.py H.npy clusters.json OUT_PREFIX [ids...] [--k 4] [--rows 8] [--split 0.75] [--late] [--minsupport 0.05] [--tile 224]
(v2: also reads the complete validation runs in /workspace/drone_seen_full, whose survey views are native L2)
"""
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

H = np.load(sys.argv[1]); CL = json.load(open(sys.argv[2])); OUT = sys.argv[3]
argv = sys.argv[4:]


def opt(name, default):
    return argv[argv.index(name) + 1] if name in argv else default


K = int(opt('--k', 4)); ROWS = int(opt('--rows', 8)); TILE = int(opt('--tile', 224)); NCOL = int(opt('--ncol', 1))  # clusters per sheet row
SPLIT = float(opt('--split', 0)) if '--split' in argv else None
LATE = '--late' in argv
MINS = float(opt('--minsupport', 0.0))
ids = [int(a) for a in argv if a.isdigit() and (argv.index(a) == 0 or argv[argv.index(a) - 1] not in ('--k', '--rows', '--split', '--minsupport', '--tile', '--ncol'))]
C = {c['id']: c for c in CL}
if not ids:
    for c in CL:
        if c.get('drop') or c['support'] < MINS: continue
        top = next(iter(c['vote'].values()))
        fam = c.get('fam', {})
        disagree = 'ft' in fam and 'v2' in fam and fam['ft'][0] != fam['v2'][0] and fam['v2'][2] >= 0.5
        if (SPLIT is not None and (top < SPLIT or disagree)) or (LATE and c['tref'] >= 151):
            ids.append(c['id'])
if '--idfile' in argv: ids += json.load(open(opt('--idfile', '')))
ids = sorted(set(ids), key=lambda i: -C[i]['support'])
print('clusters to sheet', len(ids))
Hi = np.linalg.inv(H); _pw = {}


def mpow(k):
    if k not in _pw: _pw[k] = np.linalg.matrix_power(H, k) if k >= 0 else np.linalg.matrix_power(Hi, -k)
    return _pw[k]


def warp_pts(M, pts):
    ph = np.hstack([pts, np.ones((len(pts), 1))]) @ M.T
    return ph[:, :2] / ph[:, 2:3]


def warp_box(M, box):
    x1, y1, x2, y2 = box
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    p = warp_pts(M, np.array([[cx, cy], [cx - w / 2, cy], [cx + w / 2, cy], [cx, cy - h / 2], [cx, cy + h / 2]]))
    nw = np.linalg.norm(p[2] - p[1]); nh = np.linalg.norm(p[4] - p[3])
    return np.array([p[0][0] - nw / 2, p[0][1] - nh / 2, p[0][0] + nw / 2, p[0][1] + nh / 2])


SC = {0: 4, 1: 2, 2: 1}
views = {}
for root in ('/workspace/drone_seen', '/workspace/.holdout/drone_seen_heldout', '/workspace/drone_seen_full'):
    for p in sorted(glob.glob(f'{root}/*/*.png')):
        b = os.path.basename(p); m = re.match(r'(\d+)_L(\d)_(\d+)_(\d+)\.png', b)
        if not m or b in views or 'local' in p: continue
        f, l, cx, cy = map(int, m.groups()); s = SC[l]
        views[b] = (f, l, (cx - 480 * s, cy - 270 * s, cx + 480 * s, cy + 270 * s), p)
byf = {}
for b, v in views.items(): byf.setdefault(v[0], []).append(v)


def box_at(c, t):
    a, bb = np.array(c['resid'][0]), np.array(c['resid'][1]); dt = t - c['tref']; sh = a + bb * dt
    return warp_box(mpow(dt), np.array(c['box_ref']) + np.array([sh[0], sh[1], sh[0], sh[1]]))


rows = []; sheet_no = 0; done = []


def flush():
    global rows, sheet_no
    if not rows: return
    while len(rows) % NCOL: rows.append(np.zeros_like(rows[0]))
    grid = [np.hstack([r if j == 0 else np.hstack([np.full((r.shape[0], 6, 3), 255, np.uint8), r]) for j, r in enumerate(rows[i:i + NCOL])])
            for i in range(0, len(rows), NCOL)]
    cv2.imwrite(f'{OUT}_{sheet_no}.jpg', np.vstack(grid), [cv2.IMWRITE_JPEG_QUALITY, 88]); print('wrote', f'{OUT}_{sheet_no}.jpg', len(rows), 'rows')
    sheet_no += 1; rows = []


for cid in ids:
    c = C[cid]; cands = []
    for t in range(max(1, c['tmin'] - 3), min(249, c['tmax'] + 3) + 1):
        b = box_at(c, t)
        for f, l, r, p in byf.get(t, []):
            if b[0] >= r[0] + 3 and b[1] >= r[1] + 3 and b[2] <= r[2] - 3 and b[3] <= r[3] - 3:
                cands.append((l, f, r, p, b))
    if not cands:
        print('no view for', cid); continue
    cands.sort(key=lambda x: (-x[0], x[1]))
    best_level = cands[0][0]; pool = [x for x in cands if x[0] == best_level]
    pick = [pool[int(i)] for i in sorted(set(np.linspace(0, len(pool) - 1, min(K, len(pool))).astype(int)))]
    if len(pick) < K: pick += [x for x in cands if x[0] != best_level][:K - len(pick)]
    tiles = []
    for l, f, r, p, b in pick:
        img = cv2.imread(p); s = SC[l]
        vx1, vx2 = (b[0] - r[0]) / s, (b[2] - r[0]) / s; vy1, vy2 = (b[1] - r[1]) / s, (b[3] - r[1]) / s
        cx, cy = (vx1 + vx2) / 2, (vy1 + vy2) / 2; half = max(28, max(vx2 - vx1, vy2 - vy1) * 1.2)
        x1, y1 = int(max(0, cx - half)), int(max(0, cy - half)); x2, y2 = int(min(960, cx + half)), int(min(540, cy + half))
        cr = img[y1:y2, x1:x2]
        if cr.size == 0: continue
        sc = TILE / max(x2 - x1, y2 - y1)
        cr = cv2.resize(cr, (int((x2 - x1) * sc), int((y2 - y1) * sc)), interpolation=cv2.INTER_CUBIC)
        pad = np.zeros((TILE, TILE, 3), np.uint8); pad[:cr.shape[0], :cr.shape[1]] = cr; cr = pad
        cv2.rectangle(cr, (int((vx1 - x1) * sc), int((vy1 - y1) * sc)), (int((vx2 - x1) * sc), int((vy2 - y1) * sc)), (0, 255, 255), 1)
        v = list(c['vote'].items())
        lab = f"#{cid} t{c['t_star']:.1f} x{c['x_star']:.0f} {v[0][0][:10]} {v[0][1]:.2f}" + (f" {v[1][0][:8]} {v[1][1]:.2f}" if len(v) > 1 else '')
        for txt, y, col in ((lab, 12, (0, 255, 255)), (f'L{l} f{f} {b[2]-b[0]:.0f}x{b[3]-b[1]:.0f} s{c["support"]:.2f} n{c["nviews"]}', TILE - 5, (255, 255, 255))):
            cv2.putText(cr, txt, (2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 3); cv2.putText(cr, txt, (2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, col, 1)
        tiles.append(cr)
    while len(tiles) < K: tiles.append(np.zeros((TILE, TILE, 3), np.uint8))
    rows.append(np.hstack(tiles[:K])); done.append(cid)
    if len(rows) >= ROWS * NCOL: flush()
flush()
print('SHEETS_DONE', done)
