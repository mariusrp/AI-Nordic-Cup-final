"""Eyeball crops for specific valcity candidates: for each id (a cands.json cluster id, or NAME@t:x1,y1,x2,y2 in
source px) take up to K recorded views (frames <= 150, finest level first, spread over the object's life) that
contain the box, crop with margin at native view resolution, upscale x4 and tile them into one sheet.
Usage: python crops.py H.npy SEEN_DIR cands.json OUT.jpg id1 id2 ... [--k 3]"""
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

H = np.load(sys.argv[1]); SEEN = sys.argv[2]; C = {str(c['id']): c for c in json.load(open(sys.argv[3]))}
OUT = sys.argv[4]; IDS = [a for a in sys.argv[5:] if not a.startswith('--')]
K = int(sys.argv[sys.argv.index('--k') + 1]) if '--k' in sys.argv else 3
SC = {0: 4, 1: 2, 2: 1}
F = {0: np.eye(3)}
for t in range(1, 151):
    F[t] = F[t - 1] @ H


def pt(M, x, y):
    q = M @ [x, y, 1.0]
    return q[0] / q[2], q[1] / q[2]


views, seen = [], set()
for p in sorted(glob.glob(f'{SEEN}/*/*.png')):
    m = re.match(r'(\d+)_L(\d)_(\d+)_(\d+)\.png', os.path.basename(p))
    if not m or m.groups() in seen:
        continue
    seen.add(m.groups()); f, l, cx, cy = map(int, m.groups()); s = SC[l]
    if 1 <= f <= 150:
        views.append((f, l, (cx - 480 * s, cy - 270 * s, cx + 480 * s, cy + 270 * s), p))


def box_at(spec, t):
    if '@' in spec:  # NAME@t0:x1,y1,x2,y2 -> propagate from t0
        name, rest = spec.split('@'); t0, b = rest.split(':'); t0 = int(t0); x1, y1, x2, y2 = map(float, b.split(','))
        M = F[t] @ np.linalg.inv(F[t0])
        (ax, ay), (bx, by) = pt(M, x1, y1), pt(M, x2, y2)
        return name, (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))
    c = C[spec]; x, y = pt(F[t], c['gx'], c['gy']); w, h = c['w'], c['h']
    return f"#{spec} {max(c.get('votes', {}), key=c.get('votes', {}).get) if c.get('votes') else ''}", (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


TILE = 256
rows = []
for spec in IDS:
    cands = []
    for f, l, r, p in views:
        name, b = box_at(spec, f)
        if b[0] >= r[0] + 4 and b[1] >= r[1] + 4 and b[2] <= r[2] - 4 and b[3] <= r[3] - 4:
            cands.append((l, f, r, p, b, name))
    if not cands:
        print('no view for', spec); continue
    cands.sort(key=lambda c: (-c[0], c[1]))
    best_level = cands[0][0]
    pool = [c for c in cands if c[0] == best_level]
    pick = [pool[int(i)] for i in np.linspace(0, len(pool) - 1, min(K, len(pool)))]
    if len(pick) < K:
        pick += [c for c in cands if c[0] != best_level][:K - len(pick)]
    tiles = []
    for l, f, r, p, b, name in pick:
        img = cv2.imread(p); s = SC[l]
        vx1, vx2 = (b[0] - r[0]) / s, (b[2] - r[0]) / s; vy1, vy2 = (b[1] - r[1]) / s, (b[3] - r[1]) / s
        cx, cy = (vx1 + vx2) / 2, (vy1 + vy2) / 2; half = max(24, (max(vx2 - vx1, vy2 - vy1) * 1.6) / 2)
        x1, y1 = int(max(0, cx - half)), int(max(0, cy - half)); x2, y2 = int(min(960, cx + half)), int(min(540, cy + half))
        cr = img[y1:y2, x1:x2]
        if cr.size == 0:
            continue
        cr = cv2.resize(cr, (TILE, TILE), interpolation=cv2.INTER_CUBIC)
        sc = TILE / (x2 - x1)
        cv2.rectangle(cr, (int((vx1 - x1) * sc), int((vy1 - y1) * sc)), (int((vx2 - x1) * sc), int((vy2 - y1) * sc)), (0, 255, 255), 1)
        for txt, y, col in ((f'{name}', 14, (0, 255, 255)), (f'L{l} f{f} {b[2]-b[0]:.0f}x{b[3]-b[1]:.0f}src', TILE - 6, (255, 255, 255))):
            cv2.putText(cr, txt, (3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3); cv2.putText(cr, txt, (3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        tiles.append(cr)
    while len(tiles) < K:
        tiles.append(np.zeros((TILE, TILE, 3), np.uint8))
    rows.append(np.hstack(tiles[:K]))
cv2.imwrite(OUT, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
print('wrote', OUT, len(rows), 'rows')
