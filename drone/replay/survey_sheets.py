"""Hunt sheets for objects no detector fired on: the SURVEY runs' native level-2 views (drone_seen_full/<seq>/, row
y=1080) with every table box of that frame drawn on them (green = conf >= HI, yellow = LO..HI, labelled class+conf).
An unboxed vehicle / aircraft / tower / hangar on a sheet is a missed object: add it as a manual cluster
({"add": true, "cls": c, "t": frame, "box": [x1,y1,x2,y2] source px} in the overrides file; the view's source region
is printed on each tile, source px = region x1/y1 + view px).
Usage: python survey_sheets.py TABLE.json OUT_PREFIX RUN_DIR[,RUN_DIR2] [--frames 150-249] [--step 1] [--scale 0.5]
       [--cols 2] [--rows 3] [--lo 0.05] [--hi 0.3]"""
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

T = json.load(open(sys.argv[1])); OUT = sys.argv[2]; RUNS = sys.argv[3].split(',')
argv = sys.argv[4:]


def opt(n, d):
    return argv[argv.index(n) + 1] if n in argv else d


F0, F1 = map(int, opt('--frames', '1-249').split('-')); STEP = int(opt('--step', 1)); SC = float(opt('--scale', 0.5))
COLS = int(opt('--cols', 2)); ROWS = int(opt('--rows', 3)); LO = float(opt('--lo', 0.05)); HI = float(opt('--hi', 0.3))
views = []
for rd in RUNS:
    for p in sorted(glob.glob(f'{rd}/*_L2_*.png') + glob.glob(f'{rd}/*_L2_*.jpg')):
        f, l, cx, cy = map(int, re.match(r'(\d+)_L(\d)_(\d+)_(\d+)\.', os.path.basename(p)).groups())
        if F0 <= f <= F1 and (f - F0) % STEP == 0: views.append((f, cx, cy, p))
views.sort()
tiles = []; n = 0
AB = dict(small_launcher='sLa', medium_launcher='mLa', large_launcher='LLa', small_plane='sPl', medium_plane='mPl', jet_plane='jet',
          small_tower='sTw', large_tower='LTw', mine_roller='mRo', spacecraft='spc', helicopter='hel', hangar='hng', jammer='jam',
          tank='tnk', condor='cnd')


def flush():
    global tiles, n
    if not tiles: return
    while len(tiles) % COLS: tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)]
    cv2.imwrite(f'{OUT}_{n:03d}.jpg', np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85]); print('wrote', f'{OUT}_{n:03d}.jpg'); n += 1; tiles = []


for f, cx, cy, p in views:
    img = cv2.imread(p); rx, ry = cx - 480, cy - 270
    big = cv2.resize(img, None, fx=SC * 2, fy=SC * 2, interpolation=cv2.INTER_CUBIC) if SC * 2 != 1 else img.copy()
    k = SC * 2
    for c, b, s in T.get(str(f), []):
        if s < LO: continue
        x1, y1, x2, y2 = b[0] * 3840 - rx, b[1] * 2160 - ry, b[2] * 3840 - rx, b[3] * 2160 - ry
        if x2 < 0 or y2 < 0 or x1 > 960 or y1 > 540: continue
        col = (0, 255, 0) if s >= HI else (0, 220, 255)
        cv2.rectangle(big, (int(x1 * k), int(y1 * k)), (int(x2 * k), int(y2 * k)), col, 1)
        cv2.putText(big, f'{AB.get(c, c)}{s:.2f}', (int(x1 * k), max(8, int(y1 * k) - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
    lab = f'f{f} region x{rx} y{ry} {os.path.basename(os.path.dirname(p))[:4]}'
    cv2.putText(big, lab, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3); cv2.putText(big, lab, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    tiles.append(big)
    if len(tiles) >= COLS * ROWS: flush()
flush()
print('SURVEY_SHEETS_DONE', len(views))
