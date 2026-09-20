"""B5 step 2: YOLO training views from the 25 labelled Helsinki frames - the only ground-truth data we have from a
city OTHER than the flown validation city, and therefore the only training signal for the final flight.

Each 4K frame is tiled into 960x540 camera views exactly the way the evaluator renders one (crop the level region,
INTER_AREA downscale by 4 / 2 / 1 for level 0 / 1 / 2), and the frame's GT boxes are projected into each tile.
Truncated boxes (< VIS visible) are painted grey (114) instead of being taught as background, like table_views.py.

Frames listed in --exclude are never tiled: they are the held-out evaluation frames. Note that make_synth.py's
HOLDOUT_FRAMES = {4, 12, 20} are the only Helsinki frames whose objects were never cut out for the synthetic set,
so they are the only frames no existing checkpoint has seen in any form - keep them inside --exclude.

Usage: python helsinki_views.py SCENE_DIR OUT_DIR DATA_YAML [--exclude 4,12,20,22,24] [--levels 0,1,2] [--split train]
"""
import json
import os
import re
import sys

import cv2
import numpy as np

SCENE, OUT, YAML = sys.argv[1], sys.argv[2], sys.argv[3]
argv = sys.argv[4:]


def opt(name, default):
    return argv[argv.index(name) + 1] if name in argv else default


EXCL = {int(v) for v in opt('--exclude', '4,12,20,22,24').split(',') if v != ''}
LEVELS = [int(v) for v in opt('--levels', '0,1,2').split(',')]
SPLIT = opt('--split', 'train')
VIS = float(os.environ.get('VIS', 0.6))
MIN_PX = float(os.environ.get('MIN_PX', 6))
REG = {0: (3840, 2160), 1: (1920, 1080), 2: (960, 540)}


def class_index(data_yaml):
    names = {}
    for line in open(data_yaml).read().split('names:')[1].splitlines():
        m = re.match(r'\s+(\d+):\s*(\S+)', line)
        if m:
            names[int(m.group(1))] = m.group(2)
    return {n: i for i, n in names.items()}


def tiles(level, overlap=1.0):
    """Non-overlapping cover of the 4K frame at `level` (overlap 1.0), or det_eval.py's 1.3x overlapped cover."""
    rw, rh = REG[level]
    nx = max(1, int(np.ceil(3840 / rw * overlap)))
    ny = max(1, int(np.ceil(2160 / rh * overlap)))
    xs = np.linspace(0, 3840 - rw, nx).astype(int)
    ys = np.linspace(0, 2160 - rh, ny).astype(int)
    return [(int(x), int(y), int(x + rw), int(y + rh)) for y in ys for x in xs]


CI = class_index(YAML)
os.makedirs(f'{OUT}/images/{SPLIT}', exist_ok=True)
os.makedirs(f'{OUT}/labels/{SPLIT}', exist_ok=True)
frames = sorted(int(re.findall(r'(\d+)', f)[0]) for f in os.listdir(f'{SCENE}/images') if f.endswith('.png'))
use = [f for f in frames if f not in EXCL]
print('helsinki frames', len(frames), 'excluded', sorted(EXCL & set(frames)), 'tiled', len(use), flush=True)

n_img = n_box = 0
per_class = {}
for f in use:
    img4k = cv2.imread(f'{SCENE}/images/frame_{f:06d}.png')
    if img4k is None:
        continue
    ann = json.load(open(f'{SCENE}/annotations/frame_{f:06d}.json'))['annotations']
    for L in LEVELS:
        s = {0: 4, 1: 2, 2: 1}[L]
        for (x1, y1, x2, y2) in tiles(L):
            view = img4k[y1:y2, x1:x2]
            if L < 2:
                view = cv2.resize(view, (960, 540), interpolation=cv2.INTER_AREA)
            else:
                view = view.copy()
            lines, grey = [], []
            for a in ann:
                if a['object_id'] not in CI:
                    continue
                b = a['bbox']
                v = [(b[0] - x1) / s, (b[1] - y1) / s, (b[2] - x1) / s, (b[3] - y1) / s]
                q = [max(0.0, v[0]), max(0.0, v[1]), min(960.0, v[2]), min(540.0, v[3])]
                if q[2] <= q[0] or q[3] <= q[1]:
                    continue
                vis = (q[2] - q[0]) * (q[3] - q[1]) / max(1e-6, (v[2] - v[0]) * (v[3] - v[1]))
                if vis >= VIS and q[2] - q[0] >= MIN_PX and q[3] - q[1] >= MIN_PX:
                    lines.append(f'{CI[a["object_id"]]} {(q[0] + q[2]) / 1920:.6f} {(q[1] + q[3]) / 1080:.6f} '
                                 f'{(q[2] - q[0]) / 960:.6f} {(q[3] - q[1]) / 540:.6f}')
                    per_class[a['object_id']] = per_class.get(a['object_id'], 0) + 1
                else:
                    grey.append(q)
            P = [list(map(float, ln.split()[1:])) for ln in lines]
            for q in grey:
                gx1, gy1 = int(q[0]), int(q[1])
                gx2, gy2 = int(np.ceil(q[2])), int(np.ceil(q[3]))
                patch = np.zeros((540, 960), bool)
                patch[gy1:gy2, gx1:gx2] = True
                for p in P:
                    px1, py1 = int((p[0] - p[2] / 2) * 960), int((p[1] - p[3] / 2) * 540)
                    px2, py2 = int(np.ceil((p[0] + p[2] / 2) * 960)), int(np.ceil((p[1] + p[3] / 2) * 540))
                    patch[max(0, py1 - 1):py2 + 1, max(0, px1 - 1):px2 + 1] = False
                view[patch] = 114
            name = f'hel{f:03d}_L{L}_{x1}_{y1}'
            cv2.imwrite(f'{OUT}/images/{SPLIT}/{name}.png', view)
            open(f'{OUT}/labels/{SPLIT}/{name}.txt', 'w').write('\n'.join(lines) + ('\n' if lines else ''))
            n_img += 1
            n_box += len(lines)
print('HELSINKI_VIEWS views', n_img, 'boxes', n_box)
print('HELSINKI_VIEWS per class', json.dumps(dict(sorted(per_class.items(), key=lambda kv: -kv[1]))))
