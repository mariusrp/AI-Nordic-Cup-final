"""B5 step 1: a REAL-image YOLO training set for the flown validation city, labelled from a replay answer table.

Every recorded view (PNG `{frame:06d}_L{level}_{cx}_{cy}.png` with a meta.jsonl beside it) is a real 960x540 camera
image of the deterministic validation flight. The answer table (drone/replay/fuse.py output,
`{"<frame>": [[cls, [x1,y1,x2,y2] as fractions of 3840x2160, conf], ...]}`) is the offline, eye-checked
reconstruction of that flight, so unlike bb3/make_targets.py (frames <= 150, H-chained v1_plus clusters) it labels
ALL 249 frames.

Per table row, projected into the view with its level scale (source px per view px: L0 4, L1 2, L2 1):
    conf >= POS_CONF (0.5)              -> positive box, if >= MIN_PX (6) px wide/high in the view and >= VIS visible
    IGN_LO (0.15) <= conf < POS_CONF    -> ignore: painted grey (114), so a real-but-unsure object is never taught
                                           as background (same device as bb3/make_targets.py / build_views.py)
    conf < IGN_LO                       -> left as background. In table_v2b those rows are the clusters the eye pass
                                           DEMOTED to support 0.02-0.1 (roofs, bushes, blurry photogrammetry terrain);
                                           they are the FP-suppression signal we need on an unseen city, and there are
                                           29-87 of them per frame, so painting them would grey out the image.
A positive that is too small or too truncated is painted instead of dropped. Grey is never painted over a positive.

Views are deduplicated by (frame, level, cx, cy) across every recording root.
Usage: python table_views.py TABLE.json OUT_DIR DATA_YAML ROOT [ROOT ...]
  ROOT: a directory of recording dirs (e.g. /workspace/drone_seen); a subdir named 'local' is skipped (Helsinki replay,
  whose frame numbers would collide with the validation flight).
Env: POS_CONF=0.5 IGN_LO=0.15 MIN_PX=6 VIS=0.6 SPLIT=train
"""
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

TABLE, OUT, YAML = sys.argv[1], sys.argv[2], sys.argv[3]
ROOTS = sys.argv[4:]
POS_CONF = float(os.environ.get('POS_CONF', 0.5))
IGN_LO = float(os.environ.get('IGN_LO', 0.15))
MIN_PX = float(os.environ.get('MIN_PX', 6))
VIS = float(os.environ.get('VIS', 0.6))
SPLIT = os.environ.get('SPLIT', 'train')
W, H = 3840.0, 2160.0
SC = {0: 4, 1: 2, 2: 1}


def class_index(data_yaml):
    """names: {0: hangar, ...} out of the synthetic data.yaml = the upstream OBJECT_CLASSES order."""
    names = {}
    for line in open(data_yaml).read().split('names:')[1].splitlines():
        m = re.match(r'\s+(\d+):\s*(\S+)', line)
        if m:
            names[int(m.group(1))] = m.group(2)
    return {n: i for i, n in names.items()}


CI = class_index(YAML)
T = json.load(open(TABLE))
os.makedirs(f'{OUT}/images/{SPLIT}', exist_ok=True)
os.makedirs(f'{OUT}/labels/{SPLIT}', exist_ok=True)

# ------------------------------------------------------------------ collect the views (deduplicated)
views = {}   # (frame, level, cx, cy) -> (png path, region)
for root in ROOTS:
    for meta in sorted(glob.glob(f'{root}/*/meta.jsonl')):
        d = os.path.dirname(meta)
        if os.path.basename(d) == 'local':
            continue
        for line in open(meta):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            f, lv = int(r['frame']), int(r['level'])
            if not 1 <= f <= 249 or 'file' not in r:
                continue
            p = os.path.join(d, r['file'])
            if not os.path.exists(p):
                continue
            key = (f, lv, int(r['cx']), int(r['cy']))
            views.setdefault(key, (p, [float(v) for v in r['region']]))
print('recording roots', len(ROOTS), 'unique views', len(views), flush=True)

# ------------------------------------------------------------------ write one YOLO sample per view
n_img = n_pos = n_ign = n_small = 0
per_class = {}
for (f, lv, cx, cy), (path, region) in sorted(views.items()):
    img = cv2.imread(path)
    if img is None or img.shape[0] != 540 or img.shape[1] != 960:
        continue
    s = SC[lv]
    rx, ry = region[0], region[1]
    lines, grey = [], []
    for cls, b, conf in T.get(str(f), []):
        if conf < IGN_LO or cls not in CI:
            continue
        a = [(b[0] * W - rx) / s, (b[1] * H - ry) / s, (b[2] * W - rx) / s, (b[3] * H - ry) / s]
        q = [max(0.0, a[0]), max(0.0, a[1]), min(960.0, a[2]), min(540.0, a[3])]
        if q[2] <= q[0] or q[3] <= q[1]:
            continue                                     # outside this view
        if conf < POS_CONF:
            grey.append(q)                               # unsure -> ignore region
            continue
        vis = (q[2] - q[0]) * (q[3] - q[1]) / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
        if vis >= VIS and q[2] - q[0] >= MIN_PX and q[3] - q[1] >= MIN_PX:
            lines.append(f'{CI[cls]} {(q[0] + q[2]) / 1920:.6f} {(q[1] + q[3]) / 1080:.6f} '
                         f'{(q[2] - q[0]) / 960:.6f} {(q[3] - q[1]) / 540:.6f}')
            per_class[cls] = per_class.get(cls, 0) + 1
        else:
            grey.append(q)                               # truncated or below MIN_PX: neither positive nor background
            n_small += 1
    # paint the ignore regions, never over a positive box
    P = [list(map(float, ln.split()[1:])) for ln in lines]
    for q in grey:
        x1, y1 = int(q[0]), int(q[1])
        x2, y2 = int(np.ceil(q[2])), int(np.ceil(q[3]))
        patch = np.zeros((540, 960), bool)
        patch[y1:y2, x1:x2] = True
        for p in P:
            px1, py1 = int((p[0] - p[2] / 2) * 960), int((p[1] - p[3] / 2) * 540)
            px2, py2 = int(np.ceil((p[0] + p[2] / 2) * 960)), int(np.ceil((p[1] + p[3] / 2) * 540))
            patch[max(0, py1 - 1):py2 + 1, max(0, px1 - 1):px2 + 1] = False
        img[patch] = 114
    name = f'vc{f:03d}_L{lv}_{cx}_{cy}'
    cv2.imwrite(f'{OUT}/images/{SPLIT}/{name}.png', img)
    open(f'{OUT}/labels/{SPLIT}/{name}.txt', 'w').write('\n'.join(lines) + ('\n' if lines else ''))
    n_img += 1
    n_pos += len(lines)
    n_ign += len(grey)
print('TABLE_VIEWS views', n_img, 'pos boxes', n_pos, 'ignore regions', n_ign, 'demoted-to-ignore', n_small)
print('TABLE_VIEWS per class', json.dumps(dict(sorted(per_class.items(), key=lambda kv: -kv[1]))))
