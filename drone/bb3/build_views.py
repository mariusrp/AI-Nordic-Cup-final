"""B3 step 2: YOLO dataset from RECORDED validation-city views (frames <= 150 only) + targets from make_targets.py.
Each recorded view <frame>_L<level>_<cx>_<cy>.png is a real 960x540 camera image. Label = target box clipped to the view
if >= VIS of it is inside; partially visible targets and ignore regions are painted grey (114) so they are neither
positives nor negatives. Duplicate views (same frame/level/centre from different runs) are written once.
Split: frames in [TRAIN_LO, TRAIN_HI] -> train, frames in [VAL_LO, VAL_HI] -> val (temporal fold with a buffer).
Usage: python build_views.py SEEN_DIR TARGETS.json OUT_DIR TRAIN_LO TRAIN_HI VAL_LO VAL_HI CLASSNAMES_JSON"""
import sys, os, re, glob, json, numpy as np, cv2
SEEN, TG, OUT = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3]
TLO, THI, VLO, VHI = map(int, sys.argv[4:8]); NAMES = json.load(open(sys.argv[8]))
CI = {n: i for i, n in enumerate(NAMES)}
SC = {0: 4, 1: 2, 2: 1}; VIS = 0.6
MAXF = int(os.environ.get('MAXF', 150))
for s in ('train', 'val'):
    os.makedirs(f'{OUT}/images/{s}', exist_ok=True); os.makedirs(f'{OUT}/labels/{s}', exist_ok=True)
seen = set(); n = {'train': 0, 'val': 0}; nb = {'train': 0, 'val': 0}
for p in sorted(glob.glob(f'{SEEN}/*/*.png')):
    m = re.match(r'(\d+)_L(\d)_(\d+)_(\d+)\.png', os.path.basename(p))
    if not m: continue
    f, l, cx, cy = map(int, m.groups())
    if f > MAXF or f < 1: continue  # frames > MAXF are never used (default 150: frames >= 151 are holdout)
    split = 'train' if TLO <= f <= THI else 'val' if VLO <= f <= VHI else None
    if split is None or (f, l, cx, cy) in seen: continue
    seen.add((f, l, cx, cy))
    img = cv2.imread(p)
    s = SC[l]; rx, ry = cx - 480 * s, cy - 270 * s
    t = TG[str(f)]; lines = []; grey = []
    for c, x1, y1, x2, y2 in t['pos']:
        a = [(x1 - rx) / s, (y1 - ry) / s, (x2 - rx) / s, (y2 - ry) / s]
        b = [max(0, a[0]), max(0, a[1]), min(960, a[2]), min(540, a[3])]
        if b[2] <= b[0] or b[3] <= b[1]: continue
        vis = (b[2] - b[0]) * (b[3] - b[1]) / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
        if vis >= VIS and b[2] - b[0] >= 2 and b[3] - b[1] >= 2:
            lines.append(f'{CI[c]} {(b[0] + b[2]) / 1920:.6f} {(b[1] + b[3]) / 1080:.6f} {(b[2] - b[0]) / 960:.6f} {(b[3] - b[1]) / 540:.6f}')
        else: grey.append(b)
    for x1, y1, x2, y2 in t['ign']:
        b = [max(0, (x1 - rx) / s), max(0, (y1 - ry) / s), min(960, (x2 - rx) / s), min(540, (y2 - ry) / s)]
        if b[2] > b[0] and b[3] > b[1]: grey.append(b)
    # never grey over a positive box
    P = [list(map(float, ln.split()[1:])) for ln in lines]
    for b in grey:
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(np.ceil(b[2])), int(np.ceil(b[3]))
        patch = np.zeros((540, 960), bool); patch[y1:y2, x1:x2] = True
        for q in P:
            qx1, qy1 = int((q[0] - q[2] / 2) * 960), int((q[1] - q[3] / 2) * 540)
            qx2, qy2 = int(np.ceil((q[0] + q[2] / 2) * 960)), int(np.ceil((q[1] + q[3] / 2) * 540))
            patch[max(0, qy1 - 1):qy2 + 1, max(0, qx1 - 1):qx2 + 1] = False
        img[patch] = 114
    name = f'vc_{f:03d}_L{l}_{cx}_{cy}'
    cv2.imwrite(f'{OUT}/images/{split}/{name}.png', img)
    open(f'{OUT}/labels/{split}/{name}.txt', 'w').write('\n'.join(lines) + ('\n' if lines else ''))
    n[split] += 1; nb[split] += len(lines)
print('views', n, 'boxes', nb)
