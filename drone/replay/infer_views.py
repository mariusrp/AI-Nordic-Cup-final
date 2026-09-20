"""Offline multi-model / multi-scale inference on EVERY unique recorded view of the validation flight (no latency
limit). Views are the 960x540 PNGs `{frame:06d}_L{level}_{cx}_{cy}.png` of /workspace/drone_seen/<rec>/ (frames <= 150)
and /workspace/.holdout/drone_seen_heldout/<rec>/ (151-249); identical file names across recordings are the same
image (deterministic flight) and are run once. Boxes are lifted to SOURCE pixels (3840x2160) via the view region
(level scale 4/2/1 source px per view px). One jsonl per (model, imgsz): {file, frame, level, region, dets:[[cls, x1,y1,x2,y2, conf]]}.
Usage: python infer_views.py OUT_DIR name=weights.pt [name=weights.pt ...] --imgsz 1280,1920 [--conf 0.01] [--flip]
       [--roots dirA:dirB] (default: drone_seen + holdout + drone_seen_full) [--levels 2] (only these levels) [--batch 16]
       [--skip-dets DIR] (skip views already present in DIR/*.jsonl, e.g. the first run's dets)
drone_seen_full/<seq>/ holds complete validation runs (all 249 frames), incl. SURVEY runs whose level-2 views sweep
the centre row y=1080 at native resolution.
"""
import glob
import json
import os
import re
import sys
import time

import numpy as np

OUT = sys.argv[1]
models = [a.split('=', 1) for a in sys.argv[2:] if '=' in a and not a.startswith('--')]
imgszs = [int(v) for v in (sys.argv[sys.argv.index('--imgsz') + 1] if '--imgsz' in sys.argv else '1280').split(',')]
CONF = float(sys.argv[sys.argv.index('--conf') + 1]) if '--conf' in sys.argv else 0.01
FLIP = '--flip' in sys.argv
SEEN = (sys.argv[sys.argv.index('--roots') + 1].split(':') if '--roots' in sys.argv else
        ['/workspace/drone_seen', '/workspace/.holdout/drone_seen_heldout', '/workspace/drone_seen_full'])
LEVELS = {int(v) for v in sys.argv[sys.argv.index('--levels') + 1].split(',')} if '--levels' in sys.argv else {0, 1, 2}
SKIP = set()
if '--skip-dets' in sys.argv:
    for fp in glob.glob(sys.argv[sys.argv.index('--skip-dets') + 1] + '/*.jsonl'):
        for line in open(fp):
            SKIP.add(json.loads(line)['file'])
        break  # every jsonl of a dets dir covers the same views
SC = {0: 4, 1: 2, 2: 1}
os.makedirs(OUT, exist_ok=True)

views = {}
for root in SEEN:
    for p in sorted(glob.glob(f'{root}/*/*.png')):
        b = os.path.basename(p)
        m = re.match(r'(\d+)_L(\d)_(\d+)_(\d+)\.png', b)
        if not m or b in views or b in SKIP or os.path.basename(os.path.dirname(p)) == 'local':
            continue
        f, l, cx, cy = map(int, m.groups())
        if not 1 <= f <= 249 or l not in LEVELS:
            continue
        s = SC[l]
        views[b] = dict(file=p, frame=f, level=l, region=[cx - 480 * s, cy - 270 * s, cx + 480 * s, cy + 270 * s])
names_ref = json.load(open('/workspace/ft/names.json'))
files = sorted(views)
print('unique views', len(files), 'L0/1/2', [sum(1 for v in views.values() if v['level'] == l) for l in (0, 1, 2)], flush=True)

from ultralytics import YOLO  # noqa: E402

for name, w in models:
    model = YOLO(w, task='detect')
    names = model.names
    missing = set(names.values()) ^ set(names_ref)
    print(name, w, 'classes', len(names), 'symdiff vs names.json', sorted(missing), flush=True)
    for imgsz in imgszs:
        for flip in ([False, True] if FLIP else [False]):
            tag = f'{name}_{imgsz}' + ('_flip' if flip else '')
            op = f'{OUT}/{tag}.jsonl'
            if os.path.exists(op) and os.path.getsize(op) > 0:
                print('skip existing', op, flush=True); continue
            t0 = time.time(); n = 0
            with open(op + '.tmp', 'w') as fo:
                B = int(sys.argv[sys.argv.index('--batch') + 1]) if '--batch' in sys.argv else 16
                for i in range(0, len(files), B):
                    chunk = files[i:i + B]
                    src = [views[b]['file'] for b in chunk]
                    if flip:
                        import cv2
                        src = [cv2.imread(p)[:, ::-1].copy() for p in src]
                    res = model.predict(src, imgsz=imgsz, conf=CONF, iou=0.6, max_det=300, half=True, verbose=False, device=0)
                    for b, r in zip(chunk, res):
                        v = views[b]; s = SC[v['level']]; rx, ry = v['region'][:2]
                        dets = []
                        if r.boxes is not None and len(r.boxes):
                            xyxy = r.boxes.xyxy.cpu().numpy(); cls = r.boxes.cls.cpu().numpy().astype(int); cf = r.boxes.conf.cpu().numpy()
                            if flip:
                                xyxy = xyxy.copy(); x1 = 960 - xyxy[:, 2]; x2 = 960 - xyxy[:, 0]; xyxy[:, 0] = x1; xyxy[:, 2] = x2
                            for j in range(len(cf)):
                                dets.append([names[int(cls[j])], round(float(rx + xyxy[j, 0] * s), 1), round(float(ry + xyxy[j, 1] * s), 1),
                                             round(float(rx + xyxy[j, 2] * s), 1), round(float(ry + xyxy[j, 3] * s), 1), round(float(cf[j]), 4)])
                        n += len(dets)
                        fo.write(json.dumps(dict(file=b, frame=v['frame'], level=v['level'], region=v['region'], dets=dets)) + '\n')
            os.rename(op + '.tmp', op)
            print(f'{tag}: {len(files)} views {n} dets {time.time() - t0:.0f}s', flush=True)
print('INFER_DONE', flush=True)
