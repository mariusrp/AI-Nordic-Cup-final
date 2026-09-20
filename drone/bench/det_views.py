"""Detector-only recall/FP on the RECORDED views of the ft_all-unseen block (frames 181-249, /workspace/.holdout, eval
only) against the table GT (mk_gt.py), with the production YoloDetector (same fast path as the server) at one or more
imgsz. Answers: which small classes does the L1 operating point lose, and does a larger inference size recover them.
A GT box counts when it lies fully inside the view region; a detection is a hit if IoU >= 0.5 with a same-class GT
box, a class-miss if IoU >= 0.5 with another class, else an FP (ignore boxes excluded). Also prints ms/view.

    cd /workspace/nacb/drone && UPSTREAM=/workspace/upstream python bench/det_views.py GT.json OUT.json \
        --imgsz 1280,1920 --thr 0.25 --levels 1 name=weights.pt ...
"""
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: F401,E402
from common import CLASSES, LEVEL_SCALE  # noqa: E402
from detector import YoloDetector  # noqa: E402

argv = sys.argv


def opt(name, default):
    return argv[argv.index(name) + 1] if name in argv else default


GT_PATH, OUT = argv[1], argv[2]
IMGSZ = [int(v) for v in opt('--imgsz', '1280').split(',')]
THR = float(opt('--thr', '0.25'))
LEVELS = [int(v) for v in opt('--levels', '1').split(',')]
F0, F1 = int(opt('--f0', '181')), int(opt('--f1', '249'))
RUNS = [a.split('=', 1) for a in argv[3:] if '=' in a and not a.startswith('--')]
G = json.load(open(GT_PATH))
GT = {int(k): v for k, v in G['gt'].items()}
IGN = {int(k): v for k, v in G['ignore'].items()}
HOLD = opt('--root', '/workspace/.holdout/drone_seen_heldout')   # or /workspace/drone_seen_full (L2 survey sweeps)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


views = []
for mp in sorted(glob.glob(f'{HOLD}/*/meta.jsonl')):
    for line in open(mp):
        m = json.loads(line)
        if F0 <= m['frame'] <= F1 and m['level'] in LEVELS and m['frame'] in GT:
            m['_path'] = os.path.join(os.path.dirname(mp), m['file'])
            views.append(m)
print('views', len(views), 'levels', LEVELS, 'frames', F0, F1, flush=True)

res = {}
for name, w in RUNS:
    for sz in IMGSZ:
        det = YoloDetector(w, imgsz=sz)
        tag = f'{name}@{sz}'
        st = {c: dict(n=0, hit=0, cmiss=0) for c in CLASSES}
        fp = 0
        ts = []
        for m in views:
            img = cv2.imread(m['_path'])
            if img is None:
                continue
            L, reg = m['level'], m['region']
            t = time.perf_counter()
            dets = det(img, L)
            ts.append(1000 * (time.perf_counter() - t))
            s = LEVEL_SCALE[L]
            boxes = []
            for d in dets:
                if d['conf'] < THR:
                    continue
                x1, y1, x2, y2 = d['box']
                boxes.append(([reg[0] + x1 * s, reg[1] + y1 * s, reg[0] + x2 * s, reg[1] + y2 * s],
                              CLASSES[int(np.argmax(d['probs']))], float(d['conf'])))
            gts = [g for g in GT[m['frame']] if g['bbox'][0] >= reg[0] and g['bbox'][1] >= reg[1]
                   and g['bbox'][2] <= reg[2] and g['bbox'][3] <= reg[3]]
            taken = set()
            for b, c, cf in sorted(boxes, key=lambda x: -x[2]):
                best, bj = 0.0, -1
                for j, g in enumerate(gts):
                    v = iou(b, g['bbox'])
                    if v > best:
                        best, bj = v, j
                if best >= 0.5:
                    if bj in taken:
                        fp += 1
                        continue
                    taken.add(bj)
                    if gts[bj]['object_id'] == c:
                        st[c]['hit'] += 1
                    else:
                        st[gts[bj]['object_id']]['cmiss'] += 1
                elif not any(iou(b, g) >= 0.3 for g in IGN.get(m['frame'], [])):
                    fp += 1
            for g in gts:
                st[g['object_id']]['n'] += 1
        ngt = sum(v['n'] for v in st.values())
        hit = sum(v['hit'] for v in st.values())
        res[tag] = dict(recall=hit / max(1, ngt), fp_per_view=fp / max(1, len(views)), ms_p50=float(np.median(ts)),
                        ms_p90=float(np.percentile(ts, 90)), per_class={c: v for c, v in st.items() if v['n']})
        print(f"{tag:14s} GT-in-view {ngt:4d} recall@{THR} {hit / max(1, ngt):.3f}  FP/view {fp / max(1, len(views)):.2f}  "
              f"det ms p50 {np.median(ts):.0f} p90 {np.percentile(ts, 90):.0f}", flush=True)
        for c, v in st.items():
            if v['n']:
                print(f"    {c:16s} n={v['n']:4d} hit {v['hit'] / v['n']:.2f} class-miss {v['cmiss'] / v['n']:.2f}")
        del det
json.dump(res, open(OUT, 'w'), indent=1)
print('DET_VIEWS_DONE')
