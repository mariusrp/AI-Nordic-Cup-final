"""Time trend of live platform answers (frames <= 150) against valcity_plus (drone understanding lane, cycle 5).

Per 50-frame block and arm: answered boxes per frame, TP boxes (same class, IoU >= .5 with a GT box), FP boxes, and
'confident' FPs (conf >= 0.1), plus the block mAP. Shows whether an arm's clutter grows with time inside the visible
frames (frames > 150 are never read).
Usage: python live_trend.py UPSTREAM_DIR arm:name=path.jsonl [...]
"""
import sys, os, json, collections
import numpy as np
up = sys.argv[1]
sys.path.insert(0, up); os.chdir(up)
import local_evaluator as le  # noqa: E402
SCENE = 'valcity_v1_plus'
GT = {f: le.load_annotations(f, SCENE) for f in le.frame_numbers(SCENE)}
real_fn, real_load = le.frame_numbers, le.load_annotations


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


BL = [(1, 50), (51, 100), (101, 150)]
agg = collections.defaultdict(lambda: collections.defaultdict(list))
for a in sys.argv[2:]:
    an, p = a.split('=', 1); arm, name = an.split(':')
    P = {}
    for line in open(p):
        r = json.loads(line)
        if r['frame'] <= 150:
            P[r['frame']] = [dict(object_id=c, bbox=tuple(v * (3840 if i % 2 == 0 else 2160) for i, v in enumerate(b)), confidence=s) for c, b, s in r['ann']]
    for lo, hi in BL:
        fr = [f for f in sorted(GT) if lo <= f <= hi]
        n = tp = fp = cfp = 0
        for f in fr:
            for d in P.get(f, []):
                n += 1
                ok = any(g['object_id'] == d['object_id'] and iou(g['bbox'], d['bbox']) >= .5 for g in GT[f])
                tp += ok; fp += (not ok); cfp += (not ok and d['confidence'] >= 0.1)
        vids = list(range(1, len(fr) + 1))
        le.frame_numbers = lambda s: vids
        le.load_annotations = lambda v, s: GT[fr[v - 1]]
        m = le.score(SCENE, {v: P.get(f, []) for v, f in zip(vids, fr)})[0]
        le.frame_numbers, le.load_annotations = real_fn, real_load
        agg[arm][(lo, hi)].append((n / len(fr), tp / len(fr), fp / len(fr), cfp / len(fr), m))
for arm, D in agg.items():
    for blk, L in D.items():
        A = np.array(L).mean(0)
        print(f'{arm:5s} frames {blk[0]:3d}-{blk[1]:3d}: boxes/frame {A[0]:5.1f}  TP {A[1]:4.2f}  FP {A[2]:5.1f}  FP(conf>=.1) {A[3]:4.2f}  block mAP {A[4]:.4f} (n={len(L)})')
