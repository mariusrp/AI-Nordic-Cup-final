"""Per-time-block valcity_plus mAP of replay variants vs a base (drone understanding lane, cycle 5).

The platform scores frames 1-249, but valcity covers 1-150 and clutter tracks accumulate with time (live_trend.py), so
the LAST visible block (101-150) is the closest proxy for the unseen frames 151-249. For each recording and block, prints
mAP(variant) - mAP(base) and the per-class AP change in that block; then the mean over recordings.
Usage: python block_diff.py UPSTREAM_DIR base_variant variant[,variant...] DIR rec [rec ...]
"""
import sys, os, json, collections
import numpy as np
up, base, vs, d = sys.argv[1], sys.argv[2], sys.argv[3].split(','), sys.argv[4]
recs = sys.argv[5:]
sys.path.insert(0, up); os.chdir(up)
import local_evaluator as le  # noqa: E402
SCENE = 'valcity_v1_plus'
GT = {f: le.load_annotations(f, SCENE) for f in le.frame_numbers(SCENE)}
real_fn, real_load = le.frame_numbers, le.load_annotations
BL = [(1, 50), (51, 100), (101, 150)]


def load(p):
    P = {}
    for line in open(p):
        r = json.loads(line)
        if r['frame'] <= 150:
            P[r['frame']] = [dict(object_id=c, bbox=tuple(v * (3840 if i % 2 == 0 else 2160) for i, v in enumerate(b)), confidence=s) for c, b, s in r['ann']]
    return P


def sc(P, fr):
    vids = list(range(1, len(fr) + 1))
    le.frame_numbers = lambda s: vids
    le.load_annotations = lambda v, s: GT[fr[v - 1]]
    try:
        return le.score(SCENE, {v: P.get(f, []) for v, f in zip(vids, fr)})
    finally:
        le.frame_numbers, le.load_annotations = real_fn, real_load


D = collections.defaultdict(list)
for rec in recs:
    Pb = load(os.path.join(d, f'{rec}_{base}.jsonl'))
    for v in vs:
        Pv = load(os.path.join(d, f'{rec}_{v}.jsonl'))
        for lo, hi in BL:
            fr = [f for f in sorted(GT) if lo <= f <= hi]
            mb, cb = sc(Pb, fr); mv, cv = sc(Pv, fr)
            D[(v, lo)].append(mv - mb)
            moved = ' '.join(f'{c}{cv[c] - cb[c]:+.2f}' for c in sorted(cb) if abs(cv[c] - cb[c]) >= .02)
            print(f'{rec} {v:9s} {lo:3d}-{hi:3d}: base {mb:.4f} diff {mv - mb:+.4f}  {moved}')
print('\nMEAN over recordings (diff vs', base + ')')
for v in vs:
    print(f'  {v:9s} ' + '  '.join(f'{lo}-{hi}: {np.mean(D[(v, lo)]):+.4f} ({sum(x > 0 for x in D[(v, lo)])}/{len(D[(v, lo)])} >0)' for lo, hi in BL))
