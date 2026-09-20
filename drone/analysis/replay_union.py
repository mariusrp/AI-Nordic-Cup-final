"""(1) Does the valcity replay rank recorded runs like the platform?  (2) How complementary are two detectors?

(1) scores every recorded run (frames <= 150 only) on a valcity label set with the unmodified upstream scorer and
    prints it next to the platform validation score of that run (full 249 frames; from `nac.py status`).
(2) 'union' of two recorded runs of the SAME flight: per frame, the answers of both runs are pooled and de-duplicated
    within a class (IoU > .5 keeps the higher confidence). Both runs saw the same source frames, so the pooled boxes are
    valid answers for that frame. This approximates running both detectors in one server (a 2-model ensemble), with an
    optimistic bias because the valcity labels were built from the union of runs.
'A+B/demote' appends B's boxes below all of A's (conf x 0.001): what 'demote instead of drop' can add.
Usage: python replay_union.py UPSTREAM_DIR SCENE name=meta.jsonl@platform_score [...] -- unionA+unionB [...]
"""
import sys, os, json
import numpy as np

up, scene = sys.argv[1], sys.argv[2]
args = sys.argv[3:]
cut = args.index('--') if '--' in args else len(args)
runs = [a.split('=', 1) for a in args[:cut]]
unions = [a.split('+') for a in args[cut + 1:]]
sys.path.insert(0, up); os.chdir(up)
import local_evaluator as le  # noqa: E402
FR = [f for f in le.frame_numbers(scene) if f <= 150]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def load(mp):
    P = {}
    for line in open(mp):
        r = json.loads(line)
        if r['frame'] > 150:
            continue
        P[r['frame']] = [dict(object_id=c, bbox=tuple(v * (3840 if i % 2 == 0 else 2160) for i, v in enumerate(b)), confidence=s)
                         for c, b, s in r['ann']]
    return P


P, plat = {}, {}
for name, spec in runs:
    mp, _, ps = spec.partition('@')
    P[name] = load(mp); plat[name] = float(ps) if ps else float('nan')
res = {}
for name in P:
    m, per = le.score(scene, {f: P[name][f] for f in FR if f in P[name]})
    res[name] = (m, per)
    print(f'{name:12s} valcity {m:.4f}  platform {plat[name]:.4f}  ' + ' '.join(f'{k[:7]}:{v:.2f}' for k, v in sorted(per.items())))
names = [n for n in P if plat[n] == plat[n]]
if len(names) >= 3:
    a = np.array([res[n][0] for n in names]); b = np.array([plat[n] for n in names])
    ra = a.argsort().argsort(); rb = b.argsort().argsort()
    rho = np.corrcoef(ra, rb)[0, 1]
    conc = sum(1 for i in range(len(names)) for j in range(i + 1, len(names)) if (a[i] - a[j]) * (b[i] - b[j]) > 0)
    tot = len(names) * (len(names) - 1) // 2
    print(f'Spearman rho valcity vs platform over {len(names)} runs: {rho:.2f}; concordant pairs {conc}/{tot}')

for A, B in unions:
    U = {}
    demote = B.endswith('/demote')  # 'A+B/demote': B's boxes appended BELOW every box of A (conf x 0.001)
    B = B.split('/')[0]
    for f in FR:
        pb = [dict(d, confidence=d['confidence'] * 0.001) for d in P[B].get(f, [])] if demote else P[B].get(f, [])
        pool = sorted(P[A].get(f, []) + pb, key=lambda d: -d['confidence'])
        keep = []
        for d in pool:
            if any(k['object_id'] == d['object_id'] and iou(k['bbox'], d['bbox']) > 0.5 for k in keep):
                continue
            keep.append(d)
        if f in P[A] or f in P[B]:
            U[f] = keep
    m, per = le.score(scene, U)
    best = {c: max(res[A][1].get(c, 0), res[B][1].get(c, 0)) for c in per}
    print(f'union {A}+{B}{"/demote" if demote else ""}: {m:.4f} (A {res[A][0]:.4f}, B {res[B][0]:.4f}, per-class best-of {np.mean(list(best.values())):.4f})  ' +
          ' '.join(f'{k[:7]}:{v:.2f}' for k, v in sorted(per.items())))
