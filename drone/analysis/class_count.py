"""How many classes does the platform's validation flight score?  (drone understanding lane, cycle 2)

The platform reports only one number: mean AP over the classes PRESENT in the flight (absent classes are skipped,
see score_semantics.py T2). valcity_v1_core labels 11 classes (19 objects) in frames 1-150. If the platform flight
also contains the 5 classes nobody ever detected (condor, large_tower, medium_launcher, medium_plane, spacecraft),
each run scores ~0 there and the platform number is ~ sum(valcity AP) / 16, not / 11.
This compares both hypotheses over every recorded run that has a platform score (frames <= 150 only).
Usage: python class_count.py UPSTREAM_DIR SCENE name=meta.jsonl@platform [...]
"""
import sys, os, json
import numpy as np
up, scene = sys.argv[1], sys.argv[2]
sys.path.insert(0, up); os.chdir(up)
import local_evaluator as le  # noqa: E402
FR = [f for f in le.frame_numbers(scene) if f <= 150]


def load(mp):
    P = {}
    for line in open(mp):
        r = json.loads(line)
        if r['frame'] > 150:
            continue
        P[r['frame']] = [dict(object_id=c, bbox=tuple(v * (3840 if i % 2 == 0 else 2160) for i, v in enumerate(b)), confidence=s)
                         for c, b, s in r['ann']]
    return P


rows = []
for a in sys.argv[3:]:
    name, spec = a.split('=', 1)
    mp, _, ps = spec.partition('@')
    P = load(mp)
    m, per = le.score(scene, {f: P[f] for f in FR if f in P})
    s = sum(per.values())
    rows.append((name, float(ps), m, s))
    print(f'{name:10s} platform {float(ps):.4f} | valcity mAP(11) {m:.4f} sumAP {s:.3f} -> /16 {s / 16:.4f}  /11 {s / 11:.4f} | '
          f'platform/valcity {float(ps) / m:.2f}')
p = np.array([r[1] for r in rows]); s = np.array([r[3] for r in rows])
for k in (11, 12, 13, 14, 15, 16):
    err = p - s / k
    print(f'N={k:2d}: mean(platform - sumAP/N) {err.mean():+.4f}  rmse {np.sqrt((err ** 2).mean()):.4f}')
# least squares platform = sumAP / N  (no intercept)
N = (s @ s) / (s @ p)
print(f'best-fit N (platform = sumAP/N, no intercept): {N:.1f}')
A = np.vstack([s, np.ones_like(s)]).T
coef, *_ = np.linalg.lstsq(A, p, rcond=None)
print(f'with intercept: platform = {coef[0]:.4f} * sumAP + {coef[1]:+.4f}  (1/slope = {1 / coef[0]:.1f})  corr {np.corrcoef(s, p)[0, 1]:.2f}')
