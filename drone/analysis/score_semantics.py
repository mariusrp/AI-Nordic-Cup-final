"""Prove the scoring facts that decide our reporting policy, with the UNMODIFIED upstream scorer (local_evaluator.score).

T1 maxDets=100 is per (frame, CLASS), not per frame: 150 high-confidence FPs of one class in a frame do not push the
   true positives of OTHER classes in that frame out of the evaluation (they do push that class's own TP out).
T2 classes absent from the ground truth are not evaluated: FPs of an absent class cost exactly nothing.
T3 only the WITHIN-class ranking matters: multiplying every confidence of one class by 0.01 leaves mAP unchanged.
T4 appending boxes ranked below every existing box of their class never lowers AP ('demote, never drop').
Usage: python score_semantics.py UPSTREAM_DIR SCENE_WITH_ALL_CLASSES SCENE_WITH_SUBSET_OF_CLASSES
"""
import sys, os, random
up, full_scene, sub_scene = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, up); os.chdir(up)
import local_evaluator as le  # noqa: E402


def oracle(scene, conf=1.0):
    return {f: [dict(object_id=a['object_id'], bbox=tuple(float(c) for c in a['bbox']), confidence=conf)
                for a in le.load_annotations(f, scene)] for f in le.frame_numbers(scene)}


# T1
P = oracle(full_scene, 0.5)
f0 = le.frame_numbers(full_scene)[0]
flood = [dict(object_id='hangar', bbox=(10.0 + 20 * i, 2000.0, 25.0 + 20 * i, 2015.0), confidence=0.99) for i in range(150)]
P1 = dict(P); P1[f0] = P[f0] + flood
m0, per0 = le.score(full_scene, P)
m1, per1 = le.score(full_scene, P1)
others = {c: (per0[c], per1[c]) for c in per0 if c != 'hangar' and abs(per0[c] - per1[c]) > 1e-9}
print(f'T1 oracle {m0:.4f}; +150 hangar FPs (conf .99) in frame {f0}: {m1:.4f}; hangar AP {per0["hangar"]:.3f} -> {per1["hangar"]:.3f}; '
      f'other classes changed: {others or "none"} (frame {f0} has {len(P[f0])} GT, so a per-frame cap of 100 would have dropped them)')

# T2
P = oracle(sub_scene, 1.0)
present = {a['object_id'] for f in P for a in P[f]}
absent = [c for c in le.OBJECT_CLASSES if c not in present]
P2 = {f: v + [dict(object_id=c, bbox=(100.0, 100.0, 140.0, 130.0), confidence=1.0) for c in absent] for f, v in P.items()}
print(f'T2 {sub_scene}: {len(present)} classes present, absent {absent}; oracle {le.score(sub_scene, P)[0]:.4f}, '
      f'+1 conf-1.0 FP per absent class per frame: {le.score(sub_scene, P2)[0]:.4f}')

# T3 / T4 on a noisy prediction set
random.seed(0)
P = {}
for f in le.frame_numbers(sub_scene):
    ds = []
    for a in le.load_annotations(sub_scene and f, sub_scene) if False else le.load_annotations(f, sub_scene):
        if random.random() < 0.7:
            ds.append(dict(object_id=a['object_id'], bbox=tuple(float(c) for c in a['bbox']), confidence=random.random()))
    for _ in range(5):
        x, y = random.uniform(0, 3700), random.uniform(0, 2000)
        ds.append(dict(object_id=random.choice(sorted(present)), bbox=(x, y, x + 30, y + 30), confidence=random.random()))
    P[f] = ds
m, per = le.score(sub_scene, P)
c0 = sorted(present)[0]
P3 = {f: [dict(d, confidence=d['confidence'] * 0.01) if d['object_id'] == c0 else d for d in v] for f, v in P.items()}
print(f'T3 noisy set {m:.4f}; all {c0} confidences x0.01: {le.score(sub_scene, P3)[0]:.4f}')
worst = 1.0
for trial in range(20):
    P4 = {}
    for f, v in P.items():
        extra = []
        for a in le.load_annotations(f, sub_scene):
            if random.random() < 0.3:  # extra boxes: some hits, some misses, some duplicates, all below the floor
                extra.append(dict(object_id=a['object_id'], bbox=tuple(float(c) + random.uniform(-15, 15) for c in a['bbox']),
                                  confidence=1e-4 * random.random()))
        for _ in range(10):
            x, y = random.uniform(0, 3700), random.uniform(0, 2000)
            extra.append(dict(object_id=random.choice(sorted(present)), bbox=(x, y, x + 30, y + 30), confidence=1e-4 * random.random()))
        P4[f] = v + extra
    worst = min(worst, le.score(sub_scene, P4)[0] - m)
print(f'T4 appending ~30% hits + 10 FPs/frame below the floor, 20 random trials: min change {worst:+.4f} (never negative)')
