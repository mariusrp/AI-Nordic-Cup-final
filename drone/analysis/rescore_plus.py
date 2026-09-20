"""Re-score lane replays on valcity_plus next to valcity_v1_core (drone understanding lane, cycle 3).

For each recording: every variant's full-scene mAP and per-class AP on BOTH scenes, and the paired 15-frame block
bootstrap diff vs that recording's base (same resampling for all variants, unmodified upstream local_evaluator.score).
Answers: does a lane's valcity_v1_core verdict survive the 2 objects v1_core misses (third small_plane, second hangar)?
Frames > 150 are dropped on load (holdout), even if a replay file carries them.
Usage: python rescore_plus.py UPSTREAM_DIR NBOOT B rec:variant=path.jsonl [...]   (first variant of each rec = its base)
"""
import sys, os, json, collections
import numpy as np
up, NB, B = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
groups = collections.OrderedDict()
for a in sys.argv[4:]:
    rv, p = a.split('=', 1); rec, var = rv.split(':')
    groups.setdefault(rec, []).append((var, p))
sys.path.insert(0, up); os.chdir(up)
import local_evaluator as le  # noqa: E402
real_fn, real_load = le.frame_numbers, le.load_annotations
SCENES = os.environ.get('VC_SCENES', 'valcity_v1_core,valcity_v1_plus').split(',')
GT = {s: {f: real_load(f, s) for f in real_fn(s)} for s in SCENES}


def load(p):
    d = {}
    for line in open(p):
        r = json.loads(line)
        if r['frame'] > 150:
            continue
        d[r['frame']] = [dict(object_id=c, bbox=tuple(v * (3840 if i % 2 == 0 else 2160) for i, v in enumerate(b)), confidence=s)
                         for c, b, s in r['ann']]
    return d


def score_on(scene, fr, P):
    vids = list(range(1, len(fr) + 1))
    le.frame_numbers = lambda s: vids
    le.load_annotations = lambda v, s: GT[scene][fr[v - 1]]
    return {n: le.score(scene, {v: P[n][f] for v, f in zip(vids, fr) if f in P[n]}) for n in P}


for rec, vs in groups.items():
    P = collections.OrderedDict((v, load(p)) for v, p in vs)
    base = vs[0][0]
    for scene in SCENES:
        frames = sorted(GT[scene])
        full = score_on(scene, frames, P)
        blocks = [frames[i:i + B] for i in range(0, len(frames), B)]
        rng = np.random.default_rng(0); S = collections.defaultdict(list)
        for _ in range(NB):
            fr = [f for k in rng.integers(0, len(blocks), len(blocks)) for f in blocks[k]]
            try:
                o = score_on(scene, fr, P)
            except ValueError:
                continue
            for n in o:
                S[n].append(o[n][0])
        print(f'\n=== {rec} {scene}')
        for n in P:
            d = np.array(S[n]) - np.array(S[base]); m = full[n][0] - full[base][0]
            print(f'  {n:6s} mAP {full[n][0]:.4f}  diff {m:+.4f} +- {d.std():.4f} ({m / d.std() if d.std() > 0 else 0:+.1f} se)')
        cls = sorted(full[base][1])
        print('  per-class AP: ' + ' '.join(f'{c[:9]:>9s}' for c in cls))
        for n in P:
            print(f'  {n:12s} ' + ' '.join(f'{full[n][1][c]:9.3f}' for c in cls))
