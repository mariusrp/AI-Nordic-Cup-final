"""Score what the server ACTUALLY answered on the platform, frames <= 150, on valcity_plus (drone understanding lane, cycle 5).

Input: per-run jsonl of the recorder's meta (frame, level, ann = [cls, normalized xyxy bbox, conf rounded to 4 dp]),
filtered to frames <= 150 ON THE POD before download (frames > 150 are never read). Several views of one frame: the last
answer sent for that frame counts (the platform keeps one answer per frame). Missing frames score as empty answers.
Prints per-run mAP and per-class AP, and base-vs-cand means with the between-run se.
Usage: python score_live.py UPSTREAM_DIR arm:name=path.jsonl [...]
"""
import sys, os, json, collections
import numpy as np
up = sys.argv[1]
sys.path.insert(0, up); os.chdir(up)
import local_evaluator as le  # noqa: E402
SCENE = os.environ.get('VC_SCENE', 'valcity_v1_plus')
runs = collections.OrderedDict()
for a in sys.argv[2:]:
    an, p = a.split('=', 1); arm, name = an.split(':')
    P = {}
    for line in open(p):
        r = json.loads(line)
        if r['frame'] > 150:
            continue
        P[r['frame']] = [dict(object_id=c, bbox=tuple(v * (3840 if i % 2 == 0 else 2160) for i, v in enumerate(b)), confidence=s)
                         for c, b, s in r['ann']]
    runs[(arm, name)] = P
res = {k: le.score(SCENE, P) for k, P in runs.items()}
cls = sorted(next(iter(res.values()))[1])
print('run'.ljust(30), 'mAP   ', ' '.join(f'{c[:9]:>9s}' for c in cls))
for (arm, name), (m, pc) in res.items():
    print(f'{arm + ":" + name:30s} {m:.4f} ' + ' '.join(f'{pc[c]:9.3f}' for c in cls))
by = collections.defaultdict(list)
for (arm, _), (m, pc) in res.items():
    by[arm].append((m, pc))
for arm, L in by.items():
    ms = np.array([m for m, _ in L])
    print(f'MEAN {arm:25s} {ms.mean():.4f} +- {ms.std(ddof=1) / np.sqrt(len(ms)) if len(ms) > 1 else 0:.4f} (n={len(ms)}) ' +
          ' '.join(f'{np.mean([pc[c] for _, pc in L]):9.3f}' for c in cls))
