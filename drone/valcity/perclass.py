"""Per-class AP50 of replayed answer files on a valcity scene with the UNMODIFIED upstream local_evaluator.score.
Usage: python perclass.py UPSTREAM_DIR SCENE name1=meta1.jsonl name2=meta2.jsonl ..."""
import sys, json, os
up, scene = sys.argv[1], sys.argv[2]
runs = [a.split('=', 1) for a in sys.argv[3:]]
sys.path.insert(0, up); os.chdir(up)
import local_evaluator as le  # noqa: E402
res = {}
for name, mp in runs:
    d = {}
    for line in open(mp):
        r = json.loads(line)
        d[r['frame']] = [dict(object_id=c, bbox=tuple(v * (3840 if i % 2 == 0 else 2160) for i, v in enumerate(b)), confidence=s)
                         for c, b, s in r['ann']]
    fr = set(le.frame_numbers(scene))
    res[name] = le.score(scene, {f: p for f, p in d.items() if f in fr})
cls = sorted(set(k for _, pc in res.values() for k in pc))
print('class'.ljust(18) + ''.join(n[:9].rjust(10) for n, _ in runs))
print('mAP'.ljust(18) + ''.join(f'{res[n][0]:10.3f}' for n, _ in runs))
for c in cls:
    print(c[:18].ljust(18) + ''.join(f'{res[n][1].get(c, float("nan")):10.3f}' for n, _ in runs))
