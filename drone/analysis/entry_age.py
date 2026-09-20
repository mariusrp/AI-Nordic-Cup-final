"""Hit rate by object age (frames since it entered the 4K frame) and by vertical position (cycle 3).

On the validation flight every point moves DOWN (Happrox: +53 px/frame at the top edge, +85 at the bottom, scale
1.006-1.013 per frame), so objects enter at the top, where they are smallest, and cross the frame in ~32 frames.
This measures how much of each object's life is lost before the pipeline first answers it, per recorded run.
Hit = same-class answer box at IoU >= .5 (any confidence). Frames <= 150 only.
Usage: python entry_age.py SCENE_DIR name=meta.jsonl [...]
"""
import sys, os, json, glob, collections
scene = sys.argv[1]
GT = {}
for f in glob.glob(os.path.join(scene, 'annotations', '*.json')):
    d = json.load(open(f))
    if d['frame'] <= 150:
        GT[d['frame']] = d['annotations']
first = {}
for t in sorted(GT):
    for g in GT[t]:
        first.setdefault((g['object_id'], str(g['valcity_cluster'])), t)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


AB = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 99)]
YB = [(0, 720), (720, 1440), (1440, 2160)]
print(f'{"run":8s} ' + ' '.join(f'age{a}-{b - 1 if b < 99 else "+"}' for a, b in AB) + ' | ' + ' '.join(f'y{a}-{b}' for a, b in YB))
for arg in sys.argv[2:]:
    name, mp = arg.split('=', 1)
    rows = {}
    for line in open(mp):
        r = json.loads(line)
        if r['frame'] <= 150:
            rows[r['frame']] = r
    A = collections.defaultdict(list); Y = collections.defaultdict(list)
    for t in sorted(GT):
        r = rows.get(t)
        for g in GT[t]:
            k = (g['object_id'], str(g['valcity_cluster'])); b = g['bbox']
            if first[k] == 1:      # objects already in frame 1 have no observed entry
                continue
            hit = r is not None and any(c == g['object_id'] and iou([bb[0] * 3840, bb[1] * 2160, bb[2] * 3840, bb[3] * 2160], b) >= .5
                                        for c, bb, s in r['ann'])
            age = t - first[k]; yc = (b[1] + b[3]) / 2
            A[next(i for i, (a, z) in enumerate(AB) if a <= age < z)].append(hit)
            Y[next(i for i, (a, z) in enumerate(YB) if a <= yc < z or (i == 2 and yc >= z))].append(hit)
    f = lambda v: f'{sum(v) / len(v):4.0%}({len(v):3d})' if v else '   -     '
    print(f'{name:8s} ' + ' '.join(f'{f(A[i]):>9s}' for i in range(len(AB))) + ' | ' + ' '.join(f'{f(Y[i]):>9s}' for i in range(len(YB))))
