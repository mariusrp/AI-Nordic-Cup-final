"""Where recall is lost: camera coverage vs memory vs detector vs class vs box (drone understanding lane, cycle 3).

For every GT instance of a valcity scene (frames <= 150) and each recorded run, classify the frame's answer:
  hit      same-class box with IoU >= .5 (any confidence)
  wrongcls only another class's box at IoU >= .5
  loc      best same-or-other-class IoU in [.1, .5)
  none     nothing with IoU >= .1
and the object's camera state at that frame (the view the answer came from):
  view     object centre inside the current view (with its level)
  mem      not in view, but inside some earlier view of this run (answer comes from the tracker's memory)
  never    never inside any view of this run so far
Also, per object: first frame in GT, first frame in view, first frame hit (delay = frames lost at entry).
Usage: python coverage.py SCENE_DIR name=meta.jsonl [...]
"""
import sys, os, json, glob, collections
scene = sys.argv[1]
GT = {}
for f in glob.glob(os.path.join(scene, 'annotations', '*.json')):
    d = json.load(open(f))
    if d['frame'] <= 150:
        GT[d['frame']] = d['annotations']


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def inside(b, r):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return r[0] <= cx <= r[2] and r[1] <= cy <= r[3]


tot = collections.Counter()
for arg in sys.argv[2:]:
    name, mp = arg.split('=', 1)
    rows = {}
    for line in open(mp):
        r = json.loads(line)
        if r['frame'] <= 150:
            rows[r['frame']] = r      # last row per frame wins
    seen = set(); first = {}; C = collections.Counter(); per = collections.defaultdict(collections.Counter)
    for t in sorted(GT):
        r = rows.get(t)
        for g in GT[t]:
            k = (g['object_id'], str(g['valcity_cluster'])); b = g['bbox']
            fi = first.setdefault(k, dict(gt=t, view=None, hit=None, n=0, nhit=0))
            fi['n'] += 1
            if r is None:
                cam, res = 'skip', 'none'
            else:
                if inside(b, r['region']):
                    cam = f"view L{r['level']}"; seen.add(k)
                    if fi['view'] is None:
                        fi['view'] = t
                else:
                    cam = 'mem' if k in seen else 'never'
                best_same = best_other = 0
                for c, bb, s in r['ann']:
                    v = iou([bb[0] * 3840, bb[1] * 2160, bb[2] * 3840, bb[3] * 2160], b)
                    if c == g['object_id']:
                        best_same = max(best_same, v)
                    else:
                        best_other = max(best_other, v)
                res = 'hit' if best_same >= .5 else 'wrongcls' if best_other >= .5 else 'loc' if max(best_same, best_other) >= .1 else 'none'
            if res == 'hit':
                fi['nhit'] += 1
                if fi['hit'] is None:
                    fi['hit'] = t
            C[(cam, res)] += 1; per[k][res] += 1
    n = sum(C.values())
    print(f'\n=== {name}: {n} GT instances (frames <= 150)')
    cams = sorted({c for c, _ in C})
    print(f'  {"camera state":10s} {"share":>6s} ' + ' '.join(f'{x:>8s}' for x in ('hit', 'wrongcls', 'loc', 'none')))
    for cam in cams:
        m = sum(C[(cam, x)] for x in ('hit', 'wrongcls', 'loc', 'none'))
        print(f'  {cam:10s} {m / n:6.1%} ' + ' '.join(f'{C[(cam, x)] / m:8.1%}' for x in ('hit', 'wrongcls', 'loc', 'none')))
    allhit = sum(C[(c, 'hit')] for c in cams)
    print(f'  overall hit rate {allhit / n:.1%}; misses by state: ' + ', '.join(
        f'{cam} {sum(C[(cam, x)] for x in ("wrongcls", "loc", "none")) / n:.1%}' for cam in cams))
    lost_entry = 0
    print(f'  {"object":22s} {"nGT":>4s} {"hit%":>5s} {"firstGT":>7s} {"firstView":>9s} {"firstHit":>8s}')
    for k in sorted(first, key=lambda k: first[k]['gt']):
        fi = first[k]
        if fi['hit'] is not None:
            lost_entry += sum(1 for t in GT if fi['gt'] <= t < fi['hit'] and any((g['object_id'], str(g['valcity_cluster'])) == k for g in GT[t]))
        print(f'  {k[0][:14]:14s} #{k[1]:6s} {fi["n"]:4d} {fi["nhit"] / fi["n"]:5.0%} {fi["gt"]:7d} {str(fi["view"]):>9s} {str(fi["hit"]):>8s}')
    print(f'  GT instances before an object is first hit: {lost_entry} ({lost_entry / n:.1%} of all)')
