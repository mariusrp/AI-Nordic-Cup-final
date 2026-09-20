"""Per object: raw detector hit rate vs the pipeline's answer rate while the object is in view (understanding lane, cycle 2).

For every valcity object and each recorded run: in frames where the object lies fully inside the run's own L1 view,
the fraction of frames with a same-class answer box at IoU >= .5 (any confidence). Next to it, the raw YOLO hit rate
(same class, conf >= .05) on the views of one run from det_probe.py output. The difference shows what the tracker /
verifier add or delete relative to the detector. Frames <= 150 only.
Usage: python inview_rates.py SCENE_DIR PROBE.json PROBE_RUN8 name=meta.jsonl [...]
"""
import sys, json, glob, os, collections
scene, probe, prun = sys.argv[1:4]
GT = {}
for f in glob.glob(os.path.join(scene, 'annotations', '*.json')):
    d = json.load(open(f)); GT[d['frame']] = d['annotations']


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


raw = collections.defaultdict(list)
for x in json.load(open(probe)):
    if x['level'] == 1 and x['run'] == prun:
        raw[x['obj']].append(x['same_conf'] >= 0.05)
cols = []
for a in sys.argv[4:]:
    name, mp = a.split('=', 1)
    out = collections.defaultdict(list)
    for line in open(mp):
        r = json.loads(line); t = r['frame']
        if t > 150 or r['level'] != 1:
            continue
        reg = r['region']
        for g in GT.get(t, []):
            b = g['bbox']
            if not (b[0] >= reg[0] and b[1] >= reg[1] and b[2] <= reg[2] and b[3] <= reg[3]):
                continue
            hit = any(c == g['object_id'] and iou([bb[0] * 3840, bb[1] * 2160, bb[2] * 3840, bb[3] * 2160], b) >= 0.5 for c, bb, s in r['ann'])
            out[(g['object_id'], str(g['valcity_cluster']))].append(hit)
    cols.append((name, out))
keys = sorted(set().union(*[set(o) for _, o in cols]))
f = lambda v: f'{sum(v):2d}/{len(v):2d}' if v else '  -  '
print(f'{"object":22s} raw@.05({prun}) ' + ' '.join(f'{n:>8s}' for n, _ in cols))
for k in keys:
    print(f'{k[0]:15s} #{k[1]:5s} {f(raw.get(k[1], [])):>14s} ' + ' '.join(f'{f(o.get(k, [])):>8s}' for _, o in cols))
