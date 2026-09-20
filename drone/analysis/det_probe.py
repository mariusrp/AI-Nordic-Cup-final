"""Raw detector recall per labelled object on recorded validation views (drone understanding lane, cycle 2).

Separates DETECTOR misses from pipeline (tracker / verifier / SAFER) losses: runs a YOLO model directly on every recorded
view (frames <= 150 ONLY) with a low threshold, maps boxes to source px, and for every valcity GT object fully inside
the view records the best same-class IoU and confidence, and the best any-class match (class confusion).
Output: per object x level: #views, hit rate (same class, IoU>=.5) at conf>=.05 / >=.25, any-class hit rate, top wrong class.
Usage: python det_probe.py WEIGHTS SEEN_DIR SCENE_DIR OUT.json run_id [...]
"""
import sys, os, json, glob, collections
import numpy as np
from ultralytics import YOLO

wts, seen, scene, out = sys.argv[1:5]
runs = sys.argv[5:]
model = YOLO(wts)
names = model.names
GT = {}
for f in glob.glob(os.path.join(scene, 'annotations', '*.json')):
    d = json.load(open(f))
    if d['frame'] <= 150:
        GT[d['frame']] = d['annotations']


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


rec = []
for r in runs:
    for line in open(os.path.join(seen, r, 'meta.jsonl')):
        m = json.loads(line)
        if m['frame'] > 150:
            continue  # never touch frames >= 151
        p = os.path.join(seen, r, m['file'])
        if not os.path.exists(p):
            continue
        reg = m['region']; s = (reg[2] - reg[0]) / 960.0
        inside = [g for g in GT.get(m['frame'], []) if g['bbox'][0] >= reg[0] and g['bbox'][1] >= reg[1] and g['bbox'][2] <= reg[2] and g['bbox'][3] <= reg[3]]
        if not inside:
            continue
        res = model.predict(p, imgsz=1280, conf=0.01, iou=0.6, verbose=False, max_det=300)[0]
        B = res.boxes.xyxy.cpu().numpy() * s + [reg[0], reg[1], reg[0], reg[1]]
        C = [names[int(c)] for c in res.boxes.cls.cpu().numpy()]
        S = res.boxes.conf.cpu().numpy()
        for g in inside:
            best_same, best_any = (0.0, 0.0), (0.0, 0.0, None)
            for b, c, sc in zip(B, C, S):
                o = iou(b, g['bbox'])
                if o >= 0.5 and c == g['object_id'] and sc > best_same[1]:
                    best_same = (o, float(sc))
                if o >= 0.5 and sc > best_any[1]:
                    best_any = (o, float(sc), c)
            rec.append(dict(run=r[:8], frame=m['frame'], level=m['level'], obj=str(g['valcity_cluster']), cls=g['object_id'],
                            same_conf=best_same[1], any_conf=best_any[1], any_cls=best_any[2]))
json.dump(rec, open(out, 'w'))
agg = collections.defaultdict(list)
for x in rec:
    agg[(x['cls'], x['obj'], x['level'])].append(x)
print(f'weights {wts}: {len(rec)} object-views')
for k in sorted(agg):
    v = agg[k]
    n = len(v)
    h05 = sum(x['same_conf'] >= 0.05 for x in v) / n; h25 = sum(x['same_conf'] >= 0.25 for x in v) / n
    a05 = sum(x['any_conf'] >= 0.05 for x in v) / n
    wrong = collections.Counter(x['any_cls'] for x in v if x['any_conf'] >= 0.05 and x['same_conf'] < 0.05)
    print(f'  {k[0]:15s} #{k[1]:5s} L{k[2]} views {n:3d}  hit@.05 {h05:.2f}  hit@.25 {h25:.2f}  any-class@.05 {a05:.2f}  wrong: {dict(wrong.most_common(2))}')
