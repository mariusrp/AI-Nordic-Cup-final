"""FINALS: phantom detections per view on object-free terrain no checkpoint has ever seen.

mAP saturates on our offline scenes; what decides an unseen city is how many boxes the detector invents on
new ground texture. Every view here is guaranteed object-free (bb3/mk_negatives.py), so EVERY detection is a
false positive and no ground truth is needed. Reported per confidence threshold, which is exactly the
track-birth threshold DRONE_NEW_THR the live tracker uses.

    python fp_negatives.py /dev/shm/neg/images/train v2=/workspace/drone_weights/v2.pt ft_all=...

An ensemble is scored on the same views without a second GPU pass, by naming runs already on the command line:
    python fp_negatives.py DIR ft5r=A.pt v2=B.pt w21=fuse:wbf:ft5r,v2:2,1:0.55
(MODE = wbf | avg | union, exactly as in eval_scene.py - fusion happens per view, like server.py.)
"""
import collections
import glob
import json
import os
import sys

from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fuse_boxes import wbf  # noqa: E402

SRC = sys.argv[1]
RUNS = [a.split('=', 1) for a in sys.argv[2:] if '=' in a]
BASE = [(n, w) for n, w in RUNS if not w.startswith('fuse:')]
FUSE = [(n, w) for n, w in RUNS if w.startswith('fuse:')]
THRS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.70]
imgs = sorted(glob.glob(f'{SRC}/*.jpg') + glob.glob(f'{SRC}/*.png'))
print('views', len(imgs), flush=True)

dets = {}  # name -> list over views of [(box, class, score)]
for name, w in BASE:
    m = YOLO(w, task='detect')
    names = {int(k): v for k, v in m.names.items()}
    per_view = []
    for i in range(0, len(imgs), 16):
        for r in m.predict(imgs[i:i + 16], imgsz=1280, conf=min(THRS), max_det=300, verbose=False, device=0):
            b = r.boxes
            if b is None or len(b) == 0:
                per_view.append([])
                continue
            xy = b.xyxy.cpu().numpy()
            cf = b.conf.cpu().numpy()
            cl = b.cls.cpu().numpy().astype(int)
            per_view.append([(tuple(float(v) for v in xy[j]), names[int(cl[j])], float(cf[j])) for j in range(len(cf))])
    dets[name] = per_view
    print('ran', name, w, flush=True)

for name, spec in FUSE:
    mode, srcs, ws, iou = spec.split(':')[1:5]
    srcs = srcs.split(',')
    ws = [float(v) for v in ws.split(',')]
    dets[name] = [wbf([dets[s][i] for s in srcs], ws, float(iou), mode) for i in range(len(imgs))]
    print('fused', name, spec, flush=True)

out = {}
for name, _ in RUNS:
    n = collections.Counter()
    percls = collections.Counter()
    top = []
    for dv in dets[name]:
        if not dv:
            continue
        for t in THRS:
            n[t] += sum(1 for _, _, s in dv if s >= t)
        for _, c, s in dv:
            if s >= 0.25:
                percls[c] += 1
        top.append(max(s for _, _, s in dv))
    out[name] = {'fp_per_view': {str(t): round(n[t] / len(imgs), 3) for t in THRS},
                 'views_with_det@0.25': sum(1 for t in top if t >= 0.25),
                 'per_class@0.25': dict(percls.most_common(8))}
    print(name, json.dumps(out[name]), flush=True)
print('FP_NEG ' + json.dumps(out))
