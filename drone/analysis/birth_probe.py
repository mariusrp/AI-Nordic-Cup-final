"""Does the verifier act as a track-BIRTH filter? (drone understanding lane, cycle 3; analysis only, production untouched)

Production feeds the tracker the VERIFIED confidence (conf^.5 * p_proto(cls)^.5) and births a track only if that is
>= new_thr = 0.25, a threshold tuned on RAW YOLO confidences (SAFER, before I4). So a real object with YOLO .40 and
p_proto .10 (verified .20) never gets a track. This replays recorded views (frames <= 150) through the real
detector + verifier + tracker with three tracker variants that share one YOLO + DINOv2 pass:
  base  production (birth on verified conf >= .25; p_bg > .5 dropped)
  RB    birth on RAW YOLO conf >= .25; everything else (scores, drop) as production
  RBD   RB + p_bg-dropped detections are kept at verified conf x 1e-3 (demote, never drop)
  N<x>  (--thr x) production with new_thr = x, as a shadow stream for floor_band.py
It also logs every detection that matches a valcity_plus GT box (IoU >= .5, any class): raw conf, verified conf,
p_bg, dropped -> how many GT births the verifier blocks.
Usage (gpu3): python birth_probe.py --drone /workspace/drone_fast_r4_2/drone --scene SCENE_DIR --run DIR [...] --out OUT
"""
import argparse, inspect, json, os, sys, time, glob
import numpy as np
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('--drone', required=True); ap.add_argument('--scene', required=True)
ap.add_argument('--run', action='append', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--thr', type=float, action='append', default=[])
ap.add_argument('--weights', default='/workspace/drone_weights/v2.pt'); ap.add_argument('--bank', default='/workspace/i4/bankAll.pt')
a = ap.parse_args()
sys.path.insert(0, a.drone)
from common import CLASSES, H as FH, W as FW, LEVEL_SCALE, VIEW_H, VIEW_W  # noqa: E402
from detector import YoloDetector  # noqa: E402
import proto_verify  # noqa: E402
import tracker as trk  # noqa: E402

src = inspect.getsource(trk.Tracker.update)
assert src.count('conf < self.new_thr') == 1
ns = {}
exec(compile('\n'.join(l[4:] for l in src.replace('conf < self.new_thr', 'self._birth[j] < self.new_thr').splitlines()), 'rb', 'exec'),
     trk.__dict__, ns)


class RawBirthTracker(trk.Tracker):
    update = ns['update']


GT = {}
for f in glob.glob(os.path.join(a.scene, 'annotations', '*.json')):
    d = json.load(open(f)); GT[d['frame']] = d['annotations']


def iou(p, q):
    ix = max(0, min(p[2], q[2]) - max(p[0], q[0])); iy = max(0, min(p[3], q[3]) - max(p[1], q[1])); i = ix * iy
    u = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i
    return i / u if u > 0 else 0


def src_box(box, level, region):
    s = LEVEL_SCALE[level]; x1, y1, x2, y2 = box
    x1, x2 = max(0.0, min(VIEW_W, x1)), max(0.0, min(VIEW_W, x2)); y1, y2 = max(0.0, min(VIEW_H, y1)), max(0.0, min(VIEW_H, y2))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None, False
    m = 1.5
    trunc = ((x1 < m and region[0] > 0) or (y1 < m and region[1] > 0) or (x2 > VIEW_W - m and region[2] < FW) or (y2 > VIEW_H - m and region[3] < FH))
    return np.array([region[0] + x1 * s, region[1] + y1 * s, region[0] + x2 * s, region[1] + y2 * s]), bool(trunc)


def ann(tr):
    outs = sorted(tr.outputs(), key=lambda o: -o[2])[:100]
    return [[CLASSES[c], [round(float(v), 6) for v in (b[0] / FW, b[1] / FH, b[2] / FW, b[3] / FH)], round(float(min(1.0, cf)), 4)] for b, c, cf in outs]


yolo = YoloDetector(a.weights)
emb = proto_verify.Embedder()
ver = proto_verify.Verifier(__import__('torch').load(a.bank, map_location=emb.dev), emb=emb)  # production defaults
has = ver.has.cpu().numpy()
os.makedirs(a.out, exist_ok=True)
for run in a.run:
    rid = os.path.basename(run.rstrip('/'))[:8]
    metas = sorted((json.loads(l) for l in open(os.path.join(run, 'meta.jsonl'))), key=lambda m: (m['frame'], m['t']))
    assert all(m['frame'] <= 150 for m in metas), 'holdout frames present'
    motion = trk.MotionModel()
    T = {'base': trk.Tracker(), 'RB': RawBirthTracker(), 'RBD': RawBirthTracker()}
    for x in a.thr:   # extra shadow trackers: birth on verified conf >= x
        T[f'N{x}'] = trk.Tracker(new_thr=x)
    W = {k: open(os.path.join(a.out, f'{rid}_{k}.jsonl'), 'w') for k in T}
    log = open(os.path.join(a.out, f'{rid}_gtdets.jsonl'), 'w'); t0 = time.perf_counter()
    for m in metas:
        img = cv2.imread(os.path.join(run, m['file']), cv2.IMREAD_COLOR)
        if img is None:
            continue
        level, region, frame = m['level'], tuple(int(v) for v in m['region']), m['frame']
        motion.observe(frame, img, level, region)
        raw = yolo(img, level)
        P = None
        if level in ver.levels and raw:
            P = ver.probs(emb.embed([proto_verify.crop_box(img, d['box']) for d in raw])).cpu().numpy()
        kept, demoted = [], []   # (srcbox, probs, verified conf, trunc, raw conf)
        for i, d in enumerate(raw):
            b, tr_ = src_box(d['box'], level, region)
            if b is None:
                continue
            yc = int(np.argmax(d['probs'])); rc = float(d['conf']); vc, drop, pbg = rc, False, -1.0
            if P is not None:
                p = P[i]; pbg = float(p[-1])
                if pbg > ver.drop:
                    drop = True
                if not (not has[yc] and d['probs'][yc] >= 0.5):
                    vc = float(rc ** ver.a * p[yc] ** (1 - ver.a))
            (demoted if drop else kept).append((b, d['probs'], vc, tr_, rc))
            for g in GT.get(frame, []):
                if iou(b, g['bbox']) >= .5:
                    log.write(json.dumps(dict(frame=frame, level=level, obj=f"{g['object_id']}#{g['valcity_cluster']}", det=CLASSES[yc],
                                              raw=round(rc, 4), ver=round(vc, 4), pbg=round(pbg, 4), drop=drop, trunc=tr_)) + '\n')
        for k, tr in T.items():
            dets = kept + ([(b, pr, vc * 1e-3, t_, rc) for b, pr, vc, t_, rc in demoted] if k == 'RBD' else [])
            tr._birth = [d[4] for d in dets]
            tr.predict(frame, motion.H)
            tr.update([d[:4] for d in dets], level, region, frame)
            W[k].write(json.dumps(dict(frame=frame, level=level, ann=ann(tr))) + '\n')
    for w in W.values():
        w.close()
    log.close(); print(rid, 'done', round(time.perf_counter() - t0), 's', flush=True)
