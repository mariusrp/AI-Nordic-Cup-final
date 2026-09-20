"""Per-class birth threshold (FL-r5-B B3) against / on top of the floor band, on the CLASS-ROUTED stack, plus a
raw-detection log for the per-object loss funnel (drone understanding lane, cycle 5).

drone-fast-r5-2 handed off B3 (DRONE_NEW_THR_CLS=mine_roller:0.12,ta-ta:0.12,tank:0.12, +0.055 valcity_plus on the v2-only
stack). Cycle 4 measured the floor band (+0.051) on the routed stack (g2-2 DRONE_ROUTE2, the likely next production).
Both recover the same blocked births (mine_roller, ta-ta, tank), B3 inside the main ranking, the floor band below it.
This replays recorded views (frames <= 150) ONCE per frame through v2, r11 and the DINOv2 verifier (production defaults)
and feeds these trackers:
  R_base   routed dets, production tracker (new_thr .25)          must reproduce cycle 4's R_base box for box
  R_base5  routed dets, r5-2 tracker with new_thr_cls {}          must equal R_base (the r5-2 tracker is a no-op by default)
  R_N06    routed dets, production tracker new_thr .06            floor-band source (cycle 4)
  R_B3     routed dets, r5-2 tracker, new_thr_cls mine_roller/ta-ta/tank .12   (B3 on the routed stack)
  P_B3     production (v2-only) dets, r5-2 tracker, same per-class thresholds  (cross-check of the fast lane's B3 numbers)
  RL_N06   routed dets + large_launcher from r11, new_thr .06     (large_launcher-only shadow source, open question 5)
  P_base   production (v2-only) dets, production tracker          (fidelity check against what a live server answered)
The r5-2 tracker is `git show drone-fast-r5-2:drone/tracker.py` (ca1dcf4), passed as --r5b.
It also writes every raw detection of BOTH detectors (conf >= .05, before the verifier) with its class probabilities,
the verifier's p_bg, drop flag and verified conf, for the per-object funnel (funnel5.py).
Usage (gpu3): python route_probe5.py --drone DRONE_DIR --r5b tracker_r5b.py --run DIR [...] --out OUT
"""
import argparse, importlib.util, json, os, sys, time
import numpy as np
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('--drone', required=True); ap.add_argument('--r5b', required=True)
ap.add_argument('--run', action='append', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--v2', default='/workspace/drone_weights/v2.pt'); ap.add_argument('--r11', default='/workspace/drone_weights/r11.pt')
ap.add_argument('--bank', default='/workspace/i4/bankAll.pt')
a = ap.parse_args()
sys.path.insert(0, a.drone)
import torch  # noqa: E402
torch.set_num_threads(2); cv2.setNumThreads(1)
from common import CLASSES, CLS_INDEX, H as FH, W as FW, LEVEL_SCALE, VIEW_H, VIEW_W  # noqa: E402
from detector import YoloDetector  # noqa: E402
import proto_verify  # noqa: E402
import tracker as trk  # noqa: E402
os.environ.pop('DRONE_NEW_THR_CLS', None); os.environ.pop('DRONE_NEW_THR', None); os.environ.pop('DRONE_L2_CLS_W', None)
spec = importlib.util.spec_from_file_location('tracker_r5b', a.r5b); t5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t5)

ROUTE = {CLS_INDEX[c] for c in ('helicopter', 'jet_plane', 'small_plane')}
ROUTE_LL = ROUTE | {CLS_INDEX['large_launcher']}
B3 = {CLS_INDEX[c]: 0.12 for c in ('mine_roller', 'ta-ta', 'tank')}


def src_box(box, level, region):
    s = LEVEL_SCALE[level]; x1, y1, x2, y2 = box
    x1, x2 = max(0.0, min(VIEW_W, x1)), max(0.0, min(VIEW_W, x2)); y1, y2 = max(0.0, min(VIEW_H, y1)), max(0.0, min(VIEW_H, y2))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None, False
    m = 1.5
    trunc = ((x1 < m and region[0] > 0) or (y1 < m and region[1] > 0) or (x2 > VIEW_W - m and region[2] < FW) or (y2 > VIEW_H - m and region[3] < FH))
    return np.array([region[0] + x1 * s, region[1] + y1 * s, region[0] + x2 * s, region[1] + y2 * s]), bool(trunc)


def norm(b):
    return [round(float(v), 6) for v in (b[0] / FW, b[1] / FH, b[2] / FW, b[3] / FH)]


def ann(tr):
    outs = sorted(tr.outputs(), key=lambda o: -o[2])[:100]
    return [[CLASSES[c], norm(b), round(float(min(1.0, cf)), 8)] for b, c, cf in outs]


yolo2, yolo11 = YoloDetector(a.v2), YoloDetector(a.r11)
emb = proto_verify.Embedder()
ver = proto_verify.Verifier(torch.load(a.bank, map_location=emb.dev), emb=emb)  # production defaults
has = ver.has.cpu().numpy()


def verify(img, level, raw, region):
    """-> list of (srcbox, probs, conf, trunc, yolo_cls, dropped, raw_conf, p_bg, p_cls); production Verifier.__call__ semantics."""
    P = ver.probs(emb.embed([proto_verify.crop_box(img, d['box']) for d in raw])).cpu().numpy() if (level in ver.levels and raw) else None
    out = []
    for i, d in enumerate(raw):
        b, tr_ = src_box(d['box'], level, region)
        if b is None:
            continue
        yc = int(np.argmax(d['probs'])); rc = float(d['conf']); vc, drop = rc, False
        pbg, pc = -1.0, -1.0
        if P is not None:
            p = P[i]; pbg, pc = float(p[-1]), float(p[yc])
            if p[-1] > ver.drop:
                drop = True
            elif not (not has[yc] and d['probs'][yc] >= 0.5):
                vc = float(rc ** ver.a * p[yc] ** (1 - ver.a))
        out.append((b, d['probs'], vc, tr_, yc, drop, rc, pbg, pc))
    return out


def rawlog(L):
    return [[CLASSES[d[4]], norm(d[0]), round(d[6], 4), round(d[2], 4), int(d[5]), round(d[7], 3), round(d[8], 3), int(d[3]),
             [round(float(v), 3) for v in d[1]]] for d in L]


os.makedirs(a.out, exist_ok=True)
for run in a.run:
    rid = os.path.basename(run.rstrip('/'))[:8]
    metas = sorted((json.loads(l) for l in open(os.path.join(run, 'meta.jsonl'))), key=lambda m: (m['frame'], m['t']))
    assert all(m['frame'] <= 150 for m in metas), 'holdout frames present'
    motion = trk.MotionModel()
    T = {'R_base': trk.Tracker(), 'R_N06': trk.Tracker(new_thr=0.06), 'RL_N06': trk.Tracker(new_thr=0.06),
         'R_base5': t5.Tracker(new_thr=0.25, new_thr_cls={}), 'R_B3': t5.Tracker(new_thr=0.25, new_thr_cls=dict(B3)),
         'P_B3': t5.Tracker(new_thr=0.25, new_thr_cls=dict(B3)), 'P_base': trk.Tracker()}
    W = {k: open(os.path.join(a.out, f'{rid}_{k}.jsonl'), 'w') for k in T}
    WR = open(os.path.join(a.out, f'{rid}_raw.jsonl'), 'w'); t0 = time.perf_counter(); tt = {k: 0.0 for k in T}
    for m in metas:
        img = cv2.imread(os.path.join(run, m['file']), cv2.IMREAD_COLOR)
        if img is None:
            continue
        level, region, frame = m['level'], tuple(int(v) for v in m['region']), m['frame']
        motion.observe(frame, img, level, region)
        V2 = verify(img, level, yolo2(img, level), region)
        R11 = verify(img, level, yolo11(img, level), region)
        keep = lambda L: [(d[0], d[1], d[2], d[3]) for d in L if not d[5]]  # noqa: E731
        dP = keep(V2)
        dR = keep([d for d in V2 if d[4] not in ROUTE] + [d for d in R11 if d[4] in ROUTE])
        dRL = keep([d for d in V2 if d[4] not in ROUTE_LL] + [d for d in R11 if d[4] in ROUTE_LL])
        feed = {'R_base': dR, 'R_N06': dR, 'RL_N06': dRL, 'R_base5': dR, 'R_B3': dR, 'P_B3': dP, 'P_base': dP}
        for k, tr in T.items():
            s0 = time.perf_counter()
            tr.predict(frame, motion.H)
            tr.update(feed[k], level, region, frame)
            o = ann(tr); tt[k] += time.perf_counter() - s0
            W[k].write(json.dumps(dict(frame=frame, level=level, ann=o)) + '\n')
        WR.write(json.dumps(dict(frame=frame, level=level, region=list(region), v2=rawlog(V2), r11=rawlog(R11))) + '\n')
    for w in list(W.values()) + [WR]:
        w.close()
    n = len(metas)
    print(rid, 'done', round(time.perf_counter() - t0), 's views', n, 'tracker ms/view',
          {k: round(1000 * v / max(1, n), 2) for k, v in tt.items()}, flush=True)
