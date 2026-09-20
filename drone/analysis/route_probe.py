"""Floor-band package on the CLASS-ROUTED stack, plus the fast lane r5 variants' side effects (understanding lane, cycle 4).

drone-evolve-g2-2 (handed off, platform 0.1459 vs 0.1236) routes helicopter/jet_plane/small_plane to r11 and every
other class to v2, then runs the I4 verifier and the SAFER tracker. Cycle 3's floor band + floor hedges were measured on
the v2-only production stack. This replays recorded views (frames <= 150) ONCE per frame through v2, r11 and the
DINOv2 verifier (production defaults) and feeds several trackers:
  P_base   production (v2 -> verifier -> tracker new_thr .25)            reproduces birth_probe.py base
  P_N06    production dets, shadow tracker new_thr .06                    (floor-band source)
  P_DEM    production + verifier-dropped boxes fed to the tracker at raw conf x 1e-3 (FL-r5-B B2 'demote' mode)
  R_base   routed (g2-2 DRONE_ROUTE2 default), new_thr .25
  R_N06    routed dets, shadow new_thr .06
  RL_base  routed + large_launcher also from r11, new_thr .25
  U_N06    union of BOTH detectors' verified boxes (all classes), shadow new_thr .06 (2-detector floor-band source)
It also logs every frame's post-verifier detections (P and R) so the stateless FL-r5-B floor band can be rebuilt offline.
Usage (gpu3): python route_probe.py --drone DRONE_DIR --run DIR [...] --out OUT
"""
import argparse, json, os, sys, time
import numpy as np
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('--drone', required=True); ap.add_argument('--run', action='append', required=True); ap.add_argument('--out', required=True)
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

ROUTE = {CLS_INDEX[c] for c in ('helicopter', 'jet_plane', 'small_plane')}
ROUTE_LL = ROUTE | {CLS_INDEX['large_launcher']}


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
    """-> list of (srcbox, probs, conf, trunc, yolo_cls, dropped, raw_conf); production Verifier.__call__ semantics."""
    P = ver.probs(emb.embed([proto_verify.crop_box(img, d['box']) for d in raw])).cpu().numpy() if (level in ver.levels and raw) else None
    out = []
    for i, d in enumerate(raw):
        b, tr_ = src_box(d['box'], level, region)
        if b is None:
            continue
        yc = int(np.argmax(d['probs'])); rc = float(d['conf']); vc, drop = rc, False
        if P is not None:
            p = P[i]
            if p[-1] > ver.drop:
                drop = True
            elif not (not has[yc] and d['probs'][yc] >= 0.5):
                vc = float(rc ** ver.a * p[yc] ** (1 - ver.a))
        out.append((b, d['probs'], vc, tr_, yc, drop, rc))
    return out


os.makedirs(a.out, exist_ok=True)
for run in a.run:
    rid = os.path.basename(run.rstrip('/'))[:8]
    metas = sorted((json.loads(l) for l in open(os.path.join(run, 'meta.jsonl'))), key=lambda m: (m['frame'], m['t']))
    assert all(m['frame'] <= 150 for m in metas), 'holdout frames present'
    motion = trk.MotionModel()
    T = {k: trk.Tracker() for k in ('P_base', 'P_DEM', 'R_base', 'RL_base')}
    T.update({k: trk.Tracker(new_thr=0.06) for k in ('P_N06', 'R_N06', 'U_N06')})
    W = {k: open(os.path.join(a.out, f'{rid}_{k}.jsonl'), 'w') for k in T}
    WD = open(os.path.join(a.out, f'{rid}_dets.jsonl'), 'w'); t0 = time.perf_counter()
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
        dU = keep(V2 + R11)
        dDEM = dP + [(d[0], d[1], d[6] * 1e-3, d[3]) for d in V2 if d[5]]
        feed = {'P_base': dP, 'P_N06': dP, 'P_DEM': dDEM, 'R_base': dR, 'R_N06': dR, 'RL_base': dRL, 'U_N06': dU}
        for k, tr in T.items():
            tr.predict(frame, motion.H)
            tr.update(feed[k], level, region, frame)
            W[k].write(json.dumps(dict(frame=frame, level=level, ann=ann(tr))) + '\n')
        WD.write(json.dumps(dict(frame=frame, level=level,
                                 P=[[CLASSES[int(np.argmax(d[1]))], norm(d[0]), round(d[2], 6)] for d in dP],
                                 R=[[CLASSES[int(np.argmax(d[1]))], norm(d[0]), round(d[2], 6)] for d in dR])) + '\n')
    for w in list(W.values()) + [WD]:
        w.close()
    print(rid, 'done', round(time.perf_counter() - t0), 's', flush=True)
