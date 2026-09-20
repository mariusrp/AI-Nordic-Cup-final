"""B5 step 4: score several detector checkpoints on the SAME held-out 4K frames with the organisers' own scorer.

For every frame we tile the whole 4K image with camera views rendered exactly like local_evaluator.render_view
(crop the level region, INTER_AREA downscale by 4 / 2 / 1), run each checkpoint on every tile, lift the boxes back to
source pixels, do a class-wise NMS across tiles and levels, and hand the result to the UNMODIFIED
local_evaluator.score (COCO mAP@0.5, macro over the classes present in those frames). Every checkpoint sees exactly the
same tiles, so the comparison is paired.

This is a DETECTOR metric: no tracker, no planner, no answer table. It is the honest stand-in for the final flight
over an unseen city, and its only ground truth from a different city is the 25 Helsinki frames.

Two numbers per checkpoint:
  mAP@0.5     - the organisers' metric, from local_evaluator.score itself.
  mAP@.5:.95  - the same predictions through faster_coco_eval over the standard IoU sweep. On a small, easy scene
                mAP@0.5 saturates (both current baselines score 0.986 on 5 Helsinki frames, so it cannot rank
                anything); the IoU sweep still separates checkpoints by localisation quality.

Usage: python eval_scene.py UPSTREAM_DIR SCENE_DIR [--frames 4,12,20,22,24] [--levels 0,1,2] [--overlap 1.3]
                            [--conf 0.05] [--imgsz 1280] [--perlevel] [--fprec 0.9] [--perclass95] name=weights.pt ...
       name=stack:weights.pt runs the production stack (DRONE_WEIGHTS2 routing, DRONE_VERIFY verifier) instead of raw YOLO
       name=fuse:MODE:A,B:wA,wB:iou  fuses the PER-VIEW output of two runs already named on the command line
                                     (no extra GPU pass, and the singles stay bit-identical, so it is paired).
                                     MODE=wbf   weighted box fusion with the consensus rescale
                                                (box = score-weighted mean, score = sum_m u_m*s_m over the models
                                                 present in the cluster, u = w/sum(w) -> a box only one model found
                                                 keeps only its share of the score);
                                     MODE=avg   same weighted box, score = max over models (no consensus penalty);
                                     MODE=union boxes pooled unchanged (= detector.py UnionDetector / DRONE_UNION,
                                                whose merge_classes keeps the higher confidence on same-class overlap,
                                                here done by the cross-tile class-wise NMS that every run goes through).
"""
import json
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fuse_boxes import wbf  # noqa: E402

UP, SCENE = sys.argv[1], sys.argv[2]
argv = sys.argv[3:]


def opt(name, default):
    return argv[argv.index(name) + 1] if name in argv else default


LEVELS = [int(v) for v in opt('--levels', '0,1,2').split(',')]
OVERLAP = float(opt('--overlap', 1.3))
CONF = float(opt('--conf', 0.05))
IMGSZ = int(opt('--imgsz', 1280))
PERLEVEL = '--perlevel' in argv
FPREC = float(opt('--fprec', '0.9'))
PERCLASS95 = '--perclass95' in argv
RUNS = [a.split('=', 1) for a in argv if '=' in a and not a.startswith('--')]
BASE = [(n, w) for n, w in RUNS if not w.startswith('fuse:')]
FUSE = [(n, w) for n, w in RUNS if w.startswith('fuse:')]
REG = {0: (3840, 2160), 1: (1920, 1080), 2: (960, 540)}

IMGS = {int(re.findall(r'(\d+)', f)[0]): f'{SCENE}/images/{f}'
        for f in os.listdir(f'{SCENE}/images') if f.endswith(('.png', '.jpg'))}
all_frames = sorted(IMGS)
fsel = opt('--frames', '')
FRAMES = [int(v) for v in fsel.split(',')] if fsel else all_frames
FRAMES = [f for f in FRAMES if f in all_frames]

sys.path.insert(0, os.path.join(UP, 'drone-flyby'))
from dtos import OBJECT_CLASSES  # noqa: E402
import local_evaluator as le     # noqa: E402


def tiles(level):
    rw, rh = REG[level]
    nx = max(1, int(np.ceil(3840 / rw * OVERLAP)))
    ny = max(1, int(np.ceil(2160 / rh * OVERLAP)))
    xs = np.linspace(0, 3840 - rw, nx).astype(int)
    ys = np.linspace(0, 2160 - rh, ny).astype(int)
    return [(int(x), int(y), int(x + rw), int(y + rh)) for y in ys for x in xs]


def nmsd(per_frame):
    """NMS once per frame; every metric below then reuses it (greedy NMS commutes with a score threshold,
    because a box is only ever suppressed by a higher-scoring one, so this is exact, just ~50x cheaper)."""
    return {f: nms(per_frame[f]) for f in per_frame}


def nms(dets, thr=0.5):
    """dets: [(bbox xyxy source px, class name, score)] -> class-wise greedy NMS."""
    out = []
    for d in sorted(dets, key=lambda d: -d[2]):
        keep = True
        for o in out:
            if o[1] != d[1]:
                continue
            a, b = o[0], d[0]
            ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
            if u > 0 and ix * iy / u > thr:
                keep = False
                break
        if keep:
            out.append(d)
    return out


# ---------------------------------------------------------------- run every checkpoint on identical tiles
from ultralytics import YOLO  # noqa: E402

models = {}
for name, w in BASE:
    if w.startswith('stack:'):
        # the PRODUCTION stack (detector.build_detector): routing DRONE_WEIGHTS2, the DINOv2 verifier
        # DRONE_VERIFY, ... exactly as server.py builds it, so a knob can be A/B'd on these scenes.
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from detector import build_detector  # noqa: E402
        models[name] = ('stack', build_detector(w[6:], os.environ.get('DRONE_WEIGHTS2', '')))
    else:
        m = YOLO(w, task='detect')
        models[name] = ('yolo', (m, {int(k): v for k, v in m.names.items()}))
    print('loaded', name, w, flush=True)

preds = {name: {L: {f: [] for f in FRAMES} for L in LEVELS} for name, _ in RUNS}
tp = {name: {L: {} for L in LEVELS} for name, _ in BASE}  # per-TILE detections, so fusion is per view like server.py
for f in FRAMES:
    img4k = cv2.imread(IMGS[f])
    if img4k is None:
        print('missing frame', f, flush=True)
        continue
    for L in LEVELS:
        s = {0: 4, 1: 2, 2: 1}[L]
        views, origins, metas = [], [], []
        for (x1, y1, x2, y2) in tiles(L):
            v = img4k[y1:y2, x1:x2]
            views.append(cv2.resize(v, (960, 540), interpolation=cv2.INTER_AREA) if L < 2 else v.copy())
            origins.append((x1, y1))
            metas.append(dict(resolution_level=L, center_x=(x1 + x2) // 2, center_y=(y1 + y2) // 2,
                              source_region_xyxy=(x1, y1, x2, y2), width=960, height=540))
        for name, (kind, obj) in models.items():
            per_tile = [[] for _ in origins]
            if kind == 'stack':
                # production contract: det(img_bgr, level) -> [{'box': xyxy view px, 'probs': [NC], 'conf': c}]
                for ti, ((ox, oy), mt, v) in enumerate(zip(origins, metas, views)):
                    if hasattr(obj, 'set_context'):
                        obj.set_context(f, mt['source_region_xyxy'])
                    for d in obj(v, L):
                        b = d['box']
                        ci = int(np.argmax(d['probs']))
                        per_tile[ti].append(((ox + b[0] * s, oy + b[1] * s, ox + b[2] * s, oy + b[3] * s),
                                             OBJECT_CLASSES[ci], float(d['conf']) * float(d['probs'][ci])))
            else:
                m, names = obj
                res = m.predict(views, imgsz=IMGSZ, conf=CONF, max_det=150, verbose=False, device=0)
                for ti, ((ox, oy), r) in enumerate(zip(origins, res)):
                    bb = r.boxes
                    if bb is None or len(bb) == 0:
                        continue
                    xy = bb.xyxy.cpu().numpy()
                    cl = bb.cls.cpu().numpy().astype(int)
                    cf = bb.conf.cpu().numpy()
                    for b, c, sc in zip(xy, cl, cf):
                        per_tile[ti].append(((ox + b[0] * s, oy + b[1] * s, ox + b[2] * s, oy + b[3] * s),
                                             names[int(c)], float(sc)))
            tp[name][L][f] = per_tile
            preds[name][L][f] = [d for t in per_tile for d in t]
    print('frame', f, 'done', flush=True)

# ---------------------------------------------------------------- fusion candidates (no extra GPU pass)
for name, spec in FUSE:
    mode, srcs, ws, iou = spec.split(':')[1:5]
    srcs = srcs.split(',')
    ws = [float(v) for v in ws.split(',')]
    iou = float(iou)
    for L in LEVELS:
        for f in FRAMES:
            if any(f not in tp[s][L] for s in srcs):
                continue
            nt = len(tp[srcs[0]][L][f])
            preds[name][L][f] = [d for ti in range(nt)
                                 for d in wbf([tp[s][L][f][ti] for s in srcs], ws, iou, mode)]
    print('fused', name, spec, flush=True)

# ---------------------------------------------------------------- score with the organisers' code
GT = {f: json.load(open(f'{SCENE}/annotations/frame_{f:06d}.json'))['annotations'] for f in FRAMES}
vids = list(range(1, len(FRAMES) + 1))
le.frame_numbers = lambda s: vids
gt_by_v = {v: [dict(object_id=a['object_id'], bbox=list(a['bbox'])) for a in GT[f]] for v, f in zip(vids, FRAMES)}
le.load_annotations = lambda v, s: gt_by_v[v]
os.chdir(os.path.join(UP, 'drone-flyby'))


def coco_sweep(per_frame):
    """The same predictions at IoU 0.50:0.95 (macro over the present classes), which does not saturate."""
    from faster_coco_eval import COCO, COCOeval_faster
    cid = {c: i + 1 for i, c in enumerate(OBJECT_CLASSES)}
    present = sorted({a['object_id'] for f in FRAMES for a in GT[f]}, key=OBJECT_CLASSES.index)
    anns, k = [], 1
    for v, f in zip(vids, FRAMES):
        for a in GT[f]:
            x1, y1, x2, y2 = a['bbox']
            anns.append(dict(id=k, image_id=v, category_id=cid[a['object_id']], bbox=[x1, y1, x2 - x1, y2 - y1],
                             area=(x2 - x1) * (y2 - y1), iscrowd=0))
            k += 1
    g = dict(images=[dict(id=v, width=3840, height=2160, file_name=str(f)) for v, f in zip(vids, FRAMES)],
             categories=[dict(id=cid[c], name=c) for c in OBJECT_CLASSES], annotations=anns)
    dts = [dict(image_id=v, category_id=cid[c], bbox=[b[0], b[1], b[2] - b[0], b[3] - b[1]], score=sc)
           for v, f in zip(vids, FRAMES) for b, c, sc in per_frame[f]]
    if not dts:
        return 0.0
    cg = COCO(g)
    ev = COCOeval_faster(cg, cg.loadRes(dts), 'bbox')
    ev.params.catIds = [cid[c] for c in present]
    ev.evaluate(); ev.accumulate()
    P = ev.eval['precision'][:, :, :, 0, -1]
    vals, per = [], {}
    for i, c in enumerate(present):
        if (P[:, :, i] > -1).any():
            v = float(P[:, :, i][P[:, :, i] > -1].mean())
            vals.append(v)
            per[c] = v
    return (float(np.mean(vals)) if vals else 0.0), per


def fp_recall(per_frame, thr=0.25, iou=0.3):
    """High-ranked false positives per frame and class-aware recall at `thr` - the number COCO AP hides and the one
    that killed the detector on the unseen validation city (FP flood on urban clutter)."""
    fp = hit = ngt = 0
    for f in FRAMES:
        dets = [d for d in per_frame[f] if d[2] >= thr]
        gts = [(a['bbox'], a['object_id']) for a in GT[f]]
        ngt += len(gts)
        taken = set()
        for b, c, sc in sorted(dets, key=lambda d: -d[2]):
            best, bj = 0.0, -1
            for j, (g, gc) in enumerate(gts):
                if j in taken or gc != c:
                    continue
                ix = max(0.0, min(b[2], g[2]) - max(b[0], g[0]))
                iy = max(0.0, min(b[3], g[3]) - max(b[1], g[1]))
                u = (b[2] - b[0]) * (b[3] - b[1]) + (g[2] - g[0]) * (g[3] - g[1]) - ix * iy
                v = ix * iy / u if u > 0 else 0.0
                if v > best:
                    best, bj = v, j
            if best >= iou:
                taken.add(bj); hit += 1
            else:
                fp += 1
    return fp / max(1, len(FRAMES)), hit / max(1, ngt)


def fp_at_recall(per_frame, target=0.9, iou=0.3):
    """Sweep the confidence threshold; report (FP/frame, recall, thr) at the HIGHEST threshold whose class-aware
    recall still reaches `target`. If the detector never reaches it, report its best recall at conf=CONF."""
    best = None
    for thr in [round(t, 3) for t in np.arange(0.95, CONF - 1e-9, -0.01)]:
        fp, rec = fp_recall(per_frame, thr=thr, iou=iou)
        if rec >= target:
            return fp, rec, thr
        best = (fp, rec, thr)
    return best if best else (0.0, 0.0, CONF)


def score(per_frame):
    p = {v: [dict(object_id=c, bbox=list(b), confidence=sc) for b, c, sc in per_frame[f]]
         for v, f in zip(vids, FRAMES)}
    return le.score('x', p)


ngt = {}
for f in FRAMES:
    for a in GT[f]:
        ngt[a['object_id']] = ngt.get(a['object_id'], 0) + 1
rows, sweep, fpr, sweep_c, fprec = {}, {}, {}, {}, {}
for name, _ in RUNS:
    pooled = nmsd({f: sum((preds[name][L][f] for L in LEVELS), []) for f in FRAMES})
    rows[name] = score(pooled)
    sweep[name], sweep_c[name] = coco_sweep(pooled)
    fpr[name] = fp_recall(pooled)
    fprec[name] = fp_at_recall(pooled, FPREC)
    if PERLEVEL:
        for L in LEVELS:
            pl = nmsd(preds[name][L])
            rows[f'{name}@L{L}'] = score(pl)
            sweep[f'{name}@L{L}'], sweep_c[f'{name}@L{L}'] = coco_sweep(pl)
            fpr[f'{name}@L{L}'] = fp_recall(pl)
            fprec[f'{name}@L{L}'] = fp_at_recall(pl, FPREC)

order = [n for n, _ in RUNS] + ([f'{n}@L{L}' for n, _ in RUNS for L in LEVELS] if PERLEVEL else [])
print()
print(f'SCENE {SCENE} frames {FRAMES} levels {LEVELS} overlap {OVERLAP} conf {CONF} imgsz {IMGSZ}')
print('class'.ljust(18) + 'nGT'.rjust(5) + ''.join(n[:13].rjust(14) for n in order))
print('mAP@0.5'.ljust(18) + ''.rjust(5) + ''.join(f'{rows[n][0]:14.4f}' for n in order))
print('mAP@.5:.95'.ljust(18) + ''.rjust(5) + ''.join(f'{sweep[n]:14.4f}' for n in order))
print('FP/frame@.25'.ljust(18) + ''.rjust(5) + ''.join(f'{fpr[n][0]:14.2f}' for n in order))
print('recall@.25'.ljust(18) + ''.rjust(5) + ''.join(f'{fpr[n][1]:14.3f}' for n in order))
print(f'FP/frame@R{FPREC:.2f}'.ljust(18) + ''.rjust(5) + ''.join(f'{fprec[n][0]:14.2f}' for n in order))
print('  (recall,thr)'.ljust(18) + ''.rjust(5) + ''.join(f'{fprec[n][1]:.2f}/{fprec[n][2]:.2f}'.rjust(14) for n in order))
for c in OBJECT_CLASSES:
    if c not in ngt:
        continue
    print(c[:18].ljust(18) + str(ngt[c]).rjust(5) + ''.join(f'{rows[n][1].get(c, float("nan")):14.3f}' for n in order))
if PERCLASS95:
    print('-- per-class AP@.5:.95 --')
    for c in OBJECT_CLASSES:
        if c not in ngt:
            continue
        print(c[:18].ljust(18) + str(ngt[c]).rjust(5) + ''.join(f'{sweep_c[n].get(c, float("nan")):14.3f}' for n in order))
print('EVAL_SCENE ' + json.dumps({n: round(float(rows[n][0]), 4) for n in order}))
print('EVAL_SCENE_SWEEP ' + json.dumps({n: round(float(sweep[n]), 4) for n in order}))
print('EVAL_SCENE_FP ' + json.dumps({n: [round(fpr[n][0], 2), round(fpr[n][1], 3)] for n in order}))
print('EVAL_SCENE_FPREC ' + json.dumps({n: [round(fprec[n][0], 2), round(fprec[n][1], 3), fprec[n][2]] for n in order}))
print('EVAL_SCENE_PERCLASS95 ' + json.dumps({n: {c: round(v, 4) for c, v in sweep_c[n].items()} for n in order}))
