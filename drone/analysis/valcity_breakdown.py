"""Where does a recorded validation run lose mAP?  (drone understanding lane, cycle 1)

Replays recorded platform answers (meta.jsonl 'ann', frames <= 150 ONLY) against the valcity labels and decomposes the
loss with the UNMODIFIED upstream scorer (local_evaluator.score):

  actual        the answers as sent
  rank_oracle   same boxes, every COCO true positive moved above every false positive  -> loss due to ranking/precision
  cls_oracle    + every box with IoU>=.5 to some GT relabelled to that GT's class (keeps conf) -> loss due to class errors
  loc_oracle    + every right-class box with .1<=IoU<.5 snapped to its GT box            -> loss due to box errors
  recall_max    fraction of GT instances matched by a right-class box (upper bound of rank_oracle)

Per GT instance it also classifies the miss: TP / wrong-class / loose-box (.1-.5) / nothing, and whether the object was
inside the current camera view (and at which level), or has ever been inside an L1/L2 view before.
Usage: python valcity_breakdown.py UPSTREAM_DIR SCENE name=meta.jsonl [...]
Never pass frames >= 151: rows with frame > 150 are dropped on read.
"""
import sys, os, json, collections
import numpy as np

up, scene = sys.argv[1], sys.argv[2]
runs = [a.split('=', 1) for a in sys.argv[3:]]
sys.path.insert(0, up); os.chdir(up)
import local_evaluator as le  # noqa: E402

FR = [f for f in le.frame_numbers(scene) if f <= 150]
GT = {f: le.load_annotations(f, scene) for f in FR}
CLS = le.OBJECT_CLASSES


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def load(mp):
    P, V = {}, {}
    for line in open(mp):
        r = json.loads(line)
        if r['frame'] > 150:
            continue
        P[r['frame']] = [dict(object_id=c, bbox=tuple(v * (3840 if i % 2 == 0 else 2160) for i, v in enumerate(b)), confidence=s)
                         for c, b, s in r['ann']]
        V[r['frame']] = (r['level'], r['region'])
    return P, V


def coco_match(P):
    """COCO greedy matching per (frame, class): returns set of (frame, det_idx) TPs and dict gt->det."""
    tp, g2d = set(), {}
    for f in FR:
        dets = P.get(f, [])
        for c in CLS:
            gts = [(gi, g) for gi, g in enumerate(GT[f]) if g['object_id'] == c]
            ds = sorted([(di, d) for di, d in enumerate(dets) if d['object_id'] == c], key=lambda x: -x[1]['confidence'])[:100]
            used = set()
            for di, d in ds:
                best, bj = 0.5, None
                for gi, g in gts:
                    if gi in used:
                        continue
                    o = iou(d['bbox'], g['bbox'])
                    if o >= best:
                        best, bj = o, gi
                if bj is not None:
                    used.add(bj); tp.add((f, di)); g2d[(f, bj)] = di
    return tp, g2d


def score(P):
    m, per = le.score(scene, {f: P[f] for f in FR if f in P})
    return m, per


def rank_oracle(P):
    tp, _ = coco_match(P)
    return {f: [dict(d, confidence=(0.5 + 0.5 * d['confidence']) if (f, i) in tp else 0.5 * d['confidence'])
                for i, d in enumerate(ds)] for f, ds in P.items()}


def cls_oracle(P):
    out = {}
    for f, ds in P.items():
        nd = []
        for d in ds:
            best, bc = 0.5, None
            for g in GT[f]:
                o = iou(d['bbox'], g['bbox'])
                if o >= best:
                    best, bc = o, g['object_id']
            nd.append(dict(d, object_id=bc) if bc else d)
        out[f] = nd
    return out


def loc_oracle(P):
    out = {}
    for f, ds in P.items():
        nd = []
        for d in ds:
            best, bb = 0.1, None
            for g in GT[f]:
                if g['object_id'] != d['object_id']:
                    continue
                o = iou(d['bbox'], g['bbox'])
                if 0.1 <= o < 0.5 and o >= best:
                    best, bb = o, tuple(float(v) for v in g['bbox'])
            nd.append(dict(d, bbox=bb) if bb else d)
        out[f] = nd
    return out


def inside(box, reg, frac=0.5):
    ix = max(0.0, min(box[2], reg[2]) - max(box[0], reg[0])); iy = max(0.0, min(box[3], reg[3]) - max(box[1], reg[1]))
    return ix * iy >= frac * (box[2] - box[0]) * (box[3] - box[1])


nG = sum(len(GT[f]) for f in FR)
per_cls_n = collections.Counter(g['object_id'] for f in FR for g in GT[f])
print(f'scene {scene}: frames {len(FR)}, GT instances {nG}, classes {len(per_cls_n)}: ' +
      ' '.join(f'{c}:{n}' for c, n in sorted(per_cls_n.items())))
sizes = collections.defaultdict(list)
for f in FR:
    for g in GT[f]:
        b = g['bbox']; sizes[g['object_id']].append(min(b[2] - b[0], b[3] - b[1]))
print('median GT short side (source px):', ' '.join(f'{c}:{np.median(v):.0f}' for c, v in sorted(sizes.items())))

for name, mp in runs:
    P, V = load(mp)
    base, per = score(P)
    r1, per_r = score(rank_oracle(P))
    C = cls_oracle(P); r2, per_c = score(rank_oracle(C))
    Lc = loc_oracle(C); r3, per_l = score(rank_oracle(Lc))
    tp, g2d = coco_match(P)
    ndet = sum(len(v) for v in P.values())
    # per-GT-instance outcome
    cat = collections.Counter(); cat_cls = collections.defaultdict(collections.Counter)
    seen_close = {}  # object key -> first frame it was inside an L1/L2 view
    vis = collections.Counter()
    for f in FR:
        lv, reg = V.get(f, (None, None))
        for gi, g in enumerate(GT[f]):
            key = g.get('valcity_cluster', gi)
            if reg is not None and lv in (1, 2) and inside(g['bbox'], reg):
                seen_close.setdefault(key, f)
            dets = P.get(f, [])
            if (f, gi) in g2d:
                k = 'TP'
            else:
                best_any = max([iou(d['bbox'], g['bbox']) for d in dets] or [0])
                best_same = max([iou(d['bbox'], g['bbox']) for d in dets if d['object_id'] == g['object_id']] or [0])
                if f not in P:
                    k = 'frame_skipped'
                elif best_same >= 0.5:
                    k = 'dup_or_cap'
                elif best_any >= 0.5:
                    k = 'wrong_class'
                elif best_same >= 0.1:
                    k = 'loose_box'
                elif best_any >= 0.1:
                    k = 'loose_wrongcls'
                else:
                    k = 'no_box'
            now = 'L%d_inview' % lv if (reg is not None and inside(g['bbox'], reg)) else 'out_of_view'
            ever = 'seen_close_before' if (key in seen_close and seen_close[key] <= f) else 'never_close_yet'
            cat[k] += 1; cat_cls[g['object_id']][k] += 1; vis[(k, now, ever)] += 1
    tpn = cat['TP']
    fp = ndet - len(tp)
    print(f'\n=== {name}: mAP {base:.4f} | rank_oracle {r1:.4f} | +cls_oracle {r2:.4f} | +loc_oracle {r3:.4f} | '
          f'dets {ndet} ({ndet / max(1, len(P)):.1f}/frame) TP {len(tp)} FP {fp} | GT {nG} recall {tpn / nG:.3f}')
    print('  GT outcome:', ' '.join(f'{k}:{v}' for k, v in cat.most_common()))
    print('  per class  AP actual / rank_oracle / +cls / +loc | outcome counts')
    for c in sorted(per_cls_n):
        print(f'   {c:15s} {per.get(c, 0):.3f} / {per_r.get(c, 0):.3f} / {per_c.get(c, 0):.3f} / {per_l.get(c, 0):.3f} | ' +
              ' '.join(f'{k}:{v}' for k, v in cat_cls[c].most_common()))
    print('  outcome x view x coverage:')
    for (k, now, ever), v in sorted(vis.items(), key=lambda x: (x[0][0], -x[1])):
        print(f'   {k:15s} {now:12s} {ever:18s} {v}')
