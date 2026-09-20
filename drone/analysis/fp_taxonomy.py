"""What kind of false positives cost AP, and which frames were never answered?  (drone understanding lane, cycle 2)

For each recorded run (frames <= 150 only) on a valcity label set, with the UNMODIFIED upstream scorer:
  * COCO-style greedy matching per (frame, class) at IoU .5 (same as the scorer; ties resolved by confidence order).
  * every false positive is typed:
      dup      same class, IoU >= .5 with a GT that an earlier (higher-conf) box already matched  (a duplicate)
      wrongcls IoU >= .5 with a GT of ANOTHER class                                               (class confusion)
      loose    same class, .1 <= IoU < .5 with a GT                                                (box/drift error)
      near     any class, .1 <= IoU < .5 with some GT                                               (drift + class)
      clutter  IoU < .1 with every GT                                                               (background, OR an unlabelled object)
  * AP at stake per type: rescore with that type of FP removed (an oracle deletion; upper bound of what suppressing it buys).
  * frames with no answer (skipped/timeouts) and server time (meta 'ms').
Usage: python fp_taxonomy.py UPSTREAM_DIR SCENE name=meta.jsonl [...]
"""
import sys, os, json, collections
import numpy as np
up, scene = sys.argv[1], sys.argv[2]
sys.path.insert(0, up); os.chdir(up)
import local_evaluator as le  # noqa: E402
FR = [f for f in le.frame_numbers(scene) if f <= 150]
GT = {f: le.load_annotations(f, scene) for f in FR}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def load(mp):
    P, M = {}, {}
    for line in open(mp):
        r = json.loads(line)
        if r['frame'] > 150:
            continue
        P[r['frame']] = [dict(object_id=c, bbox=tuple(v * (3840 if i % 2 == 0 else 2160) for i, v in enumerate(b)), confidence=s)
                         for c, b, s in r['ann']]
        M[r['frame']] = r
    return P, M


def typed(P):
    T = {}
    for f in FR:
        ds = P.get(f, [])
        for c in {d['object_id'] for d in ds}:
            idx = sorted([i for i, d in enumerate(ds) if d['object_id'] == c], key=lambda i: -ds[i]['confidence'])
            gts = [g for g in GT[f] if g['object_id'] == c]
            used = set()
            for rank, i in enumerate(idx):
                d = ds[i]
                if rank >= 100:
                    T[(f, i)] = 'capped'; continue
                best, bj = 0.5, None
                for j, g in enumerate(gts):
                    if j in used:
                        continue
                    o = iou(d['bbox'], g['bbox'])
                    if o >= best:
                        best, bj = o, j
                if bj is not None:
                    used.add(bj); T[(f, i)] = 'TP'; continue
                same = max([iou(d['bbox'], g['bbox']) for g in gts] or [0])
                other = max([iou(d['bbox'], g['bbox']) for g in GT[f] if g['object_id'] != c] or [0])
                if same >= 0.5:
                    T[(f, i)] = 'dup'
                elif other >= 0.5:
                    T[(f, i)] = 'wrongcls'
                elif same >= 0.1:
                    T[(f, i)] = 'loose'
                elif other >= 0.1:
                    T[(f, i)] = 'near'
                else:
                    T[(f, i)] = 'clutter'
    return T


for a in sys.argv[3:]:
    name, mp = a.split('=', 1)
    P, M = load(mp)
    base, per = le.score(scene, {f: P[f] for f in FR if f in P})
    T = typed(P)
    cnt = collections.Counter(T.values())
    # conf-rank position of FPs: how many FPs outrank the median TP of their class?
    print(f'\n=== {name}: valcity mAP {base:.4f}; answered frames {len(P)}/{len(FR)}; boxes {sum(len(v) for v in P.values())}: ' +
          ' '.join(f'{k}:{v}' for k, v in cnt.most_common()))
    ms = [M[f]['ms'] for f in M if M[f].get('ms') is not None]
    missing = [f for f in FR if f not in P]
    print(f'  server ms p50 {np.median(ms):.0f} p90 {np.percentile(ms, 90):.0f} max {max(ms):.0f}; unanswered frames {missing}')
    for k in ('dup', 'wrongcls', 'loose', 'near', 'clutter'):
        Q = {f: [d for i, d in enumerate(P[f]) if T.get((f, i)) != k] for f in P if f in FR}
        m, per2 = le.score(scene, Q)
        gain = {c: per2[c] - per[c] for c in per if abs(per2[c] - per[c]) > 0.005}
        print(f'  remove all {k:8s} FPs ({cnt[k]:4d}): mAP {m:.4f} ({m - base:+.4f})  ' + ' '.join(f'{c[:8]}:{v:+.2f}' for c, v in sorted(gain.items(), key=lambda x: -x[1])))
    # per class: FPs ranked above the class's k-th TP
    rows = []
    for c in sorted(per):
        confs = sorted([(d['confidence'], T[(f, i)]) for f in P if f in FR for i, d in enumerate(P[f]) if d['object_id'] == c and (f, i) in T], key=lambda x: -x[0])
        tps = [i for i, (s, t) in enumerate(confs) if t == 'TP']
        if not tps:
            rows.append(f'{c[:10]}: TP 0, FP {len(confs)}'); continue
        above = collections.Counter(t for s, t in confs[:tps[len(tps) // 2]] if t != 'TP')
        rows.append(f'{c[:10]}: TP {len(tps)} FP {len(confs) - len(tps)}, above median TP: ' + ','.join(f'{k}{v}' for k, v in above.most_common()))
    print('  ' + '\n  '.join(rows))
