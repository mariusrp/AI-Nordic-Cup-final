"""Per-class loss funnel from route_probe5.py raw logs (drone understanding lane, cycle 5).

For every valcity_plus GT box of the chosen classes that lies >= 90% inside a recorded view (frames <= 150):
does ANY raw detection (v2 or r11, conf >= .05, before the verifier) overlap it (IoU >= .5)? With which argmax class,
raw conf, verifier drop flag and verified conf? Counts the stage where the object is lost:
  none      no raw box of any class at IoU >= .5
  wrongcls  raw boxes exist but none with the right argmax class (lists the classes it is called)
  dropped   right class, but the verifier dropped every one (p_bg > .5)
  lowconf   right class kept, but verified conf < .25 (the birth threshold) in this view
  birthable right class, verified conf >= .25
Usage: python funnel5.py UPSTREAM_DIR RAW_DIR cls[,cls] rec [rec ...]
"""
import sys, os, json, collections
up, d, classes, recs = sys.argv[1], sys.argv[2], sys.argv[3].split(','), sys.argv[4:]
sys.path.insert(0, up); os.chdir(up)
import local_evaluator as le  # noqa: E402
GT = {f: le.load_annotations(f, 'valcity_v1_plus') for f in le.frame_numbers('valcity_v1_plus')}


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def inside(g, reg):
    ix = max(0, min(g[2], reg[2]) - max(g[0], reg[0])); iy = max(0, min(g[3], reg[3]) - max(g[1], reg[1]))
    return ix * iy >= 0.9 * (g[2] - g[0]) * (g[3] - g[1])


for cls in classes:
    st = collections.Counter(); called = collections.Counter(); best = collections.defaultdict(float); lv = collections.Counter()
    for rec in recs:
        for line in open(os.path.join(d, f'{rec}_raw.jsonl')):
            r = json.loads(line)
            if r['frame'] > 150:
                continue
            reg = r['region']
            for g in GT.get(r['frame'], []):
                if g['object_id'] != cls or not inside(g['bbox'], reg):
                    continue
                lv[r['level']] += 1
                hits = []
                for det in ('v2', 'r11'):
                    for c, b, rc, vc, drop, pbg, pc, trunc, probs in r[det]:
                        bb = [b[0] * 3840, b[1] * 2160, b[2] * 3840, b[3] * 2160]
                        if iou(g['bbox'], bb) >= .5:
                            hits.append((det, c, rc, vc, drop))
                if not hits:
                    st['none'] += 1; continue
                right = [h for h in hits if h[1] == cls]
                if not right:
                    st['wrongcls'] += 1
                    for h in hits:
                        called[f'{h[0]}:{h[1]}'] += 1
                    continue
                kept = [h for h in right if not h[4]]
                if not kept:
                    st['dropped'] += 1; continue
                m = max(h[3] for h in kept); best[rec] = max(best[rec], m)
                st['birthable' if m >= .25 else 'lowconf'] += 1
    n = sum(st.values())
    print(f'{cls}: {n} in-view object-views (L0/L1/L2 {lv[0]}/{lv[1]}/{lv[2]}) ' + ' '.join(f'{k} {v}' for k, v in st.most_common()) +
          f' | called as: {dict(called.most_common(5))} | best kept verified conf per rec: {dict((k, round(v, 3)) for k, v in best.items())}')
