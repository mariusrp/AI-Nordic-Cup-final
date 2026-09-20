"""Floor-band merge: keep a base answer stream untouched and append a recall variant's EXTRA boxes below it (cycle 3).

Scoring fact (score_semantics.py): boxes appended below a class's lowest-ranked box can never lower that class's AP,
so the extras can only add recall at the tail. Extra = a variant box with no same-class base box at IoU >= .5 in the
same frame. Two versions:
  oracle  extras scaled into (0, min base conf of that class over the whole recording) -- an upper bound
  eps     extras' conf x EPS (online-feasible: needs only EPS below the tracker's lowest reported confidence)
Writes <out>_oracle.jsonl and <out>_eps.jsonl (frames <= 150 only).
Usage: python floor_band.py base.jsonl variant.jsonl out_prefix [EPS=0.001]
"""
import sys, json, collections
bp, vp, out = sys.argv[1:4]
EPS = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-3


def rd(p):
    return {r['frame']: r for r in (json.loads(l) for l in open(p)) if r['frame'] <= 150}


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


B, V = rd(bp), rd(vp)
mn = collections.defaultdict(lambda: 1.0); vmax = collections.defaultdict(float)
extra = {}
for f, r in B.items():
    for c, b, s in r['ann']:
        mn[c] = min(mn[c], s)
for f, r in V.items():
    base = B.get(f, dict(ann=[]))['ann']
    ex = [(c, b, s) for c, b, s in r['ann'] if not any(c == c2 and iou(b, b2) >= .5 for c2, b2, _ in base)]
    extra[f] = ex
    for c, b, s in ex:
        vmax[c] = max(vmax[c], s)
n = sum(len(v) for v in extra.values())
for tag in ('oracle', 'eps'):
    with open(f'{out}_{tag}.jsonl', 'w') as w:
        for f in sorted(set(B) | set(V)):
            ann = list(B.get(f, dict(ann=[]))['ann'])
            for c, b, s in extra.get(f, []):
                k = (0.999 * mn[c] / vmax[c]) if (tag == 'oracle' and c in mn) else EPS
                ann.append([c, b, round(s * k, 8)])
            w.write(json.dumps(dict(frame=f, level=(B.get(f) or V.get(f))['level'], ann=ann)) + '\n')
print(f'{n} extra boxes; base min conf by class: ' + ', '.join(f'{c} {mn[c]:.4f}' for c in sorted(mn)))
