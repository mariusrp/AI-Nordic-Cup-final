"""Class-hedge copies on an answer stream, with no-harm placement (drone understanding lane, cycle 3).

  condor -> jet_plane copy at the SAME conf: v2 calls the validation jets condor in 95-100% of L1 views and emits no
    jet_plane box of its own, so jet_plane's ranking is exactly condor's ranking (it cannot hurt any other class).
  medium_plane -> small_plane copy at conf x 0.0029: below the 0.003 output floor, so small_plane's own boxes keep
    their ranking (scoring fact 1: appended-below boxes never lower a class's AP).
A copy is skipped when the frame already has a same-class box at IoU >= .5. Frames <= 150 only.
Usage: python hedge_floor.py in.jsonl out.jsonl
"""
import sys, json
ip, op = sys.argv[1:3]
RULES = [('condor', 'jet_plane', 1.0), ('medium_plane', 'small_plane', 0.0029)]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


n = 0
with open(op, 'w') as w:
    for line in open(ip):
        r = json.loads(line)
        if r['frame'] > 150:
            continue
        ann = list(r['ann'])
        for src, dst, k in RULES:
            for c, b, s in r['ann']:
                if c == src and not any(c2 == dst and iou(b, b2) >= .5 for c2, b2, _ in ann):
                    ann.append([dst, b, round(s * k, 8)]); n += 1
        r['ann'] = ann
        w.write(json.dumps(r) + '\n')
print(n, 'hedge copies')
