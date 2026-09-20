"""hold181 bench, step 1: ground truth for the ft_all-UNSEEN block of the validation flight from the best answer table.

table_v3_spc.json scored 0.6623 on the portal (deterministic, all 249 frames), so its confident rows are by far the best
stand-in for the real GT we have on frames 151-249 (the eye-verified valcity labels hold no sure box past ~150, and the
n3 consensus GT was the live stack's own boxes). Rows: confident objects sit at conf >= 0.5 (eye-confirmed clusters);
0.2-0.5 = unsure; below 0.2 = hedges (box-scale copies, runner-up class copies, demoted clutter).

    python mk_gt.py /workspace/replay/tables/table_v3_spc.json OUT.json [--pos 0.5] [--ign 0.2] [--f0 151] [--f1 249]

OUT = {"gt": {frame: [{object_id, bbox px}]}, "ignore": {frame: [bbox px]}}
"""
import json
import sys

FW, FH = 3840.0, 2160.0
argv = sys.argv


def opt(name, default):
    return type(default)(argv[argv.index(name) + 1]) if name in argv else default


POS, IGN, F0, F1 = opt('--pos', 0.5), opt('--ign', 0.2), opt('--f0', 151), opt('--f1', 249)
table = json.load(open(argv[1]))
gt, ign, ncls = {}, {}, {}
for k, rows in table.items():
    f = int(k)
    if not (F0 <= f <= F1):
        continue
    g, i = [], []
    for c, b, conf in rows:
        px = [b[0] * FW, b[1] * FH, b[2] * FW, b[3] * FH]
        if conf >= POS:
            g.append(dict(object_id=c, bbox=px))
            ncls[c] = ncls.get(c, 0) + 1
        elif conf >= IGN:
            i.append(px)
    gt[f], ign[f] = g, i
json.dump(dict(gt=gt, ignore=ign, pos=POS, ign=IGN, src=argv[1]), open(argv[2], 'w'))
print('frames', len(gt), 'gt boxes', sum(len(v) for v in gt.values()), 'ignore boxes', sum(len(v) for v in ign.values()))
print('per class', dict(sorted(ncls.items(), key=lambda kv: -kv[1])))
