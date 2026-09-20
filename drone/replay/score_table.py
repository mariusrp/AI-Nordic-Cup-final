"""Score an answer table (fuse.py format, or a recorder meta.jsonl) against a GT json with the organisers' UNMODIFIED
local_evaluator.score (COCO mAP@0.5, faster_coco_eval). GT formats: {"gt": {frame: [{object_id, bbox}]}} (n3 files),
targets_plus.json ({frame: {pos: [[cls,x1,y1,x2,y2]], ign: [...]}}) or a fuse table. Options: --frames A-B, --agn
(class-agnostic: every class -> 'x'), --ignore targets_plus.json (drop predictions whose centre lies in an ignore box or
on an unsure label; those are real-but-unlabelled objects, FPs only against the eye GT).
Usage: python score_table.py UPSTREAM_DIR GT.json [--frames 100-150] [--agn] [--ignore T.json] name=table.json [...]"""
import json
import os
import sys

import numpy as np

up, gtp = sys.argv[1], sys.argv[2]
argv = sys.argv[3:]
FR = argv[argv.index('--frames') + 1] if '--frames' in argv else '1-249'
F0, F1 = map(int, FR.split('-'))
AGN = '--agn' in argv
IGN = argv[argv.index('--ignore') + 1] if '--ignore' in argv else None
runs = [a.split('=', 1) for a in argv if '=' in a and not a.startswith('--')]
sys.path.insert(0, up); os.chdir(up)
import local_evaluator as le  # noqa: E402


def load_any(p):
    """-> {frame:int: [dict(object_id, bbox xyxy px, confidence)]}"""
    out = {}
    if p.endswith('.jsonl'):
        for line in open(p):
            r = json.loads(line)
            out[r['frame']] = [dict(object_id=c, bbox=[b[0] * 3840, b[1] * 2160, b[2] * 3840, b[3] * 2160], confidence=s) for c, b, s in r['ann']]
        return out
    d = json.load(open(p))
    if 'gt' in d: d = d['gt']
    for k, v in d.items():
        if not k.isdigit(): continue
        t = int(k)
        if isinstance(v, dict) and 'pos' in v:
            out[t] = [dict(object_id=c, bbox=[x1, y1, x2, y2], confidence=1.0) for c, x1, y1, x2, y2 in v['pos']]
        elif v and isinstance(v[0], dict):
            out[t] = [dict(object_id=a['object_id'], bbox=list(a['bbox']), confidence=a.get('confidence', 1.0)) for a in v]
        else:
            out[t] = [dict(object_id=c, bbox=[b[0] * 3840, b[1] * 2160, b[2] * 3840, b[3] * 2160], confidence=s) for c, b, s in v]
    return out


GT = load_any(gtp)
frames = [t for t in range(F0, F1 + 1) if t in GT]
ign = {}
if IGN:
    tj = json.load(open(IGN))
    for k, v in tj.items(): ign[int(k)] = v['ign']


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def filt(preds, t):
    """COCO-crowd-like ignore: a prediction whose centre lies in an ignore box and that matches no GT box (IoU < 0.3) is dropped."""
    if t not in ign: return preds
    out = []
    for p in preds:
        cx, cy = (p['bbox'][0] + p['bbox'][2]) / 2, (p['bbox'][1] + p['bbox'][3]) / 2
        if any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in ign[t]) and all(_iou(p['bbox'], g['bbox']) < 0.3 for g in GT.get(t, [])):
            continue
        out.append(p)
    return out


vids = list(range(1, len(frames) + 1))
le.frame_numbers = lambda s: vids
gt_by_v = {v: [dict(a, object_id='ta-ta' if AGN else a['object_id']) for a in GT[f]] for v, f in zip(vids, frames)}
le.load_annotations = lambda v, s: gt_by_v[v]
res = {}
for name, p in runs:
    P = load_any(p)
    preds = {v: [dict(a, object_id='ta-ta' if AGN else a['object_id']) for a in filt(P.get(f, []), f)] for v, f in zip(vids, frames)}
    res[name] = le.score('x', preds)
cls = sorted({k for _, pc in res.values() for k in pc})
ngt = {}
for f in frames:
    for a in GT[f]: ngt[a['object_id']] = ngt.get(a['object_id'], 0) + 1
print(f'GT {gtp.split("/")[-1]} frames {F0}-{F1} ({len(frames)} with GT) boxes {sum(ngt.values())}' + (' AGN' if AGN else '') + (' IGN' if IGN else ''))
print('class'.ljust(18) + 'nGT'.rjust(6) + ''.join(n[:11].rjust(12) for n, _ in runs))
print('mAP'.ljust(18) + ''.rjust(6) + ''.join(f'{res[n][0]:12.4f}' for n, _ in runs))
for c in cls:
    print(c[:18].ljust(18) + str(ngt.get(c, 0)).rjust(6) + ''.join(f'{res[n][1].get(c, float("nan")):12.3f}' for n, _ in runs))
