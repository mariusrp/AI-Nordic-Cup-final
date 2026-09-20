"""hold181 bench, step 3: score replay dumps (replay_multi.py: one JSONL line per served frame with the SERVED annotations)
of several arms on the same recorded flights against the table GT (mk_gt.py), with the organisers' unmodified
local_evaluator.score. Every (recording, frame) in [F0, F1] is one image; predictions that overlap an ignore box
(unsure table rows) are dropped first. Reports class-aware mAP@0.5 (the metric), class-agnostic AP@0.5 (robust to
label noise in the table), per-class AP, and a paired bootstrap over recordings vs the FIRST arm.

    UPSTREAM=/workspace/upstream python score_hold.py GT.json DUMP_DIR F0 F1 NBOOT name=TAG [name2=TAG2 ...]
"""
import glob
import json
import os
import sys

import numpy as np

UP = os.path.join(os.environ.get("UPSTREAM", "/workspace/upstream"), "drone-flyby")
sys.path.insert(0, UP)
os.chdir(UP)
import local_evaluator as le  # noqa: E402
from dtos import OBJECT_CLASSES  # noqa: E402

GT_PATH, DUMP_DIR, F0, F1, NB = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
ARMS = [a.split('=', 1) for a in sys.argv[6:]]
FW, FH = 3840.0, 2160.0
G = json.load(open(GT_PATH))
GT = {int(k): v for k, v in G['gt'].items()}
IGN = {int(k): v for k, v in G['ignore'].items()}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def load(tag):
    d = {}
    for p in sorted(glob.glob(os.path.join(DUMP_DIR, f"{tag}_*.jsonl"))):
        seq = os.path.basename(p)[len(tag) + 1:-6]
        for line in open(p):
            r = json.loads(line)
            f = r['frame']
            if not (F0 <= f <= F1) or f not in GT:
                continue
            preds = []
            for c, b, conf in r['ann']:
                px = [b[0] * FW, b[1] * FH, b[2] * FW, b[3] * FH]
                if any(iou(px, g) >= 0.3 for g in IGN.get(f, [])):
                    continue
                preds.append(dict(object_id=c, bbox=px, confidence=float(conf)))
            d[(seq, f)] = preds
    return d


def score_set(keys, preds_by_key, agnostic=False):
    """keys: list of (seq, frame) -> (mAP, per-class) via the upstream scorer; each key is one image."""
    ids = list(range(1, len(keys) + 1))
    gt_by_id, pr_by_id = {}, {}
    for i, k in zip(ids, keys):
        g = [dict(object_id=('tank' if agnostic else a['object_id']), bbox=list(a['bbox'])) for a in GT[k[1]]]
        p = [dict(object_id=('tank' if agnostic else a['object_id']), bbox=list(a['bbox']), confidence=a['confidence'])
             for a in preds_by_key.get(k, [])]
        gt_by_id[i], pr_by_id[i] = g, p
    le.frame_numbers = lambda s: ids
    le.load_annotations = lambda v, s: gt_by_id[v]
    try:
        return le.score('hold', pr_by_id)
    except ValueError:
        return float('nan'), {}


arms = {name: load(tag) for name, tag in ARMS}
keys = sorted(set.intersection(*(set(d) for d in arms.values())))
seqs = sorted({k[0] for k in keys})
print(f"GT {GT_PATH} frames {F0}-{F1}: {len(keys)} images common to all arms, {len(seqs)} recordings, "
      f"{sum(len(GT[k[1]]) for k in keys)} GT boxes")
ncls = {}
for k in keys:
    for a in GT[k[1]]:
        ncls[a['object_id']] = ncls.get(a['object_id'], 0) + 1

res, agn = {}, {}
for name in arms:
    res[name] = score_set(keys, arms[name])
    agn[name] = score_set(keys, arms[name], agnostic=True)[0]
    nb = sum(len(v) for k, v in arms[name].items() if k in keys) / max(1, len(keys))
    print(f"{name:12s} mAP@0.5 {res[name][0]:.4f}  agnostic {agn[name]:.4f}  boxes/frame {nb:.1f}")

order = [n for n, _ in ARMS]
print()
print('class'.ljust(18) + 'nGT'.rjust(5) + ''.join(n[:12].rjust(13) for n in order))
for c in OBJECT_CLASSES:
    if c in ncls:
        print(c[:18].ljust(18) + str(ncls[c]).rjust(5) + ''.join(f"{res[n][1].get(c, float('nan')):13.3f}" for n in order))

if NB > 0 and len(order) > 1:
    rng = np.random.default_rng(0)
    by_seq = {s: [k for k in keys if k[0] == s] for s in seqs}
    diffs = {n: [] for n in order[1:]}
    diffs_a = {n: [] for n in order[1:]}
    for _ in range(NB):
        pick = rng.choice(seqs, size=len(seqs), replace=True)
        kk = [k for s in pick for k in by_seq[s]]
        # duplicate images get distinct ids through the position in kk
        base = score_set(kk, arms[order[0]])[0]
        base_a = score_set(kk, arms[order[0]], agnostic=True)[0]
        for n in order[1:]:
            diffs[n].append(score_set(kk, arms[n])[0] - base)
            diffs_a[n].append(score_set(kk, arms[n], agnostic=True)[0] - base_a)
    print()
    for n in order[1:]:
        d, da = np.array(diffs[n]), np.array(diffs_a[n])
        print(f"{n:12s} vs {order[0]}: aware {res[n][0] - res[order[0]][0]:+.4f} "
              f"[{np.percentile(d, 2.5):+.4f},{np.percentile(d, 97.5):+.4f}] P(>0) {np.mean(d > 0):.2f} | "
              f"agnostic {agn[n] - agn[order[0]]:+.4f} [{np.percentile(da, 2.5):+.4f},{np.percentile(da, 97.5):+.4f}] "
              f"P(>0) {np.mean(da > 0):.2f}")
print('SCORE_HOLD ' + json.dumps({n: [round(res[n][0], 4), round(agn[n], 4)] for n in order}))
