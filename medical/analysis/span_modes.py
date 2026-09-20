"""Loss-mode breakdown of predicted evidence spans vs gold, per arm (medical-final lane).

    UPSTREAM=/workspace/.holdout/upstream-full python3 analysis/span_modes.py out/b1/base.rep0.json ...

Classes over the gold-yes rows (the only rows the scorer's tIoU touches):
  exact   tIoU >= 0.9
  under   the span lies inside the gold (both edges within tol) and tIoU < 0.9
  over    the gold lies inside the span and tIoU < 0.9
  shift   overlapping but neither contained (edges drift on one side)
  wrong   overlap < 10% of the union (different passage)
  none    no span sent at all
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import offline_eval as OE  # noqa: E402

sys.path.insert(0, os.path.join(os.environ.get('UPSTREAM', '/workspace/.holdout/upstream-full'),
                                'medical-appointment'))
from utils import evidence_interval, gold_evidence, temporal_iou  # noqa: E402

TOL = 0.25


def classify(gold, sp):
    if sp is None:
        return 'none'
    iou = temporal_iou(gold, sp)
    if iou >= 0.9:
        return 'exact'
    gs, ge = float(gold[0]), float(gold[1])
    ps, pe = float(sp[0]), float(sp[1])
    inter = max(0.0, min(ge, pe) - max(gs, ps))
    union = max(ge, pe) - min(gs, ps)
    if union <= 0 or inter / union < 0.10:
        return 'wrong'
    if ps >= gs - TOL and pe <= ge + TOL:
        return 'under'
    if ps <= gs + TOL and pe >= ge - TOL:
        return 'over'
    return 'shift'


def run(path):
    preds = json.load(open(path))
    counts = {k: 0 for k in ('exact', 'under', 'over', 'shift', 'wrong', 'none')}
    tot_iou, n = 0.0, 0
    for fn, rows in OE.group_questions_by_conversation():
        p = preds.get(fn)
        if p is None:
            continue
        for qi, row in enumerate(rows):
            if int(row['label']) != 1:
                continue
            gold = gold_evidence(row)
            sp = evidence_interval(p['evidence_start'][qi], p['evidence_end'][qi]) if p['answers'][qi] else None
            counts[classify(gold, sp)] += 1
            tot_iou += temporal_iou(gold, sp) if sp else 0.0
            n += 1
    return {'file': os.path.basename(path), 'n_gold_yes': n,
            'mean_tiou_yesrows': round(tot_iou / n, 4) if n else 0.0, **counts}


if __name__ == '__main__':
    for a in sys.argv[1:]:
        print(json.dumps(run(a)))
