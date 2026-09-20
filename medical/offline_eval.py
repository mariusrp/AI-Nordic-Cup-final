"""Score a predictions file against question_train.csv exactly like the upstream
local_evaluator (same Statistics class, same tIoU), without a server.

Predictions JSON: {"<audio_filename or transcript_id>": {"answers": [...],
                   "evidence_start": [...], "evidence_end": [...]}, ...}
Lists are positional, in the CSV's question order for that conversation
(the order the evaluator sends them).

    python offline_eval.py preds.json                 # all conversations in the file
    python offline_eval.py preds.json --split test    # held-out half only
    python offline_eval.py preds.json --strict        # missing conversations count as wrong
    python offline_eval.py preds.json --verbose
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional

UPSTREAM = os.environ.get('UPSTREAM', '/home/claude/Nordic-AI-Cup-2026')
MED = os.path.join(UPSTREAM, 'medical-appointment')
if not os.path.isdir(MED):
    MED = os.path.join('/workspace/upstream', 'medical-appointment')
sys.path.insert(0, MED)

import warnings  # noqa: E402
warnings.filterwarnings('ignore')
from local_evaluator import UNANSWERED, Statistics  # noqa: E402
from utils import evidence_interval, gold_evidence, group_questions_by_conversation  # noqa: E402


def split_ids(which: str) -> Optional[set]:
    """Deterministic 50/50 split of the 39 conversations (by transcript id)."""
    if which == 'all':
        return None
    ids = sorted({fn for fn, _ in group_questions_by_conversation()})
    random.Random(2026).shuffle(ids)
    dev = set(ids[: len(ids) // 2])
    return dev if which == 'dev' else set(ids) - dev


def score(preds: Dict[str, dict], split: str = 'all', strict: bool = False,
          verbose: bool = False) -> Statistics:
    stats = Statistics()
    keep = split_ids(split)
    for fn, rows in group_questions_by_conversation():
        if keep is not None and fn not in keep:
            continue
        tid = rows[0]['transcript_id']
        p = preds.get(fn) or preds.get(tid) or preds.get(fn.replace('.mp3', ''))
        n = len(rows)
        if p is None and not strict:
            continue
        answers: List[int] = [UNANSWERED] * n
        spans: List = [None] * n
        failed = True
        if p is not None:
            try:
                a, s, e = p['answers'], p['evidence_start'], p['evidence_end']
                if len(a) == len(s) == len(e) == n:
                    answers = [int(bool(x)) for x in a]
                    spans = [evidence_interval(x, y) for x, y in zip(s, e)]
                    failed = False
            except Exception:
                pass
        stats.record_request(n, p.get('latency_ms') if isinstance(p, dict) else None, failed)
        for row, pred, sp in zip(rows, answers, spans):
            label = int(row['label'])
            iou = stats.record(row['question_type'], label, pred, gold_evidence(row), sp)
            if verbose:
                mark = 'ok  ' if pred == label else 'WRONG'
                ev = f' tIoU {iou:.3f}' if label == 1 else ''
                print(f"  {mark} {row['question_id']:<24} {row['question_type']:<14} "
                      f"said {({1: 'yes', 0: 'no'}).get(pred, '-'):<3} wanted {row['answer']:<3}{ev}  {row['question']}")
    return stats


def summary(stats: Statistics) -> dict:
    return {'score': round(stats.final_score, 4), 'accuracy': round(stats.accuracy, 4),
            'tiou': round(stats.mean_tiou, 4), 'tiou_yes': round(stats.mean_tiou_answered_yes, 4),
            'n': stats.total, **{k: round(v[0] / v[1], 3) for k, v in stats.by_type.items() if v[1]}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('preds')
    ap.add_argument('--split', default='all', choices=['all', 'dev', 'test'])
    ap.add_argument('--strict', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--json', action='store_true', help='print one-line JSON summary only')
    a = ap.parse_args()
    preds = json.load(open(a.preds))
    st = score(preds, a.split, a.strict, a.verbose)
    if a.json:
        print(json.dumps(summary(st)))
    else:
        print(st.report())
        n_yes = sum(1 for p in preds.values() if isinstance(p, dict) for x in p.get('answers', []) if x)
        n_all = sum(len(p.get('answers', [])) for p in preds.values() if isinstance(p, dict))
        print(f'\nyes-rate in predictions: {n_yes}/{n_all}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
