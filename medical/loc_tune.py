"""Tune the CPU no-LLM fallback (lexical top-K prefilter + bge cross-encoder) on the dev split.

Input: full-candidate scores from `locator.py score` (every phrase run x question) on the
production transcripts. For each K the prefilter keeps K runs per question; span knobs
(alpha, target, shift_s, shift_e) are tuned with gold answers (tIoU on dev), then the yes
threshold on dev score. Test is reported, never used for selection.

    UPSTREAM=... python loc_tune.py --scores scores.json --tx-dir TX [--heur heur.json]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import locator as L  # noqa: E402
import spans as S  # noqa: E402
from offline_eval import score as oscore, summary  # noqa: E402


def main():
    up = os.environ.get('UPSTREAM', '/home/claude/Nordic-AI-Cup-2026')
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', required=True)
    ap.add_argument('--tx-dir', required=True)
    ap.add_argument('--csv', default=os.path.join(up, 'medical-appointment/data/question_train.csv'))
    ap.add_argument('--ks', default='4,6,8,12,16,24,0')
    ap.add_argument('--out')
    a = ap.parse_args()
    scores = json.load(open(a.scores))
    by = L._rows(a.csv)
    units_by = {tid: S.build_units(S.load_transcript(os.path.join(a.tx_dir, f'conversation_{tid}.json')), 'phrase')
                for tid in scores}

    def build(K, thr, alpha, target, ss, se, gold_answers=False):
        P = {}
        for tid, d in scores.items():
            us, runs = units_by[tid], [tuple(x) for x in d['runs']]
            ans, st, en = [], [], []
            for r in by[tid]:
                q = r['question']
                idx = L.lexical_topk(us, runs, q, K) if K else list(range(len(runs)))
                sub = [runs[k] for k in idx]
                sc = [d['qs'][r['question_id']][k] for k in idx]
                yes, sp, _ = L.fallback_decide(us, sub, sc, thr, alpha, target, ss, se)
                if gold_answers:
                    yes = bool(int(r['label']))
                ans.append(bool(yes))
                st.append(sp[0] if yes and sp else None)
                en.append(sp[1] if yes and sp else None)
            P[f'conversation_{tid}.mp3'] = {'answers': ans, 'evidence_start': st, 'evidence_end': en}
        return P

    ev = {sp: (lambda P, sp=sp: summary(oscore(P, sp))) for sp in ('dev', 'test', 'all')}
    res = {}
    for K in [int(x) for x in a.ks.split(',')]:
        # stage 1: span knobs with gold answers (dev tIoU)
        g1 = []
        for alpha, target in itertools.product((0.0, 1.0, 2.0, 3.0, 4.0, 6.0), (1.5, 2.0, 3.0, 4.0)):
            P = build(K, 0, alpha, target, 0.0, 0.0, True)
            g1.append((ev['dev'](P)['tiou'], alpha, target))
        _, alpha, target = max(g1)
        g2 = []
        for ss, se in itertools.product((0.0, 0.1, 0.2, 0.3, 0.4), (-0.1, 0.0, 0.1, 0.2)):
            P = build(K, 0, alpha, target, ss, se, True)
            g2.append((ev['dev'](P)['tiou'], ss, se))
        _, ss, se = max(g2)
        # stage 2: yes threshold on dev score
        g3 = []
        for thr in [x / 2 for x in range(-14, 7)]:
            P = build(K, thr, alpha, target, ss, se)
            g3.append((ev['dev'](P)['score'], thr))
        _, thr = max(g3)
        P = build(K, thr, alpha, target, ss, se)
        r = {sp: ev[sp](P) for sp in ('dev', 'test', 'all')}
        res[K] = {'params': {'k': K, 'thr': thr, 'alpha': alpha, 'target': target, 'shift_s': ss, 'shift_e': se},
                  **r}
        print(f'K={K:3d} thr {thr:+.1f} alpha {alpha} target {target} shift {ss:+.1f}/{se:+.1f}  '
              + '  '.join(f'{sp} {v["score"]:.4f} (acc {v["accuracy"]:.3f} tIoU {v["tiou"]:.3f})'
                          for sp, v in r.items()), flush=True)
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
