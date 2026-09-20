"""Where do we lose points AFTER the onset + quote-gate 'any' merge? Loss buckets (loss_breakdown.breakdown) on the
SERVED spans of the real pipeline code (deploy_check.py outputs) for the 6 fresh production-prompt runs, production
config vs deploy config, plus the per-question view: which gold-yes questions miss (no overlap) in every run.
    python3 post_merge_breakdown.py <dc_prod.json> <dc_deploy.json>
"""
import collections
import json
import sys

import common as C
import loss_breakdown as LB

NEG = ('free of', 'absent', 'no sign', 'without', 'not ', "n't", 'no ', 'normal', 'clear')


def main():
    cv = C.convs()
    for path in sys.argv[1:]:
        sv = json.load(open(path))
        tot = collections.Counter()
        N = 0
        miss_by_q = collections.Counter()
        tiou_by_q = collections.defaultdict(list)
        for tag in sorted(sv):
            L, n = LB.breakdown(sv[tag], cv)
            tot.update(L)
            N += n
            for fn, rows in cv.items():
                for r, o in zip(rows, sv[tag][fn]):
                    g = C.gold(r)
                    if int(r['label']) != 1 or g is None:
                        continue
                    t = C.tiou(g, o['span'] if o['answer'] else None)
                    tiou_by_q[r['question_id']].append(t)
                    miss_by_q[r['question_id']] += int(o['answer'] and o['span'] is not None and t == 0)
        loss = sum(tot.values()) / N
        runs = len(sv)
        print(f'{path.split("/")[-1]}: {runs} runs, mean tIoU {1 - loss:.4f}; tIoU loss {loss:.4f} = score {0.6 * loss:.4f}')
        for k, v in tot.most_common():
            print(f'  {k:18s} tIoU loss {v / N:.4f}  score {0.6 * v / N:.4f}  ({100 * v / sum(tot.values()):.0f}%)')
        always = [q for q, m in miss_by_q.items() if m == runs]
        some = [q for q, m in miss_by_q.items() if 0 < m < runs]
        lost_always = sum(1 - sum(tiou_by_q[q]) / runs for q in always) / len(tiou_by_q)
        print(f'  gold-yes questions: {len(tiou_by_q)}; missed in all {runs} runs: {len(always)} '
              f'(score {0.6 * lost_always:.4f}); missed in some runs: {len(some)}')
        # per-question tIoU distribution (mean over runs)
        mq = {q: sum(v) / len(v) for q, v in tiou_by_q.items()}
        bins = collections.Counter('<0.3' if t < 0.3 else ('0.3-0.7' if t < 0.7 else '>=0.7') for t in mq.values())
        print('  per-question mean tIoU bins:', dict(bins))
        qtext = {r['question_id']: r['question'] for rows in cv.values() for r in rows}
        neg = [q for q in mq if any(w in qtext[q].lower() for w in NEG[:4])]
        print(f'  negation-worded ("free of/absent/no sign/without"): n={len(neg)} mean tIoU '
              f'{sum(mq[q] for q in neg) / max(1, len(neg)):.3f} vs rest '
              f'{sum(mq[q] for q in mq if q not in neg) / max(1, len(mq) - len(neg)):.3f}')


if __name__ == '__main__':
    main()
