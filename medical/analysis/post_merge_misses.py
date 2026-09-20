"""Anatomy of the location misses that remain AFTER the onset + gate-any merge (served spans of the deploy config,
deploy_check.py output, 6 fresh runs). Per miss instance (answered yes, span does not overlap gold):
  sibling  our span overlaps the gold span of ANOTHER gold-yes question in the same conversation (borrowed evidence)
  cited    not sibling, but a unit the LLM cited overlaps the gold span (the right line was cited, the quote went elsewhere)
  elsewhere neither
Also: gold before/after our span, and per unique question how many runs miss it. Prints the always-missed questions.
    python3 post_merge_misses.py <dc_deploy.json>
"""
import collections
import glob
import json
import os
import sys

import common as C
import fresh_stack as FS


def ov(a, b):
    return a is not None and b is not None and min(a[1], b[1]) - max(a[0], b[0]) > 0


def text_in(W, s, e):
    return ' '.join(w['w'].strip() for w in W if w['e'] > s + 0.04 and w['s'] < e - 0.04)


def main():
    served = json.load(open(sys.argv[1]))
    runs = [rp for rp in sorted(glob.glob(os.path.join(C.WORK, 'fr5', 'fr5_rep*', 'arm*.detail.json')))
            if not rp.endswith('arm2.detail.json')]
    cv = C.convs()
    cds = FS.conv_data(sorted(cv), 'sentence')
    kinds = collections.Counter()
    where = collections.Counter()
    per_q = collections.defaultdict(list)
    for rp in runs:
        det = json.load(open(rp))
        tag = '/'.join(rp.split('/')[-2:])
        for fn, rows in cv.items():
            units = cds[fn]['units']
            golds = [(r['question_id'], C.gold(r)) for r in rows if int(r['label']) == 1 and C.gold(r)]
            for r, o, d in zip(rows, served[tag][fn], det[fn]):
                g = C.gold(r)
                if int(r['label']) != 1 or g is None or not o['answer'] or not o['span']:
                    continue
                sp = tuple(o['span'])
                if ov(g, sp):
                    per_q[r['question_id']].append(0)
                    continue
                per_q[r['question_id']].append(1)
                sib = any(ov(sp, g2) for q2, g2 in golds if q2 != r['question_id'])
                cited = any(ov(g, (units[i]['start'], units[i]['end'])) for i in (d['ids'] or []) if 0 <= i < len(units))
                k = 'sibling' if sib else ('cited' if cited else 'elsewhere')
                kinds[k] += 1
                where['gold_before' if g[1] <= sp[0] else 'gold_after'] += 1
    n = sum(kinds.values())
    print(f'miss instances over {len(runs)} runs: {n} ({n / len(runs):.1f} per run)')
    print('  kind:', {k: f'{v} ({100 * v / n:.0f}%)' for k, v in kinds.most_common()})
    print('  gold position:', dict(where))
    always = sorted(q for q, v in per_q.items() if len(v) == len(runs) and all(v))
    print(f'  always missed ({len(always)}):', ', '.join(q.replace('sample_', 's') for q in always))
    qtext = {r['question_id']: (fn, r) for fn, rows in cv.items() for r in rows}
    tag0 = '/'.join(runs[0].split('/')[-2:])
    for q in always:
        fn, r = qtext[q]
        W = C.words(C.tx(fn))
        k = [x['question_id'] for x in cv[fn]].index(q)
        sp = served[tag0][fn][k]['span']
        g = C.gold(r)
        print(f'\n{q}: {r["question"]}\n  OURS {sp[0]:.1f}-{sp[1]:.1f}: {text_in(W, sp[0] - 0.2, sp[1])[:160]!r}'
              f'\n  GOLD {g[0]:.1f}-{g[1]:.1f}: {text_in(W, *g)[:160]!r}')


if __name__ == '__main__':
    main()
