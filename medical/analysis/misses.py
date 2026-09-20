"""Production 'miss' cases (answered yes, span does not overlap gold): print question, our quote, and the gold text
(turbo words inside the gold span), plus whether gold is before/after our span, to see how annotators pick WHICH mention.
    python3 misses.py [rep]
"""
import collections
import json
import sys

import common as C


def text_in(W, s, e):
    return ' '.join(w['w'].strip() for w in W if w['e'] > s + 0.04 and w['s'] < e - 0.04)


def main():
    rep = sys.argv[1] if len(sys.argv) > 1 else 'a1_r1'
    det = json.load(open(f'{C.WORK}/med-evolve-g3-1A/out/{rep}/arm0.detail.json'))
    cnt = collections.Counter()
    for fn, rows in C.convs().items():
        W = C.words(C.tx(fn))
        for r, o in zip(rows, det[fn]):
            g = C.gold(r)
            if int(r['label']) != 1 or g is None or not o['answer'] or not o['span']:
                continue
            ps, pe = o['span']
            if min(g[1], pe) - max(g[0], ps) > 0:
                continue
            where = 'gold_before' if g[1] <= ps else 'gold_after'
            cnt[where] += 1
            print(f"\n[{where} {abs((g[0] if where == 'gold_after' else g[1]) - (pe if where == 'gold_after' else ps)):.1f}s] {r['question_id']}: {r['question']}\n"
                  f"  OURS {ps:.1f}-{pe:.1f}: {text_in(W, ps - 0.2, pe)!r}\n  GOLD {g[0]:.1f}-{g[1]:.1f}: {text_in(W, *g)!r}")
    print(cnt)


if __name__ == '__main__':
    main()
