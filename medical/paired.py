"""Paired per-conversation comparison of two prediction files with the frozen scorer
(offline_eval.score on one conversation at a time), per split: mean diff +- se.

    python paired.py base.json cand.json
"""
import json
import math
import sys

import offline_eval as O


def conv_scores(P, split):
    ids = O.split_ids(split)
    out = {}
    for fn, v in P.items():
        tid = fn.replace('conversation_', '').replace('.mp3', '')
        if ids is not None and tid not in ids and fn not in ids:
            continue
        out[fn] = O.summary(O.score({fn: v}, 'all'))['score']
    return out


def main():
    a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
    for split in ('dev', 'test', 'all'):
        sa, sb = conv_scores(a, split), conv_scores(b, split)
        keys = sorted(set(sa) & set(sb))
        d = [sb[k] - sa[k] for k in keys]
        m = sum(d) / len(d)
        se = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d)) if len(d) > 1 else float('nan')
        fa, fb = O.summary(O.score(a, split))['score'], O.summary(O.score(b, split))['score']
        print(f'{split:4s} n={len(keys):2d}  base {fa:.4f}  cand {fb:.4f}  paired diff {m:+.4f} +- {se:.4f} '
              f'({m / se if se else 0:.1f} se)')


if __name__ == '__main__':
    main()
