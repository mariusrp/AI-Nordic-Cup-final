"""Coverage-driven extent rules (cycle 2). Hypothesis: the gold passage covers EVERY content word/fact of the
question, so when our span misses question tokens that the adjacent sentence contains, the gold extends there.

Base = cycle-1 recommendation (onset start + sentstart4, turbo end), replayed on 4 cached production runs.
Rules (applied once, to the immediately adjacent sentence only):
  nextcov(k): extend the end over the next sentence if it contains >= k question content tokens that our span lacks
  prevcov(k): extend the start back over the previous sentence, same test
  prevq_lex : extend back over the previous sentence only if it is a question ('?') AND contains >= 1 missing token
    python3 coverage_extend.py
"""
import collections
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, '/home/claude/nac/medical')
import spans as S  # noqa: E402

import common as C  # noqa: E402
import extent_units as X  # noqa: E402


def toks(W, a, b):
    return set(S.content_tokens(' '.join(W[k]['w'] for k in range(a, b + 1))))


def rule(ctx, ps, pe, q, opts):
    W, ss, sid = ctx['W'], ctx['ss'], ctx['sid']
    i = int(np.argmin(np.abs(ctx['S'] - (ps - 0.2))))
    j = max(i, int(np.argmin(np.abs(ctx['E'] - pe))))
    a = ss[sid[i]][0]
    if i - a <= 4:
        i = a
    qt = set(S.content_tokens(q))
    missing = qt - toks(W, i, j)
    fired = False
    if opts.get('next') and sid[j] + 1 < len(ss) and ss[sid[j]][1] == j:
        na, nb = ss[sid[j] + 1]
        if len(missing & toks(W, na, nb)) >= opts['next'] and W[na]['s'] - W[j]['e'] < opts.get('gap', 9):
            j, fired = nb, True
    if opts.get('prev') and sid[i] > 0 and ss[sid[i]][0] == i:
        pa, pb = ss[sid[i] - 1]
        isq = W[pb]['w'].strip().endswith('?')
        if len(missing & toks(W, pa, pb)) >= opts['prev'] and (isq or not opts.get('qonly')):
            i, fired = pa, True
    s = X.start_time(ctx, i)
    return (s, max(W[j]['e'], s + 0.3)), fired


def main():
    cv = C.convs()
    dev = C.split_dev()
    ctxs = {fn: X.ctx_for(fn) for fn in cv}
    dets = [json.load(open(p)) for p in X.RUNS if os.path.exists(p)]
    rules = {'base': {}, 'nextcov1': {'next': 1}, 'nextcov2': {'next': 2}, 'nextcov1 gap<1s': {'next': 1, 'gap': 1.0},
             'prevcov1': {'prev': 1}, 'prevcov2': {'prev': 2}, 'prevq_lex1': {'prev': 1, 'qonly': 1},
             'next1+prev1': {'next': 1, 'prev': 1}, 'next2+prev2': {'next': 2, 'prev': 2}}
    per = {k: collections.defaultdict(list) for k in rules}
    fires = collections.Counter()
    better = collections.Counter()
    for det in dets:
        for fn, rows in cv.items():
            for r, o in zip(rows, det[fn]):
                g = C.gold(r)
                if int(r['label']) != 1 or g is None:
                    continue
                b = None
                for name, opts in rules.items():
                    if not (o['answer'] and o['span']):
                        per[name][fn].append(0.0)
                        continue
                    sp, f = rule(ctxs[fn], *o['span'], r['question'], opts)
                    if name == 'base':
                        b = sp
                    fires[name] += f
                    if f:
                        better[name] += np.sign(C.tiou(g, sp) - C.tiou(g, b))
                    per[name][fn].append(C.tiou(g, sp))
    for name in rules:
        line = f'{name:16s} fires {fires[name] / len(dets):5.1f}/run net better {better[name] / len(dets):+5.1f}/run'
        for half in ('dev', 'test', 'all'):
            fns = [fn for fn in cv if half == 'all' or ((fn in dev) == (half == 'dev'))]
            n = sum(len(per[name][f]) for f in fns)
            d = np.array([sum(per[name][f]) - sum(per['base'][f]) for f in fns])
            line += f' | {half} {sum(sum(per[name][f]) for f in fns) / n:.4f} d {d.sum() / n:+.4f}+-{d.std(ddof=1) * math.sqrt(len(fns)) / n:.4f}'
        print(line)


if __name__ == '__main__':
    main()
