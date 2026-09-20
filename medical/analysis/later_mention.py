"""Which mention does the annotator pick when a fact is stated more than once? (cycle 2)

1. Consistency of misses: per gold-yes question, in how many of the 4 cached production runs is the span a miss
   (no overlap)? Consistent misses cannot be fixed by sampling/agreement.
2. Direction-restricted lexical switch, replayed on the cycle-1 base spans (onset + sentstart4):
   score every window of 1-2 sentence units by IDF coverage of the question's content tokens (lexical_location.cov_fn).
   Switch to the best window that lies entirely AFTER our span (or BEFORE, for the control) when
   cov(window) >= cov(ours) + m. m < 0 means 'later even if slightly weaker'.
    python3 later_mention.py
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
from lexical_location import cov_fn, windows  # noqa: E402


def main():
    cv = C.convs()
    dev = C.split_dev()
    dets = [json.load(open(p)) for p in X.RUNS if os.path.exists(p)]
    ctxs = {fn: X.ctx_for(fn) for fn in cv}
    units = {fn: S.build_units(C.tx(fn), 'sentence') for fn in cv}
    covs = {fn: cov_fn(units[fn]) for fn in cv}
    configs = [('base', None, 0)]
    for direction in ('after', 'before'):
        for m in (-0.1, 0.0, 0.1, 0.2, 0.3):
            configs.append((f'{direction} m{m:+.1f}', direction, m))
    per = {c[0]: collections.defaultdict(list) for c in configs}
    nsw = collections.Counter()
    good = collections.Counter()
    for det in dets:
        for fn, rows in cv.items():
            U, cov = units[fn], covs[fn]
            for r, o in zip(rows, det[fn]):
                g = C.gold(r)
                if int(r['label']) != 1 or g is None:
                    continue
                base = X.rule_span(ctxs[fn], *o['span'], {}) if (o['answer'] and o['span']) else None
                for name, direction, m in configs:
                    sp = base
                    if base is not None and direction:
                        ui = [k for k, u in enumerate(U) if u['end'] > base[0] - 0.15 and u['start'] < base[1] - 0.05]
                        if ui:
                            ours = cov(r['question'], min(ui), max(ui))
                            cands = [(cov(r['question'], i, j), i, j) for i, j in windows(U)
                                     if (i > max(ui) if direction == 'after' else j < min(ui))]
                            if cands:
                                c, i, j = max(cands)
                                if c >= ours + m and c > 0:
                                    on = ctxs[fn]['on']
                                    st = U[i]['start']
                                    oc = on[(on >= st - 0.1) & (on <= U[i]['words'][0]['e'] - 0.05)]
                                    sp = (float(oc[0]) if len(oc) else st, U[j]['end'])
                                    nsw[name] += 1
                                    good[name] += int(C.tiou(g, sp) > C.tiou(g, base))
                    per[name][fn].append(C.tiou(g, sp))
    for name, _, _ in configs:
        line = f'{name:12s} switches {nsw[name] / len(dets):5.1f}/run (better {good[name] / len(dets):4.1f})'
        for half in ('dev', 'test', 'all'):
            fns = [fn for fn in cv if half == 'all' or ((fn in dev) == (half == 'dev'))]
            n = sum(len(per[name][f]) for f in fns)
            d = np.array([sum(per[name][f]) - sum(per['base'][f]) for f in fns])
            line += f' | {half} {sum(sum(per[name][f]) for f in fns) / n:.4f} d {d.sum() / n:+.4f}+-{d.std(ddof=1) * math.sqrt(len(fns)) / n:.4f}'
        print(line)


if __name__ == '__main__':
    main()
