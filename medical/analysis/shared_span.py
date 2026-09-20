"""Cross-question consistency: when two questions of one conversation get OVERLAPPING predicted spans, is one wrong?

Gold: two gold-yes spans of the same conversation overlap in 6.2% of pairs, and then they are nearly the same span
(19 of 21 pairs: the annotators reused one passage for two questions). Production: 17 overlapping yes-span pairs per run.
For each overlapping predicted pair (both answered yes; gold labels NOT used to find pairs) this script reports whether
the gold spans also overlap, whether one prediction is a miss (tIoU 0), and whether the question with the LOWER
IDF-weighted content-token coverage of the shared span (lexical_location.cov_fn over sentence units) is the missed one.
Then a replay: move the lower-coverage question to its best lexical 1-2 sentence window that does not touch the shared
span, only when its coverage there beats the shared span (zero fitted parameters). Paired per-question tIoU vs
production, se over conversations, 7 cached production runs.
    python3 shared_span.py
"""
import glob
import itertools
import json
import sys

import numpy as np

import common as C
import lexical_location as L

sys.path.insert(0, '/home/claude/nac/medical')
import spans as S  # noqa: E402

W = C.WORK
RUNS = (sorted(glob.glob(f'{W}/fr5/fr5_rep*/arm0.detail.json'))
        + [f'{W}/med-evolve-g3-1A/out/{r}/arm0.detail.json' for r in ('a1_r1', 'a1_r2', 'a1_r3')]
        + [f'{W}/medical-fast-r4-1/out/r4_1/base.detail.json'])


def ov(a, b):
    return min(a[1], b[1]) - max(a[0], b[0]) > 0


def span_units(units, sp):
    ui = [k for k, u in enumerate(units) if u['end'] > sp[0] - 0.15 and u['start'] < sp[1] - 0.05]
    return (min(ui), max(ui)) if ui else None


def main():
    cv = C.convs()
    dev = C.split_dev()
    ctx = {}
    for fn in cv:
        units = S.build_units(C.tx(fn), 'sentence')
        ctx[fn] = (units, L.cov_fn(units))
    stats = {'pairs': 0, 'gold_ov': 0, 'has_miss': 0, 'suspect_is_miss': 0, 'suspect_is_miss_or_worse': 0}
    per_b, per_c = {fn: [] for fn in cv}, {fn: [] for fn in cv}
    nsw = 0
    for p in RUNS:
        D = json.load(open(p))
        for fn, rs in cv.items():
            units, cov = ctx[fn]
            yes = [qi for qi, o in enumerate(D[fn]) if o['answer'] and o['span']]
            new = {qi: tuple(D[fn][qi]['span']) for qi in yes}
            for a, b in itertools.combinations(yes, 2):
                sa, sb = D[fn][a]['span'], D[fn][b]['span']
                if not ov(sa, sb):
                    continue
                ua, ub = span_units(units, sa), span_units(units, sb)
                ca = cov(rs[a]['question'], *ua) if ua else 0.0
                cb = cov(rs[b]['question'], *ub) if ub else 0.0
                sus, keep = (a, b) if ca < cb else (b, a)
                ga, gb = C.gold(rs[a]), C.gold(rs[b])
                if ga and gb:  # evaluation only
                    stats['pairs'] += 1
                    stats['gold_ov'] += ov(ga, gb)
                    ta, tb = C.tiou(ga, sa), C.tiou(gb, sb)
                    t = {a: ta, b: tb}
                    if min(ta, tb) == 0:
                        stats['has_miss'] += 1
                        stats['suspect_is_miss'] += t[sus] == 0
                    stats['suspect_is_miss_or_worse'] += t[sus] <= t[keep]
                # replay: move the suspect to its best lexical window away from the shared span
                shared = span_units(units, D[fn][sus]['span'])
                cands = [w for w in L.windows(units) if shared is None or w[1] < shared[0] or w[0] > shared[1]]
                if not cands:
                    continue
                i, j = max(cands, key=lambda w: cov(rs[sus]['question'], *w))
                cur = cov(rs[sus]['question'], *shared) if shared else 0.0
                if cov(rs[sus]['question'], i, j) > cur:
                    new[sus] = (units[i]['start'] + 0.2, units[j]['end'])
                    nsw += 1
            for qi, r in enumerate(rs):
                g = C.gold(r)
                if int(r['label']) != 1 or g is None:
                    continue
                o = D[fn][qi]
                per_b[fn].append(C.tiou(g, o['span'] if o['answer'] and o['span'] else None))
                per_c[fn].append(C.tiou(g, new.get(qi)))
    s = stats
    print(f"overlapping predicted pairs with both gold-yes: {s['pairs']} ({s['pairs'] / len(RUNS):.1f}/run); gold overlaps too "
          f"{s['gold_ov'] / s['pairs']:.0%}; one is a miss {s['has_miss'] / s['pairs']:.0%}; lower-coverage question is the miss "
          f"{s['suspect_is_miss']}/{s['has_miss']}; lower-coverage question has the lower tIoU {s['suspect_is_miss_or_worse']}/{s['pairs']}")
    line = f'replay (move suspect to best lexical window elsewhere): switches {nsw / len(RUNS):.1f}/run'
    for half in ('dev', 'test', 'all'):
        fns = [fn for fn in cv if half == 'all' or ((fn in dev) == (half == 'dev'))]
        d = np.array([sum(per_c[f]) - sum(per_b[f]) for f in fns])
        k = sum(len(per_b[f]) for f in fns)
        line += f' | {half} d {d.sum() / k:+.4f}+-{d.std(ddof=1) * np.sqrt(len(fns)) / k:.4f}'
    print(line)


if __name__ == '__main__':
    main()
