"""Do gold evidence passages share the question's distinctive words? (Questions look generated from the evidence
passage: 'Is the diabetes considered stable?' <- 'my assessment is that your diabetes is stable'.)

Score every window of 1-2 turbo sentence units by IDF-weighted coverage of the question's content tokens
(spans.content_tokens; IDF over the conversation's sentence units). Report how often the lexical argmax window
overlaps gold, and a replay switching rule on production preds: switch our span to the argmax window when
cov(argmax) - cov(ours) >= m. Paired tIoU vs production (3 cached reps), dev/test.
    python3 lexical_location.py
"""
import json
import math
import sys

import numpy as np

sys.path.insert(0, '/home/claude/nac/medical')
import spans as S  # noqa: E402

import common as C  # noqa: E402


def windows(units):
    out = []
    for i in range(len(units)):
        for j in (i, i + 1):
            if j < len(units):
                out.append((i, j))
    return out


def cov_fn(units):
    toks = [set(S.content_tokens(u['text'])) for u in units]
    df = {}
    for ts in toks:
        for t in ts:
            df[t] = df.get(t, 0) + 1
    n = len(units)

    def cov(q, i, j):
        qt = set(S.content_tokens(q))
        if not qt:
            return 0.0
        idf = {t: math.log((n + 1) / (df.get(t, 0) + 0.5)) for t in qt}
        ts = set().union(*toks[i:j + 1])
        return sum(idf[t] for t in qt if t in ts) / (sum(idf.values()) + 1e-9) - (0.05 if j > i else 0.0)
    return cov


def main():
    cv = C.convs()
    dev = C.split_dev()
    hit = n = 0
    ctx = {}
    for fn, rows in cv.items():
        units = S.build_units(C.tx(fn), 'sentence')
        cov = cov_fn(units)
        ctx[fn] = (units, cov)
        for r in rows:
            g = C.gold(r)
            if g is None:
                continue
            n += 1
            i, j = max(windows(units), key=lambda w: cov(r['question'], *w))
            hit += C.tiou(g, (units[i]['start'] + 0.2, units[j]['end'])) > 0
    print(f'lexical argmax window overlaps gold: {hit}/{n} = {hit / n:.2f}')
    reps = ('a1_r1', 'a1_r2', 'a1_r3')
    dets = {rep: json.load(open(f'{C.WORK}/med-evolve-g3-1A/out/{rep}/arm0.detail.json')) for rep in reps}
    for m in (None, 0.1, 0.2, 0.3, 0.4, 0.5):
        per_b, per_c = {}, {}
        nsw = 0
        for fn, rows in cv.items():
            units, cov = ctx[fn]
            vb, vc = [], []
            for rep in reps:
                for r, o in zip(rows, dets[rep][fn]):
                    g = C.gold(r)
                    if g is None:
                        continue
                    sp = tuple(o['span']) if (o['answer'] and o['span']) else None
                    vb.append(C.tiou(g, sp))
                    if sp is None or m is None:
                        vc.append(C.tiou(g, sp))
                        continue
                    # our span's units
                    ui = [k for k, u in enumerate(units) if u['end'] > sp[0] - 0.15 and u['start'] < sp[1] - 0.05]
                    ours = cov(r['question'], min(ui), max(ui)) if ui else 0.0
                    i, j = max(windows(units), key=lambda w: cov(r['question'], *w))
                    best = cov(r['question'], i, j)
                    if best - ours >= m and not (ui and i <= max(ui) and j >= min(ui)):
                        nsw += 1
                        vc.append(C.tiou(g, (units[i]['start'] + 0.2, units[j]['end'])))
                    else:
                        vc.append(C.tiou(g, sp))
            per_b[fn], per_c[fn] = vb, vc
        line = f'switch margin {m}: switches {nsw}/3 runs'
        for half in ('dev', 'test', 'all'):
            fns = [fn for fn in cv if half == 'all' or ((fn in dev) == (half == 'dev'))]
            d = np.array([sum(per_c[f]) - sum(per_b[f]) for f in fns])
            k = sum(len(per_b[f]) for f in fns)
            line += f' | {half} d {d.sum() / k:+.4f}+-{d.std(ddof=1) * math.sqrt(len(fns)) / k:.4f}'
        print(line)


if __name__ == '__main__':
    main()
