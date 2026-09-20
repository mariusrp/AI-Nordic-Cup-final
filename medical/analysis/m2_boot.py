#!/usr/bin/env python3
"""m2_boot.py (n3 M2/M1 agent, Sat 19 Sep): conversation-level paired bootstrap of every arm vs a reference arm
(default g015 = production) from the report.rep*.json files of m2_arms_local.py. Local only (no portal runs
exist for the new arms). Score of a resample = the frozen scorer's formula on the pooled rows
(0.4 * sum(correct)/sum(n) + 0.6 * sum(tiou)/sum(n_yes)); per-conversation stats are averaged over reps first.
--exclude drops conversations (e.g. sample_23, whose rows were used for the pass-2 hard examples).

Usage: python3 m2_boot.py --reports out/report.rep0.json out/report.rep1.json --ref g015 [--exclude conversation_sample_23.mp3]
"""
from __future__ import annotations

import argparse
import json
import random
from typing import Dict, List


def load(reports: List[str], exclude: set):
    reps = [json.load(open(p)) for p in reports]
    arms = list(reps[0]['arms'])
    names = [n for n in reps[0]['names'] if n not in exclude]
    for r in reps:
        assert r['names'] == reps[0]['names'] and list(r['arms']) == arms, 'reports differ in conversations/arms'
    stats: Dict[str, Dict[str, dict]] = {}
    for arm in arms:
        stats[arm] = {}
        for fn in names:
            rows = [r['arms'][arm]['per_conv'][fn] for r in reps]
            stats[arm][fn] = {'n': rows[0]['n'], 'n_yes': rows[0]['n_yes'],
                              'correct': sum(x['correct'] for x in rows) / len(rows),
                              'sum_tiou': sum(x['sum_tiou'] for x in rows) / len(rows),
                              'split': rows[0]['split'], 'scores': [x['score'] for x in rows]}
    return reps, arms, names, stats


def score_set(st, fns):
    n = sum(st[f]['n'] for f in fns)
    c = sum(st[f]['correct'] for f in fns)
    ny = sum(st[f]['n_yes'] for f in fns)
    t = sum(st[f]['sum_tiou'] for f in fns)
    return 0.4 * (c / n if n else 0.0) + 0.6 * (t / ny if ny else 0.0)


def ci(xs, lo=0.025, hi=0.975):
    s = sorted(xs)
    return s[int(lo * (len(s) - 1))], s[int(hi * (len(s) - 1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reports', nargs='+', required=True)
    ap.add_argument('--ref', default='g015')
    ap.add_argument('--exclude', nargs='*', default=[])
    ap.add_argument('--boot', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    reps, arms, names, stats = load(a.reports, set(a.exclude))
    n_reps = len(reps)
    dev = [f for f in names if stats[arms[0]][f]['split'] == 'dev']
    test = [f for f in names if stats[arms[0]][f]['split'] == 'test']
    print(f'{len(names)} conversations (dev {len(dev)} / test {len(test)}), {n_reps} rep(s), arms {arms}, '
          f'ref {a.ref}, excluded {a.exclude}')
    print(f'{"arm":<7} {"local":>7} {"acc":>6} {"tiou":>6} {"dev":>7} {"test":>7} {"per-rep (31)":<16} {"lat mean/p95/max":>18}')
    out = {'arms': {}, 'boot': {}, 'n_conv': len(names), 'excluded': a.exclude, 'ref': a.ref}
    for arm in arms:
        st = stats[arm]
        n = sum(st[f]['n'] for f in names)
        acc = sum(st[f]['correct'] for f in names) / n
        tiou = sum(st[f]['sum_tiou'] for f in names) / sum(st[f]['n_yes'] for f in names)
        rep_scores = [r['arms'][arm]['all']['score'] for r in reps]
        lat = [r['arms'][arm]['latency_s'] for r in reps]
        lm = max(l['mean'] for l in lat); lp = max(l.get('p95', l['max']) for l in lat); lx = max(l['max'] for l in lat)
        out['arms'][arm] = {'local': score_set(st, names), 'acc': acc, 'tiou': tiou, 'dev': score_set(st, dev),
                            'test': score_set(st, test), 'per_rep': rep_scores, 'lat_mean': lm, 'lat_p95': lp, 'lat_max': lx}
        print(f'{arm:<7} {score_set(st, names):7.4f} {acc:6.4f} {tiou:6.4f} {score_set(st, dev):7.4f} {score_set(st, test):7.4f} '
              f'{"/".join(f"{x:.4f}" for x in rep_scores):<16} {lm:5.1f}/{lp:4.1f}/{lx:4.1f}')
    rng = random.Random(a.seed)
    idx = list(range(len(names)))
    draws = {arm: [] for arm in arms}
    for _ in range(a.boot):
        samp = [names[rng.choice(idx)] for _ in idx]
        for arm in arms:
            draws[arm].append(score_set(stats[arm], samp))
    print()
    print(f'paired bootstrap ({a.boot} resamples of {len(names)} conversations; diff = arm - {a.ref}; '
          f'pc = per-conversation paired mean +- se over conversations, mean of reps)')
    print(f'{"arm":<7} {"mean":>8} {"CI95":>20} {"P(arm>ref)":>10} {"pc mean+-se":>18} {"w/l/t":>9} {"per-rep diff":<16}')
    for arm in arms:
        d = [x - y for x, y in zip(draws[arm], draws[a.ref])]
        m = sum(d) / len(d)
        lo, hi = ci(d)
        p = sum(1 for x in d if x > 0) / len(d)
        pc = [sum(stats[arm][f]['scores']) / n_reps - sum(stats[a.ref][f]['scores']) / n_reps for f in names]
        pm = sum(pc) / len(pc)
        se = (sum((x - pm) ** 2 for x in pc) / (len(pc) - 1) / len(pc)) ** 0.5
        w = sum(1 for x in pc if x > 1e-9); l = sum(1 for x in pc if x < -1e-9)
        rd = [r['arms'][arm]['all']['score'] - r['arms'][a.ref]['all']['score'] for r in reps]
        out['boot'][arm] = {'mean': m, 'ci95': [lo, hi], 'p_better': p, 'pc_mean': pm, 'pc_se': se, 'wins': w,
                            'losses': l, 'ties': len(pc) - w - l, 'per_rep_diff': rd}
        print(f'{arm:<7} {m:+8.4f} [{lo:+.4f},{hi:+.4f}] {p:10.3f} {pm:+9.4f}+-{se:.4f} {w:>3}/{l:<3}/{len(pc) - w - l:<2} '
              f'{"/".join(f"{x:+.4f}" for x in rd):<16}')
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)
        print('wrote', a.out)


if __name__ == '__main__':
    main()
