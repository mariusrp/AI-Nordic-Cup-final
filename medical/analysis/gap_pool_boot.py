#!/usr/bin/env python3
"""gap_pool_boot.py (n3 medgap agent): pool the local 31-conversation arm scores (report.rep*.json from
gap_arms_local.py) with the portal runs, question-weighted, and run a conversation-level paired bootstrap
of every arm's difference vs gap 0.5 and vs sentence units.

Local score of a set of conversations = the frozen scorer's formula on the pooled rows:
    0.4 * sum(correct)/sum(n) + 0.6 * sum(tiou over gold-yes rows)/sum(n_yes)
so a bootstrap resample is scored exactly as offline_eval would score that resampled set. With several
reps the per-conversation sufficient statistics are averaged over reps first (paired_run.py convention).

Pooled estimate per arm = question-weighted mean of the local score (31 conv x 10 q = 310 q) and each
portal run (19 conv x 10 q = 190 q each).

Usage: python3 gap_pool_boot.py --reports out/report.rep0.json out/report.rep1.json --out-dir . [--boot 4000]
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List

PORTAL = {   # 19-conversation portal validations (medical/LEDGER.tsv + medical/out/n3_island_arms.tsv)
    'sentence': [0.7618, 0.7596, 0.7562, 0.7575],   # AEQ sentence units (production before the switch)
    'g015': [0.7805, 0.7720],
    'g030': [0.7752, 0.7731],
    'g040': [0.7840],
    'g050': [0.7858, 0.7858],
    'g070': [],                                     # 0.7311 CONTAMINATED (mid-run restart): excluded
}
PORTAL_Q = 190   # questions per portal run
LOCAL_Q = 310    # questions in the local set (31 x 10)


def load(reports: List[str]):
    reps = [json.load(open(p)) for p in reports]
    arms = list(reps[0]['arms'])
    names = reps[0]['names']
    for r in reps:
        assert r['names'] == names and list(r['arms']) == arms, 'reports differ in conversations/arms'
    # per-conv sufficient stats averaged over reps
    stats: Dict[str, Dict[str, dict]] = {}
    for arm in arms:
        stats[arm] = {}
        for fn in names:
            rows = [r['arms'][arm]['per_conv'][fn] for r in reps]
            stats[arm][fn] = {'n': rows[0]['n'], 'n_yes': rows[0]['n_yes'],
                              'correct': sum(x['correct'] for x in rows) / len(rows),
                              'sum_tiou': sum(x['sum_tiou'] for x in rows) / len(rows),
                              'split': rows[0]['split'],
                              'scores': [x['score'] for x in rows]}
    return reps, arms, names, stats


def score_set(st: Dict[str, dict], fns: List[str]) -> float:
    n = sum(st[f]['n'] for f in fns)
    c = sum(st[f]['correct'] for f in fns)
    ny = sum(st[f]['n_yes'] for f in fns)
    t = sum(st[f]['sum_tiou'] for f in fns)
    return 0.4 * (c / n if n else 0.0) + 0.6 * (t / ny if ny else 0.0)


def acc_tiou(st, fns):
    n = sum(st[f]['n'] for f in fns)
    c = sum(st[f]['correct'] for f in fns)
    ny = sum(st[f]['n_yes'] for f in fns)
    t = sum(st[f]['sum_tiou'] for f in fns)
    return c / n, t / ny


def ci(xs: List[float], lo=0.025, hi=0.975):
    s = sorted(xs)
    return s[int(lo * (len(s) - 1))], s[int(hi * (len(s) - 1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reports', nargs='+', required=True)
    ap.add_argument('--out-dir', default='.')
    ap.add_argument('--boot', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=2026)
    a = ap.parse_args()
    reps, arms, names, stats = load(a.reports)
    n_reps = len(reps)
    dev = [f for f in names if stats[arms[0]][f]['split'] == 'dev']
    test = [f for f in names if stats[arms[0]][f]['split'] == 'test']

    out = {'n_reps': n_reps, 'reports': a.reports, 'n_conv': len(names), 'boot': a.boot, 'portal': PORTAL,
           'arms': {}}
    print(f'{len(names)} conversations (dev {len(dev)} / test {len(test)}), {n_reps} rep(s), arms {arms}')
    print(f'{"arm":<9} {"local":>7} {"acc":>6} {"tiou":>6} {"dev":>7} {"test":>7} {"per-rep":<18} '
          f'{"portal runs":<32} {"portal mean":>11} {"pooled":>7} {"n_q":>5}')
    for arm in arms:
        st = stats[arm]
        loc = score_set(st, names)
        acc, tiou = acc_tiou(st, names)
        rep_scores = [r['arms'][arm]['all']['score'] for r in reps]
        portal = PORTAL.get(arm, [])
        num = LOCAL_Q * loc + PORTAL_Q * sum(portal)
        den = LOCAL_Q + PORTAL_Q * len(portal)
        pooled = num / den
        pm = sum(portal) / len(portal) if portal else float('nan')
        out['arms'][arm] = {'local': loc, 'local_acc': acc, 'local_tiou': tiou,
                            'dev': score_set(st, dev), 'test': score_set(st, test),
                            'per_rep': rep_scores, 'portal': portal, 'portal_mean': pm,
                            'pooled': pooled, 'pooled_n_q': den,
                            'per_conv': {f: {'score': sum(st[f]['scores']) / n_reps, 'split': st[f]['split']}
                                         for f in names}}
        print(f'{arm:<9} {loc:7.4f} {acc:6.4f} {tiou:6.4f} {out["arms"][arm]["dev"]:7.4f} '
              f'{out["arms"][arm]["test"]:7.4f} {"/".join(f"{x:.4f}" for x in rep_scores):<18} '
              f'{"/".join(f"{x:.4f}" for x in portal) or "-":<32} {pm:11.4f} {pooled:7.4f} {den:5d}')

    # conversation-level paired bootstrap: the same resample of conversations scores every arm
    rng = random.Random(a.seed)
    idx = list(range(len(names)))
    draws = {arm: [] for arm in arms}
    draws_dev = {arm: [] for arm in arms}
    draws_test = {arm: [] for arm in arms}
    for _ in range(a.boot):
        samp = [names[rng.choice(idx)] for _ in idx]
        sd = [dev[rng.randrange(len(dev))] for _ in dev]
        stt = [test[rng.randrange(len(test))] for _ in test]
        for arm in arms:
            draws[arm].append(score_set(stats[arm], samp))
            draws_dev[arm].append(score_set(stats[arm], sd))
            draws_test[arm].append(score_set(stats[arm], stt))
    out['bootstrap'] = {}
    print()
    print(f'paired bootstrap ({a.boot} resamples of the {len(names)} conversations; '
          f'diff = arm - reference, 95% percentile CI; pc = per-conversation paired mean +- se)')
    print(f'{"arm":<9} {"vs g050 mean":>13} {"CI95":>20} {"P(arm>ref)":>10} {"pc mean+-se":>18} | {"vs sentence mean":>17} {"CI95":>20} '
          f'{"P(arm>ref)":>10} {"pc mean+-se":>18} | {"dev d(g050)":>12} {"test d(g050)":>12} {"w/l vs g050":>11}')
    for arm in arms:
        rec = {}
        for ref in ('g050', 'sentence'):
            d = [x - y for x, y in zip(draws[arm], draws[ref])]
            m = sum(d) / len(d)
            lo, hi = ci(d)
            p = sum(1 for x in d if x > 0) / len(d)
            # per-conversation paired diffs (mean of per-conv scores), for the wins count and a plain se
            pc = [sum(stats[arm][f]['scores']) / n_reps - sum(stats[ref][f]['scores']) / n_reps for f in names]
            pm_ = sum(pc) / len(pc)
            se = (sum((x - pm_) ** 2 for x in pc) / (len(pc) - 1) / len(pc)) ** 0.5
            wins = sum(1 for x in pc if x > 1e-9)
            losses = sum(1 for x in pc if x < -1e-9)
            dd = [x - y for x, y in zip(draws_dev[arm], draws_dev[ref])]
            dt = [x - y for x, y in zip(draws_test[arm], draws_test[ref])]
            rec[ref] = {'mean': m, 'ci95': [lo, hi], 'p_better': p, 'per_conv_mean': pm_, 'per_conv_se': se,
                        'wins': wins, 'losses': losses, 'ties': len(pc) - wins - losses,
                        'dev_mean': sum(dd) / len(dd), 'dev_ci95': list(ci(dd)),
                        'test_mean': sum(dt) / len(dt), 'test_ci95': list(ci(dt))}
        out['bootstrap'][arm] = rec
        r5, rs = rec['g050'], rec['sentence']
        print(f'{arm:<9} {r5["mean"]:+13.4f} [{r5["ci95"][0]:+.4f},{r5["ci95"][1]:+.4f}] {r5["p_better"]:10.3f} '
              f'{r5["per_conv_mean"]:+9.4f}+-{r5["per_conv_se"]:.4f} | '
              f'{rs["mean"]:+17.4f} [{rs["ci95"][0]:+.4f},{rs["ci95"][1]:+.4f}] {rs["p_better"]:10.3f} '
              f'{rs["per_conv_mean"]:+9.4f}+-{rs["per_conv_se"]:.4f} | '
              f'{r5["dev_mean"]:+12.4f} {r5["test_mean"]:+12.4f} {r5["wins"]:>5}/{r5["losses"]:<5}')

    # tied range: gaps whose CI (vs the locally best gap) overlaps zero
    gaps = {g: v for g, v in {'g015': 0.15, 'g030': 0.3, 'g040': 0.4, 'g050': 0.5, 'g070': 0.7}.items() if g in arms}
    best = max(gaps, key=lambda g: out['arms'][g]['local'])
    tied = []
    for g in gaps:
        d = [x - y for x, y in zip(draws[g], draws[best])]
        lo, hi = ci(d)
        if hi >= 0:
            tied.append(g)
    tied_gaps = sorted(gaps[g] for g in tied)
    centre = (tied_gaps[0] + tied_gaps[-1]) / 2 if tied_gaps else gaps[best]
    best_pooled = max(gaps, key=lambda g: out['arms'][g]['pooled'])
    out['tied'] = {'best_local': best, 'best_pooled': best_pooled, 'tied_with_best_local': tied,
                   'tied_range': [tied_gaps[0], tied_gaps[-1]] if tied_gaps else None, 'centre': centre}
    print()
    print(f'best local gap {gaps[best]} ; best pooled gap {gaps[best_pooled]} ; tied with the local best (CI95 of the '
          f'difference reaches 0): {tied_gaps} -> centre {centre}')
    json.dump(out, open(os.path.join(a.out_dir, 'pooled_bootstrap.json'), 'w'), indent=1)
    with open(os.path.join(a.out_dir, 'per_conv_scores.tsv'), 'w') as f:
        f.write('conversation\tsplit\t' + '\t'.join(arms) + '\n')
        for fn in names:
            f.write(fn + '\t' + stats[arms[0]][fn]['split'] + '\t' +
                    '\t'.join(f'{sum(stats[arm][fn]["scores"]) / n_reps:.4f}' for arm in arms) + '\n')
    print(f'wrote {a.out_dir}/pooled_bootstrap.json and per_conv_scores.tsv')


if __name__ == '__main__':
    main()
