"""Anatomy of the U rule (MED_MULTI_FALLBACK=unit) on the 6 fresh sentence-unit fast-r5-2 runs (on top of onset).

For each yes-answer whose span U changes: tIoU before/after vs gold, grouped by unique question, and what the change
is (quote -> whole cluster of cited units). Also a counterfactual 'whole cluster for single-cluster citations' to test
whether the multi-cluster condition matters. No LLM calls.
    python3 u_rule_anatomy.py
"""
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fresh_stack as F  # noqa: E402

S, OE = F.S, F.OE


def tiou(a, b):
    if a is None or b is None:
        return 0.0
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    u = max(a[1], b[1]) - min(a[0], b[0])
    return i / u if u > 0 else 0.0


def main():
    runs = [rp for rp in sorted(glob.glob(os.path.join(F.WORK, 'fr5', 'fr5_rep*', 'arm*.detail.json')))
            if not rp.endswith('arm2.detail.json')]
    rows = {fn: rs for fn, rs in OE.group_questions_by_conversation()}
    fns = sorted(rows)
    cd = F.conv_data(fns, 'sentence')
    dev = OE.split_ids('dev')
    perq = collections.defaultdict(list)
    single = []  # counterfactual on single-cluster yes: whole cluster unit span vs quote span (onset applied to both)
    kinds = collections.Counter()
    for rp in runs:
        det = json.load(open(rp))
        for fn in fns:
            c = cd[fn]
            for r0, r in zip(rows[fn], det[fn]):
                if not r['answer']:
                    continue
                g = (float(r0['evidence_start']), float(r0['evidence_end'])) if r0['evidence_start'] else None
                a, _ = F.span_for(c, r, r0['question'], 'onset')
                b, src = F.span_for(c, r, r0['question'], 'onset+U')
                ids = [i for i in r['ids'] if isinstance(i, int) and 0 <= i < len(c['units'])]
                cl = S.clusters(ids, 2) if ids else []
                if a != b:
                    key = (fn, r0['question_id'])
                    perq[key].append((tiou(b, g) - tiou(a, g), a, b, g, len(cl), r['quote']))
                    kinds['fire_goldyes' if g else 'fire_goldno'] += 1
                elif len(cl) >= 2:
                    kinds['multi_nochange'] += 1
                if len(cl) == 1 and g:
                    whole = S.span_for_ids(c['units'], cl[0], duration=c['dur'])
                    if whole is not None:
                        whole = S.onset_sentstart(whole, c['words'], c['onsets'], 4, -0.04)
                        single.append((tiou(whole, g) - tiou(a, g), fn in dev, whole != a))
    print('fire kinds:', dict(kinds))
    tot = 0
    print(f'{"question":34s} half  n  mean dtIoU  per-run   | gold len / ours before -> after (first run)')
    for key, v in sorted(perq.items(), key=lambda kv: sum(x[0] for x in kv[1])):
        m = sum(x[0] for x in v) / len(v)
        tot += sum(x[0] for x in v)
        d, a, b, g, ncl, q = v[0]
        glen = (g[1] - g[0]) if g else float('nan')
        print(f'{key[1]:34s} {"dev " if key[0] in dev else "test"} {len(v):2d} {m:+.3f}  '
              f'{" ".join(f"{x[0]:+.2f}" for x in v):40s} | g {glen:4.1f}s  {a[1]-a[0]:4.1f}s -> {b[1]-b[0]:4.1f}s  '
              f'ncl {ncl} gold {g} ours {a} -> {b}')
    n_yes = sum(1 for fn in fns for r0 in rows[fn] if r0['evidence_start']) * len(runs)
    print(f'unique questions fired: {len(perq)}; summed dtIoU {tot:+.2f} over {len(runs)} runs '
          f'= {tot / n_yes:+.4f} mean tIoU per gold-yes question')
    ch = [x for x in single if x[2]]
    for half in (True, False):
        s = [x[0] for x in ch if x[1] == half]
        print(f'counterfactual whole-cluster on SINGLE-cluster yes ({"dev" if half else "test"}): n changed {len(s)}, '
              f'mean dtIoU {sum(s) / max(1, len(s)):+.4f}, better {sum(1 for x in s if x > 0.01)}, worse {sum(1 for x in s if x < -0.01)}')


def unique_q(arm, ref='onset'):
    """Per unique question: mean dtIoU of `arm` vs `ref` over the fresh runs where it changes the span."""
    runs = [rp for rp in sorted(glob.glob(os.path.join(F.WORK, 'fr5', 'fr5_rep*', 'arm*.detail.json')))
            if not rp.endswith('arm2.detail.json')]
    rows = {fn: rs for fn, rs in OE.group_questions_by_conversation()}
    fns = sorted(rows)
    cd = F.conv_data(fns, 'sentence')
    perq = collections.defaultdict(list)
    for rp in runs:
        det = json.load(open(rp))
        for fn in fns:
            for r0, r in zip(rows[fn], det[fn]):
                if not r['answer']:
                    continue
                g = (float(r0['evidence_start']), float(r0['evidence_end'])) if r0['evidence_start'] else None
                a, _ = F.span_for(cd[fn], r, r0['question'], ref)
                b, _ = F.span_for(cd[fn], r, r0['question'], arm)
                if a != b:
                    perq[r0['question_id']].append(tiou(b, g) - tiou(a, g))
    ms = [sum(v) / len(v) for v in perq.values()]
    n = len(ms)
    if n:
        mu = sum(ms) / n
        sd = (sum((x - mu) ** 2 for x in ms) / max(1, n - 1)) ** 0.5
        print(f'{arm} vs {ref}: {n} unique questions, {sum(len(v) for v in perq.values())} fires; '
              f'per-question mean dtIoU {mu:+.3f} (t {mu / (sd / n ** 0.5) if sd else 0:+.2f}); '
              f'better {sum(1 for x in ms if x > 0.01)}, worse {sum(1 for x in ms if x < -0.01)}')
        for q, v in sorted(perq.items(), key=lambda kv: sum(kv[1])):
            print(f'   {q:30s} n {len(v)} mean {sum(v) / len(v):+.3f}')


if __name__ == '__main__':
    main()
    for arm in ('onset+U', 'onset+cov', 'onset+any'):
        unique_q(arm)
