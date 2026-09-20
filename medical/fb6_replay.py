"""FB6-B zero-LLM replay: stack the near-miss zero-LLM span tweaks on top of the onset
(-0.04 + sentstart4) start rule, and score with the REAL frozen offline_eval.py (not a tIoU
proxy). No LLM calls: every arm is rebuilt from cached production detail.json reps (answer/ids/
quote per question, from the shared vLLM) using spans.quote_span + spans.onset_sentstart +
spans.coverage_next_end + spans.calibrate_span, on the cached large-v3-turbo transcripts and
decoded PCM (for onset detection).

Reps (pre-registered, the same 4 cached PRODUCTION-config LLM runs used to validate the onset
rule itself: LESSONS.md cycle-2 header + medical/analysis/extent_units.py RUNS):
  med-evolve-g3-1A/out/{a1_r1,a1_r2,a1_r3}/arm0.detail.json  (arm0 of medical-evolve-g3-1)
  medical-fast-r4-1/out/r4_1/base.detail.json

Arms (pre-registered before this run; BASE is the fast-r6-2 base = onset(-0.04)+sentstart4,
SHIFT_S=0.0, quote_gate=first/legacy, exactly medical-fast-r5-onset):
  base   : onset only (the fast-lane baseline for this batch)
  fb6-1  : onset + MED_MULTI_FALLBACK=unit ('U' rule: multi-cluster yes -> whole unit span of
           the cluster the quote lands in)
  fb6-2  : onset + coverage-next end extension (extend the end over the adjacent NEXT sentence
           only when it holds question content tokens our span lacks)
  fb6-3  : onset + MED_QUOTE_GATE=any (R0: keep quotes that align inside ANY cited cluster)
  fb6-4  : onset + fb6-1 + fb6-2 + fb6-3 together (the pre-registered combo)

Usage (on POD=gpu or locally, no GPU/LLM needed):
    UPSTREAM=/home/claude/upstream-work python3 fb6_replay.py \
        --tx-dir <dir>/tx/large-v3-turbo --pcm-dir <dir>/pcm \
        <dir>/med-evolve-g3-1A/out/a1_r1/arm0.detail.json \
        <dir>/med-evolve-g3-1A/out/a1_r2/arm0.detail.json \
        <dir>/med-evolve-g3-1A/out/a1_r3/arm0.detail.json \
        <dir>/medical-fast-r4-1/out/r4_1/base.detail.json \
        --out fb6_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('UPSTREAM', '/home/claude/upstream-work')
import offline_eval as OE  # noqa: E402
import spans as S  # noqa: E402

# onset(-0.04)+sentstart4, SHIFT_S=0.0 (the onset rule replaces the global start shift; see
# medical-fast-r5-onset / spans.onset_sentstart docstring). calibrate_span otherwise off.
ONSET_K, ONSET_OFF = 4, -0.04
CALSHIFT = (0.0, 0.0, 0, 0.3, 0.0, 0.0)  # shift_s, shift_e, snap, snap_pause, clamp_len, clamp_w

ARMS = {
    'base': dict(gate='first', fallback='first', coverage=False),
    'fb6-1': dict(gate='first', fallback='unit', coverage=False),
    'fb6-2': dict(gate='first', fallback='first', coverage=True),
    'fb6-3': dict(gate='any', fallback='first', coverage=False),
    'fb6-4': dict(gate='any', fallback='unit', coverage=True),
}


def build_span(units, words, onsets, ids, quote, question, dur, cfg):
    span, src = S.quote_span(units, ids, quote or '', gate=cfg['gate'], fallback=cfg['fallback'],
                             question=question, duration=dur)
    if span is not None:
        span = S.onset_sentstart(span, words, onsets, ONSET_K, ONSET_OFF)
    if span is not None and cfg['coverage']:
        span = S.coverage_next_end(span, words, question)
    if span is not None:
        span = S.calibrate_span(span, words, *CALSHIFT, duration=dur)
        s, e = float(span[0]), float(span[1])
        if e <= s:
            e = s + 0.3
        span = (round(s, 2), round(e, 2))
    return span, src


def load_conv_data(tx_dir, pcm_dir, filenames):
    data = {}
    for fn in filenames:
        base = fn.replace('.mp3', '')
        tx = S.load_transcript(os.path.join(tx_dir, base + '.json'))
        units = S.build_units(tx, 'sentence')
        words = [w for u in units for w in u['words']]
        dur = tx.get('duration')
        onsets = None
        pcm_path = os.path.join(pcm_dir, base + '.npy')
        if os.path.exists(pcm_path):
            try:
                onsets = S.energy_onsets(np.load(pcm_path))
            except Exception:
                onsets = None
        data[fn] = {'units': units, 'words': words, 'dur': dur, 'onsets': onsets}
    return data


def build_preds(detail, conv, qs, arm):
    cfg = ARMS[arm]
    preds, meta = {}, {}
    for fn, rows in detail.items():
        cd = conv[fn]
        a, s, e, m = [], [], [], []
        for q, r in zip(qs[fn], rows):
            sp, src = build_span(cd['units'], cd['words'], cd['onsets'], r['ids'], r['quote'], q,
                                 cd['dur'], cfg)
            yes = bool(r['answer'])
            a.append(yes)
            s.append(sp[0] if yes and sp else None)
            e.append(sp[1] if yes and sp else None)
            m.append({'src': src, 'span': sp})
        preds[fn] = {'answers': a, 'evidence_start': s, 'evidence_end': e}
        meta[fn] = m
    return preds, meta


def per_conv_score(preds, split):
    keep = OE.split_ids(split)
    return {fn: OE.score({fn: p}, split).final_score for fn, p in preds.items()
           if keep is None or fn in keep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tx-dir', required=True)
    ap.add_argument('--pcm-dir', required=True)
    ap.add_argument('--out')
    ap.add_argument('details', nargs='+')
    a = ap.parse_args()

    qs = {}
    for fn, rows in OE.group_questions_by_conversation():
        qs[fn] = [r['question'] for r in rows]

    filenames = sorted(qs)
    conv = load_conv_data(a.tx_dir, a.pcm_dir, filenames)

    # pooled-over-reps per-conversation scores, per arm, per half
    pooled = {arm: {'dev': {}, 'test': {}, 'all': {}} for arm in ARMS}
    per_rep_summary = {arm: [] for arm in ARMS}
    fired_by_arm = {arm: 0 for arm in ARMS}

    for path in a.details:
        detail = json.load(open(path))
        detail = {fn: rows for fn, rows in detail.items() if fn in qs}
        print(f'\n== {path} ({len(detail)} conversations)')
        base_preds, base_meta = build_preds(detail, conv, qs, 'base')
        for arm in ARMS:
            preds, meta = build_preds(detail, conv, qs, arm)
            fired = 0
            for fn in meta:
                for r, m0, m1 in zip(detail[fn], base_meta[fn], meta[fn]):
                    if r['answer'] and m0['span'] != m1['span']:
                        fired += 1
            fired_by_arm[arm] += fired
            row = {}
            for half in ('dev', 'test', 'all'):
                summ = OE.summary(OE.score(preds, half))
                p1 = per_conv_score(preds, half)
                for fn, v in p1.items():
                    pooled[arm][half].setdefault(fn, []).append(v)
                row[half] = summ['score']
            per_rep_summary[arm].append(row)
            print(f'  {arm:6s} fired {fired:3d}  dev {row["dev"]:.4f}  test {row["test"]:.4f}')

    # Pool over reps the SAME way as the validated onset replay (medical/analysis/replay_rules.py
    # report()): sum each conversation's diff over all its reps, then take the conversation as the
    # independent sampling unit for se (cluster by conversation, not by rep or question).
    base_pool = {half: pooled['base'][half] for half in ('dev', 'test', 'all')}

    report = {'reps': a.details, 'fired': fired_by_arm, 'per_rep': per_rep_summary, 'arms': {}}
    print('\n== Pooled over reps (paired vs base, clustered by conversation)')
    for arm in ARMS:
        arm_report = {}
        for half in ('dev', 'test', 'all'):
            fns = sorted(pooled[arm][half])
            per_conv_diffs = {fn: [av - bv for av, bv in zip(pooled[arm][half][fn], base_pool[half][fn])]
                              for fn in fns}
            conv_sum = [sum(per_conv_diffs[fn]) for fn in fns]
            n = sum(len(per_conv_diffs[fn]) for fn in fns)  # reps x conversations
            mean = sum(conv_sum) / n
            se = (st.stdev(conv_sum) * (len(fns) ** 0.5) / n) if len(fns) > 1 else float('nan')
            arm_report[half] = {'gain': mean, 'se': se, 'se_mult': (mean / se if se else float('nan')), 'n_reps_x_conv': n}
            print(f'  {arm:6s} {half:4s} gain {mean:+.4f} +- {se:.4f}  ({mean / se if se else float("nan"):+.2f} se, n={n})')
        report['arms'][arm] = arm_report

    if a.out:
        json.dump(report, open(a.out, 'w'), indent=1)
        print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
