"""Allmquote-g4-2 zero-LLM replay: cluster-aware quote gate / fallback on cached LLM output.

For every cached pipeline detail.json (answer, ids, quote per question) the span is rebuilt from
the cached turbo transcript with spans.quote_span + spans.calibrate_span (production shift 0.2),
once per pre-registered arm, and scored with the frozen offline_eval.py. The legacy arm must
reproduce the cached pipeline spans (sanity check).

    UPSTREAM=/home/claude/upstream-work python g4_2_replay.py --tx-dir TX detail1.json [detail2.json ...]

Arms (pre-registered): L = legacy (production); R0 = gate any cluster; R0b = R0 + ellipsis pieces;
R0+R2 / R0b+R2 = fallback to max question-overlap cluster; R0+R3 / R0b+R3 = fallback to max
quote-overlap cluster; U = the proposal's original rule (multi-cluster yes -> whole unit span of
the cluster containing the aligned quote, first cluster if none). Pick on dev of the FIRST
detail file; test once; the remaining files are confirmation runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('UPSTREAM', '/home/claude/upstream-work')
import offline_eval as OE  # noqa: E402
import spans as S  # noqa: E402

SHIFT = (0.2, 0.0, 0, 0.3, 0.0, 0.0)
ARMS = {'L': dict(gate='first', ellipsis=0, fallback='first'),
        'R0': dict(gate='any', ellipsis=0, fallback='first'),
        'R0b': dict(gate='any', ellipsis=1, fallback='first'),
        'R0+R2': dict(gate='any', ellipsis=0, fallback='qoverlap'),
        'R0+R3': dict(gate='any', ellipsis=0, fallback='quoteoverlap'),
        'R0b+R2': dict(gate='any', ellipsis=1, fallback='qoverlap'),
        'R0b+R3': dict(gate='any', ellipsis=1, fallback='quoteoverlap'),
        'U': 'U',
        # post-hoc (added after seeing that most dropped quotes join sentences WITHOUT '...'):
        'R0c': dict(gate='any', ellipsis=2, fallback='first'),
        'R0c+R2': dict(gate='any', ellipsis=2, fallback='qoverlap')}


def questions():
    rows = list(csv.DictReader(open(os.path.join(os.environ['UPSTREAM'],
                                                 'medical-appointment/data/question_train.csv'))))
    g = {}
    for r in rows:
        g.setdefault(f"conversation_{r['transcript_id']}.mp3", []).append(r['question'])
    return g


def unit_rule(units, r, dur):
    """U: whole unit span of the cluster containing the (R0-gated) quote; legacy otherwise."""
    ids = [i for i in r['ids'] if isinstance(i, int) and 0 <= i < len(units)]
    cl = S.clusters(ids, 2) if ids else []
    if len(cl) < 2:
        return S.quote_span(units, r['ids'], r['quote'] or '', duration=dur)
    sp, src = S.quote_span(units, r['ids'], r['quote'] or '', gate='any', duration=dur)
    pick = cl[0]
    if src == 'quote':
        for c in cl:
            a, b = units[c[0]]['start'], units[c[-1]]['end']
            if not (sp[1] < a - 3 or sp[0] > b + 3):
                pick = c
                break
    return S.span_for_ids(units, pick, duration=dur), 'unit_U'


def build(detail, tx, qs, arm):
    preds, meta = {}, {}
    for fn, rows in detail.items():
        units = tx[fn]['units']
        words = [w for u in units for w in u['words']]
        dur = tx[fn]['dur']
        a, s, e, m = [], [], [], []
        for q, r in zip(qs[fn], rows):
            if arm == 'U':
                sp, src = unit_rule(units, r, dur)
            else:
                sp, src = S.quote_span(units, r['ids'], r['quote'] or '', question=q, duration=dur, **ARMS[arm])
            if sp is not None:
                sp = S.calibrate_span(sp, words, *SHIFT, duration=dur)
                sp = (round(float(sp[0]), 2), round(float(sp[1]), 2))
                if sp[1] <= sp[0]:
                    sp = (sp[0], round(sp[0] + 0.3, 2))
            yes = bool(r['answer'])
            a.append(yes)
            s.append(sp[0] if yes and sp else None)
            e.append(sp[1] if yes and sp else None)
            ids = [i for i in r['ids'] if isinstance(i, int) and 0 <= i < len(units)]
            m.append({'src': src, 'multi': len(S.clusters(ids, 2)) >= 2 if ids else False, 'span': sp})
        preds[fn] = {'answers': a, 'evidence_start': s, 'evidence_end': e}
        meta[fn] = m
    return preds, meta


def pc(preds, split):
    keep = OE.split_ids(split)
    return {fn: OE.score({fn: p}, split).final_score for fn, p in preds.items() if fn in keep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tx-dir', required=True)
    ap.add_argument('details', nargs='+')
    ap.add_argument('--out')
    a = ap.parse_args()
    qs = questions()
    tx = {}
    for f in os.listdir(a.tx_dir):
        t = S.load_transcript(os.path.join(a.tx_dir, f))
        tx[f.replace('.json', '.mp3')] = {'units': S.build_units(t, 'sentence'), 'dur': t.get('duration')}
    report = {}
    for path in a.details:
        detail = json.load(open(path))
        res = {}
        base_preds, base_meta = build(detail, tx, qs, 'L')
        # sanity: legacy replay vs cached spans (cached shift may be 0.2 or 0)
        n = same = 0
        for fn, rows in detail.items():
            for r, mm in zip(rows, base_meta[fn]):
                if r.get('cand_span') is not None and mm['span'] is not None:
                    n += 1
                    same += abs(r['cand_span'][0] - mm['span'][0]) < 0.02 and abs(r['cand_span'][1] - mm['span'][1]) < 0.02
        print(f'\n== {path}\n legacy reproduces cached cand_span (shift 0.2): {same}/{n}')
        pcb = {sp: pc(base_preds, sp) for sp in ('dev', 'test')}
        for arm in ARMS:
            preds, meta = build(detail, tx, qs, arm)
            fired = mig = 0
            migs = {}
            for fn in meta:
                for r, m0, m1 in zip(detail[fn], base_meta[fn], meta[fn]):
                    if r['answer'] and m0['span'] != m1['span']:
                        fired += 1
                        k = f"{m0['src']}->{m1['src']}"
                        migs[k] = migs.get(k, 0) + 1
            row = {'fired': fired, 'migration': migs}
            for sp in ('dev', 'test'):
                summ = OE.summary(OE.score(preds, sp))
                p1 = pc(preds, sp)
                d = [p1[k] - pcb[sp][k] for k in p1]
                row[sp] = {'score': summ['score'], 'tiou': summ.get('tiou', summ.get('mean_tiou')),
                           'gain': st.mean(d), 'se': st.stdev(d) / len(d) ** 0.5}
            res[arm] = row
            print(f" {arm:7s} fired {fired:3d}  dev {row['dev']['score']:.4f} ({row['dev']['gain']:+.4f}+-{row['dev']['se']:.4f})"
                  f"  test {row['test']['score']:.4f} ({row['test']['gain']:+.4f}+-{row['test']['se']:.4f})  {migs}")
        nyes = sum(bool(r['answer']) for rows in detail.values() for r in rows)
        nmulti = sum(1 for fn in detail for r, m in zip(detail[fn], base_meta[fn]) if r['answer'] and m['multi'])
        nfb = sum(1 for fn in detail for r, m in zip(detail[fn], base_meta[fn])
                  if r['answer'] and m['multi'] and m['src'] == 'units' and r['quote'])
        print(f' yes {nyes}, multi-cluster yes {nmulti}, of which quote dropped by legacy gate {nfb}')
        report[path] = {'arms': res, 'legacy_repro': [same, n], 'yes': nyes, 'multi_yes': nmulti, 'legacy_dropped': nfb}
    if a.out:
        json.dump(report, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
