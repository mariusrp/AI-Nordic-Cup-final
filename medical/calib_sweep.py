"""Sweep spans.calibrate_span on cached pipeline output (no LLM calls).

    python calib_sweep.py <run.detail.json> <tx_dir> [--split dev]

Replays the span step: takes each question's `cand_span` (the quote/unit span before
calibration, written by pipeline.py with calibration off), applies calibrate_span with
the grid settings, and scores with the frozen offline_eval.score on the chosen split.
Prints the best settings and, for each, the paired per-conversation SE of the gain.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import statistics as st

import offline_eval as OE
import spans as S


def build_preds(detail, words, cfg):
    preds = {}
    for fn, res in detail.items():
        a, s, e = [], [], []
        for r in res:
            a.append(r['answer'])
            sp = r.get('cand_span')
            if r['answer'] and sp:
                sp = S.calibrate_span(tuple(sp), words[fn], *cfg)
                s.append(sp[0])
                e.append(sp[1])
            else:
                s.append(None)
                e.append(None)
        preds[fn] = {'answers': a, 'evidence_start': s, 'evidence_end': e}
    return preds


def per_conv(preds, split):
    out = {}
    for fn in preds:
        out[fn] = OE.score({fn: preds[fn]}, split).final_score
    return {k: v for k, v in out.items() if not math.isnan(v)} if out else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('detail')
    ap.add_argument('tx')
    ap.add_argument('--split', default='dev')
    ap.add_argument('--top', type=int, default=15)
    a = ap.parse_args()
    detail = json.load(open(a.detail))
    words = {}
    for fn in detail:
        tx = S.load_transcript(os.path.join(a.tx, fn.replace('.mp3', '.json')))
        words[fn] = [w for u in S.build_units(tx, 'sentence') for w in u['words']]
    keep = OE.split_ids(a.split)
    detail = {k: v for k, v in detail.items() if keep is None or k in keep}

    grid = {
        'shift_s': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'shift_e': [-0.1, 0.0, 0.1, 0.2, 0.3],
        'snap': [0, 1, 2],
        'snap_pause': [0.15, 0.3, 0.5],
        'clamp_len': [0.0, 2.0, 3.0],
        'clamp_w': [0.25, 0.5, 1.0],
    }
    base_cfg = (0.0, 0.0, 0, 0.3, 0.0, 0.0)
    base = per_conv(build_preds(detail, words, base_cfg), a.split)
    results = []
    seen = set()
    for cfg in itertools.product(*grid.values()):
        ss, se, snap, sp, cl, cw = cfg
        if snap == 0:
            sp = 0.3
        if cl == 0.0:
            cw = 0.0
        cfg = (ss, se, snap, sp, cl, cw)
        if cfg in seen:
            continue
        seen.add(cfg)
        pc = per_conv(build_preds(detail, words, cfg), a.split)
        d = [pc[k] - base[k] for k in base]
        mean = st.mean(pc.values())
        se_d = st.stdev(d) / math.sqrt(len(d))
        nparam = (ss != 0) + (se != 0) + (snap != 0) + (cl != 0)
        results.append((mean, st.mean(d), se_d, nparam, cfg))
    results.sort(key=lambda r: -r[0])
    print(f'split={a.split} n_conv={len(base)} base={st.mean(base.values()):.4f}')
    for r in results[: a.top]:
        print(f'{r[0]:.4f} gain {r[1]:+.4f} +- {r[2]:.4f} nparam {r[3]} cfg {r[4]}')
    best = results[0]
    ok = [r for r in results if r[0] >= best[0] - best[2]]
    ok.sort(key=lambda r: (r[3], -r[0]))
    print('simplest within 1 SE of best:', ok[0])
    for k in range(4):
        cand = [r for r in results if r[3] == k]
        if cand:
            print(f'best with {k} params:', cand[0])


if __name__ == '__main__':
    main()
