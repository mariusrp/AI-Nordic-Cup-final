"""medical-final lane (Sun 20 Sep): interleaved paired arms on ALL 39 TRAINING conversations.

Base = the production rules config (MED_ORDER=aeq MED_SPAN_COVERAGE_NEXT=1 MED_USE_QUOTE=0
MED_UNIT_MODE=island MED_ISLAND_GAP=0.15 MED_ISLAND_THR=-35 MED_RULES=1 MED_RULES_V=3
MED_SPAN_CONT=1), taken from the environment; every arm differs from it in ONE knob group:

    base  production (rules v3)
    R4    MED_RULES_V=4            tightened rules wording (location tie-break + closing check)
    SC3   MED_SC_N=3 MED_SC_MODE=island MED_SC_AGREE=2   per-line self-consistency over 3 samples
    VC    MED_VC=1                 contraction-only verifier on the cited cluster

Arms are interleaved per conversation (all arms answer conversation k before conversation k+1), so
vLLM run-to-run noise hits every arm equally inside a replicate. Nothing is stacked.

The set is the 39 TRAINING conversations only (UPSTREAM=/workspace/.holdout/upstream-full, whose
question_train.csv holds 31 visible + 8 holdout); the 19 portal-validation conversations are never
loaded. Transcripts come from MED_TX_DIR, audio (for the energy-onset start rule) from the upstream
data/audio dir, exactly like the live server.

    UPSTREAM=/workspace/.holdout/upstream-full MED_TX_DIR=/workspace/medical_val/tx \
        python3 rules_arms.py --reps 3 --out-dir out/final --arms base,R4,SC3,VC
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import offline_eval as OE  # noqa: E402
import pipeline as P  # noqa: E402
import spans as S  # noqa: E402

UPSTREAM = os.environ.get('UPSTREAM', '/workspace/.holdout/upstream-full')
TX_DIRS = [d for d in os.environ.get('MED_TX_DIR', '/workspace/medical_val/tx').split(',') if d]
AUDIO_DIRS = [d for d in os.environ.get(
    'MED_AUDIO_DIR',
    os.path.join(UPSTREAM, 'medical-appointment/data/audio') + ',/workspace/.holdout/med_holdout_audio'
).split(',') if d]

ARMS = {
    'base': dict(rules_v=3, sc_n=1, sc_mode='cluster', sc_agree=2, vc=0, span_trim=0),
    'R4':   dict(rules_v=4, sc_n=1, sc_mode='cluster', sc_agree=2, vc=0),
    'SC3':  dict(rules_v=3, sc_n=3, sc_mode='island', sc_agree=2, vc=0),
    'VC':   dict(rules_v=3, sc_n=1, sc_mode='cluster', sc_agree=2, vc=1),
    'R4SC3': dict(rules_v=4, sc_n=3, sc_mode='island', sc_agree=2, vc=0),
    'R4VC': dict(rules_v=4, sc_n=1, sc_mode='cluster', sc_agree=2, vc=1),
    # med-trim lane (Sun 20 Sep): clause-level contraction of the finished span (MED_SPAN_TRIM,
    # pipeline.clause_trim). span_trim is the ONLY difference from base.
    'TRIM':  dict(rules_v=3, sc_n=1, sc_mode='cluster', sc_agree=2, vc=0, span_trim=1),
    'TRIM2': dict(rules_v=3, sc_n=1, sc_mode='cluster', sc_agree=2, vc=0, span_trim=2),
    'TRIM3': dict(rules_v=3, sc_n=1, sc_mode='cluster', sc_agree=2, vc=0, span_trim=3),
}


def set_arm(name: str):
    for k, v in ARMS[name].items():
        P.CFG[k] = v


def load_pool():
    """{fn: {'tx': ..., 'questions': [...]}} for every conversation in the 39-question CSV."""
    from faster_whisper.audio import decode_audio
    qs = {}
    for fn, rows in OE.group_questions_by_conversation():
        qs[fn] = [r['question'] for r in rows]
    pool = {}
    for fn in sorted(qs):
        base = fn.replace('.mp3', '')
        txp = next((os.path.join(d, base + '.json') for d in TX_DIRS
                    if os.path.exists(os.path.join(d, base + '.json'))), None)
        if txp is None:
            print(f'WARN no transcript for {fn}', flush=True)
            continue
        tx = dict(S.load_transcript(txp))
        ap = next((os.path.join(d, fn) for d in AUDIO_DIRS if os.path.exists(os.path.join(d, fn))), None)
        if ap:
            try:
                tx['_pcm'] = decode_audio(ap, sampling_rate=16000)
            except Exception as e:
                print(f'WARN pcm {fn}: {e}', flush=True)
        else:
            print(f'WARN no audio for {fn} (onset rule degraded)', flush=True)
        pool[fn] = {'tx': tx, 'questions': qs[fn]}
    return pool


def conv_score(pred: dict, fn: str) -> float:
    return OE.summary(OE.score({fn: pred}, 'all'))['score']


def boot_ci(diffs, b=4000, seed=2026):
    """Percentile bootstrap CI of the mean over conversations (the resampling unit)."""
    if not diffs:
        return float('nan'), float('nan'), float('nan')
    rnd = random.Random(seed)
    n = len(diffs)
    ms = []
    for _ in range(b):
        ms.append(sum(diffs[rnd.randrange(n)] for _ in range(n)) / n)
    ms.sort()
    p_pos = sum(1 for m in ms if m > 0) / b
    return ms[int(0.025 * (b - 1))], ms[int(0.975 * (b - 1))], p_pos


def run(reps, out_dir, arms, limit=0):
    os.makedirs(out_dir, exist_ok=True)
    pool = load_pool()
    names = sorted(pool)
    if limit:
        names = names[:limit]
    print(f'{len(names)} conversations x {len(arms)} arms x {reps} reps; base cfg '
          f"order={P.CFG['order']} unit={P.CFG['unit_mode']} rules={P.CFG['rules']} "
          f"cont={P.CFG['span_cont']} covnext={P.CFG['span_coverage_next']}", flush=True)
    ans = P.Answerer('llm')
    if ans.llm is None:
        raise SystemExit('LLM unavailable; refusing to run without the real backend')

    preds = {a: [dict() for _ in range(reps)] for a in arms}
    lat = {a: [] for a in arms}
    t0 = time.time()
    for rep in range(reps):
        for k, fn in enumerate(names):
            tx, qq = pool[fn]['tx'], pool[fn]['questions']
            for arm in arms:
                set_arm(arm)
                t = time.time()
                res = ans.answer(tx, qq, deadline=time.time() + P.CFG['deadline'])
                dt = time.time() - t
                lat[arm].append(dt)
                preds[arm][rep][fn] = {
                    'answers': [r['answer'] for r in res],
                    'evidence_start': [r['span'][0] if r['answer'] and r['span'] else None for r in res],
                    'evidence_end': [r['span'][1] if r['answer'] and r['span'] else None for r in res],
                    'latency_ms': dt * 1000,
                }
            if (k + 1) % 5 == 0 or k + 1 == len(names):
                print(f'rep {rep} {k + 1}/{len(names)}  {time.time() - t0:.0f}s', flush=True)
        for arm in arms:
            json.dump(preds[arm][rep], open(os.path.join(out_dir, f'{arm}.rep{rep}.json'), 'w'))
            sc = OE.summary(OE.score(preds[arm][rep], 'all'))
            print(f'-- rep {rep} {arm}: {sc}', flush=True)

    report = {'reps': reps, 'n_conv': len(names), 'arms': {a: ARMS[a] for a in arms},
              'summary': {}, 'paired_vs_base': {},
              'latency_s': {a: {'mean': sum(lat[a]) / len(lat[a]),
                                'max': max(lat[a])} for a in arms}}
    for arm in arms:
        report['summary'][arm] = {}
        for split in ('all', 'dev', 'test'):
            ss = [OE.summary(OE.score(preds[arm][r], split))['score'] for r in range(reps)]
            report['summary'][arm][split] = {'reps': ss, 'mean': round(sum(ss) / len(ss), 4)}
        full = [OE.summary(OE.score(preds[arm][r], 'all')) for r in range(reps)]
        report['summary'][arm]['acc'] = round(sum(f['accuracy'] for f in full) / reps, 4)
        report['summary'][arm]['tiou'] = round(sum(f['tiou'] for f in full) / reps, 4)

    cs = {a: {fn: [conv_score(preds[a][r][fn], fn) for r in range(reps)] for fn in names} for a in arms}
    dev = OE.split_ids('dev')
    for arm in arms:
        if arm == 'base':
            continue
        entry = {}
        for split, keys in (('all', names), ('dev', [f for f in names if f in dev]),
                            ('test', [f for f in names if f not in dev])):
            d = [sum(cs[arm][fn]) / reps - sum(cs['base'][fn]) / reps for fn in keys]
            n = len(d)
            m = sum(d) / n
            se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n) if n > 1 else float('nan')
            lo, hi, ppos = boot_ci(d)
            entry[split] = {'n': n, 'mean': round(m, 4), 'se': round(se, 4),
                            'ci95': [round(lo, 4), round(hi, 4)], 'p_positive': round(ppos, 3),
                            'conv_better': sum(1 for x in d if x > 1e-9),
                            'conv_worse': sum(1 for x in d if x < -1e-9)}
        entry['per_conv'] = {fn: round(sum(cs[arm][fn]) / reps - sum(cs['base'][fn]) / reps, 4) for fn in names}
        entry['sign_test'] = sign_test(preds['base'], preds[arm], names, reps)
        report['paired_vs_base'][arm] = entry
    json.dump(report, open(os.path.join(out_dir, 'report.json'), 'w'), indent=1)
    out = {k: v for k, v in report.items() if k != 'paired_vs_base'}
    out['paired_vs_base'] = {a: {k: v for k, v in e.items() if k != 'per_conv'}
                             for a, e in report['paired_vs_base'].items()}
    print(json.dumps(out, indent=1), flush=True)
    return report


def sign_test(base_reps, cand_reps, names, reps):
    """Per-question tIoU delta on gold-yes rows, averaged over reps (questions touched in >= 1 rep)."""
    sys.path.insert(0, os.path.join(UPSTREAM, 'medical-appointment'))
    from utils import evidence_interval, gold_evidence, temporal_iou
    rows_by_conv = {fn: rows for fn, rows in OE.group_questions_by_conversation()}
    deltas = {}
    for fn in names:
        for qi, row in enumerate(rows_by_conv.get(fn, [])):
            if int(row['label']) != 1:
                continue
            gold = gold_evidence(row)
            bt, ct, fired = [], [], False
            for r in range(reps):
                bp, cp = base_reps[r].get(fn), cand_reps[r].get(fn)
                if bp is None or cp is None:
                    continue
                bs = evidence_interval(bp['evidence_start'][qi], bp['evidence_end'][qi]) if bp['answers'][qi] else None
                cps = evidence_interval(cp['evidence_start'][qi], cp['evidence_end'][qi]) if cp['answers'][qi] else None
                if bs != cps or bp['answers'][qi] != cp['answers'][qi]:
                    fired = True
                bt.append(temporal_iou(gold, bs) if bs else 0.0)
                ct.append(temporal_iou(gold, cps) if cps else 0.0)
            if fired and bt:
                deltas[row.get('question_id', f'{fn}:{qi}')] = sum(ct) / len(ct) - sum(bt) / len(bt)
    better = sum(1 for d in deltas.values() if d > 1e-9)
    worse = sum(1 for d in deltas.values() if d < -1e-9)
    return {'touched': len(deltas), 'better': better, 'worse': worse,
            'mean_dtiou_on_touched': round(sum(deltas.values()) / len(deltas), 4) if deltas else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--out-dir', default='out/final')
    ap.add_argument('--arms', default='base,R4,SC3,VC')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    run(a.reps, a.out_dir, [x for x in a.arms.split(',') if x], a.limit)


if __name__ == '__main__':
    main()
