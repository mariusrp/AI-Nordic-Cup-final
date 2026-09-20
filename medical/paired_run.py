"""FR3-B fast-lane batch: BASE + 3 cheap LLM-in-loop variants (FR3-B1 cluster verifier,
FR3-B2 quote-repair re-query, FR3-B3 sibling-aware context replication), scored in ONE
interleaved paired run against the shared vLLM (:8001).

BASE = main deploy env (MED_SPAN_ONSET_SENTSTART=1, MED_SPAN_SHIFT_S=0, from medical/env.sh) +
the FR3-A1 port's MED_QUOTE_GATE=any (paired_pool.set_base). Every arm is re-run FRESH (no
cached predictions reused) and arms are interleaved per conversation, i.e. for each conversation
the base arm and all variant arms are called in the same pass, then the next conversation -- not
one arm run start-to-finish before the next -- so LLM run-to-run noise (vLLM batching) affects
every arm equally within a replicate.

Guardrails (per Adrian's directive): stop immediately if the base arm's running score drifts more
than 0.015 from 0.795, or if any arm/question hits the deadline (MED_DEADLINE, 45s default).
Also logs p95 per-conversation latency per arm (must stay < 15s on contended :8001).

Usage (on POD=gpu, light client against the shared :8001):
    UPSTREAM=/workspace/upstream MED_TX_DIR=/workspace/tx/large-v3-turbo \\
        python3 paired_run.py --reps 2 --out-dir out/fr3b
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import offline_eval as OE  # noqa: E402
import paired_pool as PP  # noqa: E402
import pipeline as P  # noqa: E402

BASE_EXPECT = 0.795
DRIFT_BAR = 0.015


def run(reps: int, out_dir: str, limit: int = 0):
    os.makedirs(out_dir, exist_ok=True)
    PP.set_base()
    pool = PP.load_pool()
    names = sorted(pool)
    if limit:
        names = names[:limit]
    print(f'{len(names)} conversations, {len(PP.ARMS)} arms x {reps} reps '
          f'({len(names) * len(PP.ARMS) * reps} conversation-arm-reps); base quote_gate={P.CFG["quote_gate"]}',
          flush=True)

    ans = P.Answerer('llm')
    if ans.llm is None:
        raise SystemExit('LLM unavailable; refusing to run FR3-B with the heuristic-only fallback')

    # preds[arm][rep] = {fn: pred_dict}; detail[arm][rep] = {fn: [row,...]}
    preds = {a: [dict() for _ in range(reps)] for a in PP.ARMS}
    detail = {a: [dict() for _ in range(reps)] for a in PP.ARMS}
    lat_ms = {a: [] for a in PP.ARMS}
    deadline_misses = 0
    n_done = 0

    t_start = time.time()
    for rep in range(reps):
        for fn in names:
            entry = pool[fn]
            tx, qs = entry['tx'], entry['questions']
            for arm in PP.ARMS:
                PP.set_arm(arm)
                t0 = time.time()
                res = ans.answer(tx, qs, deadline=time.time() + P.CFG['deadline'])
                dt = time.time() - t0
                lat_ms[arm].append(dt * 1000)
                if dt > P.CFG['deadline']:
                    deadline_misses += 1
                    print(f'DEADLINE MISS arm={arm} rep={rep} {fn} {dt:.1f}s', flush=True)
                preds[arm][rep][fn] = {
                    'answers': [r['answer'] for r in res],
                    'evidence_start': [r['span'][0] if r['answer'] and r['span'] else None for r in res],
                    'evidence_end': [r['span'][1] if r['answer'] and r['span'] else None for r in res],
                    'latency_ms': dt * 1000,
                }
                detail[arm][rep][fn] = res
            n_done += 1
            if n_done % 5 == 0 or fn == names[-1]:
                elapsed = time.time() - t_start
                print(f'rep {rep} conv {n_done}/{len(names)}  elapsed {elapsed:.0f}s', flush=True)

        # guardrail: check the base arm's running score after each full replicate
        base_all = OE.summary(OE.score(preds['base'][rep], 'all'))['score']
        print(f'-- rep {rep} base arm score(all)={base_all:.4f} (expect ~{BASE_EXPECT}, '
              f'bar +-{DRIFT_BAR})', flush=True)
        if abs(base_all - BASE_EXPECT) > DRIFT_BAR:
            print(f'STOP: base arm drifted {base_all - BASE_EXPECT:+.4f} from {BASE_EXPECT} '
                  f'(> {DRIFT_BAR}); aborting further reps', flush=True)
            reps = rep + 1
            break
        if deadline_misses:
            print(f'STOP: {deadline_misses} deadline miss(es); aborting further reps', flush=True)
            reps = rep + 1
            break

    # merge reps into one preds/detail file per arm (last rep wins) for record-keeping/handoff,
    # and save every rep separately for the paired stats below
    for arm in PP.ARMS:
        for rep in range(reps):
            json.dump(preds[arm][rep], open(os.path.join(out_dir, f'{arm}.rep{rep}.json'), 'w'))
            json.dump(detail[arm][rep], open(os.path.join(out_dir, f'{arm}.rep{rep}.detail.json'), 'w'),
                      default=str)
        json.dump(preds[arm][reps - 1], open(os.path.join(out_dir, f'{arm}.json'), 'w'))

    report = score_report(preds, detail, reps, names, lat_ms)
    json.dump(report, open(os.path.join(out_dir, 'report.json'), 'w'), indent=1)
    print(json.dumps(report['summary'], indent=1))
    print(json.dumps(report['latency'], indent=1))
    return report


def conv_scores(p: dict, split: str) -> dict:
    keep = OE.split_ids(split)
    out = {}
    for fn, v in p.items():
        if keep is not None and fn not in keep:
            continue
        out[fn] = OE.summary(OE.score({fn: v}, 'all'))['score']
    return out


def paired(base_reps, cand_reps, split: str, names):
    """Per-conversation diff, averaged over reps first (n = n_conversations in this split),
    then mean +- se over conversations. Matches medical/paired.py's per-split convention."""
    keep = OE.split_ids(split)
    keys = [fn for fn in names if keep is None or fn in keep]
    diffs = []
    for fn in keys:
        rep_d = []
        for base_p, cand_p in zip(base_reps, cand_reps):
            if fn not in base_p or fn not in cand_p:
                continue
            sb = OE.summary(OE.score({fn: base_p[fn]}, 'all'))['score']
            sc = OE.summary(OE.score({fn: cand_p[fn]}, 'all'))['score']
            rep_d.append(sc - sb)
        if rep_d:
            diffs.append(sum(rep_d) / len(rep_d))
    n = len(diffs)
    m = sum(diffs) / n if n else float('nan')
    se = math.sqrt(sum((x - m) ** 2 for x in diffs) / (n - 1) / n) if n > 1 else float('nan')
    return {'n': n, 'mean': m, 'se': se, 'se_mult': (m / se) if se else float('nan'),
            'per_conv': dict(zip(keys, diffs))}


def sign_test(base_preds_reps, cand_preds_reps, names, rows_by_conv):
    """Per-question sign test on tIoU (only over gold-yes rows, since only those have a span to
    score), averaged over reps. Reports the unique questions touched (span or answer differs in
    >= 1 rep) and a binomial-style sign count over those questions' mean tIoU delta.
    LESSONS.md: 'a rule that fires on few questions fires on the SAME questions every run' -- so
    replicates are not independent replication, only the unique-question count + sign matter."""
    from utils import evidence_interval, gold_evidence, temporal_iou

    touched = set()
    per_q_deltas: dict = {}
    for fn in names:
        rows = rows_by_conv.get(fn)
        if not rows:
            continue
        for qi, row in enumerate(rows):
            if int(row['label']) != 1:
                continue
            gold = gold_evidence(row)
            b_tious, c_tious = [], []
            fired = False
            for base_p, cand_p in zip(base_preds_reps, cand_preds_reps):
                bp, cp = base_p.get(fn), cand_p.get(fn)
                if bp is None or cp is None:
                    continue
                bs = evidence_interval(bp['evidence_start'][qi], bp['evidence_end'][qi]) if bp['answers'][qi] else None
                cs = evidence_interval(cp['evidence_start'][qi], cp['evidence_end'][qi]) if cp['answers'][qi] else None
                if bs != cs or bp['answers'][qi] != cp['answers'][qi]:
                    fired = True
                b_tious.append(temporal_iou(gold, bs) if bs else 0.0)
                c_tious.append(temporal_iou(gold, cs) if cs else 0.0)
            if fired and b_tious:
                qid = row.get('question_id', f'{fn}:q{qi}')
                touched.add(qid)
                per_q_deltas[qid] = (sum(c_tious) / len(c_tious)) - (sum(b_tious) / len(b_tious))
    better = sum(1 for d in per_q_deltas.values() if d > 1e-9)
    worse = sum(1 for d in per_q_deltas.values() if d < -1e-9)
    neutral = len(per_q_deltas) - better - worse
    mean_d = sum(per_q_deltas.values()) / len(per_q_deltas) if per_q_deltas else 0.0
    return {'n_unique_touched': len(touched), 'better': better, 'worse': worse, 'neutral': neutral,
            'mean_dtiou_on_touched': mean_d, 'unique_touched': sorted(touched)}


def fire_counts(detail_reps):
    """Count rows whose span_src shows the variant fired (and, for cluster-verify, actually
    switched away from the served cluster): 'quote_cv'/'units_cv' = fr3-b1 fired and switched;
    'quote_repair' = fr3-b2's repair aligned successfully."""
    c = {'cluster_verify_switched': 0, 'quote_repair_aligned': 0}
    for rep_detail in detail_reps:
        for rows in rep_detail.values():
            for r in rows:
                src = r.get('span_src')
                if src in ('quote_cv', 'units_cv'):
                    c['cluster_verify_switched'] += 1
                elif src == 'quote_repair':
                    c['quote_repair_aligned'] += 1
    return c


def pctl(xs, p):
    if not xs:
        return float('nan')
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p * (len(xs) - 1)))))
    return xs[k]


def score_report(preds, detail, reps, names, lat_ms):
    rep_range = range(reps)
    summary = {}
    for arm in PP.ARMS:
        summary[arm] = {}
        for split in ('dev', 'test', 'all'):
            scores = [OE.summary(OE.score(preds[arm][r], split))['score'] for r in rep_range]
            summary[arm][split] = {'scores': scores, 'mean': sum(scores) / len(scores)}
    latency = {arm: {'mean_ms': sum(lat_ms[arm]) / len(lat_ms[arm]) if lat_ms[arm] else float('nan'),
                     'p95_ms': pctl(lat_ms[arm], 0.95), 'max_ms': max(lat_ms[arm]) if lat_ms[arm] else float('nan')}
               for arm in PP.ARMS}
    rows_by_conv = {fn: rows for fn, rows in OE.group_questions_by_conversation() if fn in names}
    paired_vs_base = {}
    for arm in PP.ARMS:
        if arm == 'base':
            continue
        paired_vs_base[arm] = {split: paired(preds['base'][:reps], preds[arm][:reps], split, names)
                               for split in ('dev', 'test', 'all')}
        paired_vs_base[arm]['sign_test'] = sign_test(preds['base'][:reps], preds[arm][:reps],
                                                      names, rows_by_conv)
        paired_vs_base[arm]['fire_counts'] = fire_counts(detail[arm][:reps])
    return {'summary': summary, 'paired_vs_base': paired_vs_base, 'latency': latency,
            'reps': reps, 'n_conv': len(names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--out-dir', default='out/fr3b')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    run(a.reps, a.out_dir, a.limit)


if __name__ == '__main__':
    main()
