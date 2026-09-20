"""APerquesti-g1-2: verdict-first decoding (MED_ORDER=aeq) vs production (eqa), paired.

run (POD=gpu, light client of the shared vLLM :8001):
    UPSTREAM=/workspace/upstream MED_TX_DIR=/workspace/tx/large-v3-turbo \\
        python3 aeq_run.py run --reps 2 --out out/aeq [--items analysis/think_probe_items.json]
  Arms are interleaved per conversation (base then aeq, same pass). Production config otherwise
  (onset start rule, quote_gate=any, pipeline defaults). Per question we keep p (P(yes)),
  cand_span (the span we would emit on yes), src and raw reply, so the yes threshold can be
  re-picked per arm on DEV afterwards (critique (1): answer-first shifts the P(yes) distribution).
  --items restricts to the probe questions (subset probe).

score (anywhere with the upstream CSV):
    python3 aeq_run.py score out/aeq [--items analysis/think_probe_items.json]
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

ARMS = ('eqa', 'aeq')          # default; --arms eqa,aeq,aeqmc adds the minimal-clause arm (g2-1)
BASE = 'eqa'
THR_GRID = [round(0.15 + 0.01 * i, 2) for i in range(26)]   # 0.15 .. 0.40
PROD_THR = 1 / (1 + math.exp(1.2))                            # yes_bias 1.2 -> 0.2315


def cmd_run(a):
    import offline_eval as OE
    import pipeline as P
    import spans as S
    from faster_whisper.audio import decode_audio
    tx_dir = os.environ.get('MED_TX_DIR', '/workspace/tx/large-v3-turbo')
    audio_dir = os.path.join(os.environ.get('UPSTREAM', '/workspace/upstream'), 'medical-appointment/data/audio')
    conv = {fn: [r['question'] for r in rows] for fn, rows in OE.group_questions_by_conversation()}
    sel = None
    if a.items:
        sel = {}
        for it in json.load(open(a.items)):
            sel.setdefault(it['fn'], []).append(it['qi'])
    names = sorted(fn for fn in conv if os.path.exists(os.path.join(tx_dir, fn.replace('.mp3', '.json'))))
    if sel is not None:
        names = [fn for fn in names if fn in sel]
    arms = tuple(a.arms.split(','))
    assert arms[0] == BASE
    os.makedirs(a.out, exist_ok=True)
    ans = P.Answerer('llm')
    if ans.llm is None:
        raise SystemExit('LLM unavailable')
    import hashlib
    meta = {'arms': arms, 'reps': a.reps, 'cfg': {k: v for k, v in P.CFG.items() if k != 'order'},
            'md5': {f: hashlib.md5(open(os.path.join(HERE, f), 'rb').read()).hexdigest()
                    for f in ('pipeline.py', 'spans.py', 'offline_eval.py')}}
    print(json.dumps(meta, default=str), flush=True)
    print(f'{len(names)} conversations x {len(arms)} arms x {a.reps} reps; quote_gate={P.CFG["quote_gate"]} '
          f'onset={P.CFG["span_onset_sentstart"]}', flush=True)
    rec = {arm: [dict() for _ in range(a.reps)] for arm in arms}
    rec['_meta'] = meta
    t0 = time.time()
    for rep in range(a.reps):
        for n, fn in enumerate(names):
            tx = dict(S.load_transcript(os.path.join(tx_dir, fn.replace('.mp3', '.json'))))
            try:
                tx['_pcm'] = decode_audio(os.path.join(audio_dir, fn), sampling_rate=16000)
            except Exception as e:
                print('no pcm', fn, e, flush=True)
            qidx = sel[fn] if sel is not None else list(range(len(conv[fn])))
            qs = [conv[fn][i] for i in qidx]
            order = arms if (rep + n) % 2 == 0 else arms[::-1]      # alternate who goes first
            for arm in order:
                P.CFG['order'] = arm
                t = time.time()
                res = ans.answer(dict(tx), qs, deadline=time.time() + P.CFG['deadline'])
                dt = time.time() - t
                rec[arm][rep][fn] = {'qidx': qidx, 'latency_s': dt,
                                     'rows': [{'p': r['p'], 'src': r['src'], 'cand_span': r['cand_span'],
                                               'span_src': r['span_src'], 'ids': r['ids'], 'quote': r['quote'],
                                               'raw': r['raw']} for r in res]}
            if (n + 1) % 5 == 0 or n == len(names) - 1:
                print(f'rep {rep} conv {n + 1}/{len(names)} elapsed {time.time() - t0:.0f}s lat ' +
                      ' '.join(f'{x} {rec[x][rep][fn]["latency_s"]:.1f}' for x in arms), flush=True)
        json.dump(rec, open(os.path.join(a.out, 'rec.json'), 'w'), default=str)
    print('done', time.time() - t0, flush=True)


# --------------------------------------------------------------------------- scoring
def to_preds(recs_rep, thr, n_by_fn):
    """Full-conversation prediction dicts (only for full runs)."""
    out = {}
    for fn, v in recs_rep.items():
        n = n_by_fn[fn]
        ans, st, en = [False] * n, [None] * n, [None] * n
        for qi, r in zip(v['qidx'], v['rows']):
            yes = r['p'] > thr if r['src'] == 'llm' else r['p'] > 0.5
            sp = r['cand_span']
            ans[qi] = bool(yes)
            if yes and sp:
                st[qi], en[qi] = sp[0], sp[1]
        out[fn] = {'answers': ans, 'evidence_start': st, 'evidence_end': en, 'latency_ms': v['latency_s'] * 1000}
    return out


def cmd_score(a):
    import offline_eval as OE
    from utils import evidence_interval, gold_evidence, temporal_iou
    rec = json.load(open(os.path.join(a.dir, 'rec.json')))
    rec.pop('_meta', None)
    for extra in (a.pool or []):             # pool reps from other runs (same code/CFG checked by caller)
        r2 = json.load(open(extra))
        r2.pop('_meta', None)
        for arm, reps_ in r2.items():
            if arm in rec:
                rec[arm] = rec[arm] + reps_     # appended: a new-only arm pairs with this run's eqa reps
    ARMS = tuple(x for x in ('eqa', 'aeq', 'aeqmc') if x in rec)
    # paired reps: arm x rep r is paired with eqa rep r; an arm with fewer reps uses reps 0..k-1
    rows_by = {fn: rows for fn, rows in OE.group_questions_by_conversation()}
    n_by = {fn: len(r) for fn, r in rows_by.items()}
    reps = len(rec['eqa'])
    nrep = {arm: len(rec[arm]) for arm in ARMS}
    groups = {}
    if a.items:
        for it in json.load(open(a.items)):
            groups[(it['fn'], it['qi'])] = it['group']
    rep_out = {'n_reps': nrep}

    def q_tiou(r, row, thr):
        yes = r['p'] > thr if r['src'] == 'llm' else r['p'] > 0.5
        if int(row['label']) != 1 or not yes or not r['cand_span']:
            return 0.0
        return temporal_iou(gold_evidence(row), evidence_interval(*r['cand_span']))

    # per-question stats at production threshold (accuracy / tIoU by bucket)
    for arm in ARMS:
        acc = [0, 0]
        buck = {}
        ps = []
        for rep in range(nrep[arm]):
            for fn, v in rec[arm][rep].items():
                for qi, r in zip(v['qidx'], v['rows']):
                    row = rows_by[fn][qi]
                    yes = r['p'] > PROD_THR if r['src'] == 'llm' else r['p'] > 0.5
                    acc[0] += int(yes == bool(int(row['label'])))
                    acc[1] += 1
                    ps.append(r['p'])
                    if int(row['label']) == 1:
                        g = groups.get((fn, qi), 'all_yes')
                        buck.setdefault(g, []).append(q_tiou(r, row, PROD_THR))
        rep_out[arm] = {'acc@0.23': acc[0] / acc[1], 'n_q': acc[1],
                        'tiou_by_bucket@0.23': {g: (round(sum(x) / len(x), 4), len(x)) for g, x in buck.items()},
                        'frac_p_in_0.05_0.95': sum(1 for p in ps if 0.05 < p < 0.95) / len(ps)}
        lat = [v['latency_s'] for rep in range(nrep[arm]) for v in rec[arm][rep].values()]
        rep_out[arm]['lat_mean_s'] = sum(lat) / len(lat)
        rep_out[arm]['lat_max_s'] = max(lat)
        # early-stop potential: share of reply chars after the ANSWER line on 'no' replies
        tail, tot = 0, 0
        for rep in range(nrep[arm]):
            for v in rec[arm][rep].values():
                for r in v['rows']:
                    raw = r['raw'] or ''
                    tot += len(raw)
                    if arm != 'eqa' and raw.strip().upper().startswith('ANSWER: NO'):
                        tail += len(raw) - len(raw.split('\n')[0])
        rep_out[arm]['no_tail_char_share'] = tail / tot if tot else 0.0
    if a.items:
        # per-item paired delta (mean over reps) + sign counts by group
        sgn = {}
        for key, g in groups.items():
            fn, qi = key
            row = rows_by[fn][qi]
            d = []
            for rep in range(reps):
                rb = rec['eqa'][rep].get(fn)
                rc = rec['aeq'][rep].get(fn)
                if not rb or not rc:
                    continue
                k = rb['qidx'].index(qi)
                d.append(q_tiou(rc['rows'][k], row, PROD_THR) - q_tiou(rb['rows'][k], row, PROD_THR))
            if d:
                m = sum(d) / len(d)
                s = sgn.setdefault(g, [0, 0, 0])
                s[0 if m > 1e-9 else (1 if m < -1e-9 else 2)] += 1
        rep_out['probe_sign_better_worse_same'] = sgn
        print(json.dumps(rep_out, indent=1))
        return

    # full run: threshold per arm on DEV (secondary contrast), fixed PROD_THR (primary, pre-registered)
    chosen = {}
    for arm in ARMS:
        best = None
        for thr in THR_GRID:
            sc = sum(OE.summary(OE.score(to_preds(rec[arm][r], thr, n_by), 'dev'))['score']
                     for r in range(nrep[arm])) / nrep[arm]
            if best is None or sc > best[1] + 1e-12:
                best = (thr, sc)
        chosen[arm] = best[0]
    rep_out['thr_dev'] = chosen
    splits = {sp: OE.split_ids(sp) for sp in ('dev', 'test', 'all')}
    for label, thrs in (('fixed0.23', {x: PROD_THR for x in ARMS}), ('devthr', chosen)):
        res = {}
        preds = {arm: [to_preds(rec[arm][r], thrs[arm], n_by) for r in range(nrep[arm])] for arm in ARMS}
        for arm in ARMS:
            k = nrep[arm]
            res[arm] = {sp: round(sum(OE.summary(OE.score(preds[arm][r], sp))['score'] for r in range(k)) / k, 4)
                        for sp in ('dev', 'test', 'all')}
            res[arm]['acc_all'] = round(sum(OE.summary(OE.score(preds[arm][r], 'all'))['accuracy'] for r in range(k)) / k, 4)
            res[arm]['tiou_all'] = round(sum(OE.summary(OE.score(preds[arm][r], 'all'))['tiou'] for r in range(k)) / k, 4)
        for arm in ARMS[1:]:
            k = min(nrep[arm], reps)                 # pair arm rep r with eqa rep r
            con = {'paired_reps': k}
            for sp, keep in splits.items():
                diffs = []
                for fn in rec[BASE][0]:
                    if keep is not None and fn not in keep:
                        continue
                    d = [OE.summary(OE.score({fn: preds[arm][r][fn]}))['score'] -
                         OE.summary(OE.score({fn: preds[BASE][r][fn]}))['score'] for r in range(k)]
                    diffs.append(sum(d) / len(d))
                n = len(diffs)
                m = sum(diffs) / n
                se = math.sqrt(sum((x - m) ** 2 for x in diffs) / (n - 1) / n)   # conversation-clustered
                # uq sign test on gold-yes tIoU (mean over paired reps)
                better = worse = 0
                for fn in rec[BASE][0]:
                    if keep is not None and fn not in keep:
                        continue
                    for qi, row in enumerate(rows_by[fn]):
                        if int(row['label']) != 1:
                            continue
                        dq = [q_tiou(rec[arm][r][fn]['rows'][qi], row, thrs[arm]) -
                              q_tiou(rec[BASE][r][fn]['rows'][qi], row, thrs[BASE]) for r in range(k)]
                        mq = sum(dq) / k
                        better += mq > 1e-9
                        worse += mq < -1e-9
                con[sp] = {'n': n, 'mean': round(m, 4), 'se': round(se, 4), 'z': round(m / se, 2) if se else None,
                           'uq_better_worse': [better, worse]}
            res[f'{arm}_vs_{BASE}'] = con
        rep_out[label] = res
    print(json.dumps(rep_out, indent=1))
    json.dump(rep_out, open(os.path.join(a.dir, 'report.json'), 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--reps', type=int, default=2)
    r.add_argument('--out', default='out/aeq')
    r.add_argument('--items', default='')
    r.add_argument('--arms', default=','.join(ARMS))
    s = sub.add_parser('score')
    s.add_argument('dir')
    s.add_argument('--items', default='')
    s.add_argument('--pool', nargs='*', help='extra rec.json files whose reps are appended (same code/CFG)')
    a = ap.parse_args()
    (cmd_run if a.cmd == 'run' else cmd_score)(a)


if __name__ == '__main__':
    main()
