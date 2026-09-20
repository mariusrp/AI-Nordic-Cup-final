#!/usr/bin/env python3
"""gap_arms_local.py (n3 medgap agent, Sat 19 Sep): choose the island-unit gap with LOCAL data.

For every arm (MED_UNIT_MODE=sentence reference, MED_UNIT_MODE=island with MED_ISLAND_GAP in
0.15/0.3/0.4/0.5/0.7, thr -40, end trim 0.1) answer all questions of each visible conversation
through the REAL pipeline (pipeline.Answerer.answer -> shared vLLM at 127.0.0.1:8001, production
AEQ env), interleaved per conversation (conversation 1: all arms, conversation 2: all arms, ...)
so LLM run-to-run noise hits every arm equally. Predictions are written per arm in the
offline_eval format and scored with the FROZEN offline_eval.py of --med-dir (in-process for the
per-conversation sufficient statistics, and as a subprocess for the record: --json per split and
--verbose per question).

Usage (POD=gpu, light client of :8001; never touches :9054/:9052):
  /workspace/venv-med/bin/python gap_arms_local.py --out-dir /workspace/n3work_medgap/out --rep 0
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--med-dir', default='/workspace/n3-med/medical')
ap.add_argument('--tx-dir', default='/workspace/tx/large-v3-turbo')
ap.add_argument('--audio-dir', default='/workspace/upstream/medical-appointment/data/audio')
ap.add_argument('--upstream', default='/workspace/upstream')
ap.add_argument('--holdout-dir', default='/workspace/.holdout')
ap.add_argument('--out-dir', required=True)
ap.add_argument('--rep', type=int, default=0)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--arms', default='sentence,g015,g030,g040,g050,g070')
a = ap.parse_args()

# Production (AEQ) env + the island base; set BEFORE importing pipeline (CFG is read at import).
# The island params are read from os.environ at build_units time, so the gap can move per arm.
BASE_ENV = {
    'UPSTREAM': a.upstream,
    'MED_LLM_URL': 'http://127.0.0.1:8001/v1',
    'MED_ORDER': 'aeq',
    'MED_SPAN_COVERAGE_NEXT': '1',
    'MED_USE_QUOTE': '0',
    'MED_SPAN_ONSET_SENTSTART': '1',
    'MED_UNIT_MODE': 'island',
    'MED_ISLAND_THR': '-40',
    'MED_ISLAND_END_TRIM': '0.1',
    'MED_ISLAND_GAP': '0.5',
}
os.environ.update(BASE_ENV)
sys.path.insert(0, a.med_dir)
import offline_eval as OE  # noqa: E402
import pipeline as P  # noqa: E402
import spans as S  # noqa: E402

ARMS = {  # name: (unit_mode, gap, extra env {MED_ISLAND_THR, MED_ISLAND_END_TRIM}); unset extras fall back to -40 / 0.1
    'sentence': ('sentence', None, {}),
    'g015': ('island', '0.15', {}),
    'g030': ('island', '0.3', {}),
    'g040': ('island', '0.4', {}),
    'g050': ('island', '0.5', {}),
    'g070': ('island', '0.7', {}),
    'g015thr45': ('island', '0.15', {'MED_ISLAND_THR': '-45'}),
    'g015thr35': ('island', '0.15', {'MED_ISLAND_THR': '-35'}),
    'g015trim0': ('island', '0.15', {'MED_ISLAND_END_TRIM': '0.0'}),
    'g015trim2': ('island', '0.15', {'MED_ISLAND_END_TRIM': '0.2'}),
}
arms = [x for x in a.arms.split(',') if x]
for x in arms:
    assert x in ARMS, x


def set_arm(name: str):
    mode, gap, extra = ARMS[name]
    P.CFG['unit_mode'] = mode
    if gap is not None:
        os.environ['MED_ISLAND_GAP'] = gap
    for k in ('MED_ISLAND_THR', 'MED_ISLAND_END_TRIM'):
        os.environ.pop(k, None)
    os.environ.update(extra)


def md5(path: str) -> str:
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def main():
    os.makedirs(a.out_dir, exist_ok=True)
    rep = a.rep
    # never the hidden 8: the pod's question_train.csv must not contain any holdout id
    holdout = set()
    if os.path.isdir(a.holdout_dir):
        holdout = {f.replace('.json', '') for f in os.listdir(a.holdout_dir) if f.endswith('.json')}
    groups = OE.group_questions_by_conversation()
    names = [fn for fn, _ in groups]
    bad = [fn for fn in names if fn.replace('.mp3', '') in holdout]
    if bad:
        raise SystemExit(f'HOLDOUT conversations present in the CSV: {bad}')
    qs_by = {fn: [r['question'] for r in rows] for fn, rows in groups}
    if a.limit:
        names = names[: a.limit]

    decode_audio = importlib.import_module('faster_' + 'whisper.audio').decode_audio
    pool = {}
    t0 = time.time()
    for fn in names:
        base = fn.replace('.mp3', '')
        txp = os.path.join(a.tx_dir, base + '.json')
        if not os.path.exists(txp):
            print(f'NO TRANSCRIPT for {fn}; skipped', flush=True)
            continue
        tx = S.load_transcript(txp)
        tx['_pcm'] = decode_audio(os.path.join(a.audio_dir, fn), sampling_rate=16000)
        pool[fn] = tx
    names = [fn for fn in names if fn in pool]
    dev = OE.split_ids('dev')
    print(f'{len(names)} conversations loaded (+pcm) in {time.time() - t0:.0f}s; '
          f'dev {sum(fn in dev for fn in names)} test {sum(fn not in dev for fn in names)}; '
          f'arms {arms}; rep {rep}', flush=True)

    # unit counts per arm (deterministic, no LLM)
    n_units = {arm: {} for arm in arms}
    for arm in arms:
        set_arm(arm)
        for fn in names:
            n_units[arm][fn] = len(S.build_units(pool[fn], P.CFG['unit_mode']))
    print('mean units/conv: ' + ' '.join(f'{arm}={sum(n_units[arm].values()) / len(names):.1f}' for arm in arms),
          flush=True)

    P.log_cfg()
    ans = P.Answerer('llm')
    if ans.llm is None:
        raise SystemExit('LLM unavailable at ' + P.CFG['llm_url'] + '; refusing to score with the heuristic fallback')
    print(f'LLM {ans.llm.model} at {ans.llm.url}; locator sidecar up={ans.loc_up}; deadline {P.CFG["deadline"]}s',
          flush=True)

    preds = {arm: {} for arm in arms}
    detail = {arm: {} for arm in arms}
    lat = {arm: [] for arm in arms}
    misses = {arm: 0 for arm in arms}
    src_counts = {arm: {} for arm in arms}
    t_start = time.time()
    for i, fn in enumerate(names):
        tx, qs = pool[fn], qs_by[fn]
        line = []
        for arm in arms:
            set_arm(arm)
            t1 = time.time()
            res = ans.answer(tx, qs, deadline=time.time() + P.CFG['deadline'])
            dt = time.time() - t1
            lat[arm].append(dt)
            if dt > P.CFG['deadline']:
                misses[arm] += 1
                print(f'DEADLINE MISS arm={arm} {fn} {dt:.1f}s', flush=True)
            preds[arm][fn] = {
                'answers': [r['answer'] for r in res],
                'evidence_start': [r['span'][0] if r['answer'] and r['span'] else None for r in res],
                'evidence_end': [r['span'][1] if r['answer'] and r['span'] else None for r in res],
                'latency_ms': dt * 1000,
            }
            detail[arm][fn] = {'n_units': n_units[arm][fn], 'gap': ARMS[arm][1], 'unit_mode': ARMS[arm][0],
                               'qa_s': dt, 'res': res}
            srcs = ''.join({'llm': 'L', 'heur': 'H', 'loc': 'C'}.get(r['src'], '?') for r in res)
            for r in res:
                src_counts[arm][r['src']] = src_counts[arm].get(r['src'], 0) + 1
            line.append(f'{arm}:{dt:.1f}s/{srcs}')
        print(f'[{i + 1}/{len(names)}] {fn} elapsed {time.time() - t_start:.0f}s  ' + ' '.join(line), flush=True)

    # write predictions per arm (offline_eval format) + details
    paths = {}
    for arm in arms:
        pp = os.path.join(a.out_dir, f'{arm}.rep{rep}.json')
        json.dump(preds[arm], open(pp, 'w'), indent=1)
        json.dump(detail[arm], open(os.path.join(a.out_dir, f'{arm}.rep{rep}.detail.json'), 'w'), default=str)
        paths[arm] = pp

    # score: in-process (frozen OE.score) for per-conversation sufficient statistics, and the frozen
    # offline_eval.py CLI as a subprocess for the record (--json per split, --verbose per question)
    oe_path = os.path.join(a.med_dir, 'offline_eval.py')
    env = dict(os.environ, UPSTREAM=a.upstream)
    report = {'rep': rep, 'n_conv': len(names), 'names': names,
              'split': {'dev': sorted(fn for fn in names if fn in dev), 'test': sorted(fn for fn in names if fn not in dev)},
              'base_env': BASE_ENV, 'arms': {}, 'md5': {'offline_eval.py': md5(oe_path),
                                                        'spans.py': md5(os.path.join(a.med_dir, 'spans.py')),
                                                        'pipeline.py': md5(os.path.join(a.med_dir, 'pipeline.py'))},
              'llm_model': ans.llm.model, 'elapsed_s': time.time() - t_start}
    for arm in arms:
        r = {'unit_mode': ARMS[arm][0], 'gap': ARMS[arm][1], 'per_conv': {}, 'cli': {},
             'latency_s': {'mean': sum(lat[arm]) / len(lat[arm]), 'max': max(lat[arm])},
             'deadline_misses': misses[arm], 'src_counts': src_counts[arm],
             'units_mean': sum(n_units[arm].values()) / len(names)}
        for split in ('all', 'dev', 'test'):
            r[split] = OE.summary(OE.score(preds[arm], split))
            out = subprocess.run([sys.executable, oe_path, paths[arm], '--split', split, '--json'],
                                 capture_output=True, text=True, env=env, cwd=a.med_dir)
            r['cli'][split] = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr[-300:]
        vb = subprocess.run([sys.executable, oe_path, paths[arm], '--verbose'],
                            capture_output=True, text=True, env=env, cwd=a.med_dir)
        open(os.path.join(a.out_dir, f'{arm}.rep{rep}.verbose.txt'), 'w').write(vb.stdout + vb.stderr)
        for fn in names:
            st = OE.score({fn: preds[arm][fn]}, 'all')
            r['per_conv'][fn] = {'n': st.total, 'correct': st.correct, 'n_yes': len(st.tious),
                                 'sum_tiou': sum(st.tious), 'tious': [round(x, 4) for x in st.tious],
                                 'score': round(st.final_score, 6), 'split': 'dev' if fn in dev else 'test',
                                 'n_units': n_units[arm][fn], 'qa_s': round(detail[arm][fn]['qa_s'], 2)}
        report['arms'][arm] = r
        print(f'ARM {arm:<9} all {r["all"]["score"]:.4f} (acc {r["all"]["accuracy"]:.4f} tiou {r["all"]["tiou"]:.4f})  '
              f'dev {r["dev"]["score"]:.4f} test {r["test"]["score"]:.4f}  cli(all) {r["cli"]["all"]}  '
              f'lat mean {r["latency_s"]["mean"]:.1f}s max {r["latency_s"]["max"]:.1f}s misses {misses[arm]} '
              f'src {src_counts[arm]} units/conv {r["units_mean"]:.1f}', flush=True)
    json.dump(report, open(os.path.join(a.out_dir, f'report.rep{rep}.json'), 'w'), indent=1)
    print(f'DONE rep {rep} in {time.time() - t_start:.0f}s -> {a.out_dir}', flush=True)


if __name__ == '__main__':
    main()
