"""Mock-LLM end-to-end check of a medical deploy config (no GPU, no LLM calls).

Runs the REAL pipeline code of a branch (pipeline.Answerer.answer, spans.py) the way server.py does after
transcription: transcript -> units -> LLM replies -> quote_span / onset rule / calibrate_span -> served span,
with the SERVING decode path for the onset rule (faster_whisper.audio.decode_audio(path, 16000), the exact call
in medical-fast-r6-2 pipeline.transcribe). The LLM is replaced by the cached replies (p, ids, quote, raw) of
fresh production-prompt runs, so every served span can be compared one-to-one with the zero-LLM replay
(fresh_stack.span_for) and scored with the frozen offline_eval.py against production on the SAME replies.

Env of the child process = the deploy config under test (CFG is read at import):
    CODE=<dir with the branch's pipeline.py + spans.py> MED_SPAN_ONSET_SENTSTART=1 MED_SPAN_SHIFT_S=0 \
    MED_QUOTE_GATE=any python3 deploy_check.py run <out.json>
    python3 deploy_check.py compare <out.json> <replay-arm>     # replay-arm: prod | onset+any | ...
Transcripts: $MED_UND_DIR/tx/large-v3-turbo (the cached turbo transcripts the runs used). Runs: $MED_UND_DIR/fr5.
"""
from __future__ import annotations

import concurrent.futures as cf
import glob
import json
import os
import sys
import time

WORK = os.environ.get('MED_UND_DIR', '/tmp/claude-0/-home-claude/323fa704-a9b4-5fa1-bde8-352ff3b28cce/scratchpad/med')
AUDIO = os.path.join(os.environ.get('UPSTREAM', '/home/claude/upstream-work'), 'medical-appointment', 'data', 'audio')


def runs():
    return [rp for rp in sorted(glob.glob(os.path.join(WORK, 'fr5', 'fr5_rep*', 'arm*.detail.json')))
            if not rp.endswith('arm2.detail.json')]


def run(out_path):
    code = os.environ['CODE']
    sys.path.insert(0, code)
    import pipeline as P  # noqa: E402  (CFG from this process's env)
    import spans as S  # noqa: F401,E402
    from faster_whisper.audio import decode_audio  # serving decode path
    print('CFG', {k: P.CFG[k] for k in ('span_shift_s', 'span_onset_sentstart', 'quote_gate', 'multi_fallback',
                                        'span_coverage_next', 'unit_mode') if k in P.CFG}, flush=True)
    A = P.Answerer.__new__(P.Answerer)
    A.llm = object()
    A.loc_up = False          # main (g4-2) locator sidecar: not used when every LLM reply is present
    A.pool = cf.ThreadPoolExecutor(max_workers=16)
    txs = {}
    res = {}
    for rp in runs():
        det = json.load(open(rp))
        tag = '/'.join(rp.split('/')[-2:])
        res[tag] = {}
        for fn, rows in det.items():
            if fn not in txs:
                tx = json.load(open(os.path.join(WORK, 'tx', 'large-v3-turbo', fn.replace('.mp3', '.json'))))
                tx['_pcm'] = decode_audio(os.path.join(AUDIO, fn), sampling_rate=16000)
                txs[fn] = tx
            qs = [r['question'] for r in rows] if rows and 'question' in rows[0] else None
            if qs is None:
                qs = QS[fn]
            cache = {q: (r['p'], r['ids'], r['quote'], r.get('raw', '')) for q, r in zip(qs, rows)}
            A._one = lambda units, q, timeout, _c=cache: _c[q]
            out = A.answer(dict(txs[fn]), qs, deadline=time.time() + 60)
            res[tag][fn] = [{'answer': o['answer'], 'span': o['span']} for o in out]
    json.dump(res, open(out_path, 'w'))
    print('wrote', out_path, sum(len(v) for v in res.values()), 'conversations x runs')


def compare(out_path, arm):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fresh_stack as FS  # the zero-LLM replay (r6-2 spans.py)
    served = json.load(open(out_path))
    cds = {'sentence': FS.conv_data(sorted(QS), 'sentence')}
    same = tot = ans_same = 0
    diffs = []
    for rp in runs():
        det = json.load(open(rp))
        tag = '/'.join(rp.split('/')[-2:])
        for fn, rows in det.items():
            for q, r, o in zip(QS[fn], rows, served[tag][fn]):
                ans_same += int(bool(r['answer']) == o['answer'])
                if not r['answer']:
                    continue
                sp, _ = FS.span_for(cds['sentence'][fn], r, q, arm)
                tot += 1
                srv = tuple(o['span']) if o['span'] else None
                ok = srv is not None and sp is not None and abs(sp[0] - srv[0]) < 0.011 and abs(sp[1] - srv[1]) < 0.011
                same += int(ok)
                if not ok and len(diffs) < 8:
                    diffs.append((tag, fn, q[:50], sp, srv))
    print(f'{out_path}: answers identical {ans_same}; served yes spans == replay[{arm}] {same}/{tot}')
    for d in diffs:
        print('   diff', d)


def score(paths):
    """Pooled paired offline_eval score of served configs vs the first path (se over conversations)."""
    import statistics as st
    sys.path.insert(0, os.path.join(WORK, 'r62'))
    import offline_eval as OE
    dev = OE.split_ids('dev')
    sv = [json.load(open(p)) for p in paths]
    tags = sorted(sv[0])
    fns = sorted(QS)
    per = [{fn: [] for fn in fns} for _ in paths]
    for k, s in enumerate(sv):
        for tag in tags:
            for fn in fns:
                o = s[tag][fn]
                pred = {'answers': [x['answer'] for x in o],
                        'evidence_start': [x['span'][0] if x['answer'] and x['span'] else None for x in o],
                        'evidence_end': [x['span'][1] if x['answer'] and x['span'] else None for x in o]}
                per[k][fn].append(OE.score({fn: pred}, 'all').final_score)
    for k in range(1, len(paths)):
        line = f'{os.path.basename(paths[k])} vs {os.path.basename(paths[0])}:'
        for half in ('dev', 'test', 'all'):
            hf = [fn for fn in fns if half == 'all' or ((fn in dev) == (half == 'dev'))]
            cs = [sum(a - b for a, b in zip(per[k][fn], per[0][fn])) for fn in hf]
            n = len(tags) * len(hf)
            m = sum(cs) / n
            se = st.stdev(cs) * len(hf) ** 0.5 / n
            line += f' {half} {m:+.4f}+-{se:.4f} ({m / se:+.1f} se)'
        print(line)
        absm = sum(sum(per[k][fn]) for fn in fns) / (len(tags) * len(fns))
        print(f'   mean absolute score {absm:.4f} (base {sum(sum(per[0][fn]) for fn in fns) / (len(tags) * len(fns)):.4f})')


def _qs():
    import csv
    out = {}
    for r in csv.DictReader(open(os.path.join(os.path.dirname(AUDIO), 'question_train.csv'))):
        out.setdefault(f"conversation_{r['transcript_id']}.mp3", []).append(r['question'])
    return out


QS = _qs()

if __name__ == '__main__':
    if sys.argv[1] == 'run':
        run(sys.argv[2])
    elif sys.argv[1] == 'compare':
        compare(sys.argv[2], sys.argv[3])
    else:
        score(sys.argv[2:])
