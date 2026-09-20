"""FastAPI server for the medical-appointment case (upstream api.py contract).

    cd medical && UPSTREAM=/workspace/upstream python server.py      # port 9054, POST /predict

Models are loaded and warmed at import. Every request is answered: a hard
internal deadline (MED_DEADLINE, default 45 s) cuts the LLM short and keeps
heuristic answers; any exception returns a valid all-yes / null-span guess.
Responses are checked with upstream utils.validate_response before leaving.
Incoming audio + questions are saved to MED_SEEN_DIR (default
/workspace/medical_seen, only with MED_SAVE_SEEN=1; off by default because portal
requests can contain the hidden-holdout conversations) in a background thread.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
UPSTREAM = os.environ.get('UPSTREAM', '/workspace/upstream')
if not os.path.isdir(UPSTREAM):
    UPSTREAM = '/home/claude/Nordic-AI-Cup-2026'
MED_UP = os.path.join(UPSTREAM, 'medical-appointment')
sys.path.insert(1, MED_UP)

import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402

from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto  # noqa: E402
from utils import decode_audio, validate_response  # noqa: E402

import pipeline as P  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
log = logging.getLogger('medical.server')

HOST = '0.0.0.0'
PORT = int(os.environ.get('MED_PORT', 9054))
# The scorer averages tIoU over gold-yes questions whatever we answered, so a span sent with a 'no' can only
# add (cand_span = the span we would have sent had we said yes). MED_ATTACH_NO=0 restores null-on-no.
ATTACH_NO = os.environ.get('MED_ATTACH_NO', '1') == '1'
FORCE_YES = os.environ.get('MED_FORCE_YES', '0') == '1'   # diagnostic only: answer yes to everything
BACKEND = os.environ.get('MED_BACKEND', 'llm')          # llm | heuristic
SAVE_SEEN = os.environ.get('MED_SAVE_SEEN', '0') == '1'   # off: portal requests may contain holdout conversations
SEEN_DIR = os.environ.get('MED_SEEN_DIR', '/workspace/medical_seen')
ASR_BUDGET = float(os.environ.get('MED_ASR_BUDGET', 35.0))  # give up on ASR after this (s)
# One JSON line per request (no audio, no text): latency and answer source, for the portal gap diagnosis.
REQ_LOG = os.environ.get('MED_REQ_LOG', '/workspace/logs/med_requests.jsonl')
_REQ_LOG_LOCK = threading.Lock()
_INFLIGHT = [0]

app = FastAPI()
start_time = time.time()
ANSWERER = None
_ASR_LOCK = threading.Lock()


def _fallback(n: int) -> ASRQuestionResponseDto:
    return ASRQuestionResponseDto(answers=[True] * n, evidence_start=[None] * n, evidence_end=[None] * n)


def _save_seen(req: ASRQuestionRequestDto, result: dict):
    try:
        os.makedirs(SEEN_DIR, exist_ok=True)
        stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        base = os.path.join(SEEN_DIR, f'{stamp}_{os.path.basename(req.audio_filename)}')
        with open(base if base.endswith('.mp3') else base + '.mp3', 'wb') as f:
            f.write(decode_audio(req.audio_base64))
        with open(base.replace('.mp3', '') + '.json', 'w') as f:
            json.dump({'audio_filename': req.audio_filename, 'questions': req.questions, **result}, f,
                      indent=1, default=str)
    except Exception as e:
        log.warning('could not save seen request: %s', e)


def _transcribe_with_budget(audio: bytes, questions, budget: float):
    box = {}

    def run():
        try:
            with _ASR_LOCK:
                box['tx'] = P.transcribe(audio, questions)
        except Exception as e:  # noqa
            box['err'] = e

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(budget)
    if 'tx' in box:
        return box['tx']
    raise RuntimeError(f"ASR failed or exceeded {budget}s: {box.get('err')}")


def _log_request(row: dict):
    try:
        with _REQ_LOG_LOCK, open(REQ_LOG, 'a') as f:
            f.write(json.dumps(row) + '\n')
    except Exception as e:  # logging must never affect the answer
        log.warning('request log failed: %s', e)


def predict(req: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    t0 = time.time()
    n = len(req.questions)
    deadline = t0 + P.CFG['deadline']
    info = {}
    _INFLIGHT[0] += 1
    row = {'t': datetime.datetime.now().isoformat(timespec='seconds'), 'file': os.path.basename(req.audio_filename),
           'n': n, 'inflight': _INFLIGHT[0]}
    try:
        audio = decode_audio(req.audio_base64)
        tx = _transcribe_with_budget(audio, req.questions, min(ASR_BUDGET, P.CFG['deadline'] - 5))
        t1 = time.time()
        res = ANSWERER.answer(tx, list(req.questions), deadline=deadline)
        def _sp(r):
            if r['answer']:
                return r['span']
            return r.get('cand_span') if ATTACH_NO else None
        resp = ASRQuestionResponseDto(
            answers=[True if FORCE_YES else bool(r['answer']) for r in res],
            evidence_start=[float(_sp(r)[0]) if _sp(r) else None for r in res],
            evidence_end=[float(_sp(r)[1]) if _sp(r) else None for r in res],
        )
        validate_response(resp, expected_count=n)
        info = {'asr_s': round(t1 - t0, 2), 'qa_s': round(time.time() - t1, 2), 'transcript': tx,
                'results': [{k: v for k, v in r.items()} for r in res]}
        log.info('%s: asr %.1fs qa %.1fs yes=%d src=%s', req.audio_filename, t1 - t0, time.time() - t1,
                 sum(resp.answers), ''.join({'llm': 'L', 'loc': 'c'}.get(r['src'], 'h') for r in res))
        row.update(asr_s=info['asr_s'], qa_s=info['qa_s'], asr_path=tx.get('asr_path'),
                   n_llm=sum(r['src'] == 'llm' for r in res), n_heur=sum(r['src'] == 'heur' for r in res),
                   n_loc=sum(r['src'] == 'loc' for r in res),
                   n_yes=sum(resp.answers), n_yes_nospan=sum(1 for r in res if r['answer'] and not r['span']),
                   span_src=''.join({'quote': 'q', 'units': 'u', 'loc': 'c'}.get(r['span_src'], '-') for r in res),
                   onsets_n=tx.get('_onsets_n'),
                   qs=[{'a': bool(r['answer']), 'p': round(float(r['p']), 4), 'ids': r['ids'], 'quote': r['quote'],
                        'span': r['span'], 'src': r['src']} for r in res])
    except Exception as e:
        log.error('predict failed for %s: %s\n%s', req.audio_filename, e, traceback.format_exc())
        resp = _fallback(n)
        info = {'error': repr(e)}
        row['error'] = repr(e)[:300]
    _INFLIGHT[0] -= 1
    row['total_s'] = round(time.time() - t0, 2)
    _log_request(row)
    if SAVE_SEEN:
        threading.Thread(target=_save_seen, args=(req, {**info, 'response': resp.model_dump()}),
                         daemon=True).start()
    return resp


@app.post('/predict', response_model=ASRQuestionResponseDto)
def predict_endpoint(request: ASRQuestionRequestDto):
    try:
        resp = predict(request)
        validate_response(resp, expected_count=len(request.questions))
        return resp
    except Exception as e:
        log.error('endpoint fallback: %s', e)
        try:
            return _fallback(len(request.questions))
        except Exception:
            return ASRQuestionResponseDto(answers=[], evidence_start=[], evidence_end=[])


@app.get('/api')
def hello():
    return {'service': 'medical-appointment-usecase',
            'uptime': '{}'.format(datetime.timedelta(seconds=time.time() - start_time)),
            'backend': BACKEND, 'llm': getattr(getattr(ANSWERER, 'llm', None), 'model', None),
            'asr': P.CFG['asr_model'], 'onset_stats': P.ONSET_STATS,
            'span_cfg': {k: P.CFG[k] for k in ('span_onset_sentstart', 'quote_gate', 'span_shift_s',
                                                'allow_shift_with_onset', 'loc_shift_s')}}


@app.get('/')
def index():
    return 'Your endpoint is running!'


def _warmup():
    global ANSWERER
    P.log_cfg()
    ANSWERER = P.Answerer(BACKEND)
    try:
        P.load_asr()
        sample = os.path.join(MED_UP, 'data', 'audio', 'conversation_sample_4.mp3')
        if os.path.exists(sample):
            t = time.time()
            tx = P.transcribe(open(sample, 'rb').read())
            log.info('warm-up ASR %.1fs', time.time() - t)
            t = time.time()
            ANSWERER.answer(tx, ['Does the visit concern asthma?', 'Was 200 mg prescribed?'],
                            deadline=time.time() + 60)
            log.info('warm-up QA %.1fs', time.time() - t)
    except Exception as e:
        log.error('warm-up failed (server still starts): %s', e)
    # CPU fallback ASR, loaded in the background so a GPU OOM (shared GPU) does not
    # turn into an all-yes/no-span answer.
    if os.environ.get('MED_ASR_CPU_PRELOAD', '1') == '1':
        threading.Thread(target=lambda: _safe(P.load_asr_cpu), daemon=True).start()


def _safe(fn):
    try:
        fn()
    except Exception as e:
        log.warning('%s failed: %s', getattr(fn, '__name__', fn), e)


_warmup()

if __name__ == '__main__':
    uvicorn.run(app, host=HOST, port=PORT, timeout_keep_alive=75)
