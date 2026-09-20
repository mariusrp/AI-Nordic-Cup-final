"""Medical-appointment pipeline: ASR -> numbered transcript units -> LLM verifier
(one short call per question, run in parallel against a LOCAL vLLM server) ->
yes/no + evidence span. A lexical heuristic answerer is the fallback whenever the
LLM is unavailable, too slow or unparseable.

Everything is configured by environment variables (see CFG below).

CLI (offline experiments, no FastAPI):
    # answer on cached transcripts (LLM only), write predictions + timing
    python pipeline.py --tx-dir /workspace/tx/large-v3-turbo --out preds.json
    # full run incl. ASR on the 39 training mp3s
    python pipeline.py --audio-dir $UPSTREAM/medical-appointment/data/audio --out preds.json \
        --save-tx /workspace/tx/large-v3-turbo
    # heuristic only (CPU)
    python pipeline.py --tx-dir ~/work/med/tx_small.en --backend heuristic --out h.json
then: python offline_eval.py preds.json
"""
from __future__ import annotations

import concurrent.futures as cf
import gc
import json
import logging
import math
import os
import re
import tempfile
import time
import urllib.parse
from typing import Dict, List, Optional, Sequence, Tuple

import locator as L
import spans as S

log = logging.getLogger('medical.pipeline')


def _env(name, default, cast=str):
    v = os.environ.get(name)
    if v is None or v == '':
        return default
    try:
        return cast(v)
    except Exception:
        return default


CFG = {
    'asr_model': _env('MED_ASR_MODEL', 'large-v3-turbo'),
    'asr_device': _env('MED_ASR_DEVICE', 'auto'),
    'asr_compute': _env('MED_ASR_COMPUTE', 'auto'),
    'asr_beam': _env('MED_ASR_BEAM', 5, int),
    'asr_hotwords': _env('MED_ASR_HOTWORDS', 0, int),   # bias ASR with rare words from the questions
    'llm_url': _env('MED_LLM_URL', 'http://127.0.0.1:8001/v1'),
    'llm_model': _env('MED_LLM_MODEL', ''),              # '' = ask the server (/v1/models)
    'llm_think': _env('MED_LLM_THINK', 0, int),          # 1 = enable thinking (slower)
    'llm_max_tokens': _env('MED_LLM_MAX_TOKENS', 160, int),
    'llm_timeout': _env('MED_LLM_TIMEOUT', 30.0, float),
    # Added to logit(p_yes) before thresholding at 0. Scoring favours yes: answering
    # yes is right in expectation when p_yes > 0.4/(0.8+1.2*E[tIoU]) ~= 0.23 for
    # tIoU~0.8, i.e. logit bias ~ +1.2 for a calibrated p (EXPERIMENTS.md E3).
    'yes_bias': _env('MED_YES_BIAS', 1.2, float),
    'heur_bias': _env('MED_HEUR_BIAS', 3.0, float),      # same, for the heuristic fallback (tuned on dev split)
    'unit_mode': _env('MED_UNIT_MODE', 'island'),
    'use_quote': _env('MED_USE_QUOTE', 1, int),
    'deadline': _env('MED_DEADLINE', 45.0, float),
    # post-alignment span calibration (spans.calibrate_span; EXPERIMENTS.md E8, fitted on dev)
    'span_shift_s': _env('MED_SPAN_SHIFT_S', 0.2, float),
    'span_shift_e': _env('MED_SPAN_SHIFT_E', 0.0, float),
    'span_snap': _env('MED_SPAN_SNAP', 0, int),
    'span_snap_pause': _env('MED_SPAN_SNAP_PAUSE', 0.3, float),
    'span_clamp_len': _env('MED_SPAN_CLAMP_LEN', 0.0, float),
    'span_clamp_w': _env('MED_SPAN_CLAMP_W', 0.0, float),
    # No-LLM fallback for questions the LLM did not answer (dead server, per-question error or
    # timeout, deadline): CPU cross-encoder sidecar (loc_server.py) over the K lexically best
    # phrase runs (locator.lexical_topk / fallback_decide). Unreachable or out of time ->
    # heuristic answer as before. Knobs tuned on dev by loc_tune.py. '' disables.
    'loc_url': _env('MED_LOC_URL', 'http://127.0.0.1:9061'),
    'loc_k': _env('MED_LOC_K', 12, int),
    'loc_thr': _env('MED_LOC_THR', -2.5, float),
    'loc_alpha': _env('MED_LOC_ALPHA', 3.0, float),
    'loc_target': _env('MED_LOC_TARGET', 2.0, float),
    'loc_shift_s': _env('MED_LOC_SHIFT_S', 0.3, float),
    'loc_shift_e': _env('MED_LOC_SHIFT_E', 0.0, float),
    'loc_min_s': _env('MED_LOC_MIN_S', 5.0, float),      # only start the locator with >= this left
    'loc_reserve': _env('MED_LOC_RESERVE', 6.0, float),  # LLM phase ends this early when the sidecar is up
    # Onset start rule (spans.onset_sentstart; ON by default since medical-evolve-g1-1-onset,
    # verified +0.0147 (3.6 se) with quote_gate=any on 6 fresh runs). It REPLACES the global start
    # shift: when onsets are available the effective span_shift_s is 0.0 (onset + 0.2 replays
    # -0.0125 below production), unless MED_ALLOW_SHIFT_WITH_ONSET=1. When onsets are missing
    # (decode failed, or < onset_min_rate onsets per 10 s) the legacy span_shift_s (0.2) applies.
    # The bge fallback path keeps its own loc_shift_s and never gets the onset rule.
    'span_onset_sentstart': _env('MED_SPAN_ONSET_SENTSTART', 1, int),
    'allow_shift_with_onset': _env('MED_ALLOW_SHIFT_WITH_ONSET', 0, int),
    'span_onset_min_rate': _env('MED_SPAN_ONSET_MIN_RATE', 2.0, float),   # onsets per 10 s
    'span_onset_k': _env('MED_SPAN_ONSET_K', 4, int),
    'span_onset_off': _env('MED_SPAN_ONSET_OFF', -0.04, float),
    # quote gate (Allmquote-g4-2, folded from medical-evolve-g4-2): 'first' = legacy (quote must
    # be near the FIRST cited cluster), 'any' = near any cited cluster; ellipsis 1 = multi-cluster
    # '...' quotes keep the last aligned piece; multi_fallback = which cited cluster the unit
    # fallback uses (first|qoverlap|quoteoverlap|unit). gate defaults to 'any' (verified bug fix,
    # merged with the onset rule); the others stay legacy/off.
    'quote_gate': _env('MED_QUOTE_GATE', 'any'),
    'quote_ellipsis': _env('MED_QUOTE_ELLIPSIS', 0, int),
    'multi_fallback': _env('MED_MULTI_FALLBACK', 'first'),
    # FB6-2: coverage-next end extension (spans.coverage_next_end), off by default.
    'span_coverage_next': _env('MED_SPAN_COVERAGE_NEXT', 0, int),
    # APerquesti-g1-2: reply field order. 'eqa' = legacy EVIDENCE/QUOTE/ANSWER; 'aeq' = verdict
    # first (ANSWER/EVIDENCE/QUOTE), same prompt text otherwise. Parsing is order-agnostic.
    # 'aeqmc' = aeq + minimal-clause QUOTE wording (APerquesti-g2-1 secondary arm).
    'order': _env('MED_ORDER', 'eqa'),
    # fast-r3-aeqmc: QUOTE instruction wording, independent of order. '' = legacy;
    # 'minclause' = quote the minimal clause (with its subject and verb) that states the evidence.
    'quote_wording': _env('MED_QUOTE_WORDING', ''),
    # Citation self-consistency (fable session, Sat 19 Sep): the greedy reply's cited cluster varies between
    # runs (same config scored 0.7612 and 0.7517 on the portal). With sc_n > 1, sc_n-1 extra replies are
    # sampled at temperature sc_t in ONE request (n=...), and the first cited cluster is chosen by majority
    # vote (ties -> the greedy reply). P(yes) always comes from the greedy reply. 1 = off (production).
    'sc_n': _env('MED_SC_N', 1, int),
    'sc_t': _env('MED_SC_T', 0.7, float),
    # sc_mode: how the sc_n candidate citations are combined (medical-final lane, Sun 20 Sep).
    #   'cluster' = production: majority vote over the WHOLE first cluster, ties -> greedy reply;
    #   'island'  = per-line agreement: keep a cited line only when >= sc_agree of the sc_n candidates
    #               cite it (public evidence: N=3 with a 0.75 majority lifts evidence-alignment F1).
    #               Nothing kept -> the greedy reply's ids (never empty).
    'sc_mode': _env('MED_SC_MODE', 'cluster'),
    'sc_agree': _env('MED_SC_AGREE', 2, int),
    # Contraction-only verifier (medical-final lane, Sun 20 Sep): when the cited cluster is >= vc_min lines,
    # one extra structured call re-reads ONLY those lines and returns the first/last line that still carries
    # part of the asked fact. The result is accepted ONLY when it is a SUB-range of the cited cluster, so the
    # pass can shorten an over-long span but never move or extend it. 0 = off (production).
    'vc': _env('MED_VC', 0, int),
    'vc_min': _env('MED_VC_MIN', 2, int),
    # Two-pass structured evidence selection (n3 M2, Sat 19 Sep). Runs AFTER the production reply, only for
    # questions the production pass answers yes (P(yes) after yes_bias); the yes/no answer and P(yes) are
    # always the production pass's. 0 = off (production).
    #   1 = pass 1 lists ALL candidate unit ids that mention the questioned fact (vLLM structured output,
    #       response_format json_schema with an enum of the valid ids), union with the production citation,
    #       then pass 2 picks the evidence unit(s) with explicit annotation rules + hard examples;
    #   2 = pass 2 only, candidates = the production citation's clusters (cheap; fires when >= 2 clusters).
    # Pass 2 output replaces the cited ids, so the downstream span rules (onset, coverage-next) are unchanged.
    'two_pass': _env('MED_TWO_PASS', 0, int),
    'tp_max_cands': _env('MED_TP_MAX_CANDS', 8, int),
    'tp_examples': _env('MED_TP_EXAMPLES', 1, int),     # 0 = rules only, no hard examples in pass 2
    'tp_min_cands': _env('MED_TP_MIN_CANDS', 2, int),   # pass 2 runs only with >= this many candidate units
    # 'pick' = the span is the unit(s) pass 2 picked; 'cluster' = when the pick lies inside one of the production
    # citation's clusters, keep that whole cluster (pass 2 then only chooses the LOCATION, the extent stays
    # production's; gold spans cover 2+ islands in 63/195 cases).
    'tp_extent': _env('MED_TP_EXTENT', 'pick'),
    # Rulebook-guided evidence selection (med-rules lane, Sat 19 Sep night; medical/offline/RULEBOOK.md = the
    # annotation convention of the 39 training conversations' gold). 0 = off (production). Bit flags:
    #   1 = the verdict-first prompt carries the compact convention rules (first explicit statement, continuation
    #       clips of one statement, question clip before a bare yes/no, no acknowledgements) and a multi-clip
    #       example; the reply format is unchanged (ANSWER/EVIDENCE/QUOTE, parse_reply);
    #   2 = extent pass: after the reply, for questions answered yes, the LLM sees the island lines around the
    #       FIRST cited cluster (rules_win_before / rules_win_after lines) and returns the first/last line of the
    #       evidence span per the convention (vLLM structured output, enum of the window ids). The location stays
    #       the reply's; a pick that does not overlap the cited cluster falls back to the reply's ids.
    #   3 = both.
    'rules': _env('MED_RULES', 0, int),
    'rules_win_before': _env('MED_RULES_WIN_BEFORE', 3, int),
    'rules_win_after': _env('MED_RULES_WIN_AFTER', 5, int),
    'rules_p2_all': _env('MED_RULES_P2_ALL', 0, int),     # 1 = extent pass also for 'no' verdicts (attached spans)
    # rules prompt text: 1 = first version (over-extends: cites echoes/confirmations after the fact),
    # 2 = tight version (fewest lines; continuation only while the fact is unfinished; no echo/confirmation);
    # 3 = v2 + bare-answer rule only when nothing after the bare reply states the fact + no echo after a complete
    #     sentence.
    'rules_v': _env('MED_RULES_V', 1, int),
    # dual (with rules & 1): ALSO run the production prompt; the answer and P(yes) are the production reply's,
    # the LOCATION is the rules reply's first cited cluster, and when the two first clusters overlap the
    # production reply's ids (its extent) are kept.
    'rules_dual': _env('MED_RULES_DUAL', 0, int),
    # Continuation rule (med-rules lane; RULEBOOK section 2 "continuation clips"): when the span starts or ends
    # INSIDE a sentence (the previous / its last word has no . ? !), extend it over the rest of that sentence up
    # to the island that holds the first / last question content token the span lacks (only when such a token
    # exists there). Deterministic, zero latency. 0 = off (production).
    'span_cont': _env('MED_SPAN_CONT', 0, int),
    # Clause snap (fable session): when the verbatim quote aligns, widen the span to the clause unit(s)
    # (commas / pauses > span_clause_pause s) it overlaps, i.e. between the tight quote and the whole
    # cited sentence. 0 = off.
    'span_clause_snap': _env('MED_SPAN_CLAUSE_SNAP', 0, int),
    'span_clause_pause': _env('MED_SPAN_CLAUSE_PAUSE', 1.0, float),
    # Clause-level contraction (med-trim lane, Sun 20 Sep; RULEBOOK section 3 "sub-island spans", see
    # clause_trim below). Splits the finished span at coordinating boundaries and keeps the minimal run
    # of clauses that still holds every question content token the span matched. Contraction only, and
    # gated on a margin; 0 = off (production).
    #   1 = coordinating boundaries only (', and' / ', but' / ';' / ':'), 2 = + any comma,
    #   3 = + sentence ends (question+answer pairs inside one span).
    'span_trim': _env('MED_SPAN_TRIM', 0, int),
    'trim_margin': _env('MED_TRIM_MARGIN', 2, int),        # distinct matched q tokens kept minus dropped
    'trim_max_frac': _env('MED_TRIM_MAX_FRAC', 0.75, float),  # never remove more than this share of the span
    'trim_min_keep': _env('MED_TRIM_MIN_KEEP', 0.8, float),  # never leave a span shorter than this
    'trim_min_hits': _env('MED_TRIM_MIN_HITS', 2, int),    # need this many matched q tokens to fire at all
    'trim_min_drop': _env('MED_TRIM_MIN_DROP', 0.2, float),  # below this there is nothing worth cutting
    'trim_end_strict': _env('MED_TRIM_END_STRICT', 1, int),  # end cuts only at sentence/coordinating boundaries
}


# --------------------------------------------------------------------------- #
# ASR
# --------------------------------------------------------------------------- #

_ASR = None


def load_asr():
    global _ASR
    if _ASR is None:
        from faster_whisper import WhisperModel
        dev = CFG['asr_device']
        if dev == 'auto':
            try:
                import ctranslate2
                dev = 'cuda' if ctranslate2.get_cuda_device_count() > 0 else 'cpu'
            except Exception:
                dev = 'cpu'
        comp = CFG['asr_compute']
        if comp == 'auto':
            comp = 'float16' if dev == 'cuda' else 'int8'
        t = time.time()
        _ASR = WhisperModel(CFG['asr_model'], device=dev, compute_type=comp)
        log.info('ASR %s on %s/%s loaded in %.1fs', CFG['asr_model'], dev, comp, time.time() - t)
    return _ASR


def hotwords_from_questions(questions: Sequence[str]) -> Optional[str]:
    """Capitalised / rare words from the questions (drug names etc.), never numbers,
    so ASR spelling matches the question vocabulary without biasing doses."""
    words = []
    for q in questions:
        for i, w in enumerate(re.findall(r"[A-Za-z][A-Za-z\-]+", q)):
            if i > 0 and w[0].isupper() and w.lower() not in S.STOP and w not in words:
                words.append(w)
    return ' '.join(words[:30]) or None


_ASR_CPU = None


def load_asr_cpu():
    """CPU fallback ASR (used only when the GPU model fails, e.g. CUDA OOM because another
    job grabbed the shared GPU). Model: MED_ASR_CPU_MODEL (default small.en, int8)."""
    global _ASR_CPU
    if _ASR_CPU is None:
        from faster_whisper import WhisperModel
        t = time.time()
        _ASR_CPU = WhisperModel(os.environ.get('MED_ASR_CPU_MODEL', 'small.en'), device='cpu',
                                compute_type='int8', cpu_threads=int(os.environ.get('MED_ASR_CPU_THREADS', 8)))
        log.info('CPU fallback ASR loaded in %.1fs', time.time() - t)
    return _ASR_CPU


def _run_asr(model, path: str, questions: Sequence[str], beam: int) -> dict:
    kw = dict(language='en', word_timestamps=True, beam_size=beam,
              vad_filter=False, condition_on_previous_text=False)
    if CFG['asr_hotwords'] and questions:
        hw = hotwords_from_questions(questions)
        if hw:
            kw['hotwords'] = hw
    segs, info = model.transcribe(path, **kw)
    res = [{'start': s.start, 'end': s.end, 'text': s.text,
            'words': [{'s': w.start, 'e': w.end, 'w': w.word, 'p': w.probability}
                      for w in (s.words or [])]} for s in segs]
    return {'duration': info.duration, 'segments': res}


def transcribe(audio_bytes: bytes, questions: Sequence[str] = ()) -> dict:
    """GPU ASR; on a CUDA failure (OOM / broken context) reload the GPU model once and retry,
    then fall back to the CPU model so the request still gets real spans."""
    global _ASR
    with tempfile.NamedTemporaryFile(suffix='.mp3') as f:
        f.write(audio_bytes)
        f.flush()
        try:
            tx = {**_run_asr(load_asr(), f.name, questions, CFG['asr_beam']), 'asr_path': 'gpu'}
        except Exception as e:
            log.warning('GPU ASR failed (%s); reloading and retrying', e)
            _ASR = None
            gc.collect()
            tx = None
        if tx is None:
            try:
                tx = {**_run_asr(load_asr(), f.name, questions, CFG['asr_beam']), 'asr_path': 'gpu_retry'}
            except Exception as e:
                log.warning('GPU ASR retry failed (%s); CPU fallback', e)
                _ASR = None
                gc.collect()
                tx = None
        if tx is None:
            tx = {**_run_asr(load_asr_cpu(), f.name, questions, 1), 'asr_path': 'cpu'}
        if CFG['span_onset_sentstart']:
            try:
                from faster_whisper.audio import decode_audio
                tx['_pcm'] = decode_audio(f.name, sampling_rate=16000)
            except Exception as e:
                log.warning('onset decode failed (%s); onset rule will no-op this request', e)
        return tx


ONSET_STATS = {'requests': 0, 'fired': 0, 'no_pcm': 0, 'sparse': 0, 'error': 0}


def _request_onsets(tx: dict, dur) -> Optional[List[float]]:
    """Speech onsets for the onset start rule, or None (rule off / no PCM / too few onsets)
    -> the caller then uses the legacy start shift. Every None is logged and counted."""
    if not CFG['span_onset_sentstart']:
        return None
    ONSET_STATS['requests'] += 1
    pcm = tx.get('_pcm')
    if pcm is None:
        ONSET_STATS['no_pcm'] += 1
        log.warning('onset rule: no PCM for this request -> legacy shift_s=%.2f (stats %s)',
                    CFG['span_shift_s'], ONSET_STATS)
        return None
    try:
        ons = S.energy_onsets(pcm)
    except Exception as e:
        ONSET_STATS['error'] += 1
        log.warning('onset rule: energy_onsets failed (%s) -> legacy shift (stats %s)', e, ONSET_STATS)
        return None
    length = float(dur) if dur else len(pcm) / 16000.0
    if length <= 0 or len(ons) * 10.0 / length < CFG['span_onset_min_rate']:
        ONSET_STATS['sparse'] += 1
        log.warning('onset rule: %d onsets in %.0f s (< %.1f/10 s) -> legacy shift (stats %s)',
                    len(ons), length, CFG['span_onset_min_rate'], ONSET_STATS)
        return None
    ONSET_STATS['fired'] += 1
    return ons


def log_cfg():
    eff = 0.0 if CFG['span_onset_sentstart'] and not CFG['allow_shift_with_onset'] else CFG['span_shift_s']
    log.info('CFG resolved: onset=%d gate=%s multi_fallback=%s coverage_next=%d span_shift_s=%.2f '
             '(effective with onsets %.2f, without %.2f) loc_shift_s=%.2f unit_mode=%s',
             CFG['span_onset_sentstart'], CFG['quote_gate'], CFG['multi_fallback'],
             CFG['span_coverage_next'], CFG['span_shift_s'], eff, CFG['span_shift_s'],
             CFG['loc_shift_s'], CFG['unit_mode'])


# --------------------------------------------------------------------------- #
# Heuristic answerer (fallback + CPU baseline)
# --------------------------------------------------------------------------- #

_NUMW = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7,
         'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
         'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
         'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
         'seventy': 70, 'eighty': 80, 'ninety': 90, 'hundred': 100, 'half': 0.5,
         'once': 1, 'twice': 2, 'single': 1, 'a': None}


def numbers_in(text: str) -> set:
    out = set()
    t = text.lower().replace(',', '')
    for m in re.findall(r'\d+(?:\.\d+)?', t):
        out.add(float(m))
    toks = re.findall(r'[a-z]+', t)
    i = 0
    while i < len(toks):
        w = toks[i]
        if w in _NUMW and _NUMW[w] is not None and w not in ('half',):
            v = _NUMW[w]
            if i + 1 < len(toks) and toks[i + 1] in _NUMW and _NUMW[toks[i + 1]] and _NUMW[toks[i + 1]] < 10 and v >= 20:
                v += _NUMW[toks[i + 1]]
                i += 1
            if i + 1 < len(toks) and toks[i + 1] == 'hundred':
                v *= 100
                i += 1
            out.add(float(v))
        i += 1
    return out


def heuristic_answer(units: List[dict], question: str) -> Tuple[float, List[int], str]:
    """Returns (p_yes, cited unit ids, quote)."""
    q = [t for t in S.content_tokens(question)]
    if not q or not units:
        return 0.5, [], ''
    df: Dict[str, int] = {}
    utoks = [set(S.content_tokens(u['text'])) for u in units]
    for ts in utoks:
        for t in ts:
            df[t] = df.get(t, 0) + 1
    n = len(units)
    idf = {t: math.log((n + 1) / (df.get(t, 0) + 0.5)) for t in q}
    qn = numbers_in(question)
    best, bi = -1.0, (0, 0)
    for i in range(n):
        for j in (i, i + 1):
            if j >= n:
                continue
            ts = utoks[i] | utoks[j]
            cov = sum(idf[t] for t in set(q) if t in ts) / (sum(idf[t] for t in set(q)) + 1e-9)
            if j > i:
                cov -= 0.05
            if cov > best:
                best, bi = cov, (i, j)
    text = ' '.join(units[k]['text'] for k in range(bi[0], bi[1] + 1))
    tn = numbers_in(' '.join(units[k]['text'] for k in range(max(0, bi[0] - 1), min(n, bi[1] + 2))))
    num_ok = (not qn) or qn <= tn
    score = best - (0.35 if not num_ok else 0.0)
    p = 1 / (1 + math.exp(-(score - 0.55) * 8))
    ids = list(range(bi[0], bi[1] + 1))
    if bi[1] > bi[0]:  # keep the better single unit if one clearly dominates
        a = sum(1 for t in q if t in utoks[bi[0]])
        b = sum(1 for t in q if t in utoks[bi[1]])
        if a >= 2 * max(b, 1):
            ids = [bi[0]]
        elif b >= 2 * max(a, 1):
            ids = [bi[1]]
    return p, ids, ''


# --------------------------------------------------------------------------- #
# LLM (local vLLM, OpenAI-compatible)
# --------------------------------------------------------------------------- #

SYSTEM = """You check yes/no questions against an automatic transcript of a recorded consultation between a doctor and a patient. The transcript is split into numbered lines; speakers are not labelled. Speech recognition may misspell names (e.g. drug names), so match names by sound, but treat numbers, units, doses, frequencies, durations, dates, body parts, left/right, test results and who-did-what as exact.

Answer "yes" only if the conversation explicitly establishes the statement in the question, with every detail matching. Answer "no" if any detail differs (e.g. a different dose, drug, duration, body location, side, result or plan), if it was only suggested and then rejected, or if the topic never comes up. Tag questions ("..., right?", "didn't it?") ask whether the statement is true. A question with a negation ("Are there no signs of X?") is "yes" when the transcript says there are no signs of X.

Reply with exactly three lines and nothing else:
EVIDENCE: <the id(s) of the fewest lines that settle the question, e.g. 12 or 12,13; "none" if the topic never comes up>
QUOTE: <the shortest verbatim excerpt from those lines that answers the question, copied word for word, usually one clause>
ANSWER: <yes or no>

Example. Transcript lines:
[7] So what should I take?
[8] I will prescribe amoxicillin, 500 milligrams three times a day, for seven days.
[9] And take it with food.
Question: Should the antibiotic be taken for ten days?
EVIDENCE: 8
QUOTE: for seven days
ANSWER: no
Question: Is the dose 500 mg three times daily?
EVIDENCE: 8
QUOTE: 500 milligrams three times a day
ANSWER: yes
Question: Was the patient told to take the tablets with a meal?
EVIDENCE: 9
QUOTE: take it with food
ANSWER: yes"""



def _reorder_aeq(system: str) -> str:
    """SYSTEM with every EVIDENCE/QUOTE/ANSWER triple (format spec + examples) emitted verdict
    first: ANSWER, EVIDENCE, QUOTE. Wording is untouched."""
    lines = system.split('\n')
    out, i = [], 0
    while i < len(lines):
        if (i + 2 < len(lines) and lines[i].startswith('EVIDENCE:')
                and lines[i + 1].startswith('QUOTE:') and lines[i + 2].startswith('ANSWER:')):
            out += [lines[i + 2], lines[i], lines[i + 1]]
            i += 3
        else:
            out.append(lines[i])
            i += 1
    return '\n'.join(out)


SYSTEM_AEQ = _reorder_aeq(SYSTEM)
assert SYSTEM_AEQ != SYSTEM and SYSTEM_AEQ.count('ANSWER:') == SYSTEM.count('ANSWER:') == 4


# APerquesti-g2-1 (aeqmc arm): aeq plus the external "minimal clause with subject and verb"
# wording in the QUOTE format line only (examples untouched).
SYSTEM_AEQMC = SYSTEM_AEQ.replace('answers the question, copied word for word, usually one clause>',
                                  'answers the question, copied word for word: a minimal clause with its subject and verb>')
assert SYSTEM_AEQMC != SYSTEM_AEQ


_QW_OLD = 'answers the question, copied word for word, usually one clause>'
_QW_MINCLAUSE = ('answers the question, copied word for word: the minimal clause '
                 '(with its subject and verb) that states the evidence>')
assert SYSTEM.count(_QW_OLD) == 1


# --------------------------------------------------------------------------- #
# Rulebook-guided evidence selection (MED_RULES; medical/offline/RULEBOOK.md)
# --------------------------------------------------------------------------- #

_EV_OLD = ('EVIDENCE: <the id(s) of the fewest lines that settle the question, e.g. 12 or 12,13; '
           '"none" if the topic never comes up>')
_EV_RULES = ('EVIDENCE: <the id(s) of the consecutive line(s) that state the asked fact, chosen by the rules above, '
             'e.g. 12 or 12,13,14; "none" if the topic never comes up>')
assert SYSTEM.count(_EV_OLD) == 1

RULES_BLOCK = """How to choose EVIDENCE. Each line is one spoken clip; doctor and patient usually alternate. The evidence follows the annotators' convention:
- It is the passage that states the asked fact explicitly: usually the doctor's own statement, finding or plan; the patient's own words when the question is about what the patient reports, feels, asks or wants.
- Take the FIRST line that states the fact explicitly, not a later recap or repetition. A vague lead-up ("Let us go through your results") or a line that only asks about the topic is not the evidence.
- One statement often runs over several consecutive lines of the same speaker: cite every consecutive line that carries part of the asked fact, and stop when it is complete. A prescription "For the sinuses, / penicillin. / One million units, / four times daily, / for seven days." is one run of lines for any question about it. Keep a short opening fragment of the sentence ("So,", "For the pain,").
- When the answer is a bare "Yes.", "No.", "None." or a one-word reply that does not name its subject, cite the question line before it together with the reply. A self-contained answer ("No fever.", "No redness at all.") is cited alone.
- Do not cite what follows the fact: an acknowledgement or echo ("Good.", "All right.", "Exactly that."), a confirmation of a statement ("Yes, that is right."), an emphatic repeat, or the next topic.
- For a "no" answer, cite the passage about the same topic.

"""

_EX_OLD = SYSTEM_AEQ[SYSTEM_AEQ.index('Example. Transcript lines:'):]
_EX_RULES = """Example. Transcript lines:
[7] So what should I take?
[8] I will prescribe amoxicillin,
[9] 500 milligrams three times a day,
[10] for seven days.
[11] All right.
[12] Any fever at all?
[13] No.
[14] And take it with food.
Question: Should the antibiotic be taken for ten days?
ANSWER: no
EVIDENCE: 8,9,10
QUOTE: for seven days
Question: Is the dose 500 mg three times daily?
ANSWER: yes
EVIDENCE: 8,9,10
QUOTE: 500 milligrams three times a day
Question: Is the patient free of fever?
ANSWER: yes
EVIDENCE: 12,13
QUOTE: No.
Question: Was the patient told to take the tablets with a meal?
ANSWER: yes
EVIDENCE: 14
QUOTE: take it with food"""


def _rules_system(base: str) -> str:
    """base (aeq/eqa system) + the convention rules before the reply-format paragraph, the EVIDENCE line
    pointing at the rules, and the multi-clip example (verdict-first when base is aeq)."""
    s = base.replace(_EV_OLD, _EV_RULES)
    k = s.index('Reply with exactly three lines')
    s = s[:k] + RULES_BLOCK + s[k:]
    ex = _EX_RULES if base is SYSTEM_AEQ else _reorder_eqa(_EX_RULES)
    i = s.index('Example. Transcript lines:')
    return s[:i] + ex


def _reorder_eqa(text: str) -> str:
    """ANSWER/EVIDENCE/QUOTE triples -> EVIDENCE/QUOTE/ANSWER (legacy order)."""
    lines = text.split('\n')
    out, i = [], 0
    while i < len(lines):
        if (i + 2 < len(lines) and lines[i].startswith('ANSWER:')
                and lines[i + 1].startswith('EVIDENCE:') and lines[i + 2].startswith('QUOTE:')):
            out += [lines[i + 1], lines[i + 2], lines[i]]
            i += 3
        else:
            out.append(lines[i])
            i += 1
    return '\n'.join(out)


SYSTEM_AEQ_RULES = _rules_system(SYSTEM_AEQ)
assert SYSTEM_AEQ_RULES.count('ANSWER:') == 5 and 'How to choose EVIDENCE' in SYSTEM_AEQ_RULES

_EV_RULES2 = ('EVIDENCE: <the id(s) of the fewest consecutive lines that state the asked fact, chosen by the rules above, '
              'e.g. 12 or 12,13; "none" if the topic never comes up>')

RULES_BLOCK2 = """How to choose EVIDENCE. Each line is one spoken clip; doctor and patient usually alternate.
- Location: the FIRST line that states the asked fact explicitly, usually the doctor's own statement, finding or plan; the patient's own words when the question is about what the patient reports, feels, asks or wants. A vague lead-up or a line that only asks about the topic is not the evidence; a later recap is not either.
- Extent: usually ONE line. Add the next line only while the statement is unfinished: the sentence runs on into it ("I am sending a blood sample for" / "testing."), or a prescription or plan is spread over several lines (drug, dose, frequency, duration).
- Stop as soon as the fact is complete. Do not cite the lines after it: echoes or repeats ("Rest and" / "fluids."), confirmations ("Yes.", "Correct.", "Exactly."), comments, follow-up questions, or a new clause that begins with "and".
- Answers: when the answer line is only "Yes.", "No.", "None." or a single word, cite the question line together with it. When the answer itself states the fact ("No, no headaches at all.", "No rash anywhere."), cite only the answer line(s), not the question.

"""

_EX_RULES2 = """Example. Transcript lines:
[7] So what should I take?
[8] I will prescribe amoxicillin,
[9] 500 milligrams three times a day,
[10] for seven days.
[11] Seven days.
[12] Any fever at all?
[13] No.
[14] And take it with food.
[15] With food, right.
Question: Should the antibiotic be taken for ten days?
ANSWER: no
EVIDENCE: 8,9,10
QUOTE: for seven days
Question: Is the dose 500 mg three times daily?
ANSWER: yes
EVIDENCE: 8,9,10
QUOTE: 500 milligrams three times a day
Question: Is the patient free of fever?
ANSWER: yes
EVIDENCE: 12,13
QUOTE: No.
Question: Was the patient told to take the tablets with a meal?
ANSWER: yes
EVIDENCE: 14
QUOTE: take it with food"""


def _rules_system2(base: str) -> str:
    s = base.replace(_EV_OLD, _EV_RULES2)
    k = s.index('Reply with exactly three lines')
    s = s[:k] + RULES_BLOCK2 + s[k:]
    ex = _EX_RULES2 if base is SYSTEM_AEQ else _reorder_eqa(_EX_RULES2)
    i = s.index('Example. Transcript lines:')
    return s[:i] + ex


SYSTEM_AEQ_RULES2 = _rules_system2(SYSTEM_AEQ)
assert SYSTEM_AEQ_RULES2.count('ANSWER:') == 5 and 'Stop as soon as the fact is complete' in SYSTEM_AEQ_RULES2

# v3 = v2 with the two failure modes of v2 on the training set addressed: the bare-answer rule fired on a
# "No," clip whose sentence continues on the next line ("No," / "nothing new."), and complete statements were
# still extended over the echo/confirmation lines that follow them.
RULES_BLOCK3 = RULES_BLOCK2.replace(
    '- Stop as soon as the fact is complete. Do not cite the lines after it: echoes or repeats ("Rest and" / "fluids."), '
    'confirmations ("Yes.", "Correct.", "Exactly."), comments, follow-up questions, or a new clause that begins with "and".',
    '- Stop as soon as the fact is complete. A complete sentence that states the fact is the whole evidence: do not cite '
    'the lines after it, even when they repeat, question or confirm the same fact ("There is no rash." / "So no rash at '
    'all?" / "Correct." -> only the first line). Echoes ("Rest and" / "fluids."), comments and a new clause that begins '
    'with "and" are not cited either.').replace(
    '- Answers: when the answer line is only "Yes.", "No.", "None." or a single word, cite the question line together '
    'with it. When the answer itself states the fact ("No, no headaches at all.", "No rash anywhere."), cite only the '
    'answer line(s), not the question.',
    '- Answers: when the reply to a question is only a bare "Yes.", "No.", "None." or one word ("Fine.") and nothing '
    'after it states the fact, cite the question line together with it. When the reply goes on to state the fact, '
    'even on the next line ("No," / "no headaches at all."), cite only the reply lines, starting at "No,"; the '
    'question is not cited.')
assert RULES_BLOCK3 != RULES_BLOCK2 and RULES_BLOCK3.count('even on the next line') == 1 and 'So no rash' in RULES_BLOCK3


def _rules_system3(base: str) -> str:
    return _rules_system2(base).replace(RULES_BLOCK2, RULES_BLOCK3)


SYSTEM_AEQ_RULES3 = _rules_system3(SYSTEM_AEQ)
assert 'even on the next line' in SYSTEM_AEQ_RULES3

# v4 (medical-final lane, Sun 20 Sep) = v3 plus two bullets aimed at the two remaining RULEBOOK loss modes
# of v3 on the 39 training conversations (island-range exact 108/195: wrong location 17, over 32, under 28):
# a LOCATION tie-break (RULEBOOK section 1 "first explicit mention wins", including its exception) and a
# closing CHECK that is symmetric - it asks to drop a line that carries none of the fact AND to add the rest
# of the sentence when the fact is unfinished - so it pushes neither longer nor shorter on its own.
RULES_BLOCK4 = RULES_BLOCK3[:-1] + """- More than one passage: when several passages mention the asked fact, cite the one that states it explicitly and on its own; if several do, cite the earliest of them. A line that only raises the topic, asks about it, agrees with it, or recaps it later is not the evidence.
- Check before replying: every line you cite must carry part of the asked fact - drop any line that does not - and the cited lines together must state that fact in full; when the rest of the same sentence is on the next line, cite that line too.

"""


def _rules_system4(base: str) -> str:
    return _rules_system2(base).replace(RULES_BLOCK2, RULES_BLOCK4)


SYSTEM_AEQ_RULES4 = _rules_system4(SYSTEM_AEQ)
assert ('More than one passage' in SYSTEM_AEQ_RULES4 and 'Check before replying' in SYSTEM_AEQ_RULES4
        and 'even on the next line' in SYSTEM_AEQ_RULES4
        and SYSTEM_AEQ_RULES4.count('ANSWER:') == 5)

EXT_SYSTEM = """You mark the exact evidence span for a yes/no question about a recorded consultation between a doctor and a patient. The transcript is split into numbered lines; each line is one spoken clip (speakers are not labelled; doctor and patient usually alternate). A first reader located the evidence at the cited line(s). Choose the FIRST and the LAST line of the evidence span among the candidate lines, following the annotators' convention:
1. Keep the location: the span covers the statement of the asked fact at or next to the cited line(s); only fix where it starts and ends.
2. One statement often runs over consecutive clips of the same speaker ("For the sinuses, / penicillin. / One million units, / four times daily, / for seven days."): include every following clip that continues the sentence or carries part of the asked fact (drug, dose, frequency, duration, site; a value after its name), and stop when the fact is complete.
3. Include a short opening fragment of the same sentence right before it ("So,", "For the pain,", "Taken together,").
4. If the answer is a bare "Yes.", "No.", "None." or a one-word reply that does not name its subject, the span starts at the question line right before it. A self-contained answer ("No fever.", "No, nothing new.") starts at the answer itself; the question before it is not included.
5. Do not include what follows the fact: an acknowledgement or echo ("Good.", "All right.", "Exactly that.", "Penicillin."), a confirmation of a statement ("Yes, that is right."), an emphatic repeat ("Nothing like that at all."), or the next topic.
6. Most spans are a single line; about a third are 2 to 4 lines.
Reply with JSON only: {"first": <id>, "last": <id>}."""


def _ext_schema(valid: List[int]) -> dict:
    en = {'type': 'integer', 'enum': [int(i) for i in valid]}
    schema = {'type': 'object', 'properties': {'first': en, 'last': en},
              'required': ['first', 'last'], 'additionalProperties': False}
    return {'response_format': {'type': 'json_schema',
                                'json_schema': {'name': 'evidence_range', 'schema': schema, 'strict': True}}}


def rules_extent_ids(llm, units: List[dict], q: str, ids: List[int], timeout: float) -> Tuple[List[int], str]:
    """MED_RULES extent pass: (new ids, note). The first cited cluster (max_gap 2, as span_for_ids) is the
    location; the LLM picks first/last line inside a window around it. Any failure or a pick that does not
    overlap the cluster -> the reply's ids unchanged."""
    n = len(units)
    ok = [i for i in ids if isinstance(i, int) and 0 <= i < n]
    if not ok:
        return ids, 'ext=skip(no ids)'
    c = S.clusters(ok, 2)[0]
    a, b = c[0], c[-1]
    lo, hi = max(0, a - CFG['rules_win_before']), min(n - 1, b + CFG['rules_win_after'])
    win = list(range(lo, hi + 1))
    cited = ','.join(str(i) for i in c) if len(c) <= 3 else f'{a}-{b}'
    user = (build_user(units, q) + f'\n\nCited line(s): {cited}\n\nCandidate lines:\n'
            + '\n'.join(f"[{i}] {units[i]['text']}" for i in win))
    msgs = [{'role': 'system', 'content': EXT_SYSTEM}, {'role': 'user', 'content': user}]
    ch = llm.chat(msgs, max_tokens=24, timeout=timeout, logprobs=False, extra=_ext_schema(win))
    txt = re.sub(r'<think>.*?</think>', '', (ch.get('message') or {}).get('content') or '', flags=re.S).strip()
    try:
        obj = json.loads(txt)
        f, l = int(obj['first']), int(obj['last'])
    except Exception:
        return ids, f'ext=fail({txt[:40]})'
    if f > l:
        f, l = l, f
    if f not in win or l not in win:
        return ids, f'ext=out({f},{l})'
    if l < a or f > b:          # must overlap the cited cluster (location is the reply's)
        return ids, f'ext=nooverlap({f},{l} vs {a},{b})'
    return list(range(f, l + 1)), f'ext={f}-{l} (cited {a}-{b})'


VC_SYSTEM = """You trim the evidence span for a yes/no question about a recorded consultation between a doctor and a patient. A first reader cited the numbered lines below; each line is one spoken clip. The citation is in the right place - your ONLY job is to drop lines at its edges that do not belong to the evidence.

Keep a line when it carries part of the asked fact (the statement, finding, plan, or a value belonging to it: drug, dose, frequency, duration, site), or when it is the question line right before a bare "Yes."/"No."/"None." reply, or when the sentence of a kept line runs on into it.
Drop a line when it only acknowledges or echoes the fact ("Good.", "All right.", "Exactly that.", "Penicillin."), confirms it ("Yes, that is right."), repeats it for emphasis, asks about it again, or starts a different topic.
Keep every line when they all belong to the evidence. Never choose lines outside the cited range.
Reply with JSON only: {"first": <id>, "last": <id>}."""


def verify_contract_ids(llm, units: List[dict], q: str, ids: List[int], timeout: float) -> Tuple[List[int], str]:
    """MED_VC contraction-only verifier: (new ids, note). The LLM sees ONLY the cited cluster's lines and
    returns the first/last line that still carries the fact. The reply is accepted only when it is a
    SUB-range of that cluster, so the pass can shorten the span but never move or lengthen it."""
    n = len(units)
    ok = [i for i in ids if isinstance(i, int) and 0 <= i < n]
    if not ok:
        return ids, 'vc=skip(no ids)'
    c = S.clusters(ok, 2)[0]
    if len(c) < max(2, CFG['vc_min']):
        return ids, f'vc=skip(len {len(c)})'
    a, b = c[0], c[-1]
    user = (build_user(units, q) + '\n\nCited lines:\n'
            + '\n'.join(f"[{i}] {units[i]['text']}" for i in c))
    msgs = [{'role': 'system', 'content': VC_SYSTEM}, {'role': 'user', 'content': user}]
    ch = llm.chat(msgs, max_tokens=24, timeout=timeout, logprobs=False, extra=_ext_schema(list(c)))
    txt = re.sub(r'<think>.*?</think>', '', (ch.get('message') or {}).get('content') or '', flags=re.S).strip()
    try:
        obj = json.loads(txt)
        f, l = int(obj['first']), int(obj['last'])
    except Exception:
        return ids, f'vc=fail({txt[:40]})'
    if f > l:
        f, l = l, f
    if f < a or l > b:                      # contraction only
        return ids, f'vc=notsub({f},{l} vs {a},{b})'
    keep = [i for i in c if f <= i <= l]
    if not keep:
        return ids, f'vc=empty({f},{l})'
    if keep == list(c):
        return ids, 'vc=keep(all)'
    return keep, f'vc={f}-{l} (cited {a}-{b})'


def rules_system_prompt() -> str:
    aeq = CFG['order'] in ('aeq', 'aeqmc')
    if CFG.get('rules_v', 1) == 4:
        return SYSTEM_AEQ_RULES4 if aeq else _rules_system4(SYSTEM)
    if CFG.get('rules_v', 1) == 3:
        return SYSTEM_AEQ_RULES3 if aeq else _rules_system3(SYSTEM)
    if CFG.get('rules_v', 1) == 2:
        return SYSTEM_AEQ_RULES2 if aeq else _rules_system2(SYSTEM)
    return SYSTEM_AEQ_RULES if aeq else _rules_system(SYSTEM)


def _first_cluster(ids: List[int], n: int) -> Optional[List[int]]:
    ok = [i for i in ids if isinstance(i, int) and 0 <= i < n]
    return S.clusters(ok, 2)[0] if ok else None


def dual_ids(prod_ids: List[int], rules_ids: List[int], n: int) -> Tuple[List[int], str]:
    """MED_RULES_DUAL: location from the rules reply, extent from the production reply when they agree."""
    cp, cr = _first_cluster(prod_ids, n), _first_cluster(rules_ids, n)
    if cr is None:
        return prod_ids, 'dual=prod(no rules ids)'
    if cp is None:
        return rules_ids, 'dual=rules(no prod ids)'
    if not (cp[-1] < cr[0] or cp[0] > cr[-1]):
        return prod_ids, f'dual=prod(overlap {cp[0]}-{cp[-1]} ~ {cr[0]}-{cr[-1]})'
    return rules_ids, f'dual=rules({cr[0]}-{cr[-1]} vs prod {cp[0]}-{cp[-1]})'


def _widx(words: Sequence[dict], t: float, key: str) -> int:
    return min(range(len(words)), key=lambda k: abs(words[k][key] - t))


def _island_of(units: Sequence[dict], t: float) -> Optional[dict]:
    for u in units:
        if u['start'] - 0.01 <= t <= u['end'] + 0.01:
            return u
    return None


def cont_start(span, words: Sequence[dict], units: Sequence[dict], q: str, max_words: int = 25):
    """MED_SPAN_CONT, start side: span starts mid-sentence and the sentence's earlier words hold question content
    tokens the span lacks -> start at the island holding the earliest of them."""
    if span is None or not words or not q:
        return span
    s, e = float(span[0]), float(span[1])
    i = _widx(words, s, 's')
    if i == 0 or S.SENT_END.search(words[i - 1]['w'].strip()):
        return span
    k = i - 1
    while k > 0 and i - k <= max_words and not S.SENT_END.search(words[k - 1]['w'].strip()):
        k -= 1
    if i - k > max_words:
        return span
    j = _widx(words, e, 'e')
    miss = (set(S.content_tokens(q)) - set(S.content_tokens(' '.join(words[x]['w'] for x in range(i, j + 1))))) \
        & set(S.content_tokens(' '.join(words[x]['w'] for x in range(k, i))))
    if not miss:
        return span
    m = min(x for x in range(k, i) if set(S.content_tokens(words[x]['w'])) & miss)
    u = _island_of(units, words[m]['s'])
    return (min(u['start'], words[m]['s']) if u else words[m]['s'], e)


def cont_end(span, words: Sequence[dict], units: Sequence[dict], q: str, max_words: int = 25):
    """MED_SPAN_CONT, end side: span ends mid-sentence and the rest of the sentence holds question content tokens
    the span lacks -> end at the island holding the last of them."""
    if span is None or not words or not q:
        return span
    s, e = float(span[0]), float(span[1])
    j = _widx(words, e, 'e')
    if S.SENT_END.search(words[j]['w'].strip()):
        return span
    k = j + 1
    while k < len(words) and k - j <= max_words and not S.SENT_END.search(words[k]['w'].strip()):
        k += 1
    if k >= len(words) or k - j > max_words:
        return span
    i = _widx(words, s, 's')
    miss = (set(S.content_tokens(q)) - set(S.content_tokens(' '.join(words[x]['w'] for x in range(i, j + 1))))) \
        & set(S.content_tokens(' '.join(words[x]['w'] for x in range(j + 1, k + 1))))
    if not miss:
        return span
    m = max(x for x in range(j + 1, k + 1) if set(S.content_tokens(words[x]['w'])) & miss)
    u = _island_of(units, words[m]['s'])
    return (s, max(u['end'], words[m]['e']) if u else words[m]['e'])


# --------------------------------------------------------------------------- #
# Clause-level contraction (med-trim lane, Sun 20 Sep 2026)
# --------------------------------------------------------------------------- #
# RULEBOOK.md section 3: 26/195 golds are ONE CLAUSE of a coordinated island ("Your blood pressure is
# normal, and your foot status is normal."), and word-level trims lift the island-range oracle from
# 0.891 to 0.949. Every selection arm so far only chooses WHICH islands are cited, so the "over" loss
# mode stays at 53-56/195: the cited island CONTAINS the gold and runs past it. This pass splits the
# span at coordinating boundaries and keeps the minimal run of clauses that still holds every question
# content token the span matched. Six earlier edge-moving arms lost 0.014-0.040 on unseen data
# (LESSONS.md medical dead ends: padding, lexical trim, clause snap, end shifts, start refits), so this
# one is contraction-only and evidence-gated: it never extends, the kept run must hold EVERY question
# content token the span matched, the dropped clauses must hold clearly fewer of them (margin), the cut
# is capped, and any failed gate leaves the span untouched.
# 0 = off (production).

_TRIM_CONJ = {'and', 'but', 'or', 'so', 'then', 'while', 'although', 'because', 'which',
              'plus', 'also', 'nor', 'yet', 'whereas'}
_TRIM_ALNUM = re.compile(r'[^a-z0-9]')


def _trim_word(w: str) -> str:
    return _TRIM_ALNUM.sub('', w.strip().lower())


def trim_parts(words: Sequence[dict], i: int, j: int, mode: int) -> List[Tuple[int, int, str]]:
    """Inclusive word-index ranges of the clause parts of words[i..j], each with the KIND of the
    boundary that closes it ('sent', 'conj', 'comma', or 'end' for the last part).

    mode 1: coordinating boundaries only  (', and' / ', but' / ', so' / ', then' / ';' / ':')
    mode 2: 1 + any comma
    mode 3: 2 + sentence ends (question+answer pairs inside one span)
    A part is never shorter than 2 words.
    """
    parts: List[Tuple[int, int, str]] = []
    a = i
    for x in range(i, j):
        t = words[x]['w'].strip()
        kind = ''
        if t.endswith((';', ':')):
            kind = 'conj'
        elif t.endswith(','):
            if _trim_word(words[x + 1]['w']) in _TRIM_CONJ:
                kind = 'conj'
            elif mode >= 2:
                kind = 'comma'
        elif mode >= 3 and S.SENT_END.search(t):
            kind = 'sent'
        if kind and x - a >= 1:
            parts.append((a, x, kind))
            a = x + 1
    parts.append((a, j, 'end'))
    return parts


def clause_trim(span, words: Sequence[dict], q: str, mode: int = 1, margin: int = 2,
                max_frac: float = 0.6, min_keep: float = 0.8, min_hits: int = 2,
                min_drop: float = 0.2, end_strict: int = 1):
    """MED_SPAN_TRIM: contract a cited span to the clause that answers the question.

    Returns (span, note); `span` is unchanged (and note '') whenever a gate fails. The kept range is
    the minimal contiguous run of clause parts that holds EVERY question content token present in the
    span (so a dropped clause can never hold question content the kept run lacks); it is accepted only
    when the dropped parts hold at least `margin` fewer distinct matched tokens than the kept run, the
    end cut sits on a real boundary (end_strict), and the trim removes between `min_drop` seconds and
    `max_frac` of the span while leaving >= `min_keep` seconds. Never extends either edge.

    end_strict (default on): the END may only be cut at a sentence end or a coordinating boundary
    (', and' / ', but' / ';'), never at a plain list comma - a comma inside an enumeration continues
    the same statement instead of ending it (RULEBOOK section 2: "And fluconazole, / 50 milligrams, /
    for seven days, / for the mouth." is ONE gold). Measured over the cached base reps: every end cut
    at a list comma lost tIoU, every end cut at a sentence or coordinating boundary won it.
    """
    if span is None or not words or not q or not mode:
        return span, ''
    s, e = float(span[0]), float(span[1])
    i, j = _widx(words, s, 's'), _widx(words, e, 'e')
    if j - i < 2:
        return span, ''
    parts = trim_parts(words, i, j, mode)
    if len(parts) < 2:
        return span, ''
    qt = set(S.content_tokens(q))
    ptoks = [set(S.content_tokens(' '.join(words[x]['w'] for x in range(a, b + 1)))) for a, b, _ in parts]
    matched = qt & set().union(*ptoks)
    if len(matched) < min_hits:
        return span, ''
    best = None
    for a in range(len(parts)):
        cov: set = set()
        for b in range(a, len(parts)):
            cov |= (ptoks[b] & matched)
            if cov == matched:
                key = (b - a, round(words[parts[b][1]]['e'] - words[parts[a][0]]['s'], 3), a)
                if best is None or key < best[0]:
                    best = (key, a, b)
                break
    if best is None:
        return span, ''
    _, a, b = best
    # an end cut is only allowed at a sentence end or a coordinating boundary; a plain list comma
    # continues the same statement, so hand the end back up to the next real boundary
    while end_strict and b < len(parts) - 1 and parts[b][2] == 'comma':
        b += 1
    if a == 0 and b == len(parts) - 1:
        return span, ''
    # the kept run holds every matched token by construction; require in addition that the dropped
    # clauses hold clearly fewer of them (margin), i.e. they are not a second statement of the fact
    drop = [k for k in range(len(parts)) if k < a or k > b]
    drop_hits: set = set().union(*[ptoks[k] & matched for k in drop]) if drop else set()
    if len(matched) - len(drop_hits) < margin:
        return span, ''
    # never trim away the bare answer that follows a question clip (RULEBOOK section 2)
    if b < len(parts) - 1 and words[parts[b][1]]['w'].strip().endswith('?'):
        return span, ''
    a0, b1 = parts[a][0], parts[b][1]
    # a kept clause never starts on the coordinator that joined it ("and", "but", "so", "Overall,")
    while b1 - a0 > 2 and _trim_word(words[a0]['w']) in _TRIM_CONJ | {'overall'} \
            and not (set(S.content_tokens(words[a0]['w'])) & matched):
        a0 += 1
    ns, ne = float(words[a0]['s']), float(words[b1]['e'])
    if ns < s - 0.01 or ne > e + 0.01 or ne - ns < min_keep:
        return span, ''                       # contraction only
    dropped = (e - s) - (ne - ns)
    if dropped < min_drop or (e - s) <= 0 or dropped / (e - s) > max_frac:
        return span, ''
    return (ns, ne), f'trim w{i}-{j}->{a0}-{b1} parts {len(parts)} keep {a}-{b} drop {len(drop)}'


def system_prompt() -> str:
    if CFG.get('rules', 0) & 1 and not CFG.get('rules_dual', 0):
        return rules_system_prompt()
    s = {'aeq': SYSTEM_AEQ, 'aeqmc': SYSTEM_AEQMC}.get(CFG['order'], SYSTEM)
    if CFG.get('quote_wording') == 'minclause':
        s = s.replace(_QW_OLD, _QW_MINCLAUSE)
    return s

def _assert_local(url: str):
    host = urllib.parse.urlparse(url).hostname or ''
    if host not in ('127.0.0.1', 'localhost', '0.0.0.0', '::1'):
        raise RuntimeError(f'LLM url {url} is not local; cloud calls are forbidden in /predict')


class LLM:
    def __init__(self, url: str = None, model: str = None):
        import requests
        self.url = (url or CFG['llm_url']).rstrip('/')
        _assert_local(self.url)
        self.s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=32)
        self.s.mount('http://', adapter)
        self.model = model or CFG['llm_model'] or self._discover()

    def _discover(self) -> str:
        r = self.s.get(self.url + '/models', timeout=10)
        r.raise_for_status()
        return r.json()['data'][0]['id']

    def chat(self, messages, max_tokens=None, timeout=None, logprobs=True, extra: dict = None) -> dict:
        body = {'model': self.model, 'messages': messages, 'temperature': 0.0,
                'max_tokens': max_tokens or CFG['llm_max_tokens'],
                'chat_template_kwargs': {'enable_thinking': bool(CFG['llm_think'])}}
        if logprobs:
            body['logprobs'] = True
            body['top_logprobs'] = 10
        if extra:
            body.update(extra)   # e.g. response_format (vLLM structured outputs)
        r = self.s.post(self.url + '/chat/completions', json=body, timeout=timeout or CFG['llm_timeout'])
        r.raise_for_status()
        return r.json()['choices'][0]

    def chat_n(self, messages, n, temperature, max_tokens=None, timeout=None) -> list:
        """n sampled replies in one request (shared prefill), no logprobs; returns all choices."""
        body = {'model': self.model, 'messages': messages, 'temperature': temperature, 'n': int(n),
                'max_tokens': max_tokens or CFG['llm_max_tokens'],
                'chat_template_kwargs': {'enable_thinking': bool(CFG['llm_think'])}}
        r = self.s.post(self.url + '/chat/completions', json=body, timeout=timeout or CFG['llm_timeout'])
        r.raise_for_status()
        return r.json()['choices']


def _vote_ids(cands: List[List[int]], n_units: int) -> List[int]:
    """Majority vote over candidate citations by their FIRST cluster (max_gap 2); ties go to the earliest
    candidate (the greedy reply comes first). Returns that candidate's ids."""
    keys = []
    for ids in cands:
        ok = [i for i in ids if isinstance(i, int) and 0 <= i < n_units]
        cl = S.clusters(ok, 2) if ok else []
        keys.append(tuple(cl[0]) if cl else ())
    counts: Dict[tuple, int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    best = max(counts.values())
    for ids, k in zip(cands, keys):
        if counts[k] == best:
            return ids
    return cands[0]


def _vote_islands(cands: List[List[int]], n_units: int, agree: int) -> List[int]:
    """Per-line self-consistency (MED_SC_MODE=island): keep a cited line only when at least `agree` of the
    candidate citations cite it. Returns the kept ids of the FIRST surviving cluster (max_gap 2), or []
    when nothing reaches the bar (the caller then keeps the greedy reply's ids)."""
    counts: Dict[int, int] = {}
    for ids in cands:
        for i in {j for j in ids if isinstance(j, int) and 0 <= j < n_units}:
            counts[i] = counts.get(i, 0) + 1
    kept = sorted(i for i, c in counts.items() if c >= agree)
    if not kept:
        return []
    return list(S.clusters(kept, 2)[0])


def _answer_prob(choice: dict) -> Optional[float]:
    """P(yes) at the token that follows 'ANSWER:' (normalised over yes/no)."""
    try:
        toks = choice['logprobs']['content']
    except Exception:
        return None
    if not toks:
        return None
    acc = ''
    start = None
    for i, t in enumerate(toks):
        acc = (acc + t['token'])[-24:]
        if re.search(r'ANSWER\s*:\s*\**\s*$', acc.upper()):
            start = i + 1          # keep the LAST 'ANSWER:' (thinking text may contain one)
    cand = range(start, min(len(toks), start + 3)) if start is not None else range(len(toks) - 1, -1, -1)
    for i in cand:
        tok = toks[i]['token'].strip().lower()
        if tok.startswith(('yes', 'no')) or tok in ('y', 'n'):
            py = pn = 0.0
            for alt in toks[i].get('top_logprobs') or [{'token': toks[i]['token'], 'logprob': toks[i]['logprob']}]:
                a = alt['token'].strip().lower().strip('"\'*')
                if a.startswith('yes'):
                    py += math.exp(alt['logprob'])
                elif a.startswith('no'):
                    pn += math.exp(alt['logprob'])
            if py + pn > 0:
                return py / (py + pn)
    return None


def parse_reply(text: str, n_units: int) -> Tuple[Optional[bool], List[int], str]:
    text = text or ''
    # drop any thinking block
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
    if '</think>' in text:
        text = text.split('</think>')[-1]
    ans = None
    m = re.findall(r'ANSWER\s*[:\-]\s*\**\s*(yes|no)', text, flags=re.I)
    if m:
        ans = m[-1].lower() == 'yes'
    else:
        m2 = re.findall(r'\b(yes|no)\b', text, flags=re.I)
        if m2:
            ans = m2[-1].lower() == 'yes'
    ids: List[int] = []
    me = re.search(r'EVIDENCE[ \t]*[:\-][ \t]*([^\n]*)', text, flags=re.I)
    if me:
        for a, b in re.findall(r'(\d+)(?:\s*-\s*(\d+))?', me.group(1)):
            a = int(a)
            b = int(b) if b else a
            for k in range(a, min(b, a + 6) + 1):
                if 0 <= k < n_units:
                    ids.append(k)
    quote = ''
    mq = re.search(r'QUOTE[ \t]*[:\-][ \t]*([^\n]*)', text, flags=re.I)
    if mq:
        quote = mq.group(1).strip().strip('"“”\'')
    return ans, ids, quote


def build_user(units: List[dict], question: str) -> str:
    lines = '\n'.join(f"[{u['id']}] {u['text']}" for u in units)
    return f"Transcript:\n{lines}\n\nQuestion: {question}"


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Two-pass structured evidence selection (n3 M2)
# --------------------------------------------------------------------------- #

TP1_SYSTEM = """You read a numbered transcript of a recorded consultation between a doctor and a patient (one spoken clip per line, speakers not labelled) and a yes/no question about it. List the ids of ALL lines that mention the fact the question asks about, in any form: the first vague mention, the doctor's finding or statement, later restatements, conclusions and confirmations of it. Do not list lines about other facts. Include a bare confirmation line ("Yes.", "Right.") only together with the line it confirms. Reply with JSON only: {"ids": [..]} with the ids in transcript order, at most 8; {"ids": []} if the fact is never mentioned."""

TP2_RULES = """You choose the evidence clip for a yes/no question about a recorded consultation between a doctor and a patient. The transcript is split into numbered lines (one spoken clip each; speakers are not labelled). You are given the candidate lines that mention the fact. Pick the line that STATES the fact, following these annotation rules:
1. The evidence line itself states the fact: a clause with a subject and a verb that asserts the finding, plan, symptom or event. A line that only raises the topic, asks about it or hints at it is not evidence.
2. When the fact is stated more than once, the LATER, more explicit restatement wins over a vague or brief first mention: the doctor's finding wins over the patient's first complaint, and a conclusion or summary near the end wins over the first time the fact comes up.
3. A bare confirmation ("Yes.", "Right.", "Exactly.") is not evidence on its own; pick the line whose content it confirms.
4. Keep it minimal: one line, or two adjacent lines only when the statement runs across both.
Reply with JSON only: {"evidence": [id]} or {"evidence": [id, next_id]}."""

TP2_EXAMPLES = """

Examples (candidate lines shown; the answer follows).
Question: Does the patient want the skin changes removed?
Candidates:
[11] I want them gone.
[32] then can we get on with removing them
[42] I would still rather have them off than keep catching them.
{"evidence": [42]}
(42 is the later, explicit restatement; 11 is a brief first mention and 32 is a request phrased as a question.)
Question: Was a skin wart found under the right breast?
Candidates:
[8] One under my right breast,
[13] The one under your right breast is a skin wart,
{"evidence": [13]}
(8 only raises the topic; 13 states the finding.)
Question: Is an appointment for the removal going to be arranged?
Candidates:
[34] I will arrange an appointment for the removal
[44] Then the appointment for removal is the plan.
{"evidence": [44]}
(the later conclusion restates the plan.)"""


def tp2_system() -> str:
    return TP2_RULES + (TP2_EXAMPLES if CFG['tp_examples'] else '')


def _tp_schema(valid: List[int], key: str, max_items: int) -> dict:
    """OpenAI-style response_format for vLLM structured outputs: {key: [ids from the enum]}."""
    schema = {'type': 'object',
              'properties': {key: {'type': 'array', 'items': {'type': 'integer', 'enum': [int(i) for i in valid]},
                                   'maxItems': int(max_items)}},
              'required': [key], 'additionalProperties': False}
    return {'response_format': {'type': 'json_schema',
                                'json_schema': {'name': 'evidence_' + key, 'schema': schema, 'strict': True}}}


def _tp_parse(text: str, key: str, valid: set) -> List[int]:
    try:
        obj = json.loads(re.sub(r'<think>.*?</think>', '', text or '', flags=re.S).strip())
        out = []
        for i in obj.get(key) or []:
            if isinstance(i, int) and i in valid and i not in out:
                out.append(i)
        return out
    except Exception:
        return []


def two_pass_ids(llm, units: List[dict], q: str, ids: List[int], timeout: float) -> Tuple[List[int], str]:
    """Two-pass evidence selection; returns (new ids, debug note). Falls back to the production ids on any
    failure. Mode 1: pass 1 (candidate listing, structured) + pass 2 (selection with annotation rules).
    Mode 2: pass 2 only over the production citation's clusters."""
    n = len(units)
    prod = [i for i in ids if isinstance(i, int) and 0 <= i < n]
    mode = CFG['two_pass']
    cands: List[int] = []
    note = ''
    if mode == 1:
        msgs = [{'role': 'system', 'content': TP1_SYSTEM}, {'role': 'user', 'content': build_user(units, q)}]
        ch = llm.chat(msgs, max_tokens=80, timeout=timeout, logprobs=False,
                      extra=_tp_schema(list(range(n)), 'ids', CFG['tp_max_cands']))
        cands = _tp_parse((ch.get('message') or {}).get('content') or '', 'ids', set(range(n)))
        note = f'p1={cands}'
        for i in prod:              # never lose the production citation
            if i not in cands:
                cands.append(i)
        cands = sorted(cands)[: max(CFG['tp_max_cands'], len(prod))]
    else:
        cands = sorted(set(prod))
        if len(S.clusters(cands, 2)) < 2:
            return ids, note + ' p2=skip(1 cluster)'
    if len(cands) < CFG['tp_min_cands'] or not cands:
        return ids, note + ' p2=skip(few)'
    cand_lines = '\n'.join(f"[{i}] {units[i]['text']}" for i in cands)
    user = build_user(units, q) + '\n\nCandidates:\n' + cand_lines
    msgs = [{'role': 'system', 'content': tp2_system()}, {'role': 'user', 'content': user}]
    ch = llm.chat(msgs, max_tokens=30, timeout=timeout, logprobs=False, extra=_tp_schema(cands, 'evidence', 2))
    pick = _tp_parse((ch.get('message') or {}).get('content') or '', 'evidence', set(cands))
    if not pick:
        return ids, note + ' p2=fail'
    pick = sorted(pick)
    if len(pick) == 2 and pick[1] - pick[0] > 1:   # not adjacent: keep the later one (rule 2)
        pick = [pick[1]]
    if CFG['tp_extent'] == 'cluster':
        for c in S.clusters(prod, 2):
            if any(i in c for i in pick):
                return list(c), note + f' p2={pick} ext={list(c)}'
    return pick, note + f' p2={pick}'


class Answerer:
    def __init__(self, backend: str = 'llm'):
        self.backend = backend
        self.llm = None
        if backend == 'llm':
            try:
                self.llm = LLM()
                log.info('LLM backend %s at %s', self.llm.model, self.llm.url)
            except Exception as e:
                log.error('LLM unavailable (%s); heuristic fallback only', e)
        self.pool = cf.ThreadPoolExecutor(max_workers=16)
        self.loc_up = False
        if CFG['loc_url']:
            import threading
            self._loc_check()
            threading.Thread(target=self._loc_poll, daemon=True).start()

    def _loc_check(self):
        """Sets self.loc_up from the sidecar's health endpoint; never raises."""
        try:
            import requests
            up = requests.get(CFG['loc_url'] + '/health', timeout=1.0).status_code == 200
        except Exception:
            up = False
        if up != self.loc_up:
            log.info('locator sidecar %s at %s', 'up' if up else 'DOWN', CFG['loc_url'])
        self.loc_up = up

    def _loc_poll(self):
        while True:
            time.sleep(10)
            self._loc_check()

    def _locate(self, tx: dict, questions: List[str], out: List[dict], deadline: float):
        """Replace heuristic answers (src 'heur') by the CPU cross-encoder fallback, one
        question at a time while time remains. Any failure keeps the heuristic answer."""
        import requests
        todo = [k for k, o in enumerate(out) if o['src'] == 'heur']
        if not todo or not CFG['loc_url'] or not self.loc_up or deadline - time.time() < CFG['loc_min_s']:
            return
        try:
            pu = S.build_units(tx, 'phrase')
            runs = L.candidate_runs(pu)
            if not runs:
                return
            texts = [L.run_text(pu, i, j) for i, j in runs]
            for k in todo:
                rem = deadline - time.time()
                if rem < 1.5:
                    log.warning('locator out of time; heuristic kept for %d question(s)', len(todo) - todo.index(k))
                    break
                idx = L.lexical_topk(pu, runs, questions[k], CFG['loc_k'])
                r = requests.post(CFG['loc_url'] + '/score', timeout=max(0.5, min(10.0, rem - 1.0)),
                                  json={'pairs': [[questions[k], texts[i]] for i in idx]})
                r.raise_for_status()
                sc = r.json()['scores']
                if len(sc) != len(idx):
                    break
                yes, span, mx = L.fallback_decide(pu, [runs[i] for i in idx], sc, CFG['loc_thr'], CFG['loc_alpha'],
                                                  CFG['loc_target'], CFG['loc_shift_s'], CFG['loc_shift_e'])
                z = min(max(mx - CFG['loc_thr'], -30.0), 30.0)
                out[k] = {**out[k], 'p': 1 / (1 + math.exp(-z)), 'src': 'loc', 'loc_yes': bool(yes),
                          'loc_span': span}
        except Exception as e:
            log.warning('locator fallback failed (%s); heuristic kept', e)

    def _one(self, units, q, timeout):
        msgs = [{'role': 'system', 'content': system_prompt()},
                {'role': 'user', 'content': build_user(units, q)}]
        ch = self.llm.chat(msgs, timeout=timeout)
        text = (ch.get('message') or {}).get('content') or ''
        ans, ids, quote = parse_reply(text, len(units))
        p = _answer_prob(ch)
        if p is None and ans is not None:
            p = 0.9 if ans else 0.1
        if CFG['sc_n'] > 1 and ids:
            try:
                alts = self.llm.chat_n(msgs, CFG['sc_n'] - 1, CFG['sc_t'], timeout=timeout)
                cands = [ids] + [parse_reply((c.get('message') or {}).get('content') or '', len(units))[1]
                                 for c in alts]
                cands = [c for c in cands if c]
                if not cands:
                    pass
                elif CFG.get('sc_mode') == 'island':
                    ids = _vote_islands(cands, len(units), CFG['sc_agree']) or ids
                else:
                    ids = _vote_ids(cands, len(units))
            except Exception as e:
                log.warning('self-consistency sampling failed (%s); greedy citation kept', e)
        if CFG.get('rules', 0) & 1 and CFG.get('rules_dual', 0) and units and p is not None:
            try:
                rm = [{'role': 'system', 'content': rules_system_prompt()},
                      {'role': 'user', 'content': build_user(units, q)}]
                rch = self.llm.chat(rm, timeout=timeout, logprobs=False)
                rtext = (rch.get('message') or {}).get('content') or ''
                rids = parse_reply(rtext, len(units))[1]
                ids, note = dual_ids(ids, rids, len(units))
                text += '\n[RULES] ' + rtext.replace('\n', ' | ') + '\n[DUAL] ' + note
            except Exception as e:
                log.warning('rules dual call failed (%s); production citation kept', e)
                text += '\n[DUAL] error ' + str(e)[:80]
        if CFG.get('rules', 0) & 2 and units and p is not None:
            pp = min(max(p, 1e-6), 1 - 1e-6)
            if CFG['rules_p2_all'] or math.log(pp / (1 - pp)) + CFG['yes_bias'] > 0:
                try:
                    ids, note = rules_extent_ids(self.llm, units, q, ids, timeout)
                    text += '\n[EXT] ' + note
                except Exception as e:
                    log.warning('rules extent pass failed (%s); reply citation kept', e)
                    text += '\n[EXT] error ' + str(e)[:80]
        if CFG.get('vc', 0) and units and p is not None and self.llm is not None:
            pp = min(max(p, 1e-6), 1 - 1e-6)
            if CFG['vc'] > 1 or math.log(pp / (1 - pp)) + CFG['yes_bias'] > 0:
                try:
                    ids, note = verify_contract_ids(self.llm, units, q, ids, timeout)
                    text += '\n[VC] ' + note
                except Exception as e:
                    log.warning('contraction verifier failed (%s); citation kept', e)
                    text += '\n[VC] error ' + str(e)[:80]
        if CFG['two_pass'] and units and p is not None:
            pp = min(max(p, 1e-6), 1 - 1e-6)
            if math.log(pp / (1 - pp)) + CFG['yes_bias'] > 0:   # the production verdict is yes
                try:
                    ids, note = two_pass_ids(self.llm, units, q, ids, timeout)
                    text += '\n[TP]' + note
                except Exception as e:
                    log.warning('two-pass selection failed (%s); production citation kept', e)
                    text += '\n[TP] error ' + str(e)[:80]
        return p, ids, quote, text

    def answer(self, tx: dict, questions: List[str], deadline: float) -> List[dict]:
        units = S.build_units(tx, CFG['unit_mode'])
        words = [w for u in units for w in u['words']]
        dur = tx.get('duration')
        onsets = _request_onsets(tx, dur)
        tx['_onsets_n'] = len(onsets) if onsets is not None else 0   # request log / live check
        # onset rule replaces the start shift (DEPLOY TRAP fix); no onsets -> legacy shift
        shift_s = CFG['span_shift_s']
        if onsets is not None and not CFG['allow_shift_with_onset']:
            shift_s = 0.0
        out = []
        # heuristic first: cheap and always available
        for q in questions:
            try:
                p, ids, quote = heuristic_answer(units, q)
            except Exception:
                p, ids, quote = 0.5, [], ''
            out.append({'p': p, 'ids': ids, 'quote': quote, 'src': 'heur', 'raw': ''})
        # the LLM phase leaves loc_reserve seconds for the fallback when the sidecar is up;
        # with a healthy LLM (~8 s per conversation) this never binds
        llm_deadline = deadline - (CFG['loc_reserve'] if (CFG['loc_url'] and self.loc_up) else 0.0)
        if self.llm is not None and units:
            remaining = llm_deadline - time.time()
            if remaining > 2:
                futs = {self.pool.submit(self._one, units, q, max(1.0, remaining - 0.5)): k
                        for k, q in enumerate(questions)}
                try:
                    for f in cf.as_completed(futs, timeout=max(0.5, remaining - 0.5)):
                        k = futs[f]
                        try:
                            p, ids, quote, raw = f.result()
                            if p is not None:
                                h = out[k]
                                out[k] = {'p': p, 'ids': ids or h['ids'],
                                          'quote': quote, 'src': 'llm', 'raw': raw}
                        except Exception as e:
                            log.warning('LLM call failed for q%d: %s', k, e)
                except cf.TimeoutError:
                    log.warning('LLM deadline hit; fallback answers for unfinished questions')
        self._locate(tx, questions, out, deadline)
        res = []
        clause_units = None   # built lazily for the clause snap
        for q, o in zip(questions, out):
            p = min(max(o['p'], 1e-6), 1 - 1e-6)
            logit = math.log(p / (1 - p)) + (CFG['yes_bias'] if o['src'] == 'llm' else CFG['heur_bias'])
            yes = logit > 0
            span = None
            span_src = None
            if o['src'] == 'loc':
                yes = o['loc_yes']
                logit = math.log(p / (1 - p))
                span = o['loc_span']
                span_src = 'loc' if span is not None else None
                if span is not None:
                    span = (round(float(span[0]), 2), round(float(span[1]), 2))
                    if not (math.isfinite(span[0]) and math.isfinite(span[1])) or span[1] <= span[0]:
                        span, span_src = None, None
                o['logit'] = logit
                res.append({'answer': bool(yes), 'span': span if yes else None, 'cand_span': span,
                            'p': p, 'src': 'loc', 'span_src': span_src,
                            'unit_span': S.span_for_ids(units, o['ids'], duration=dur) if o['ids'] else None,
                            'ids': o['ids'], 'quote': o['quote'], 'raw': o['raw']})
                continue
            try:
                # quote aligned to word times, gated to lie within +-3 s of the cited lines;
                # else the cited unit span (spans.quote_span; Allmquote-g4-2 knobs)
                span, span_src = S.quote_span(units, o['ids'], o['quote'] if CFG['use_quote'] else '',
                                              gate=CFG['quote_gate'], ellipsis=CFG['quote_ellipsis'],
                                              fallback=CFG['multi_fallback'], question=q,
                                              duration=dur)
                if span is not None and span_src == 'quote' and CFG['span_clause_snap']:
                    if clause_units is None:
                        clause_units = S.build_units(tx, 'clause', pause=CFG['span_clause_pause'])
                    hit = [u for u in clause_units if u['end'] > span[0] and u['start'] < span[1]]
                    if hit:
                        span = (hit[0]['start'], hit[-1]['end'])
                        span_src = 'clause'
                if span is not None and CFG.get('span_cont', 0):
                    span = cont_start(span, words, units, q)
                if span is not None and CFG['span_onset_sentstart']:
                    span = S.onset_sentstart(span, words, onsets, CFG['span_onset_k'], CFG['span_onset_off'])
                if span is not None and CFG.get('span_cont', 0):
                    span = cont_end(span, words, units, q)
                if span is not None and CFG['span_coverage_next']:
                    span = S.coverage_next_end(span, words, q)
                if span is not None and CFG.get('span_trim', 0):
                    t, note = clause_trim(span, words, q, CFG['span_trim'], CFG['trim_margin'],
                                          CFG['trim_max_frac'], CFG['trim_min_keep'],
                                          CFG['trim_min_hits'], CFG['trim_min_drop'],
                                          CFG['trim_end_strict'])
                    if note:
                        # the start moved to a mid-utterance word: re-apply the onset rule to THAT word
                        # only (k=0 = no sentence-start walk back, which would undo the trim)
                        if t[0] > span[0] + 0.01 and CFG['span_onset_sentstart']:
                            t = S.onset_sentstart(t, words, onsets, 0, CFG['span_onset_off'])
                        span = t
                if span is not None:
                    span = S.calibrate_span(span, words, shift_s, CFG['span_shift_e'],
                                            CFG['span_snap'], CFG['span_snap_pause'],
                                            CFG['span_clamp_len'], CFG['span_clamp_w'], duration=dur)
                if span is not None:
                    s, e = float(span[0]), float(span[1])
                    if not (math.isfinite(s) and math.isfinite(e)):
                        span = None
                    else:
                        if e <= s:
                            e = s + 0.3
                        span = (round(s, 2), round(e, 2))
            except Exception:
                span = None
            o['logit'] = logit
            res.append({'answer': bool(yes), 'span': span if yes else None, 'cand_span': span,
                        'p': p, 'src': o['src'], 'span_src': span_src,
                        'unit_span': S.span_for_ids(units, o['ids'], duration=dur) if o['ids'] else None,
                        'ids': o['ids'], 'quote': o['quote'], 'raw': o['raw']})
        return res


def _cli():
    import argparse
    import glob
    ap = argparse.ArgumentParser()
    ap.add_argument('--tx-dir')
    ap.add_argument('--audio-dir')
    ap.add_argument('--save-tx')
    ap.add_argument('--backend', default='llm', choices=['llm', 'heuristic'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    upstream = os.environ.get('UPSTREAM', '/home/claude/Nordic-AI-Cup-2026')
    if not os.path.isdir(upstream):
        upstream = '/workspace/upstream'
    import csv
    rows = list(csv.DictReader(open(os.path.join(upstream, 'medical-appointment/data/question_train.csv'))))
    groups: Dict[str, List[str]] = {}
    for r in rows:
        groups.setdefault(f"conversation_{r['transcript_id']}.mp3", []).append(r['question'])
    log_cfg()
    ans = Answerer(a.backend)
    preds, detail = {}, {}
    names = list(groups)[: a.limit or None]
    for fn in names:
        qs = groups[fn]
        t0 = time.time()
        if a.audio_dir:
            tx = transcribe(open(os.path.join(a.audio_dir, fn), 'rb').read(), qs)
            if a.save_tx:
                os.makedirs(a.save_tx, exist_ok=True)
                json.dump({k: v for k, v in tx.items() if k != '_pcm'},
                          open(os.path.join(a.save_tx, fn.replace('.mp3', '.json')), 'w'))
        else:
            p = os.path.join(a.tx_dir, fn.replace('.mp3', '.json'))
            if not os.path.exists(p):
                continue
            tx = S.load_transcript(p)
            if CFG['span_onset_sentstart']:   # same decode as serving (transcribe)
                try:
                    from faster_whisper.audio import decode_audio
                    tx['_pcm'] = decode_audio(os.path.join(upstream, 'medical-appointment/data/audio', fn),
                                              sampling_rate=16000)
                except Exception as e:
                    log.warning('onset decode failed for %s (%s)', fn, e)
        t1 = time.time()
        res = ans.answer(tx, qs, deadline=time.time() + CFG['deadline'])
        t2 = time.time()
        preds[fn] = {'answers': [r['answer'] for r in res],
                     'evidence_start': [r['span'][0] if r['answer'] and r['span'] else None for r in res],
                     'evidence_end': [r['span'][1] if r['answer'] and r['span'] else None for r in res],
                     'latency_ms': (t2 - t0) * 1000, 'asr_s': t1 - t0, 'qa_s': t2 - t1}
        detail[fn] = res
        log.info('%s asr %.1fs qa %.1fs yes=%d', fn, t1 - t0, t2 - t1, sum(r['answer'] for r in res))
    json.dump(preds, open(a.out, 'w'), indent=1)
    json.dump(detail, open(a.out.replace('.json', '') + '.detail.json', 'w'), indent=1, default=str)
    lat = [p['latency_ms'] / 1000 for p in preds.values()]
    if lat:
        print(f'conversations {len(lat)}  mean {sum(lat) / len(lat):.1f}s  max {max(lat):.1f}s  '
              f'asr mean {sum(p["asr_s"] for p in preds.values()) / len(lat):.1f}s  '
              f'qa mean {sum(p["qa_s"] for p in preds.values()) / len(lat):.1f}s')


if __name__ == '__main__':
    _cli()
