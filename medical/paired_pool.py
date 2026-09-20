"""Shared conversation pool + arm definitions for medical/paired_run.py (fast-lane FR3-B).

Loads the 31 visible conversations' cached large-v3-turbo transcripts (word timestamps) plus the
decoded PCM needed for the onset start rule (spans.energy_onsets), and the questions per
conversation from the upstream question_train.csv (same grouping offline_eval.py scores against).

BASE for this batch = main deploy env (MED_SPAN_ONSET_SENTSTART=1, MED_SPAN_SHIFT_S=0, set via
the pod's medical/env.sh as usual) + the FR3-A1 port's MED_QUOTE_GATE=any, set globally by
set_base() below (not a per-arm knob: every arm in ARMS shares it). Arms differ only in ONE of
the FR3-B knobs: MED_CLUSTER_VERIFY (fr3-b1), MED_QUOTE_REPAIR (fr3-b2), MED_SIBLING_CTX
(fr3-b3, unchanged code from medical-fast-r2-2-fr2b's FR2-B2). Nothing is stacked.
"""
from __future__ import annotations

import os
from typing import Dict, List

import offline_eval as OE
import pipeline as P
import spans as S

UPSTREAM = os.environ.get('UPSTREAM', '/home/claude/upstream-work')
TX_DIR = os.environ.get('MED_TX_DIR', '/workspace/tx/large-v3-turbo')
AUDIO_DIR = os.environ.get('MED_AUDIO_DIR',
                            os.path.join(UPSTREAM, 'medical-appointment/data/audio'))

# fast-r3-aeqmc batch: A0 = production (base), A1 = MED_ORDER=aeq, A2 = MED_QUOTE_WORDING=minclause,
# A3 = both. Only these two knobs move; everything else is the base.
ARMS: Dict[str, Dict[str, str]] = {
    'base': dict(order='eqa', quote_wording=''),
    'A1': dict(order='aeq', quote_wording=''),
    'A2': dict(order='eqa', quote_wording='minclause'),
    'A3': dict(order='aeq', quote_wording='minclause'),
}


def questions_by_conv() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for fn, rows in OE.group_questions_by_conversation():
        out[fn] = [r['question'] for r in rows]
    return out


def load_pool(filenames=None):
    """{fn: {'tx': dict, 'questions': [str]}}; tx carries '_pcm' when the matching audio file
    is found (decode_audio, 16 kHz mono), so onset_sentstart is live in this replay same as the
    live server (pipeline.transcribe attaches '_pcm' the same way when onset is on)."""
    qs = questions_by_conv()
    names = sorted(qs) if filenames is None else filenames
    from faster_whisper.audio import decode_audio
    pool = {}
    for fn in names:
        base = fn.replace('.mp3', '')
        txp = os.path.join(TX_DIR, base + '.json')
        if not os.path.exists(txp):
            continue
        tx = S.load_transcript(txp)
        ap = os.path.join(AUDIO_DIR, fn)
        if os.path.exists(ap):
            try:
                tx = dict(tx)
                tx['_pcm'] = decode_audio(ap, sampling_rate=16000)
            except Exception:
                pass
        pool[fn] = {'tx': tx, 'questions': qs[fn]}
    return pool


def set_base():
    """FR3-A1 port: the batch-wide base config every arm shares (on top of the pod's normal
    deploy env for onset/shift, set via medical/env.sh as usual)."""
    P.CFG['quote_gate'] = 'any'


def set_arm(name: str):
    """Mutate pipeline.CFG in place for the named arm (base knobs unchanged; only the FR3-B
    variant knobs move). Call before every Answerer.answer()."""
    cfg = ARMS[name]
    P.CFG['order'] = cfg['order']
    P.CFG['quote_wording'] = cfg['quote_wording']
