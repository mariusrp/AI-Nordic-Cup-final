"""Shared loaders for the medical understanding-lane analyses (visible 31 conversations only).

Gold: upstream question_train.csv (worker copy /home/claude/upstream-work). Transcripts: cached faster-whisper
outputs pulled from POD=gpu /workspace/tx/<model> into $MED_UND_DIR/tx/<model>. Audio: decoded with ffmpeg to
16 kHz mono float32 and cached as .npy in $MED_UND_DIR/pcm.
"""
from __future__ import annotations

import csv
import json
import os
import random
import subprocess

import numpy as np

UP = os.environ.get('UPSTREAM', '/home/claude/upstream-work')
MED = os.path.join(UP, 'medical-appointment')
CSV = os.path.join(MED, 'data', 'question_train.csv')
AUDIO = os.path.join(MED, 'data', 'audio')
WORK = os.environ.get('MED_UND_DIR', '/tmp/claude-0/-home-claude/323fa704-a9b4-5fa1-bde8-352ff3b28cce/scratchpad/med')
SR = 16000


def rows():
    return list(csv.DictReader(open(CSV)))


def convs():
    out = {}
    for r in rows():
        out.setdefault(f"conversation_{r['transcript_id']}.mp3", []).append(r)
    return out


def split_dev():
    """Same split as medical/offline_eval.split_ids (seed 2026 over the visible audio filenames)."""
    ids = sorted(convs())
    random.Random(2026).shuffle(ids)
    return set(ids[: len(ids) // 2])


def gold(r):
    if not r['evidence_start']:
        return None
    return float(r['evidence_start']), float(r['evidence_end'])


def pcm(fn):
    os.makedirs(os.path.join(WORK, 'pcm'), exist_ok=True)
    p = os.path.join(WORK, 'pcm', fn.replace('.mp3', '.npy'))
    if os.path.exists(p):
        return np.load(p)
    raw = subprocess.run(['ffmpeg', '-v', 'quiet', '-i', os.path.join(AUDIO, fn), '-ac', '1', '-ar', str(SR),
                          '-f', 'f32le', '-'], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.float32).copy()
    np.save(p, x)
    return x


def tx(fn, model='large-v3-turbo'):
    return json.load(open(os.path.join(WORK, 'tx', model, fn.replace('.mp3', '.json'))))


def words(t):
    return [w for s in t['segments'] for w in (s.get('words') or [])]


def tiou(a, b):
    if a is None or b is None:
        return 0.0
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0
