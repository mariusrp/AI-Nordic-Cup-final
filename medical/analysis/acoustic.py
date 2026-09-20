"""Energy-based speech onset/offset detection on the (near-silent-floor) conversation audio.

Frames: 25 ms window, 10 ms hop, 16 kHz. A frame is 'speech' when its RMS level is above
thr_db (absolute dBFS; pauses in this data sit at -70..-90 dBFS, speech at -10..-35).
An onset is a silence->speech transition preceded by >= min_sil seconds of silence;
an offset is a speech->silence transition followed by >= min_sil seconds of silence.
"""
from __future__ import annotations

import numpy as np

HOP = 0.01


def frame_db(x, sr=16000, win=400, hop=160):
    n = max(1, (len(x) - win) // hop)
    fr = np.lib.stride_tricks.sliding_window_view(x, win)[::hop][:n]
    rms = np.sqrt((fr.astype(np.float64) ** 2).mean(1) + 1e-12)
    return 20 * np.log10(rms + 1e-9)


def speech_mask(db, thr_db=-50.0):
    return db > thr_db


def onsets_offsets(mask, min_sil=0.15, hop=HOP):
    """Return arrays of onset times and offset times (seconds), with the silence length before
    each onset / after each offset."""
    m = mask.astype(np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    starts = np.where(d == 1)[0]          # first speech frame of a run
    ends = np.where(d == -1)[0]           # first silence frame after a run
    ons, on_sil, offs, off_sil = [], [], [], []
    prev_end = None
    for k, (s, e) in enumerate(zip(starts, ends)):
        sil_before = (s - prev_end) * hop if prev_end is not None else s * hop
        if sil_before >= min_sil:
            ons.append(s * hop)
            on_sil.append(sil_before)
        nxt = starts[k + 1] if k + 1 < len(starts) else len(m)
        sil_after = (nxt - e) * hop
        if sil_after >= min_sil:
            offs.append(e * hop)
            off_sil.append(sil_after)
        prev_end = e
    return np.array(ons), np.array(on_sil), np.array(offs), np.array(off_sil)


def silence_before(mask, t, hop=HOP, max_s=3.0):
    """Length of contiguous silence ending right before time t."""
    i = int(round(t / hop)) - 1
    n = 0
    while i >= 0 and not mask[i] and n * hop < max_s:
        n += 1
        i -= 1
    return n * hop
