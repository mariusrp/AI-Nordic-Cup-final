"""Does the SERVING decode path give the same onsets as the analysis decode path?

Analysis (replay_rules, cycle 2) decoded each MP3 with the ffmpeg CLI (-ac 1 -ar 16000 f32le, common.pcm).
The fast-lane implementation (medical-fast-r5-onset, pipeline.transcribe) decodes with
faster_whisper.audio.decode_audio (PyAV). MP3 encoder-delay handling or resampling differences could shift
every onset by a constant (e.g. ~25 ms of encoder delay), which would silently change the fitted -0.04 s lead.
Reports per conversation: sample-count difference, the lag that best aligns the two signals, the median
difference of matched onsets, then re-runs the start-rule replay with the PyAV decode.
    python3 decode_check.py
"""
import glob
import os

import numpy as np
from faster_whisper.audio import decode_audio

import acoustic as A
import common as C
import replay_rules as RR


def best_lag(a, b, max_lag=4000, n=16000 * 30):
    """Lag (samples) that maximises the correlation of b shifted against a (positive = b later)."""
    a = a[:n].astype(np.float64)
    b = b[:n].astype(np.float64)
    m = min(len(a), len(b))
    a, b = a[:m], b[:m]
    L = 1 << int(np.ceil(np.log2(2 * m)))
    c = np.fft.irfft(np.fft.rfft(b, L) * np.conj(np.fft.rfft(a, L)), L)
    lags = np.concatenate([np.arange(0, max_lag + 1), np.arange(-max_lag, 0)])
    vals = np.concatenate([c[:max_lag + 1], c[-max_lag:]])
    return int(lags[np.argmax(vals)])


def onsets(x):
    return A.onsets_offsets(A.speech_mask(A.frame_db(x), -50), 0.1)[0]


def main():
    cv = C.convs()
    pv = {}
    rows = []
    for fn in cv:
        x_ff = C.pcm(fn)
        x_pv = decode_audio(os.path.join(C.AUDIO, fn), sampling_rate=16000)
        pv[fn] = x_pv
        lag = best_lag(x_ff, x_pv)
        o_ff, o_pv = onsets(x_ff), onsets(x_pv)
        d = [o_pv[np.argmin(np.abs(o_pv - t))] - t for t in o_ff] if len(o_pv) else []
        d = np.array([v for v in d if abs(v) < 0.1])
        rows.append((fn, len(x_pv) - len(x_ff), lag, len(o_ff), len(o_pv), np.median(d) if len(d) else np.nan,
                     np.mean(np.abs(d) > 0.0051) if len(d) else np.nan))
    for r in rows:
        print('%-28s dlen %+6d samples  lag %+5d samples (%.1f ms)  onsets ff %d pv %d  median d %.3f s  frac |d|>5ms %.2f'
              % (r[0], r[1], r[2], 1000 * r[2] / 16000, r[3], r[4], r[5], r[6]))
    lags = np.array([r[2] for r in rows])
    print('lag ms: median %.2f min %.2f max %.2f' % (1000 * np.median(lags) / 16000, 1000 * lags.min() / 16000,
                                                      1000 * lags.max() / 16000))
    # replay the start rule with the PyAV decode (8 fresh fast-r5-2 runs + 4 cycle-2 runs)
    runs = sorted(glob.glob(f'{C.WORK}/fr5/fr5_rep*/arm*.detail.json'))
    rules = {'prod': RR.make_rule('prod'),
             'R ffmpeg-decode': RR.make_rule('x', sentstart=4, inword=1, minsil=0.1, on_off=-0.04)}
    out_ff, dev = RR.evaluate(rules, tuple(runs))
    C.pcm = lambda fn: pv[fn]
    rules_pv = {'prod': RR.make_rule('prod'),
                'R PyAV-decode': RR.make_rule('x', sentstart=4, inword=1, minsil=0.1, on_off=-0.04)}
    for off in (-0.02, -0.06):
        rules_pv[f'R PyAV-decode off {off}'] = RR.make_rule('x', sentstart=4, inword=1, minsil=0.1, on_off=off)
    out_pv, _ = RR.evaluate(rules_pv, tuple(runs))
    print('--- 8 fresh fast-r5-2 runs ---')
    RR.report({**out_ff, **{k: v for k, v in out_pv.items() if k != 'prod'}}, dev)


if __name__ == '__main__':
    main()
