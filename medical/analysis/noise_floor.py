"""Robustness of the onset rule's -50 dBFS speech threshold: per-conversation noise floor (10th/25th percentile of
25 ms frame RMS in dBFS), the level of pause frames, and onset density. If unseen audio had a floor near -50 dBFS
the rule would find no onsets and fall back to raw turbo starts (no +0.2 shift), which replays at -0.015 tIoU.
    python3 noise_floor.py
"""
import numpy as np

import acoustic as A
import common as C


def main():
    rows = []
    for fn in C.convs():
        x = C.pcm(fn)
        db = A.frame_db(x)
        on = A.onsets_offsets(A.speech_mask(db, -50), 0.1)[0]
        rows.append((fn, np.percentile(db, 10), np.percentile(db, 25), np.percentile(db, 50), len(on) / (len(x) / 16000) * 10,
                     20 * np.log10(np.abs(x).max() + 1e-9)))
    a = np.array([r[1:] for r in rows])
    for r in sorted(rows, key=lambda r: -r[1])[:5]:
        print('%-28s p10 %.1f p25 %.1f p50 %.1f dBFS  onsets/10s %.1f  peak %.1f dBFS' % r)
    print('ALL 31: p10 floor median %.1f (max %.1f)  p25 median %.1f (max %.1f)  onsets per 10 s min %.1f median %.1f  peak dBFS %.1f..%.1f'
          % (np.median(a[:, 0]), a[:, 0].max(), np.median(a[:, 1]), a[:, 1].max(), a[:, 3].min(), np.median(a[:, 3]),
             a[:, 4].min(), a[:, 4].max()))


if __name__ == '__main__':
    main()
