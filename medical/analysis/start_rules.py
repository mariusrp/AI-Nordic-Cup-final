"""How do gold span STARTS relate to turbo word starts and acoustic onsets?

For every gold-yes span (31 visible conversations) take the turbo word that the gold start falls in
(first word with end > gold_start + 0.04), then compare start estimators against the gold start:
  turbo      : the turbo word start
  turbo+0.2  : production rule (global +0.2 s shift)
  onset      : if an acoustic onset (silence >= min_sil before it) lies in [ws - 0.1, ws + win], use it,
               else the turbo word start + shift_mid
Also reports the END error of the turbo word the gold end falls in, and tIoU with oracle words.
    python3 start_rules.py
"""
import collections
import numpy as np

import acoustic as A
import common as C


def first_word(W, g0):
    for k, w in enumerate(W):
        if w['e'] > g0 + 0.04:
            return k
    return len(W) - 1


def last_word(W, g1):
    k = 0
    for i, w in enumerate(W):
        if w['s'] < g1 - 0.04:
            k = i
    return k


def onset_rule(ws, on, win=0.6, back=0.1, shift_mid=0.0):
    c = on[(on >= ws - back) & (on <= ws + win)]
    if len(c):
        return float(c[0]), True
    return ws + shift_mid, False


def main():
    R = [r for r in C.rows() if r['evidence_start']]
    dev = C.split_dev()
    masks = {}
    rows = []
    for r in R:
        fn = f"conversation_{r['transcript_id']}.mp3"
        if fn not in masks:
            masks[fn] = A.speech_mask(A.frame_db(C.pcm(fn)), -50)
        m = masks[fn]
        W = C.words(C.tx(fn))
        g = C.gold(r)
        i = first_word(W, g[0])
        j = max(i, last_word(W, g[1]))
        rows.append((r, fn, W, g, i, j, m))
    for thr_sil in (0.1, 0.15, 0.25):
        for win in (0.4, 0.6, 0.8):
            for shift_mid in (0.0, 0.1):
                errs = collections.defaultdict(list)
                ious = collections.defaultdict(list)
                nsnap = 0
                for r, fn, W, g, i, j, m in rows:
                    on, _, _, _ = A.onsets_offsets(m, thr_sil)
                    ws, we = W[i]['s'], W[j]['e']
                    s_on, snapped = onset_rule(ws, on, win=win, shift_mid=shift_mid)
                    nsnap += snapped
                    for name, s in (('turbo', ws), ('turbo+0.2', ws + 0.2), ('onset', s_on), ('gold', g[0])):
                        errs[name].append(s - g[0])
                        ious[name].append(C.tiou(g, (s, we)))
                print(f'minsil {thr_sil} win {win} shift_mid {shift_mid}: snapped {nsnap}/{len(rows)}  ' +
                      '  '.join(f"{k}: |err|<=.06 {np.mean(np.abs(v) <= 0.06):.2f} med {np.median(v):+.2f} tIoU(oracle words, turbo end) {np.mean(ious[k]):.4f}"
                                for k, v in errs.items() if k != 'gold') + f"  | gold-start tIoU {np.mean(ious['gold']):.4f}")
    # end errors
    de = np.array([W[j]['e'] - g[1] for r, fn, W, g, i, j, m in rows])
    print('turbo END - gold end: median %+.2f, |.|<=0.06 %.2f, |.|<=0.1 %.2f, >0.2 late %.2f, < -0.2 early %.2f' % (
        np.median(de), np.mean(np.abs(de) <= 0.06), np.mean(np.abs(de) <= 0.1), np.mean(de > 0.2), np.mean(de < -0.2)))


if __name__ == '__main__':
    main()
