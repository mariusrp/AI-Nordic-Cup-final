"""Which timing source reproduces the GOLD boundaries? (cycle 2; answers cycle-1 open question 3)

For every gold-yes span of the 31 visible conversations, take the turbo word the gold start falls in (oracle word
choice) and compare start estimators against the gold start:
  turbo        raw faster-whisper large-v3-turbo word start
  turbo+0.2    production (global +0.2 s)
  onset        cycle-1 rule: first energy onset (>= 0.1 s silence before) in [ws-0.1, we_first-0.05], else raw ws
  ctc          MMS_FA CTC word start of the same word (T-asr lane, medical-evolve-g4-3 cache atx_mms)
  ctc-0.1      T-asr arm b (shift fitted on dev)
  onset|ctc    onset rule, else CTC start (-0.1)  [mid-utterance starts go to CTC]
  onset|ctc0   onset rule, else raw CTC start
Ends: turbo word end vs CTC end vs energy offset, same oracle word.
Also the fine structure of gold - onset at 5 ms hop (is the gold start the onset itself, or a fixed lead before it?).
    MED_UND_DIR=... CTC_DIR=.../atx_mms python3 boundary_sources.py
"""
import json
import os

import numpy as np

import acoustic as A
import common as C

CTC_DIR = os.environ.get('CTC_DIR', os.path.join(C.WORK, 'ctc', 'atx_mms'))


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


def stats(name, err, ious):
    err = np.asarray(err)
    return (f'{name:12s} med {np.median(err):+.3f}  |e|med {np.median(np.abs(err)):.3f}  <=.02 {np.mean(np.abs(err) <= 0.021):.2f}'
            f'  <=.06 {np.mean(np.abs(err) <= 0.061):.2f}  <=.1 {np.mean(np.abs(err) <= 0.101):.2f}  >.3 {np.mean(np.abs(err) > 0.3):.2f}'
            f'  oracle-word tIoU {np.mean(ious):.4f}')


def main():
    dev = C.split_dev()
    seen = set()
    recs = []
    for r in C.rows():
        g = C.gold(r)
        if g is None or int(r['label']) != 1:
            continue
        fn = f"conversation_{r['transcript_id']}.mp3"
        key = (fn, g)
        if key in seen:
            continue
        seen.add(key)
        recs.append((fn, g))
    cache = {}
    res = {k: {'err': [], 'iou': [], 'half': []} for k in
           ('turbo', 'turbo+0.2', 'onset', 'ctc', 'ctc-0.1', 'onset|ctc-0.1', 'onset|ctc0', 'gold')}
    ends = {k: [] for k in ('turbo_end', 'ctc_end', 'offset')}
    fine = []
    kinds = []
    for fn, g in recs:
        if fn not in cache:
            x = C.pcm(fn)
            W = C.words(C.tx(fn))
            aW = C.words(json.load(open(os.path.join(CTC_DIR, fn.replace('.mp3', '.json')))))
            assert len(aW) == len(W)
            m = A.speech_mask(A.frame_db(x), -50)
            on, on_sil, off, off_sil = A.onsets_offsets(m, 0.1)
            # fine: 5 ms hop, 10 ms window
            dbf = A.frame_db(x, win=160, hop=80)
            cache[fn] = (W, aW, on, off, dbf)
        W, aW, on, off, dbf = cache[fn]
        i = first_word(W, g[0])
        j = max(i, last_word(W, g[1]))
        ws, we_first, we = W[i]['s'], W[i]['e'], W[j]['e']
        c = on[(on >= ws - 0.1) & (on <= we_first - 0.05)]
        s_on = float(c[0]) if len(c) else None
        cs = aW[i]['s']
        est = {'turbo': ws, 'turbo+0.2': ws + 0.2, 'onset': s_on if s_on is not None else ws,
               'ctc': cs, 'ctc-0.1': cs - 0.1,
               'onset|ctc-0.1': s_on if s_on is not None else cs - 0.1,
               'onset|ctc0': s_on if s_on is not None else cs, 'gold': g[0]}
        half = 'dev' if fn in dev else 'test'
        for k, s in est.items():
            res[k]['err'].append(s - g[0])
            res[k]['iou'].append(C.tiou(g, (s, we)))
            res[k]['half'].append(half)
        kinds.append('utt-initial' if s_on is not None else 'mid')
        ends['turbo_end'].append(we - g[1])
        ends['ctc_end'].append(aW[j]['e'] - g[1])
        k = np.argmin(np.abs(off - g[1])) if len(off) else None
        ends['offset'].append(off[k] - g[1] if k is not None else np.nan)
        # fine structure around the gold start: first 5 ms frame above -50 dBFS at/after gold-0.2
        t0 = int(max(0, (g[0] - 0.2)) / 0.005)
        above = np.where(dbf[t0:t0 + 120] > -50)[0]
        fine.append((above[0] * 0.005 + t0 * 0.005 - g[0]) if len(above) else np.nan)
    kinds = np.array(kinds)
    print(f'{len(recs)} unique gold spans; start kind: utterance-initial (onset inside first word) {np.sum(kinds == "utt-initial")}, mid {np.sum(kinds == "mid")}')
    for sel_name, sel in (('ALL', np.ones(len(kinds), bool)), ('utt-initial', kinds == 'utt-initial'), ('mid', kinds == 'mid')):
        print(f'--- {sel_name} (n={sel.sum()})')
        for k, v in res.items():
            if k == 'gold':
                continue
            print('  ' + stats(k, np.array(v['err'])[sel], np.array(v['iou'])[sel]))
        print('  ' + stats('gold', np.array(res['gold']['err'])[sel], np.array(res['gold']['iou'])[sel]))
    for half in ('dev', 'test'):
        sel = np.array(res['turbo']['half']) == half
        print(f'--- half {half} (n={sel.sum()}): oracle-word tIoU ' + '  '.join(
            f"{k} {np.mean(np.array(v['iou'])[sel]):.4f}" for k, v in res.items()))
    print('--- ENDS (oracle last word): error = estimate - gold end')
    for k, v in ends.items():
        v = np.array(v)
        v = v[np.isfinite(v)]
        print(f'  {k:10s} med {np.median(v):+.3f} |e|med {np.median(np.abs(v)):.3f} <=.02 {np.mean(np.abs(v) <= 0.021):.2f} <=.06 {np.mean(np.abs(v) <= 0.061):.2f} <=.1 {np.mean(np.abs(v) <= 0.101):.2f}')
    fine = np.array(fine)
    ui = kinds == 'utt-initial'
    f = fine[ui & np.isfinite(fine)]
    print('--- fine: first 5ms frame > -50 dBFS (searching from gold-0.2) minus gold start, utterance-initial starts')
    print('  quantiles 10/25/50/75/90: ' + ' '.join(f'{q:+.3f}' for q in np.quantile(f, [.1, .25, .5, .75, .9])))
    h = np.round(f / 0.01) * 0.01
    vals, cnt = np.unique(h, return_counts=True)
    print('  histogram (10 ms bins): ' + ' '.join(f'{v:+.2f}:{c}' for v, c in zip(vals, cnt) if abs(v) <= 0.2))


if __name__ == '__main__':
    main()
