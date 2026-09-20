"""Is MMS_FA CTC start re-timing (T-asr, medical-evolve-g4-3, 'most promising' in the g4 meta-review) worth more
reps ON TOP of the onset rule, or is it a substitute for it? Replays start rules on the 6 FRESH production-prompt runs
(fr5, arms 0/1/3, spans from the r6-2 spans.quote_span with gate 'any'), paired per conversation, frozen scorer.
Arms (ends always turbo; no start shift unless stated):
  onset       deploy rule: sentstart4 + onset inside the first word - 0.04, else raw turbo start
  ctc         CTC start of the first quoted word - 0.1 (T-asr arm b)
  onset|ctc   sentstart4 + onset inside the first word - 0.04, else CTC start - 0.1 (mid-utterance starts go to CTC)
  ctc>onset   words re-timed by CTC first, then the onset rule on the CTC-timed first word, else CTC start - 0.1
  prod        production: turbo start + 0.2
CTC word times: T-asr MMS_FA cache (atx_mms, 1:1 with the turbo words).
    python3 ctc_vs_onset.py
"""
import glob
import json
import os
import statistics as st
import sys

import numpy as np

import common as C

R62 = os.environ.get('R62', os.path.join(C.WORK, 'r62'))
sys.path.insert(0, R62)
import offline_eval as OE  # noqa: E402
import spans as S  # noqa: E402

CTC_DIR = os.path.join(C.WORK, 'ctc', 'atx_mms')


def start_rule(span, words, ctc_s, onsets, arm, ctc_e=None):
    s, e = float(span[0]), float(span[1])
    i = min(range(len(words)), key=lambda idx: abs(words[idx]['s'] - s))
    if arm == 'ctc':
        ns = ctc_s[i] - 0.1
        return (ns, e) if ns < e else (s, e)
    kk = i
    while kk > 0 and not S.SENT_END.search(words[kk - 1]['w'].strip()) and i - kk < 4:
        kk -= 1
    if kk == 0 or S.SENT_END.search(words[kk - 1]['w'].strip()):
        i = kk
    ws, we_first = words[i]['s'], words[i]['e']
    if arm == 'ctc>onset':
        ws, we_first = ctc_s[i], ctc_e[i]
    cands = [t for t in onsets if ws - 0.1 <= t <= we_first - 0.05]
    if cands:
        ns = cands[0] - 0.04
    else:
        ns = ws if arm == 'onset' else ctc_s[i] - 0.1
    return (ns, e) if ns < e else (s, e)


def main():
    runs = [rp for rp in sorted(glob.glob(os.path.join(C.WORK, 'fr5', 'fr5_rep*', 'arm*.detail.json')))
            if not rp.endswith('arm2.detail.json')]
    qs = {fn: [r['question'] for r in rows] for fn, rows in OE.group_questions_by_conversation()}
    fns = sorted(qs)
    dev = OE.split_ids('dev')
    cd = {}
    for fn in fns:
        tx = S.load_transcript(os.path.join(C.WORK, 'tx', 'large-v3-turbo', fn.replace('.mp3', '.json')))
        units = S.build_units(tx, 'sentence')
        words = [w for u in units for w in u['words']]
        tw = C.words(C.tx(fn))
        aw = C.words(json.load(open(os.path.join(CTC_DIR, fn.replace('.mp3', '.json')))))
        assert len(tw) == len(aw)
        m = {(round(w['start'] if 'start' in w else w['s'], 3)): a for w, a in zip(tw, aw)}
        ctc_s, ctc_e = [], []
        for w in words:
            a = m.get(round(w['s'], 3))
            ctc_s.append((a.get('start', a.get('s')) if a else w['s']))
            ctc_e.append((a.get('end', a.get('e')) if a else w['e']))
        ons = S.energy_onsets(np.load(os.path.join(C.WORK, 'pcm', fn.replace('.mp3', '.npy'))))
        cd[fn] = (units, words, ctc_s, ctc_e, ons, tx.get('duration'))
    arms = ('prod', 'onset', 'ctc', 'onset|ctc', 'ctc>onset')
    per = {a: {fn: [] for fn in fns} for a in arms}
    for rp in runs:
        det = json.load(open(rp))
        for fn in fns:
            units, words, ctc_s, ctc_e, ons, dur = cd[fn]
            preds = {a: {'answers': [], 'evidence_start': [], 'evidence_end': []} for a in arms}
            for q, r in zip(qs[fn], det[fn]):
                yes = bool(r['answer'])
                gate = 'first'
                for a in arms:
                    sp, _ = S.quote_span(units, r['ids'], r['quote'] or '', gate='first' if a == 'prod' else 'any',
                                         fallback='first', question=q, duration=dur)
                    if sp is not None:
                        if a == 'prod':
                            sp = S.calibrate_span(sp, words, 0.2, 0.0, 0, 0.3, 0.0, 0.0, duration=dur)
                        else:
                            sp = start_rule(sp, words, ctc_s, ons, a, ctc_e)
                            sp = S.calibrate_span(sp, words, 0.0, 0.0, 0, 0.3, 0.0, 0.0, duration=dur)
                        s, e = float(sp[0]), float(sp[1])
                        if e <= s:
                            e = s + 0.3
                        sp = (round(s, 2), round(e, 2))
                    preds[a]['answers'].append(yes)
                    preds[a]['evidence_start'].append(sp[0] if yes and sp else None)
                    preds[a]['evidence_end'].append(sp[1] if yes and sp else None)
            for a in arms:
                per[a][fn].append(OE.score({fn: preds[a]}, 'all').final_score)
    nr = len(runs)
    for a, ref in (('onset', 'prod'), ('ctc', 'prod'), ('onset|ctc', 'prod'), ('ctc', 'onset'), ('onset|ctc', 'onset'),
                    ('ctc>onset', 'prod'), ('ctc>onset', 'onset')):
        line = f'{a:10s} vs {ref:6s}'
        for half in ('dev', 'test', 'all'):
            hf = [fn for fn in fns if half == 'all' or ((fn in dev) == (half == 'dev'))]
            cs = [sum(x - y for x, y in zip(per[a][fn], per[ref][fn])) for fn in hf]
            n = nr * len(hf)
            mm = sum(cs) / n
            se = st.stdev(cs) * len(hf) ** 0.5 / n
            line += f' | {half} {mm:+.4f}+-{se:.4f} ({mm / se if se else 0:+.1f} se)'
        print(line)


if __name__ == '__main__':
    main()
