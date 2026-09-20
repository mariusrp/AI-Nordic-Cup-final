"""Replay boundary rules on cached production predictions (no LLM calls).

Production spans (arm0 of medical-evolve-g3-1 a1_r1..r3; same config as main) are mapped back to turbo words
(raw start = span start - 0.2 s), then re-bounded by a rule. Reports mean tIoU over gold-yes questions and the
paired per-conversation difference vs production (se over conversations), for dev and test halves.
Rules:
  prod           : served span
  onset(w,m,sh)  : start = first acoustic onset (>= m s silence before it) in [ws-0.1, ws+w], else ws+sh
  sentstart(k)   : move the first word back to its sentence start if <= k words away
  sentend(k)     : move the last word forward to its sentence end if <= k words away
    python3 replay_rules.py
"""
import json
import re

import numpy as np

import acoustic as A
import common as C

SENT = re.compile(r'[.?!]["\')\]]*$')
REPS = ('a1_r1', 'a1_r2', 'a1_r3')


def nearest(arr, t):
    return int(np.argmin(np.abs(arr - t)))


def make_rule(name, **kw):
    def rule(ctx, ps, pe):
        W, S, E, on = ctx['W'], ctx['S'], ctx['E'], ctx['on'][kw.get('minsil', 0.1)]
        if name == 'prod':
            return ps, pe
        i = nearest(S, ps - 0.2)
        j = max(i, nearest(E, pe))
        if kw.get('sentstart'):
            k = i
            while k > 0 and not SENT.search(W[k - 1]['w'].strip()) and i - k < kw['sentstart']:
                k -= 1
            if k == 0 or SENT.search(W[k - 1]['w'].strip()):
                i = k
        if kw.get('sentend'):
            k = j
            while k < len(W) - 1 and not SENT.search(W[k]['w'].strip()) and k - j < kw['sentend']:
                k += 1
            if SENT.search(W[k]['w'].strip()):
                j = k
        ws = W[i]['s']
        if kw.get('inword'):
            # onset (>= minsil s of silence before it) inside the first word -> the word is utterance-initial and
            # whisper clamped its start early (max-duration hack); start at the acoustic onset, else keep ws
            c = on[(on >= ws - 0.1) & (on <= W[i]['e'] - 0.05)]
            s = float(c[0]) + kw.get('on_off', 0.0) if len(c) else ws + kw.get('shift_mid', 0.0)
        elif kw.get('onset'):
            c = on[(on >= ws - 0.1) & (on <= ws + kw['onset'])]
            s = float(c[0]) if len(c) else ws + kw.get('shift_mid', 0.0)
        else:
            s = ws + kw.get('shift', 0.2)
        e = W[j]['e'] + kw.get('shift_e', 0.0)
        if e <= s:
            e = s + 0.3
        return s, e
    return rule


def evaluate(rules, reps=REPS):
    cv = C.convs()
    dev = C.split_dev()
    ctxs = {}
    for fn in cv:
        W = C.words(C.tx(fn))
        m = A.speech_mask(A.frame_db(C.pcm(fn)), -50)
        ctxs[fn] = {'W': W, 'S': np.array([w['s'] for w in W]), 'E': np.array([w['e'] for w in W]),
                    'on': {ms: A.onsets_offsets(m, ms)[0] for ms in (0.1, 0.2, 0.3)}}
    dets = {rep: json.load(open(rep if rep.endswith('.json') else f'{C.WORK}/med-evolve-g3-1A/out/{rep}/arm0.detail.json'))
            for rep in reps}
    out = {}
    for rname, rule in rules.items():
        per = {}  # fn -> list of tIoU sums over reps
        for fn, rows in cv.items():
            vals = []
            for rep in reps:
                for r, o in zip(rows, dets[rep][fn]):
                    g = C.gold(r)
                    if int(r['label']) != 1 or g is None:
                        continue
                    if o['answer'] and o['span']:
                        sp = rule(ctxs[fn], *o['span'])
                    else:
                        sp = None
                    vals.append(C.tiou(g, sp))
            per[fn] = vals
        out[rname] = per
    return out, dev


def report(out, dev, base='prod'):
    for rname, per in out.items():
        line = f'{rname:34s}'
        for half in ('dev', 'test', 'all'):
            fns = [fn for fn in per if half == 'all' or ((fn in dev) == (half == 'dev'))]
            mean = np.mean([v for fn in fns for v in per[fn]])
            d = np.array([np.sum(per[fn]) - np.sum(out[base][fn]) for fn in fns])
            n = sum(len(per[fn]) for fn in fns)
            # per-question mean diff, se from conversation totals
            md = d.sum() / n
            se = d.std(ddof=1) * np.sqrt(len(fns)) / n
            line += f' | {half} tIoU {mean:.4f} d {md:+.4f}+-{se:.4f}'
        print(line)


def main():
    rules = {'prod': make_rule('prod'),
             'shift0': make_rule('x', shift=0.0),
             'shift0.2 (re-derived)': make_rule('x', shift=0.2)}
    for w in (0.4, 0.6, 0.8):
        for ms in (0.1, 0.2):
            for sh in (0.0, 0.1):
                rules[f'onset w{w} m{ms} sh{sh}'] = make_rule('x', onset=w, minsil=ms, shift_mid=sh)
    for k in (2, 4, 8):
        rules[f'sentstart{k}'] = make_rule('x', sentstart=k)
        rules[f'sentstart{k}+onset.6'] = make_rule('x', sentstart=k, onset=0.6, minsil=0.1)
        rules[f'sentend{k}'] = make_rule('x', sentend=k)
    for ms in (0.1, 0.2):
        for sh in (0.0, 0.1):
            rules[f'inword m{ms} sh{sh}'] = make_rule('x', inword=1, minsil=ms, shift_mid=sh)
    for k in (2, 3, 4, 6):
        rules[f'sentstart{k}+inword m0.1'] = make_rule('x', sentstart=k, inword=1, minsil=0.1)
    import os
    reps = tuple(os.environ['REPS'].split(',')) if os.environ.get('REPS') else REPS
    out, dev = evaluate(rules, reps)
    report(out, dev)


if __name__ == '__main__':
    main()
