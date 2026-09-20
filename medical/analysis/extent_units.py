"""Extent errors in SENTENCE units (cycle 2). For every gold-yes question where production's span overlaps
the gold, map both spans to turbo sentences (split at words ending in .?!) and classify:
  same     : same first and last sentence
  gold_more_before / gold_more_after : gold covers extra whole sentences before/after ours
  gold_less_before / gold_less_after : ours covers extra sentences
and, inside the boundary sentence, whether gold/pred start at the sentence start and end at the sentence end.
Also: which sentence TYPE the gold adds (question '?', short reply <= 3 words, statement), and whether a pause
(>= 0.3 s acoustic silence) separates it from ours (speaker change proxy).
Then oracle-free rules replayed on the cached runs:
  base      : onset + sentstart4 (cycle-1 recommendation)
  +prevq    : also extend the start back over the previous sentence if it is a question ('?')
  +nextans  : also extend the end over the next sentence if OUR last sentence is a question
  +nextshort: also extend the end over the next sentence if it is a short reply (<= 3 words)
  +prevshort: also extend the start back if our first sentence is a short reply (<= 3 words)
    python3 extent_units.py
"""
import collections
import json
import os
import re

import numpy as np

import acoustic as A
import common as C

SENT = re.compile(r'[.?!]["\')\]]*$')
RUNS = [f'{C.WORK}/med-evolve-g3-1A/out/{r}/arm0.detail.json' for r in ('a1_r1', 'a1_r2', 'a1_r3')] + \
       [f'{C.WORK}/medical-fast-r4-1/out/r4_1/base.detail.json']


def sentences(W):
    """list of (i0, i1) word index ranges; sid[k] = sentence of word k"""
    out, i0 = [], 0
    for k, w in enumerate(W):
        if SENT.search(w['w'].strip()):
            out.append((i0, k))
            i0 = k + 1
    if i0 < len(W):
        out.append((i0, len(W) - 1))
    sid = np.zeros(len(W), int)
    for n, (a, b) in enumerate(out):
        sid[a:b + 1] = n
    return out, sid


def word_range(W, s, e):
    i = next((k for k, w in enumerate(W) if w['e'] > s + 0.04), len(W) - 1)
    j = i
    for k, w in enumerate(W):
        if w['s'] < e - 0.04:
            j = max(j, k)
    return i, j


def stype(W, a, b):
    t = W[b]['w'].strip()
    if t.endswith('?'):
        return 'question'
    if b - a + 1 <= 3:
        return 'short'
    return 'statement'


def ctx_for(fn):
    W = C.words(C.tx(fn))
    m = A.speech_mask(A.frame_db(C.pcm(fn)), -50)
    on = A.onsets_offsets(m, 0.1)[0]
    ss, sid = sentences(W)
    return {'W': W, 'm': m, 'on': on, 'ss': ss, 'sid': sid, 'S': np.array([w['s'] for w in W]),
            'E': np.array([w['e'] for w in W])}


def gap_before(ctx, i):
    """acoustic silence right before word i (uses the onset rule window)"""
    W, on = ctx['W'], ctx['on']
    c = on[(on >= W[i]['s'] - 0.1) & (on <= W[i]['e'] - 0.05)]
    return len(c) > 0


def start_time(ctx, i):
    W, on = ctx['W'], ctx['on']
    c = on[(on >= W[i]['s'] - 0.1) & (on <= W[i]['e'] - 0.05)]
    return float(c[0]) if len(c) else W[i]['s']


def rule_span(ctx, ps, pe, opts):
    W, ss, sid = ctx['W'], ctx['ss'], ctx['sid']
    i = int(np.argmin(np.abs(ctx['S'] - (ps - 0.2))))
    j = max(i, int(np.argmin(np.abs(ctx['E'] - pe))))
    # sentstart4
    a = ss[sid[i]][0]
    if i - a <= 4:
        i = a
    if opts.get('prevq') and sid[i] > 0 and ss[sid[i]][0] == i:
        pa, pb = ss[sid[i] - 1]
        if W[pb]['w'].strip().endswith('?') and pb - pa + 1 <= opts['prevq']:
            i = pa
    if opts.get('prevshort') and ss[sid[i]][0] == i and sid[i] > 0:
        a, b = ss[sid[i]]
        if b - a + 1 <= 3 and j <= b:
            pa, pb = ss[sid[i] - 1]
            i = pa
    if opts.get('nextans') and ss[sid[j]][1] == j and sid[j] + 1 < len(ss):
        if W[j]['w'].strip().endswith('?'):
            na, nb = ss[sid[j] + 1]
            if nb - na + 1 <= opts['nextans']:
                j = nb
    if opts.get('nextshort') and ss[sid[j]][1] == j and sid[j] + 1 < len(ss):
        na, nb = ss[sid[j] + 1]
        if nb - na + 1 <= 3:
            j = nb
    s = start_time(ctx, i)
    e = W[j]['e']
    return s, max(e, s + 0.3)


def classify(ctx, g, sp):
    W, ss, sid = ctx['W'], ctx['ss'], ctx['sid']
    gi, gj = word_range(W, *g)
    pi, pj = word_range(W, sp[0] - 0.2, sp[1])
    out = []
    gs0, gs1, ps0, ps1 = sid[gi], sid[gj], sid[pi], sid[pj]
    if gs0 < ps0:
        out.append(('gold_more_before', stype(W, *ss[gs0])))
    elif gs0 > ps0:
        out.append(('gold_less_before', stype(W, *ss[ps0])))
    else:
        out.append(('same_first_sentence', 'gold@start' if gi == ss[gs0][0] else 'gold@mid',
                    'pred@start' if pi == ss[ps0][0] else 'pred@mid'))
    if gs1 > ps1:
        out.append(('gold_more_after', stype(W, *ss[gs1])))
    elif gs1 < ps1:
        out.append(('gold_less_after', stype(W, *ss[ps1])))
    else:
        out.append(('same_last_sentence', 'gold@end' if gj == ss[gs1][1] else 'gold@mid',
                    'pred@end' if pj == ss[ps1][1] else 'pred@mid'))
    return out


def main():
    cv = C.convs()
    dev = C.split_dev()
    ctxs = {fn: ctx_for(fn) for fn in cv}
    dets = [json.load(open(p)) for p in RUNS if os.path.exists(p)]
    cnt = collections.Counter()
    n_ov = 0
    for det in dets[:3]:
        for fn, rows in cv.items():
            for r, o in zip(rows, det[fn]):
                g = C.gold(r)
                if int(r['label']) != 1 or g is None or not o['answer'] or not o['span']:
                    continue
                sp = o['span']
                if min(g[1], sp[1]) - max(g[0], sp[0]) <= 0:
                    continue
                n_ov += 1
                for c in classify(ctxs[fn], g, sp):
                    cnt[c] += 1
    print(f'overlapping production spans (3 runs): {n_ov}')
    for k, v in sorted(cnt.items(), key=lambda kv: (kv[0][0], -kv[1])):
        print(f'  {" / ".join(k):60s} {v:4d} ({v / n_ov:.2f})')
    # rule replay
    rules = {'prod': None, 'base(onset+ss4)': {}, '+prevq8': {'prevq': 8}, '+prevq15': {'prevq': 15},
             '+prevq99': {'prevq': 99}, '+nextans8': {'nextans': 8}, '+nextans15': {'nextans': 15},
             '+nextshort': {'nextshort': 1}, '+prevshort': {'prevshort': 1},
             '+prevq15+nextans15': {'prevq': 15, 'nextans': 15}}
    per = {k: collections.defaultdict(list) for k in rules}
    fires = collections.Counter()
    for det in dets:
        for fn, rows in cv.items():
            for r, o in zip(rows, det[fn]):
                g = C.gold(r)
                if int(r['label']) != 1 or g is None:
                    continue
                for name, opts in rules.items():
                    if not (o['answer'] and o['span']):
                        sp = None
                    elif opts is None:
                        sp = tuple(o['span'])
                    else:
                        sp = rule_span(ctxs[fn], *o['span'], opts)
                        if name != 'base(onset+ss4)':
                            b = rule_span(ctxs[fn], *o['span'], {})
                            fires[name] += int(abs(b[0] - sp[0]) > 1e-6 or abs(b[1] - sp[1]) > 1e-6)
                    per[name][fn].append(C.tiou(g, sp))
    print(f'\nrule replay on {len(dets)} cached production runs (d = paired per-question tIoU diff vs BASE, se over convs)')
    base = per['base(onset+ss4)']
    for name in rules:
        line = f'{name:22s} fires {fires[name] / len(dets):5.1f}/run'
        for half in ('dev', 'test', 'all'):
            fns = [fn for fn in cv if half == 'all' or ((fn in dev) == (half == 'dev'))]
            n = sum(len(per[name][fn]) for fn in fns)
            mean = sum(sum(per[name][fn]) for fn in fns) / n
            d = np.array([sum(per[name][fn]) - sum(base[fn]) for fn in fns])
            line += f' | {half} {mean:.4f} d {d.sum() / n:+.4f}+-{d.std(ddof=1) * np.sqrt(len(fns)) / n:.4f}'
        print(line)


if __name__ == '__main__':
    main()
