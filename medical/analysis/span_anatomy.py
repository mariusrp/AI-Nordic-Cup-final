"""Anatomy of gold spans vs production spans, in turbo words (31 visible conversations).
Start type: 'utt'  = acoustic silence >= 0.1 s right before the first word (speech onset),
            'sent' = first word opens a sentence (previous word ends with .?!) without a pause,
            'mid'  = otherwise.
End type:   'sent' = last word ends with .?!   'clause' = ends with ,;:   'mid' = otherwise.
Also: number of sentences touched, duration, and whether the span crosses a pause >= 0.4 s (likely a
speaker change: Q/A exchange).
    python3 span_anatomy.py
"""
import collections
import json
import re

import numpy as np

import acoustic as A
import common as C

SENT = re.compile(r'[.?!]["\')\]]*$')
CLAUSE = re.compile(r'[,;:]["\')\]]*$')


def word_range(W, s, e):
    i = next((k for k, w in enumerate(W) if w['e'] > s + 0.04), len(W) - 1)
    j = i
    for k, w in enumerate(W):
        if w['s'] < e - 0.04:
            j = max(j, k)
    return i, j


def anatomy(W, m, i, j):
    prev = W[i - 1]['w'].strip() if i > 0 else '.'
    ws = W[i]['s']
    # acoustic silence just before the first word's true onset: look for >=0.1 s silence in [ws-0.1, ws+0.6]
    on_sil = A.silence_before(m, ws + 0.6)  # cheap proxy computed below instead
    st = 'sent' if SENT.search(prev) else 'mid'
    last = W[j]['w'].strip()
    et = 'sent' if SENT.search(last) else ('clause' if CLAUSE.search(last) else 'mid')
    nsent = 1 + sum(1 for k in range(i, j) if SENT.search(W[k]['w'].strip()))
    return st, et, nsent


def main():
    cv = C.convs()
    G = collections.Counter()
    P = collections.Counter()
    gdur, pdur, gns, pns = [], [], [], []
    det = json.load(open(f'{C.WORK}/med-evolve-g3-1A/out/a1_r1/arm0.detail.json'))
    for fn, rows in cv.items():
        W = C.words(C.tx(fn))
        m = A.speech_mask(A.frame_db(C.pcm(fn)), -50)
        on, onsil, _, _ = A.onsets_offsets(m, 0.1)
        for r, o in zip(rows, det[fn]):
            g = C.gold(r)
            if int(r['label']) != 1 or g is None:
                continue
            i, j = word_range(W, *g)
            st, et, ns = anatomy(W, m, i, j)
            at_on = np.min(np.abs(on - g[0])) <= 0.1
            G[('start', 'utt' if at_on else st)] += 1
            G[('end', et)] += 1
            G[('nsent', min(ns, 4))] += 1
            gdur.append(g[1] - g[0])
            gns.append(ns)
            if o['answer'] and o['span']:
                ps, pe = o['span']
                i2, j2 = word_range(W, ps - 0.2, pe)
                st2, et2, ns2 = anatomy(W, m, i2, j2)
                at_on2 = np.min(np.abs(on - W[i2]['s'])) <= 0.6 and any((on >= W[i2]['s'] - 0.1) & (on <= W[i2]['s'] + 0.6))
                P[('start', 'utt' if at_on2 else st2)] += 1
                P[('end', et2)] += 1
                P[('nsent', min(ns2, 4))] += 1
                pdur.append(pe - ps)
                pns.append(ns2)
    ng, npred = len(gdur), len(pdur)
    print(f'gold n={ng}: duration median {np.median(gdur):.2f} mean {np.mean(gdur):.2f}; sentences touched mean {np.mean(gns):.2f}')
    print(f'prod n={npred}: duration median {np.median(pdur):.2f} mean {np.mean(pdur):.2f}; sentences touched mean {np.mean(pns):.2f}')
    for key in ('start', 'end', 'nsent'):
        ks = sorted({k[1] for k in list(G) + list(P) if k[0] == key}, key=str)
        print(key, ' | '.join(f'{k}: gold {G[(key, k)] / ng:.2f} prod {P[(key, k)] / npred:.2f}' for k in ks))


if __name__ == '__main__':
    main()
