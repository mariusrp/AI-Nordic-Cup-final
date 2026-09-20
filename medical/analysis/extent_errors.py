"""For production spans that overlap gold but start too late / end too early by >= 0.35 s (whole words
missing), where is the gold boundary relative to our span? Checks the sentence unit (spans.build_units
'sentence') and clause unit containing our start/end, and acoustic onsets/offsets.
    python3 extent_errors.py [--show N]
"""
import collections
import json
import sys

import numpy as np

sys.path.insert(0, '/home/claude/nac/medical')
import spans as S  # noqa: E402

import acoustic as A  # noqa: E402
import common as C  # noqa: E402


def unit_of(units, t):
    for u in units:
        if u['start'] - 0.05 <= t <= u['end'] + 0.05:
            return u
    return min(units, key=lambda u: min(abs(u['start'] - t), abs(u['end'] - t)))


def main():
    show = int(sys.argv[sys.argv.index('--show') + 1]) if '--show' in sys.argv else 0
    cv = C.convs()
    cnt = collections.Counter()
    shown = 0
    for rep in ('a1_r1',):
        det = json.load(open(f'{C.WORK}/med-evolve-g3-1A/out/{rep}/arm0.detail.json'))
        for fn, rows in cv.items():
            tx = C.tx(fn)
            sent = S.build_units(tx, 'sentence')
            cla = S.build_units(tx, 'clause')
            m = A.speech_mask(A.frame_db(C.pcm(fn)), -50)
            on, _, off, _ = A.onsets_offsets(m, 0.1)
            for r, o in zip(rows, det[fn]):
                g = C.gold(r)
                if int(r['label']) != 1 or g is None or not o['answer'] or not o['span']:
                    continue
                ps, pe = o['span']
                raw_s = ps - 0.2
                if min(g[1], pe) - max(g[0], ps) <= 0:
                    continue
                if ps - g[0] >= 0.35:
                    us = unit_of(sent, raw_s)
                    uc = unit_of(cla, raw_s)
                    at_on = np.min(np.abs(on - g[0])) <= 0.1
                    k = ('gold@sent_start' if abs(us['start'] + 0.2 - g[0]) <= 0.35 else
                         'gold@clause_start' if abs(uc['start'] + 0.2 - g[0]) <= 0.35 else
                         'gold@earlier_sentence' if g[0] < us['start'] - 0.1 else 'gold@mid_sentence')
                    cnt['START_LATE ' + k] += 1
                    cnt['START_LATE gold_at_onset=%s' % at_on] += 1
                    if shown < show:
                        shown += 1
                        W = [w for w in C.words(tx) if g[0] - 1.5 <= w['s'] <= max(g[1], pe) + 0.3]
                        print(f"\n[{k}] {r['question_id']} Q: {r['question']}\n gold {g[0]:.2f}-{g[1]:.2f} pred {ps:.2f}-{pe:.2f} quote: {o['quote']!r}\n  "
                              + ' '.join(('<<' if abs(w['s'] - g[0]) < 0.3 else '') + w['w'].strip() + f"[{w['s']:.1f}]" for w in W))
                if g[1] - pe >= 0.35:
                    ue = unit_of(sent, pe)
                    uce = unit_of(cla, pe)
                    k = ('gold@sent_end' if abs(ue['end'] - g[1]) <= 0.35 else
                         'gold@clause_end' if abs(uce['end'] - g[1]) <= 0.35 else
                         'gold@later_sentence' if g[1] > ue['end'] + 0.1 else 'gold@mid_sentence')
                    cnt['END_EARLY ' + k] += 1
    for k, v in sorted(cnt.items()):
        print(f'{k:40s} {v}')


if __name__ == '__main__':
    main()
