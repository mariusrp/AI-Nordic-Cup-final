"""Location voting across PROMPT variants of the same 35B (and optionally the 27B), zero fitted parameters.

Voters: the 4 interleaved arms of one fast-r5-2 replicate (arm0 = production prompt, arm1 no examples, arm2 pause
units, arm3 question first), optionally + the bring-up 27B run. Rule (fixed before looking at gains): keep arm0's span
unless arm0 answered yes with a span that overlaps NO other voter's span while at least two other voters' spans
overlap each other (a majority location without arm0); then take the span of the voter with the most overlaps
(ties -> lowest arm index, 27B last). Answers are unchanged (arm0's). Also reports how often same-prompt resampling
disagrees on location.
    python3 prompt_vote.py
"""
import json

import numpy as np

import common as C

W = C.WORK


def sp(o):
    return o['span'] if o['answer'] and o['span'] else None


def ov(a, b):
    return a is not None and b is not None and min(a[1], b[1]) - max(a[0], b[0]) > 0


def vote(spans):
    s0 = spans[0]
    if s0 is None:
        return s0, False
    others = list(range(1, len(spans)))
    if any(ov(s0, spans[i]) for i in others):
        return s0, False
    cnt = {i: sum(ov(spans[i], spans[j]) for j in others if j != i) for i in others if spans[i] is not None}
    if not cnt or max(cnt.values()) < 1:
        return s0, False
    best = max(cnt.items(), key=lambda t: (t[1], -t[0]))[0]
    return spans[best], True


def main():
    cv = C.convs()
    dev = C.split_dev()
    b27 = json.load(open(f'{W}/bringup/llm_q38_27b_t0.detail.json'))
    for with27 in (False, True):
        for rep in ('fr5_rep1', 'fr5_rep2'):
            arms = [json.load(open(f'{W}/fr5/{rep}/arm{a}.detail.json')) for a in range(4)] + ([b27] if with27 else [])
            per = {'base': {}, 'vote': {}}
            nsw = 0
            for fn, rs in cv.items():
                for qi, r in enumerate(rs):
                    g = C.gold(r)
                    if int(r['label']) != 1 or g is None:
                        continue
                    spans = [sp(A[fn][qi]) for A in arms]
                    s, sw = vote(spans)
                    nsw += sw
                    per['base'].setdefault(fn, []).append(C.tiou(g, spans[0]))
                    per['vote'].setdefault(fn, []).append(C.tiou(g, s))
            line = f'{rep} voters arm0-3{" +27B" if with27 else ""}: switches {nsw}'
            for half in ('dev', 'test', 'all'):
                fns = [fn for fn in cv if half == 'all' or ((fn in dev) == (half == 'dev'))]
                d = np.array([np.sum(per['vote'][fn]) - np.sum(per['base'][fn]) for fn in fns])
                n = sum(len(per['base'][fn]) for fn in fns)
                line += f' | {half} {d.sum() / n:+.4f}+-{d.std(ddof=1) * np.sqrt(len(fns)) / n:.4f}'
            print(line)


if __name__ == '__main__':
    main()
