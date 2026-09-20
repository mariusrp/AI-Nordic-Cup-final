"""Cross-model location diversity: does a second LLM (Qwen3.8-27B-AWQ, bring-up run) miss the same questions as the
production Qwen3.5-35B-A3B? And a zero-parameter fusion rule: keep the 35B span unless both models answered yes with
NON-overlapping spans, then take the 27B span (answers unchanged).
Data: 7 cached 35B production-prompt runs (fast-r5-2 arm0 x2, evolve-g3-1 arm0 x3, fast-r4-1 base, bring-up 35B t0) and
the single bring-up 27B t0 run (POD=gpu /workspace/runs/med/llm_q38_27b_t0.detail.json; bring-up prompt, same turbo
transcripts). Spans compared as served (shift differences cancel in the overlap test; tIoU of the 27B span uses its
bring-up timing, no +0.2 s shift, so the switch gain is conservative).
    python3 cross_model.py
"""
import glob
import json

import numpy as np

import common as C

W = C.WORK
R35 = (sorted(glob.glob(f'{W}/fr5/fr5_rep*/arm0.detail.json'))
       + [f'{W}/med-evolve-g3-1A/out/{r}/arm0.detail.json' for r in ('a1_r1', 'a1_r2', 'a1_r3')]
       + [f'{W}/medical-fast-r4-1/out/r4_1/base.detail.json', f'{W}/bringup/llm_q35_a3b_t0.detail.json'])
R27 = f'{W}/bringup/llm_q38_27b_t0.detail.json'


def sp(o):
    return o['span'] if o['answer'] and o['span'] else None


def main():
    cv = C.convs()
    dev = C.split_dev()
    D27 = json.load(open(R27))
    uniq = {}
    per = {h: {fn: [] for fn in cv} for h in ('base', 'fuse')}
    for p in R35:
        D = json.load(open(p))
        for fn, rs in cv.items():
            for qi, r in enumerate(rs):
                g = C.gold(r)
                if int(r['label']) != 1 or g is None:
                    continue
                sa, sb = sp(D[fn][qi]), sp(D27[fn][qi])
                ta = C.tiou(g, sa)
                tf = ta
                if sa is not None and sb is not None and min(sa[1], sb[1]) - max(sa[0], sb[0]) <= 0:
                    tf = C.tiou(g, sb)
                    uniq.setdefault((fn, qi, r['question']), []).append((ta, tf))
                per['base'][fn].append(ta)
                per['fuse'][fn].append(tf)
    print('unique questions with a 35B/27B location disagreement:', len(uniq))
    for (fn, qi, q), v in sorted(uniq.items(), key=lambda t: -len(t[1])):
        v = np.array(v)
        print(f'  {fn:28s} {"dev " if fn in dev else "test"} n_runs {len(v)}  35B {v[:, 0].mean():.2f} -> 27B {v[:, 1].mean():.2f}  {q[:60]}')
    for half in ('dev', 'test', 'all'):
        fns = [fn for fn in cv if half == 'all' or ((fn in dev) == (half == 'dev'))]
        d = np.array([np.sum(per['fuse'][fn]) - np.sum(per['base'][fn]) for fn in fns])
        n = sum(len(per['base'][fn]) for fn in fns)
        print(f'{half:4s} fusion d {d.sum() / n:+.4f} +- {d.std(ddof=1) * np.sqrt(len(fns)) / n:.4f} tIoU '
              f'(score {0.6 * d.sum() / n:+.4f}); conversations changed {int((d != 0).sum())}/{len(fns)}')


if __name__ == '__main__':
    main()
