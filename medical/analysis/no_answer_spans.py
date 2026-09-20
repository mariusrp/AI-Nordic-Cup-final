"""Scoring fact: the upstream scorer computes tIoU for every gold-yes question from the returned span,
whatever the returned answer is (local_evaluator.Statistics.record: iou = temporal_iou(gold, predicted) for
label == 1; utils.mean_temporal_iou takes gold labels + spans only, not predicted answers).
So a span attached to a NO answer still earns tIoU if the gold answer is yes.

This replays cached production predictions (arm0 = production config, 3 fresh LLM runs from
medical-evolve-g3-1) under:
  prod         : as served (span only on yes)
  attach       : same answers, cand_span attached to every NO answer too
  attach+bias b: answers re-thresholded with yes-bias b on logit(p), spans on everything
    python3 no_answer_spans.py <dir with a1_r*/arm0.detail.json>
"""
import json
import math
import sys

import numpy as np

import common as C


def score(det, cv, keep, mode, bias=1.2):
    acc = []
    ti = []
    for fn, rows in cv.items():
        if keep is not None and fn not in keep:
            continue
        res = det[fn]
        for r, o in zip(rows, res):
            p = min(max(o['p'], 1e-6), 1 - 1e-6)
            if mode == 'prod':
                yes = o['answer']
                span = o['span'] if yes else None
            else:
                if o['src'] == 'llm' and mode.startswith('bias'):
                    yes = math.log(p / (1 - p)) + bias > 0
                else:
                    yes = o['answer']
                span = o['cand_span']
            lab = int(r['label'])
            acc.append(int(yes) == lab)
            g = C.gold(r)
            if lab == 1 and g is not None:
                ti.append(C.tiou(g, tuple(span) if span else None))
    a, t = np.mean(acc), np.mean(ti)
    return 0.4 * a + 0.6 * t, a, t


def main():
    D = sys.argv[1] if len(sys.argv) > 1 else C.WORK + '/med-evolve-g3-1A/out/'
    cv = C.convs()
    dev = C.split_dev()
    test = set(cv) - dev
    for rep in ('a1_r1', 'a1_r2', 'a1_r3'):
        det = json.load(open(f'{D}/{rep}/arm0.detail.json'))
        base = score(det, cv, None, 'prod')
        line = f'{rep} prod all {base[0]:.4f} (acc {base[1]:.4f} tIoU {base[2]:.4f})'
        for mode, b in (('attach', 1.2), ('bias', 1.2), ('bias', 0.0), ('bias', -1.0), ('bias', -2.0)):
            s = score(det, cv, None, mode, b)
            sd = score(det, cv, dev, mode, b)[0] - score(det, cv, dev, 'prod')[0]
            st = score(det, cv, test, mode, b)[0] - score(det, cv, test, 'prod')[0]
            line += f' | {mode} b={b}: {s[0]:.4f} (acc {s[1]:.4f} tIoU {s[2]:.4f}; d dev {sd:+.4f} test {st:+.4f})'
        print(line)
        # where do FN / FP come from
        fn_n = fp_n = 0
        pz = []
        for fn, rows in cv.items():
            for r, o in zip(rows, det[fn]):
                lab = int(r['label'])
                if lab == 1 and not o['answer']:
                    fn_n += 1
                    pz.append(('FN', r['question_id'], round(o['p'], 3), round(C.tiou(C.gold(r), tuple(o['cand_span']) if o['cand_span'] else None), 2)))
                if lab == 0 and o['answer']:
                    fp_n += 1
                    pz.append(('FP', r['question_id'], r['question_type'], round(o['p'], 3)))
        print('   FN', fn_n, 'FP', fp_n, pz)


if __name__ == '__main__':
    main()
