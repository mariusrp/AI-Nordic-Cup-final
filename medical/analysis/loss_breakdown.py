"""Where does production lose tIoU? Decompose 1 - tIoU per gold-yes question (cached production preds,
arm0 of the 3 medical-evolve-g3-1 replicates) into:
  fn        : answered no (span not returned)
  miss      : span returned but no overlap with gold (wrong location)
  start_*   : |pred_start - gold_start| / union   (timing if < 0.35 s, else word choice)
  end_*     : |pred_end - gold_end| / union
For overlapping spans 1 - I/U = (|ds| + |de|) / U exactly.
Points = tIoU loss * 0.6 / n_gold_yes (score units).
    python3 loss_breakdown.py
"""
import collections
import json

import numpy as np

import common as C

TIMING = 0.35


def breakdown(det, cv, keep=None):
    L = collections.Counter()
    n = 0
    for fn, rows in cv.items():
        if keep is not None and fn not in keep:
            continue
        for r, o in zip(rows, det[fn]):
            g = C.gold(r)
            if int(r['label']) != 1 or g is None:
                continue
            n += 1
            sp = o['span'] if o['answer'] else None
            if not o['answer']:
                L['fn'] += 1
                continue
            if sp is None:
                L['nospan'] += 1
                continue
            ps, pe = sp
            inter = max(0.0, min(g[1], pe) - max(g[0], ps))
            if inter <= 0:
                L['miss'] += 1
                continue
            U = max(g[1], pe) - min(g[0], ps)
            ds, de = ps - g[0], pe - g[1]
            ks = 'start_timing' if abs(ds) < TIMING else ('start_early_words' if ds < 0 else 'start_late_words')
            ke = 'end_timing' if abs(de) < TIMING else ('end_late_words' if de > 0 else 'end_early_words')
            L[ks] += abs(ds) / U
            L[ke] += abs(de) / U
    return L, n


def main():
    cv = C.convs()
    tot = collections.Counter()
    N = 0
    for rep in ('a1_r1', 'a1_r2', 'a1_r3'):
        det = json.load(open(f'{C.WORK}/med-evolve-g3-1A/out/{rep}/arm0.detail.json'))
        L, n = breakdown(det, cv)
        tot.update(L)
        N += n
    loss = sum(tot.values()) / N
    print(f'production (3 runs): mean tIoU {1 - loss:.4f}; tIoU loss per gold-yes q {loss:.4f} = score {0.6 * loss:.4f}')
    for k, v in tot.most_common():
        print(f'  {k:18s} tIoU loss {v / N:.4f}  score pts {0.6 * v / N:.4f}  ({100 * v / sum(tot.values()):.0f}%)')


if __name__ == '__main__':
    main()
