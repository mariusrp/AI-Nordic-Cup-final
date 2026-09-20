"""Pool several rules_arms.py runs of the SAME two arms into one paired report.

rules_arms.py numbers its prediction files from rep0 inside each --out-dir, so a second run of the
same arms lands in a second directory. This pools all of them: per-conversation score averaged over
ALL reps, paired diff vs base, percentile bootstrap CI over the 39 conversations, dev/test halves,
and the per-question tIoU sign test on the gold-yes rows.

    UPSTREAM=/workspace/.holdout/upstream-full python3 analysis/pool_arms.py --arm TRIM3 \
        /workspace/med-trim/out/t1 /workspace/med-trim/out/t2
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import offline_eval as OE  # noqa: E402

UPSTREAM = os.environ.get('UPSTREAM', '/workspace/.holdout/upstream-full')
sys.path.insert(0, os.path.join(UPSTREAM, 'medical-appointment'))
from utils import evidence_interval, gold_evidence, temporal_iou  # noqa: E402


def load(dirs, arm):
    out = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, f'{arm}.rep*.json'))):
            out.append(json.load(open(p)))
    return out


def boot_ci(diffs, b=4000, seed=2026):
    rnd = random.Random(seed)
    n = len(diffs)
    ms = sorted(sum(diffs[rnd.randrange(n)] for _ in range(n)) / n for _ in range(b))
    return ms[int(0.025 * (b - 1))], ms[int(0.975 * (b - 1))], sum(1 for m in ms if m > 0) / b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--arm', default='TRIM3')
    ap.add_argument('--base', default='base')
    a = ap.parse_args()
    B, C = load(a.dirs, a.base), load(a.dirs, a.arm)
    assert len(B) == len(C) and B, f'{len(B)} base reps vs {len(C)} {a.arm} reps'
    reps = len(B)
    names = sorted(B[0])
    dev = OE.split_ids('dev')
    print(f'{reps} reps x {len(names)} conversations   base={a.base} arm={a.arm}')
    for tag, rr in ((a.base, B), (a.arm, C)):
        ss = [OE.summary(OE.score(r, 'all'))['score'] for r in rr]
        ty = [OE.summary(OE.score(r, 'all'))['tiou_yes'] for r in rr]
        print(f'  {tag:<6} score {sum(ss) / reps:.4f}  reps {[round(x, 4) for x in ss]}  '
              f'tiou_yes {sum(ty) / reps:.4f}')

    def cs(rr, fn):
        return sum(OE.summary(OE.score({fn: r[fn]}, 'all'))['score'] for r in rr) / reps

    d = {fn: cs(C, fn) - cs(B, fn) for fn in names}
    for split, keys in (('all', names), ('dev', [f for f in names if f in dev]),
                        ('test', [f for f in names if f not in dev])):
        v = [d[fn] for fn in keys]
        n = len(v)
        m = sum(v) / n
        se = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1) / n)
        lo, hi, pp = boot_ci(v)
        print(f'  paired {split:<4} n {n:>2}  {m:+.4f} se {se:.4f} CI [{lo:+.4f},{hi:+.4f}] '
              f'p_pos {pp:.3f}  {sum(1 for x in v if x > 1e-9)} better / {sum(1 for x in v if x < -1e-9)} worse')
    rows = {fn: r for fn, r in OE.group_questions_by_conversation()}
    dt, counts = {}, {k: [0, 0] for k in ('exact', 'under', 'over', 'shift', 'wrong', 'none')}
    for fn in names:
        for qi, row in enumerate(rows.get(fn, [])):
            if int(row['label']) != 1:
                continue
            g = gold_evidence(row)
            bt, ct, fired = [], [], False
            for r in range(reps):
                bp, cp = B[r][fn], C[r][fn]
                bs = evidence_interval(bp['evidence_start'][qi], bp['evidence_end'][qi]) if bp['answers'][qi] else None
                cps = evidence_interval(cp['evidence_start'][qi], cp['evidence_end'][qi]) if cp['answers'][qi] else None
                if bs != cps:
                    fired = True
                bt.append(temporal_iou(g, bs) if bs else 0.0)
                ct.append(temporal_iou(g, cps) if cps else 0.0)
            if fired:
                dt[row.get('question_id', f'{fn}:{qi}')] = sum(ct) / reps - sum(bt) / reps
    better = sum(1 for x in dt.values() if x > 1e-9)
    worse = sum(1 for x in dt.values() if x < -1e-9)
    print(f'  sign test on gold-yes rows: touched {len(dt)}, {better} better / {worse} worse, '
          f'mean dtIoU on touched {sum(dt.values()) / len(dt):+.4f}' if dt else '  no rows touched')


if __name__ == '__main__':
    main()
