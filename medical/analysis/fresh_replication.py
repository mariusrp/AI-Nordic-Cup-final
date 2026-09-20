"""Out-of-sample replication of the cycle-2 start rule on FRESH cached LLM runs (no LLM calls).

The rule and its settings were fixed in cycle 2 (commit 91060c9) before these runs existed:
  R = sentstart4 + onset inside the first word (min silence 0.1 s) - 0.04 s, else raw turbo start (no shift);
      ends on turbo word ends.
Runs: medical-fast-r5-2 (FB5-B, 2 interleaved replicates x 4 arms, 15:37): arm0 = production config (FC=0, shift 0.2),
arm1 = no SYSTEM examples, arm2 = MED_UNIT_MODE=pause, arm3 = question first. The rule is a post-LLM edge rule, so
every arm is a valid test; arm0 is the exact production config.
Reports the paired per-question tIoU gain of R (and its two parts) over the served span, se over conversations,
per run and pooled (conversation totals summed over runs), dev/test/all.
    python3 fresh_replication.py [dir_with_fr5_rep1_fr5_rep2]
"""
import glob
import os
import sys

import numpy as np

import common as C
import replay_rules as RR

DEF = os.path.join(C.WORK, 'fr5')  # gunzip of medical-fast-r5-2:medical/cache/fast-r5-2/fr5_rep*/arm*.detail.json.gz


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DEF
    runs = sorted(glob.glob(os.path.join(root, 'fr5_rep*', 'arm*.detail.json')))
    rules = {'prod': RR.make_rule('prod'),
             'R: sentstart4+inword-0.04': RR.make_rule('x', sentstart=4, inword=1, minsil=0.1, on_off=-0.04),
             'onset only (inword-0.04)': RR.make_rule('x', inword=1, minsil=0.1, on_off=-0.04),
             'sentstart2+inword-0.04': RR.make_rule('x', sentstart=2, inword=1, minsil=0.1, on_off=-0.04),
             'raw turbo (shift 0)': RR.make_rule('x', shift=0.0)}
    out, dev = RR.evaluate(rules, tuple(runs))
    # per-run report
    cv = C.convs()
    for rp in runs:
        tag = '/'.join(rp.split('/')[-2:])
        sub, _ = RR.evaluate({'prod': rules['prod'], 'R': rules['R: sentstart4+inword-0.04']}, (rp,))
        line = f'{tag:34s}'
        for half in ('dev', 'test', 'all'):
            fns = [fn for fn in cv if half == 'all' or ((fn in dev) == (half == 'dev'))]
            d = np.array([np.sum(sub['R'][fn]) - np.sum(sub['prod'][fn]) for fn in fns])
            n = sum(len(sub['R'][fn]) for fn in fns)
            line += f' | {half} d {d.sum() / n:+.4f}+-{d.std(ddof=1) * np.sqrt(len(fns)) / n:.4f}'
        print(line)
    print('--- pooled over', len(runs), 'runs (conversation totals summed over runs) ---')
    RR.report(out, dev)
    arm0 = [r for r in runs if r.endswith('arm0.detail.json')]
    if arm0:
        print('--- production-config arm0 only (', len(arm0), 'runs) ---')
        out0, _ = RR.evaluate(rules, tuple(arm0))
        RR.report(out0, dev)


if __name__ == '__main__':
    main()
