"""Do the fast lane's r5 variants keep the no-harm property? (drone understanding lane, cycle 4; analysis only)

Runs the REAL tracker code (drone/tracker.py of a given tree) on hand-built detections:
  (a) FL-r5-B B2 "demote" mode: a verifier-dropped box is kept at conf x 1e-3 and passed to tracker.update.
      Does it still count as a HIT (confirming an unconfirmed track, cutting its miss count)?
  (b) FL-r5-A pre-verifier hedge: the same box twice (source class + hedge class) on a fresh object.
      Do both copies spawn, merge, and leave a track CONFIRMED after one view (hits = 2)?
  (c) FL-r5-B floor band cap min(raw x 0.01, 0.99 x lowest same-class conf IN THIS FRAME): can an extra
      outrank a production box of the same class in ANOTHER frame? (needs answer streams: --streams)
Usage: python noharm_check.py [DRONE_DIR] [--streams base.jsonl ...]
"""
import sys, os, json, collections
import numpy as np

drone = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(drone))
import tracker as T  # noqa: E402
from common import CLS_INDEX, NC  # noqa: E402

REGION = (1000.0, 500.0, 2920.0, 1580.0)   # an L1 view (source px)


def probs(c, p=1.0):
    v = np.zeros(NC); v[CLS_INDEX[c]] = p; return v


def show(tr, tag):
    for t in tr.tracks:
        print(f'   {tag}: track {t.id} hits={t.hits} miss={t.miss:.2f} confirmed={t.confirmed} '
              f'score={t.score():.4f} report={t.report_score():.4f}')


box = np.array([1500.0, 900.0, 1540.0, 930.0])
print('(a) demoted detection (conf 0.30 x 1e-3) re-observing an unconfirmed track')
tr = T.Tracker()
tr.predict(1, np.eye(3)); tr.update([(box, probs('mine_roller'), 0.30, False)], 1, REGION, 1)
show(tr, 'after birth      ')
tr.predict(2, np.eye(3)); tr.update([(box + 1, probs('mine_roller'), 0.30 * 1e-3, False)], 1, REGION, 2)
show(tr, 'after demoted hit')
tr2 = T.Tracker()
tr2.predict(1, np.eye(3)); tr2.update([(box, probs('mine_roller'), 0.30, False)], 1, REGION, 1)
tr2.predict(2, np.eye(3)); tr2.update([], 1, REGION, 2)
show(tr2, 'production (dropped)')

print('(b) pre-verifier hedge: same box as condor (verified .40) and jet_plane copy (verified .30), new object')
tr = T.Tracker()
tr.predict(1, np.eye(3))
tr.update([(box, probs('condor'), 0.40, False), (box, probs('jet_plane'), 0.30, False)], 1, REGION, 1)
show(tr, 'hedged birth     ')
tr2 = T.Tracker()
tr2.predict(1, np.eye(3)); tr2.update([(box, probs('condor'), 0.40, False)], 1, REGION, 1)
show(tr2, 'production birth ')

if '--streams' in sys.argv:
    print('(c) production boxes per class with conf in [0.003, 0.01): the band an r5-B extra (raw x 0.01) can outrank')
    for p in sys.argv[sys.argv.index('--streams') + 1:]:
        n = collections.Counter(); low = collections.Counter()
        for line in open(p):
            r = json.loads(line)
            if r['frame'] > 150:
                continue
            for c, b, s in r['ann']:
                n[c] += 1
                if s < 0.01:
                    low[c] += 1
        tot = sum(n.values())
        print(f'  {os.path.basename(p)}: {sum(low.values())}/{tot} boxes < 0.01 ({100 * sum(low.values()) / max(1, tot):.0f}%); '
              + ', '.join(f'{c} {low[c]}/{n[c]}' for c in sorted(n) if low[c]))
