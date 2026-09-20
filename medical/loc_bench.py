"""Latency + fidelity of the CPU locator sidecar as the pipeline uses it: one request per
question with the K lexically best phrase runs. With --ref (locator.py score output, GPU
fp32) it also reports the max |logit difference| and yes-decision flips at --thr.

    python loc_bench.py [--url http://127.0.0.1:9061] [--tx-dir TX] [--k 8] [--nb 3] [--ref scores.json]
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import locator as L  # noqa: E402
import spans as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--url', default='http://127.0.0.1:9061')
ap.add_argument('--tx-dir', default='/workspace/tx/large-v3-turbo')
ap.add_argument('--k', type=int, default=8)
ap.add_argument('--nb', type=int, default=3)
ap.add_argument('--ref')
ap.add_argument('--thr', type=float, default=-2.0)
a = ap.parse_args()
up = os.environ.get('UPSTREAM', '/workspace/upstream')
rows = list(csv.DictReader(open(os.path.join(up, 'medical-appointment/data/question_train.csv'))))
by = {}
for r in rows:
    by.setdefault(r['transcript_id'], []).append(r)
ref = json.load(open(a.ref)) if a.ref else None
lat, diffs, flips, n = [], [], 0, 0
for p in sorted(glob.glob(a.tx_dir + '/*.json'))[:a.nb]:
    tid = os.path.basename(p)[len('conversation_'):-5]
    units = S.build_units(S.load_transcript(p), 'phrase')
    runs = L.candidate_runs(units)
    texts = [L.run_text(units, i, j) for i, j in runs]
    t0 = time.time()
    for r in by[tid]:
        idx = L.lexical_topk(units, runs, r['question'], a.k)
        t = time.time()
        sc = requests.post(a.url + '/score', json={'pairs': [[r['question'], texts[i]] for i in idx]},
                           timeout=60).json()['scores']
        lat.append(time.time() - t)
        if ref:
            rs = [ref[tid]['qs'][r['question_id']][i] for i in idx]
            diffs.append(max(abs(x - y) for x, y in zip(sc, rs)))
            flips += (max(sc) > a.thr) != (max(rs) > a.thr)
            n += 1
    print(tid, len(runs), 'runs', f'10 q in {time.time() - t0:.2f}s', flush=True)
lat.sort()
print(f'per-question latency mean {sum(lat) / len(lat):.2f}s p50 {lat[len(lat) // 2]:.2f}s max {lat[-1]:.2f}s')
if ref:
    diffs.sort()
    print(f'vs ref: max|dlogit| per q median {diffs[len(diffs) // 2]:.3f} max {diffs[-1]:.3f}; '
          f'yes flips at thr {a.thr}: {flips}/{n}')
