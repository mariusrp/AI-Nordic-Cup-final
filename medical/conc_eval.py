"""Load test: send the first N practice conversations to /predict with C parallel clients,
save predictions in offline_eval format (score with offline_eval.py) and print latency.

    python conc_eval.py --url http://127.0.0.1:9061/predict --n 10 --conc 2 --out preds.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time

UPSTREAM = os.environ.get('UPSTREAM', '/workspace/upstream')
sys.path.insert(0, os.path.join(UPSTREAM, 'medical-appointment'))
import requests  # noqa: E402
from utils import encode_audio, group_questions_by_conversation, load_sample_audio  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--conc', type=int, default=2)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.chdir(os.path.join(UPSTREAM, 'medical-appointment'))
    convs = group_questions_by_conversation()[: a.n]

    def one(item):
        fn, rows = item
        body = {'audio_base64': encode_audio(load_sample_audio(fn)), 'audio_filename': fn,
                'questions': [r['question'] for r in rows]}
        t = time.time()
        try:
            r = requests.post(a.url, json=body, timeout=60).json()
        except Exception as e:  # timeout / error counts as all wrong
            r = {'answers': [None] * len(rows), 'evidence_start': [None] * len(rows),
                 'evidence_end': [None] * len(rows), 'error': repr(e)}
        r['latency_s'] = round(time.time() - t, 2)
        return fn, r

    t0 = time.time()
    with cf.ThreadPoolExecutor(a.conc) as ex:
        preds = dict(ex.map(one, convs))
    lat = sorted(p['latency_s'] for p in preds.values())
    json.dump(preds, open(a.out, 'w'), indent=1)
    print(f'n={len(lat)} conc={a.conc} wall {time.time() - t0:.1f}s  latency mean {sum(lat) / len(lat):.1f}s '
          f'p95 {lat[int(0.95 * (len(lat) - 1))]:.1f}s max {lat[-1]:.1f}s  errors {sum("error" in p for p in preds.values())}')


if __name__ == '__main__':
    main()
