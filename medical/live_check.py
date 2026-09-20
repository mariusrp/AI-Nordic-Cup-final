"""Live /predict check of the onset deploy (I1Perquest-g1-1): POST a few visible conversations to a running server
and save the responses. Pair with the server's MED_REQ_LOG rows (onsets_n, per-question ids/quote/span) and
compare the served starts with the zero-LLM replay of the SAME LLM replies (fresh_stack.span_for 'onset+any').
    python3 live_check.py <url> <out.json> conversation_sample_19.mp3 ...
"""
import base64
import csv
import json
import os
import sys
import time

import requests

UP = os.environ.get('UPSTREAM', '/workspace/upstream')
MED = os.path.join(UP, 'medical-appointment')


def main():
    url, out, fns = sys.argv[1], sys.argv[2], sys.argv[3:]
    qs = {}
    for r in csv.DictReader(open(os.path.join(MED, 'data', 'question_train.csv'))):
        qs.setdefault(f"conversation_{r['transcript_id']}.mp3", []).append(r['question'])
    res = {}
    for fn in fns:
        b = base64.b64encode(open(os.path.join(MED, 'data', 'audio', fn), 'rb').read()).decode()
        t = time.time()
        r = requests.post(url, json={'audio_filename': fn, 'audio_base64': b, 'questions': qs[fn]}, timeout=120)
        res[fn] = {**r.json(), 'questions': qs[fn], 'wall_s': round(time.time() - t, 2)}
        print(fn, r.status_code, res[fn]['wall_s'], 's', flush=True)
    json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
