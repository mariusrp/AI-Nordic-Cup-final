"""CPU-only scoring sidecar for the no-LLM evidence locator (bge-reranker-v2-m3).

Runs as its own process (venv-vllm has torch + transformers; venv-med does not), so the
production /predict server never imports torch/transformers and never touches the GPU
for it. pipeline.py calls it only for questions the LLM failed to answer; if the sidecar
is down or slow, the pipeline keeps the heuristic answer.

    CUDA_VISIBLE_DEVICES= HF_HOME=/workspace/hf /workspace/venv-vllm/bin/python loc_server.py
    POST /score {"pairs": [[question, text], ...]} -> {"scores": [logit per pair], "s": secs}
    GET  /health -> ok
Env: MED_LOC_PORT (9061), MED_LOC_THREADS (4), MED_LOC_MODEL (BAAI/bge-reranker-v2-m3), MED_LOC_INT8 (0).
"""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')  # CPU only, never the shared GPU

import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

PORT = int(os.environ.get('MED_LOC_PORT', 9061))
THREADS = int(os.environ.get('MED_LOC_THREADS', 4))  # the pod's CPU quota is ~7.6 cores, shared
MODEL = os.environ.get('MED_LOC_MODEL', 'BAAI/bge-reranker-v2-m3')
BS = int(os.environ.get('MED_LOC_BS', 64))

torch.set_num_threads(THREADS)
TOK = AutoTokenizer.from_pretrained(MODEL)
M = AutoModelForSequenceClassification.from_pretrained(MODEL, dtype=torch.float32).eval()
INT8 = os.environ.get('MED_LOC_INT8', '0') == '1'
if INT8:  # dynamic int8 Linear layers (off: the knobs were tuned on fp32 scores, fp32 CPU matches them exactly)
    M = torch.ao.quantization.quantize_dynamic(M, {torch.nn.Linear}, dtype=torch.qint8)
LOCK = threading.Lock()  # one scoring job at a time; torch already uses THREADS cores


def score_pairs(pairs):
    """Relevance logit for every (query, text) pair; pairs sorted by length to cut padding."""
    order = sorted(range(len(pairs)), key=lambda k: len(pairs[k][0]) + len(pairs[k][1]))
    out = [0.0] * len(pairs)
    with LOCK, torch.inference_mode():
        for b in range(0, len(order), BS):
            idx = order[b:b + BS]
            enc = TOK([pairs[k][0] for k in idx], [pairs[k][1] for k in idx], padding=True,
                      truncation=True, max_length=256, return_tensors='pt')
            lg = M(**enc).logits
            lg = lg[:, 0] if lg.shape[-1] == 1 else lg[:, -1]
            for k, v in zip(idx, lg.float().tolist()):
                out[k] = float(v)
    return out


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send(200, {'ok': True, 'model': MODEL, 'threads': THREADS, 'int8': INT8})

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
            t = time.time()
            sc = score_pairs([(str(q), str(x)) for q, x in req['pairs']])
            self._send(200, {'scores': sc, 's': round(time.time() - t, 3)})
        except Exception as e:  # noqa
            self._send(500, {'error': repr(e)})


if __name__ == '__main__':
    score_pairs([('warm up question?', 'warm up text.')])
    print(f'loc_server {MODEL} cpu threads={THREADS} on :{PORT}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', PORT), H).serve_forever()
