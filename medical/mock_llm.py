"""Tiny OpenAI-compatible mock of the vLLM server, for CPU plumbing tests.

    python mock_llm.py 8001 [delay_s]

Answers with the heuristic, formatted exactly like the real model is asked to
(EVIDENCE/QUOTE/ANSWER) and with token logprobs, so the parsing, logprob and
span code paths of pipeline.py are exercised end to end.
"""
import json
import math
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pipeline as P

DELAY = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0


def reply_for(messages):
    user = messages[-1]['content']
    lines = re.findall(r'^\[(\d+)\] (.*)$', user, flags=re.M)
    q = user.rsplit('Question:', 1)[-1].strip()
    units = [{'id': int(i), 'text': t, 'start': 0, 'end': 0, 'words': []} for i, t in lines]
    p, ids, _ = P.heuristic_answer(units, q)
    p = 1 / (1 + math.exp(-(math.log(p / (1 - p)) + 3)))
    quote = ' '.join(units[ids[0]]['text'].split()[:6]) if ids else ''
    ans = 'yes' if p > 0.5 else 'no'
    text = f"EVIDENCE: {','.join(map(str, ids)) or 'none'}\nQUOTE: {quote}\nANSWER: {ans}"
    head = text.rsplit(ans, 1)[0]
    toks = [{'token': head, 'logprob': 0.0, 'top_logprobs': []},
            {'token': ans, 'logprob': math.log(max(p if ans == 'yes' else 1 - p, 1e-9)),
             'top_logprobs': [{'token': 'yes', 'logprob': math.log(max(p, 1e-9))},
                              {'token': 'no', 'logprob': math.log(max(1 - p, 1e-9))}]}]
    return text, toks


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send({'data': [{'id': 'mock-heuristic'}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        time.sleep(DELAY)
        text, toks = reply_for(body['messages'])
        self._send({'choices': [{'message': {'role': 'assistant', 'content': text},
                                 'logprobs': {'content': toks}}]})


if __name__ == '__main__':
    ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()
