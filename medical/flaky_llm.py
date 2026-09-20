"""Fault-injecting proxy in front of the shared vLLM, for fallback tests.

    python flaky_llm.py PORT MODE FRAC [UPSTREAM_URL]
MODE: 'error' -> HTTP 500 at once, 'hang' -> never answer within 120 s.
A request fails when crc32(question text) mod 1000 < FRAC*1000 (deterministic, so
repeated runs fail the same questions); the rest are forwarded unchanged.
"""
import json
import sys
import time
import urllib.request
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT, MODE, FRAC = int(sys.argv[1]), sys.argv[2], float(sys.argv[3])
UP = sys.argv[4] if len(sys.argv) > 4 else 'http://127.0.0.1:8001'


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _fwd(self, method, body=None):
        req = urllib.request.Request(UP + self.path, data=body, method=method,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._fwd('GET')

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        try:
            q = json.loads(body)['messages'][-1]['content'].rsplit('Question:', 1)[-1].strip()
        except Exception:
            q = ''
        if q and zlib.crc32(q.encode()) % 1000 < FRAC * 1000:
            if MODE == 'hang':
                time.sleep(120)
            self.send_response(500)
            self.end_headers()
            return
        self._fwd('POST', body)


if __name__ == '__main__':
    ThreadingHTTPServer(('127.0.0.1', PORT), H).serve_forever()
