#!/usr/bin/env python3
"""Pass-through OpenAI proxy forcing enable_thinking:false (arm parity).

The Hermes-side spolki bench posts chat_template_kwargs={enable_thinking:
False} on every call. goose controls its own request body, so to compare
arms across agents under the SAME model configuration, goose talks to this
proxy instead of vLLM directly: the proxy injects the flag the host cannot
set, changes nothing else, and counts LLM calls per session for the record.

Stdlib only. Port: ODR_PROXY_PORT (default 8801). Upstream: ODR_UPSTREAM
(default http://127.0.0.1:8000).
"""
import json
import os
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("ODR_UPSTREAM", "http://127.0.0.1:8000")
CALLS_PATH = os.environ.get("ODR_LLM_CALLS_FILE", "/tmp/odr_llm_calls.jsonl")
_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        try:
            req = json.loads(body)
            kw = req.setdefault("chat_template_kwargs", {})
            kw.setdefault("enable_thinking", False)
            body = json.dumps(req, ensure_ascii=False).encode()
            with _lock, open(CALLS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"model": req.get("model", "?")}) + "\n")
        except Exception:
            pass  # pass through unmodified on any parse issue
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in ("host", "content-length")}
        up = urllib.request.Request(UPSTREAM + self.path, data=body,
                                    headers=headers, method="POST")
        try:
            with urllib.request.urlopen(up, timeout=300) as r:
                data = r.read()
                self.send_response(r.status)
                for k, v in r.headers.items():
                    if k.lower() not in ("transfer-encoding", "connection"):
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except Exception as e:
            msg = str(e).encode()
            self.send_response(502)
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

    def do_GET(self):
        up = urllib.request.Request(UPSTREAM + self.path,
                                    headers={"Host": UPSTREAM.split("//")[1]})
        try:
            with urllib.request.urlopen(up, timeout=30) as r:
                data = r.read()
                self.send_response(r.status)
                self.send_header("Content-Type",
                                r.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except Exception as e:
            msg = str(e).encode()
            self.send_response(502)
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)


if __name__ == "__main__":
    port = int(os.environ.get("ODR_PROXY_PORT", "8801"))
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
