"""Live demo: full autonomous cycle against a local OpenAI-compatible server.

Spins a tiny in-process mock of POST /chat/completions (returns a buggy then
fixed solution, like a real model reacting to feedback), then drives the
supervisor for 3 cycles with --cycles. No external network, no real key.
"""

import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

STATE = {"attempts": {}}


class MockOpenAI(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][-1]["content"]
        key = prompt.split("]:")[0]
        n = STATE["attempts"].get(key, 0)
        STATE["attempts"][key] = n + 1
        # First ever attempt per key: buggy solution. Afterwards: the fix.
        fixed = "def add(a, b):\n    return a + b\n" if (n > 0 or "failure feedback:\n(none)" not in prompt) else "def add(a, b):\n    return a - b\n"
        content = f"```python\n{fixed}```"
        payload = {"choices": [{"message": {"role": "assistant", "content": content}}]}
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format, *args):  # silence request logging
        pass


def main():
    srv = HTTPServer(("127.0.0.1", 0), MockOpenAI)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    os.environ["OPENAI_API_KEY"] = "sk-demo-not-a-real-key"
    os.environ["OPENAI_BASE_URL"] = f"http://127.0.0.1:{port}/v1"

    from open_dream_rsi.cli import main as cli_main

    with tempfile.TemporaryDirectory() as tmp:
        tasks = tmp + "/tasks.json"
        with open(tasks, "w") as fh:
            json.dump([{
                "task_id": "add1", "category": "math",
                "prompt": "Implement add(a, b) returning the sum.",
                "tests": [{"call": "add(2, 3)", "expected": 5},
                          {"call": "add(-1, 1)", "expected": 0}],
                "max_attempts": 3,
            }], fh)
        mem = tmp + "/mem"
        rc = cli_main(["--memory", mem, "loop", "--tasks", tasks,
                       "--provider", "local", "--cycles", "3", "--interval", "0.1"])
        print("cli rc:", rc)
        rc2 = cli_main(["--memory", mem, "status"])
        print("status rc:", rc2)
    srv.shutdown()


if __name__ == "__main__":
    main()
