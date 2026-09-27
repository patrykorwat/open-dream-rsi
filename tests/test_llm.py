"""Tests for OpenAICompatibleClient contract bits: extra_payload + self-heal."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from open_dream_rsi.llm import LLMConfig, LLMError, OpenAICompatibleClient

SEEN = []


class Endpoint(BaseHTTPRequestHandler):
    """First POST with chat_template_kwargs -> 400; plain body -> 200."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append(body)
        if "chat_template_kwargs" in body:
            raw = json.dumps({"error": {"message":
                  "chat_template_kwargs is not supported"}}).encode()
            self.send_response(400)
        else:
            raw = json.dumps({"choices": [{"message": {
                "role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


class SelfHealTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), Endpoint)
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}/v1"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_400_on_extra_payload_retries_plain_and_persists(self):
        SEEN.clear()
        cfg = LLMConfig(base_url=self.base, api_key="***", model="m",
                        extra_payload={"chat_template_kwargs": {"enable_thinking": False}})
        client = OpenAICompatibleClient(cfg)
        out = client.chat([{"role": "user", "content": "x"}])
        self.assertEqual(out, "ok")
        self.assertIn("chat_template_kwargs", SEEN[0])   # first attempt: with flag
        self.assertNotIn("chat_template_kwargs", SEEN[1])  # retry: plain
        # flag stays dropped for the rest of the session
        client.chat([{"role": "user", "content": "y"}])
        self.assertNotIn("chat_template_kwargs", SEEN[2])
        self.assertEqual(cfg.extra_payload, {})

    def test_plain_error_still_raises(self):
        cfg = LLMConfig(base_url="http://127.0.0.1:1/v1", api_key="***", model="m")
        client = OpenAICompatibleClient(cfg)
        with self.assertRaises(LLMError):
            client.chat([{"role": "user", "content": "x"}])


if __name__ == "__main__":
    unittest.main()
