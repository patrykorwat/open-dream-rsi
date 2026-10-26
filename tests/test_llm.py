"""Tests for OpenAICompatibleClient contract bits: extra_payload + self-heal."""

import json
import os
import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

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
        cls.srv.server_close()

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
        # Dead endpoint: retries exhaust (sleep mocked — no real waits).
        with mock.patch("open_dream_rsi.llm.time.sleep") as sleep:
            cfg = LLMConfig(base_url="http://127.0.0.1:1/v1", api_key="***", model="m")
            client = OpenAICompatibleClient(cfg)
            with self.assertRaises(LLMError):
                client.chat([{"role": "user", "content": "x"}])
        self.assertEqual(sleep.call_count, 3)  # MAX_RETRIES backoffs fired


class DiscoveryEndpoint(BaseHTTPRequestHandler):
    """GET /v1/models lists two ids; POST chat/completions answers 'ok'."""

    def do_GET(self):
        if self.path.endswith("/models"):
            raw = json.dumps(
                {"data": [{"id": "served-model-A"}, {"id": "m2"}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        raw = json.dumps({"choices": [{"message": {
            "role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


class ModelDiscoveryTests(unittest.TestCase):
    """No model pinned -> the endpoint's first served id is the default."""

    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), DiscoveryEndpoint)
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}/v1"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in
                       ("OPENAI_BASE_URL", "OPENAI_API_KEY", "ODR_LLM_MODEL")}
        for k in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "ODR_LLM_MODEL"):
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_local_preset_discovers_served_model(self):
        os.environ["OPENAI_BASE_URL"] = self.base
        self.assertEqual(LLMConfig.from_preset("local").model, "served-model-A")

    def test_from_env_discovers_served_model(self):
        os.environ["OPENAI_BASE_URL"] = self.base
        self.assertEqual(LLMConfig.from_env().model, "served-model-A")

    def test_env_model_beats_discovery(self):
        os.environ["OPENAI_BASE_URL"] = self.base
        os.environ["ODR_LLM_MODEL"] = "pinned"
        self.assertEqual(LLMConfig.from_env().model, "pinned")
        self.assertEqual(LLMConfig.from_preset("local").model, "pinned")

    def test_explicit_model_beats_everything(self):
        os.environ["OPENAI_BASE_URL"] = self.base
        os.environ["ODR_LLM_MODEL"] = "pinned"
        self.assertEqual(
            LLMConfig.from_preset("local", model="explicit").model, "explicit")

    def test_discovery_failure_falls_back_to_preset_default(self):
        os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:1/v1"
        self.assertEqual(LLMConfig.from_preset("local").model, "local-model")
        self.assertEqual(LLMConfig.from_env().model, "gpt-4o-mini")


class CursorPresetTests(unittest.TestCase):
    """Preset 'cursor' targets a LOCAL OpenAI-compatible proxy over the
    Cursor login (Cursor has no public chat-completions endpoint)."""

    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), DiscoveryEndpoint)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in
                       ("OPENAI_BASE_URL", "OPENAI_API_KEY", "ODR_LLM_MODEL",
                        "CURSOR_API_KEY")}
        for k in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "ODR_LLM_MODEL",
                  "CURSOR_API_KEY"):
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_cursor_preset_defaults_to_local_proxy(self):
        cfg = LLMConfig.from_preset("cursor")
        self.assertEqual(cfg.base_url, "http://127.0.0.1:3000/v1")
        self.assertEqual(cfg.api_key_env, "CURSOR_API_KEY")
        # nothing on :3000 in the test env -> discovery fails -> preset model
        self.assertEqual(cfg.model, "cursor-model")
        # the Cursor login proxy accepts any bearer: zero-config works
        self.assertEqual(cfg.api_key, "local")

    def test_cursor_preset_discovers_served_model(self):
        os.environ["OPENAI_BASE_URL"] = f"http://127.0.0.1:{self.port}/v1"
        cfg = LLMConfig.from_preset("cursor")
        self.assertEqual(cfg.model, "served-model-A")

    def test_cursor_preset_prefers_explicit_key(self):
        os.environ["CURSOR_API_KEY"] = "crsr_test"
        cfg = LLMConfig.from_preset("cursor")
        self.assertEqual(cfg.api_key, "crsr_test")


class FlakyEndpoint(BaseHTTPRequestHandler):
    """Fails with ``status`` for the first ``fails`` POSTs, then 200 + usage."""

    fails = 0
    status = 503

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        cls = type(self)
        if cls.fails > 0:
            cls.fails -= 1
            self.send_response(cls.status)
            self.end_headers()
            return
        raw = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                      "total_tokens": 120},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


class RetryBackoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), FlakyEndpoint)
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}/v1"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def _client(self):
        return OpenAICompatibleClient(
            LLMConfig(base_url=self.base, api_key="***", model="m"))

    def test_5xx_retried_with_backoff_then_succeeds(self):
        FlakyEndpoint.fails, FlakyEndpoint.status = 2, 503
        with mock.patch("open_dream_rsi.llm.time.sleep") as sleep:
            out = self._client().chat([{"role": "user", "content": "x"}])
        self.assertEqual(out, "ok")
        delays = [c.args[0] for c in sleep.call_args_list]
        self.assertEqual(len(delays), 2)  # backoff before each retry
        for i, d in enumerate(delays):  # full jitter: 0.5x..1.5x of base*2**i
            base = 5.0 * (2 ** i)
            self.assertGreaterEqual(d, 0.5 * base)
            self.assertLessEqual(d, 1.5 * base)

    def test_429_retried_too(self):
        FlakyEndpoint.fails, FlakyEndpoint.status = 1, 429
        with mock.patch("open_dream_rsi.llm.time.sleep"):
            out = self._client().chat([{"role": "user", "content": "x"}])
        self.assertEqual(out, "ok")

    def test_4xx_is_not_retried(self):
        FlakyEndpoint.fails, FlakyEndpoint.status = 5, 400
        with mock.patch("open_dream_rsi.llm.time.sleep") as sleep:
            with self.assertRaises(LLMError):
                self._client().chat([{"role": "user", "content": "x"}])
        sleep.assert_not_called()

    def test_usage_recorded_from_response(self):
        FlakyEndpoint.fails, FlakyEndpoint.status = 0, 503
        client = self._client()
        with mock.patch("open_dream_rsi.llm.time.sleep"):
            client.chat([{"role": "user", "content": "x"}])
            client.chat([{"role": "user", "content": "y"}])
        self.assertEqual(client.usage_total, {"prompt_tokens": 200,
                                              "completion_tokens": 40,
                                              "total_tokens": 240})
        self.assertIsNotNone(client.last_usage)
        self.assertEqual(client.last_usage["prompt_tokens"], 100)

    def test_timeout_not_retried_by_default(self):
        client = self._client()
        with mock.patch("open_dream_rsi.llm.urllib.request.urlopen",
                        side_effect=socket.timeout("timed out")) as up, \
             mock.patch("open_dream_rsi.llm.time.sleep") as sleep:
            with self.assertRaises(LLMError) as ctx:
                client.chat([{"role": "user", "content": "x"}])
        self.assertIn("timed out", str(ctx.exception))
        self.assertEqual(up.call_count, 1)   # no retry on generation timeout
        sleep.assert_not_called()

    def test_timeout_retried_when_opt_in(self):
        import open_dream_rsi.llm as llm
        client = self._client()
        fake_time = mock.Mock()
        fake_time.sleep = mock.Mock()
        with mock.patch.object(llm, "RETRY_TIMEOUTS", True), \
             mock.patch.object(llm, "time", fake_time), \
             mock.patch("open_dream_rsi.llm.urllib.request.urlopen",
                        side_effect=socket.timeout("timed out")):
            with self.assertRaises(LLMError):
                client.chat([{"role": "user", "content": "x"}])
        self.assertEqual(fake_time.sleep.call_count, llm.MAX_RETRIES)


class StaleModelEndpoint(BaseHTTPRequestHandler):
    """POST 404s the pinned 'stale' model; GET /models lists the real one."""

    def do_GET(self):
        raw = json.dumps({"data": [{"id": "served-model-A"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append(body)
        if body.get("model") == "stale":
            raw = json.dumps({"error": {"message":
                "The model `stale` does not exist."}}).encode()
            self.send_response(404)
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


class StaleModelSelfHealTests(unittest.TestCase):
    """HTTP 404 'model does not exist' -> re-discover once and retry."""

    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), StaleModelEndpoint)
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}/v1"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        SEEN.clear()
        self._saved = os.environ.get("ODR_LLM_MODEL")
        os.environ.pop("ODR_LLM_MODEL", None)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("ODR_LLM_MODEL", None)
        else:
            os.environ["ODR_LLM_MODEL"] = self._saved

    def test_stale_pin_rediscovers_and_retries(self):
        cfg = LLMConfig(base_url=self.base, api_key="***", model="stale")
        client = OpenAICompatibleClient(cfg)
        self.assertEqual(client.chat([{"role": "user", "content": "x"}]), "ok")
        self.assertEqual([b["model"] for b in SEEN], ["stale", "served-model-A"])
        self.assertEqual(cfg.model, "served-model-A")
        # the discovered model sticks: no second 404 round-trip
        SEEN.clear()
        client.chat([{"role": "user", "content": "y"}])
        self.assertEqual([b["model"] for b in SEEN], ["served-model-A"])

    def test_404_without_discovery_raises(self):
        # no /models endpoint reachable -> the original error propagates
        cfg = LLMConfig(base_url="http://127.0.0.1:1/v1", api_key="***", model="stale")
        client = OpenAICompatibleClient(cfg)
        with self.assertRaises(LLMError):
            client.chat([{"role": "user", "content": "x"}])


if __name__ == "__main__":
    unittest.main()
