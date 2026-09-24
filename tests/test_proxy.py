"""Tests for the goose config resolver and the OpenAI-compatible proxy.

Uses the live_loop_demo pattern: a mock POST /chat/completions upstream on a
random port + a fabricated ~/.config/goose directory (via ODR_GOOSE_CONFIG_DIR
/ explicit config_dir), so no network and no real keys are involved.
"""

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from open_dream_rsi.proxy import ProxyState, forward_chat_completions, serve
from open_dream_rsi.utils.goose import (
    GooseConfigError, resolve_goose)

MOCK_LOG = []


class MockUpstream(BaseHTTPRequestHandler):
    """Records requests and returns an OpenAI-shaped (or Anthropic) reply."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        MOCK_LOG.append({"path": self.path, "body": body,
                         "auth": self.headers.get("Authorization"),
                         "xapi": self.headers.get("x-api-key"),
                         "system": self.headers.get("system")
                         or body.get("system")})
        if self.path.endswith("/v1/messages"):  # anthropic shape
            payload = {"id": "msg_1", "model": body["model"],
                       "content": [{"type": "text", "text": "def add(a, b):\n    return a + b\n"}],
                       "stop_reason": "end_turn",
                       "usage": {"input_tokens": 5, "output_tokens": 10}}
        else:  # openai shape
            payload = {"choices": [{"message": {
                "role": "assistant",
                "content": f"echo:{body.get('model')}:{body['messages'][-1]['content']}"},
                "finish_reason": "stop"}]}
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


def make_fake_goose_dir(root: Path, provider="openai", model="gpt-4o-mini",
                        host=None, base_path=None, secret_name="OPENAI_API_KEY",
                        secret="sk-fak...e123", custom=None):
    (root / "config.yaml").write_text(
        "GOOSE_PROVIDER: %s\nGOOSE_MODEL: %s\n" % (provider, model)
        + (f"OPENAI_HOST: {host}\n" if host else "")
        + (f"OPENAI_BASE_PATH: {base_path}\n" if base_path else ""),
        encoding="utf-8")
    if secret:
        (root / "secrets.yaml").write_text(
            f"{secret_name}: {secret}\n", encoding="utf-8")
    if custom:
        cdir = root / "custom_providers"
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / f"{custom['name']}.json").write_text(json.dumps(custom),
                                                     encoding="utf-8")
    return str(root)


class ResolverTests(unittest.TestCase):
    def test_openai_provider_reads_config_secrets(self):
        with TemporaryDirectory() as tmp:
            cdir = make_fake_goose_dir(Path(tmp))
            up = resolve_goose(cdir)
            self.assertEqual((up.provider, up.engine, up.model),
                             ("openai", "openai", "gpt-4o-mini"))
            self.assertEqual(up.base_url, "https://api.openai.com/v1")
            self.assertEqual(up.api_key, "sk-fak...e123")
            self.assertIn("secrets.yaml", up.api_key_source)
            self.assertNotIn("sk-fak...e123", json.dumps(up.masked()))  # masked!

    def test_custom_provider_json_wins_for_base_and_key(self):
        custom = {"name": "spark", "engine": "openai",
                  "base_url": "http://192.168.0.12:8000/v1/chat/completions",
                  "api_key_env": "SPARK_API_KEY",
                  "headers": {"X-Origin": "odr"}}
        with TemporaryDirectory() as tmp:
            cdir = make_fake_goose_dir(
                Path(tmp), provider="spark", model="qwen-flash",
                secret_name="SPARK_API_KEY", custom=custom)
            up = resolve_goose(cdir)
            self.assertEqual(up.engine, "openai")
            self.assertEqual(up.base_url, "http://192.168.0.12:8000/v1")
            self.assertEqual(up.model, "qwen-flash")
            self.assertEqual(up.headers, {"X-Origin": "odr"})
            self.assertEqual(up.api_key, "sk-fak...e123")

    def test_local_endpoint_gets_dummy_key(self):
        with TemporaryDirectory() as tmp:
            cdir = make_fake_goose_dir(Path(tmp), host="http://127.0.0.1:8000",
                                       secret=None)
            up = resolve_goose(cdir)
            self.assertEqual(up.base_url, "http://127.0.0.1:8000/v1")
            self.assertEqual(up.api_key, "local")

    def test_anthropic_engine_maps_messages_url(self):
        with TemporaryDirectory() as tmp:
            cdir = make_fake_goose_dir(Path(tmp), provider="anthropic",
                                       model="claude-sonnet-4-20250514",
                                       secret_name="ANTHROPIC_API_KEY")
            up = resolve_goose(cdir)
            self.assertEqual(up.engine, "anthropic")

    def test_ollama_composes_v1(self):
        with TemporaryDirectory() as tmp:
            cdir = make_fake_goose_dir(Path(tmp), provider="ollama",
                                       model="llama3", secret_name="OLLAMA_API_KEY")
            up = resolve_goose(cdir)
            self.assertTrue(up.base_url.endswith("/v1"))

    def test_missing_provider_raises(self):
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "config.yaml").write_text("OTHER: x\n", encoding="utf-8")
            with self.assertRaises(GooseConfigError):
                resolve_goose(tmp)

    def test_active_provider_desktop_format(self):
        # goose desktop app format: active_provider + nested providers: block
        with TemporaryDirectory() as tmp:
            cdir = make_fake_goose_dir(Path(tmp), secret_name="SPARK_API_KEY")
            cfg = Path(tmp, "config.yaml")
            cfg.write_text(
                "active_provider: custom_spark-27b7\n"
                "providers:\n"
                "  custom_spark-27b7:\n"
                "    enabled: true\n"
                "    model: local-inference-lab/Qwen3.8-Flash-Next-NVFP4\n"
                "    configured: true\n"
                "GOOSE_TELEMETRY_ENABLED: false\n", encoding="utf-8")
            custom = {"name": "custom_spark-27b7", "engine": "openai",
                      "base_url": "http://192.168.0.12:8000/v1",
                      "api_key_env": "SPARK_API_KEY"}
            cdir_p = Path(tmp, "custom_providers"); cdir_p.mkdir()
            (cdir_p / "custom_spark-27b7.json").write_text(json.dumps(custom),
                                                            encoding="utf-8")
            up = resolve_goose(tmp)
            self.assertEqual(up.provider, "custom_spark-27b7")
            self.assertEqual(up.model, "local-inference-lab/Qwen3.8-Flash-Next-NVFP4")
            self.assertEqual(up.base_url, "http://192.168.0.12:8000/v1")
            self.assertEqual(up.api_key, "sk-fak...e123")

    def test_custom_provider_block_only_no_json(self):
        # desktop app may keep everything in the nested providers: block
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "config.yaml").write_text(
                "active_provider: custom_spark-27b7\n"
                "providers:\n"
                "  custom_spark-27b7:\n"
                "    enabled: true\n"
                "    model: local-inference-lab/Qwen3.8-Flash-Next-NVFP4\n"
                "    base_url: http://192.168.0.12:8000/v1\n", encoding="utf-8")
            (Path(tmp) / "secrets.yaml").write_text(
                "CUSTOM_SPARK_27B7_API_KEY: ***", encoding="utf-8")
            up = resolve_goose(tmp)
            self.assertEqual(up.engine, "openai")
            self.assertEqual(up.base_url, "http://192.168.0.12:8000/v1")
            self.assertEqual(up.model, "local-inference-lab/Qwen3.8-Flash-Next-NVFP4")
            self.assertEqual(up.api_key, "sk-lab-1")

    def test_custom_provider_no_url_raises_with_hint(self):
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "config.yaml").write_text(
                "active_provider: custom_orphan\n"
                "providers:\n"
                "  custom_orphan:\n"
                "    enabled: true\n"
                "    model: m\n", encoding="utf-8")
            with self.assertRaises(GooseConfigError) as ctx:
                resolve_goose(tmp)
            self.assertIn("--base-url", str(ctx.exception))

    def test_unknown_engine_raises(self):
        with TemporaryDirectory() as tmp:
            cdir = make_fake_goose_dir(Path(tmp), provider="weird")
            with self.assertRaises(GooseConfigError):
                resolve_goose(cdir)

    def test_from_preset_goose(self):
        import os
        from open_dream_rsi.llm import LLMConfig
        with TemporaryDirectory() as tmp:
            cdir = make_fake_goose_dir(Path(tmp), host="http://127.0.0.1:9")
            old = os.environ.get("ODR_GOOSE_CONFIG_DIR")
            os.environ["ODR_GOOSE_CONFIG_DIR"] = cdir
            try:
                cfg = LLMConfig.from_preset("goose")
                self.assertEqual(cfg.model, "gpt-4o-mini")
                self.assertEqual(cfg.api_key, "sk-fak...e123")
            finally:
                if old is None:
                    del os.environ["ODR_GOOSE_CONFIG_DIR"]
                else:
                    os.environ["ODR_GOOSE_CONFIG_DIR"] = old


class ProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        MOCK_LOG.clear()
        cls.upstream = HTTPServer(("127.0.0.1", 0), MockUpstream)
        cls.up_base = f"http://127.0.0.1:{cls.upstream.server_address[1]}/v1"
        threading.Thread(target=cls.upstream.serve_forever, daemon=True).start()
        cls.proxy, cls.state = serve("127.0.0.1", 0)  # ephemeral port
        cls.proxy_port = cls.proxy.server_address[1]
        threading.Thread(target=cls.proxy.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.proxy.shutdown(); cls.upstream.shutdown()

    def _post(self, path, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.proxy_port}{path}",
            data=json.dumps(payload).encode(), method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def _get(self, path):
        with urllib.request.urlopen(
                f"http://127.0.0.1:{self.proxy_port}{path}", timeout=10) as resp:
            return resp.status, json.loads(resp.read())

    def _with_config(self, **kw):
        tmp = TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        cdir = make_fake_goose_dir(Path(tmp.name), **kw)
        # route the proxy to the mock upstream and this fake config dir
        import os
        old = os.environ.get("ODR_GOOSE_CONFIG_DIR")
        os.environ["ODR_GOOSE_CONFIG_DIR"] = cdir
        self.addCleanup(lambda: os.environ.update(
            {"ODR_GOOSE_CONFIG_DIR": old} if old is not None
            else {}))
        if old is None:
            self.addCleanup(os.environ.pop, "ODR_GOOSE_CONFIG_DIR", None)
        return cdir

    def test_health_and_models(self):
        self._with_config(host=self.up_base)
        status, data = self._get("/health")
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "ok")
        self.assertNotIn("sk-fak...e123", json.dumps(data))
        status, data = self._get("/v1/models")
        self.assertEqual(data["data"][0]["id"], "gpt-4o-mini")

    def test_chat_forwards_and_pins_model(self):
        self._with_config(host=self.up_base)
        MOCK_LOG.clear()
        status, data = self._post("/v1/chat/completions", {
            "model": "whatever-the-client-said",
            "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200)
        self.assertIn("echo:gpt-4o-mini:hi",
                      data["choices"][0]["message"]["content"])
        # upstream saw the pinned GOOSE_MODEL + config's key, not the client's
        self.assertEqual(MOCK_LOG[-1]["body"]["model"], "gpt-4o-mini")
        self.assertEqual(MOCK_LOG[-1]["auth"], "Bearer sk-fak...e123")

    def test_passthrough_models_keeps_caller_model(self):
        tmp = TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        cdir = make_fake_goose_dir(Path(tmp.name), host=self.up_base)
        import os
        os.environ["ODR_GOOSE_CONFIG_DIR"] = cdir
        self.addCleanup(os.environ.pop, "ODR_GOOSE_CONFIG_DIR", None)
        proxy, state = serve("127.0.0.1", 0, passthrough_models=True)
        port = proxy.server_address[1]
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        self.addCleanup(proxy.shutdown)
        MOCK_LOG.clear()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps({"model": "caller-model",
                             "messages": [{"role": "user", "content": "x"}]}
                            ).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            json.loads(resp.read())
        self.assertEqual(MOCK_LOG[-1]["body"]["model"], "caller-model")

    def test_completions_endpoint_mapped(self):
        self._with_config(host=self.up_base)
        status, data = self._post("/v1/completions", {
            "model": "x", "prompt": "ping"})
        self.assertEqual(status, 200)
        self.assertIn("ping", data["choices"][0]["message"]["content"])

    def test_stream_true_is_buffered(self):
        self._with_config(host=self.up_base)
        status, data = self._post("/v1/chat/completions", {
            "model": "x", "messages": [{"role": "user", "content": "s"}],
            "stream": True})
        self.assertEqual(status, 200)
        self.assertIn("content", data["choices"][0]["message"])

    def test_anthropic_translation(self):
        with TemporaryDirectory() as tmp:
            cdir = make_fake_goose_dir(
                Path(tmp), provider="anthropic", model="claude-sonnet-4-20250514",
                secret_name="ANTHROPIC_API_KEY")
            up = resolve_goose(cdir)
            # point the anthropic host at our mock (which answers /v1/messages)
            from dataclasses import replace
            up = replace(up, base_url=self.up_base.rstrip("/").rsplit("/v1", 1)[0])
            MOCK_LOG.clear()
            data = forward_chat_completions(up, {
                "model": "claude-sonnet-4-20250514",
                "messages": [{"role": "system", "content": "be terse"},
                             {"role": "user", "content": "add?"}]})
            self.assertIn("def add", data["choices"][0]["message"]["content"])
            self.assertEqual(data["choices"][0]["finish_reason"], "stop")
            self.assertEqual(MOCK_LOG[-1]["path"], "/v1/messages")
            self.assertEqual(MOCK_LOG[-1]["xapi"], "sk-fak...e123")
            self.assertEqual(MOCK_LOG[-1]["system"], "be terse")

    def test_upstream_down_returns_502_not_crash(self):
        with TemporaryDirectory() as tmp:
            cdir = make_fake_goose_dir(
                Path(tmp), host="http://127.0.0.1:1")  # nothing listening
            import os
            os.environ["ODR_GOOSE_CONFIG_DIR"] = cdir
            self.addCleanup(os.environ.pop, "ODR_GOOSE_CONFIG_DIR", None)
            proxy, state = serve("127.0.0.1", 0)
            port = proxy.server_address[1]
            threading.Thread(target=proxy.serve_forever, daemon=True).start()
            self.addCleanup(proxy.shutdown)
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps({"model": "m",
                                 "messages": [{"role": "user", "content": "x"}]}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req, timeout=10)
            self.assertEqual(ctx.exception.code, 502)
            self.assertEqual(state.errors, 1)  # counted by the served state
            self.assertEqual(state.requests, 0)


if __name__ == "__main__":
    unittest.main()
