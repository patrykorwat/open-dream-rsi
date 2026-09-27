"""Tests for the zero-config provider chain in the MCP server."""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from open_dream_rsi.mcp import _default_provider, _is_official_endpoint


class DefaultProviderTests(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in
                       ("ODR_LLM_PRESET", "OPENAI_BASE_URL", "OPENAI_API_KEY",
                        "ODR_GOOSE_CONFIG_DIR")}

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_explicit_preset_wins(self):
        os.environ["ODR_LLM_PRESET"] = "cursor"
        self.assertEqual(_default_provider(), "cursor")

    def test_openai_env_second(self):
        os.environ.pop("ODR_LLM_PRESET", None)
        os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:8799/v1"
        self.assertEqual(_default_provider(), "openai")

    def test_goose_config_third(self):
        for k in ("ODR_LLM_PRESET", "OPENAI_BASE_URL", "OPENAI_API_KEY"):
            os.environ.pop(k, None)
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "config.yaml").write_text(
                "active_provider: custom_spark\n"
                "providers:\n  custom_spark:\n    enabled: true\n"
                "    model: m\n    base_url: http://YOUR-HOST-IP:8000/v1\n",
                encoding="utf-8")
            os.environ["ODR_GOOSE_CONFIG_DIR"] = tmp
            self.assertEqual(_default_provider(), "goose")

    def test_local_fallback(self):
        for k in ("ODR_LLM_PRESET", "OPENAI_BASE_URL", "OPENAI_API_KEY"):
            os.environ.pop(k, None)
        os.environ["ODR_GOOSE_CONFIG_DIR"] = "/nonexistent-goose-dir"
        self.assertEqual(_default_provider(), "local")

    def test_official_endpoint_detection(self):
        self.assertTrue(_is_official_endpoint("https://api.openai.com/v1"))
        self.assertFalse(_is_official_endpoint("http://127.0.0.1:8799/v1"))
        self.assertFalse(_is_official_endpoint("http://YOUR-HOST-IP:8000/v1"))


if __name__ == "__main__":
    unittest.main()
