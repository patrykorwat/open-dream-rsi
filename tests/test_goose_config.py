"""Tests for the goose config.yaml extension editor (scripts/odr_goose_setup.sh core).

The editor must: replace a stale entry (inline comments included), add when
missing, stay idempotent, back up before writing, never touch foreign
entries, and work with or without the proxy envs block.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from open_dream_rsi.utils.goose_config import configure

GOOSE_CONFIG = """extensions:
  todo:
    enabled: true
    type: platform
    name: todo
    description: Enable a todo list for goose so it can keep track of what it is doing
    display_name: Todo
    bundled: true
  open-dream-rsi:
    enabled: true
    type: stdio
    name: open-dream-rsi
    cmd: /opt/homebrew/bin/python3     # stale inline comment
    args: ["-m", "open_dream_rsi", "mcp",
           "--tasks", "~/git/open-dream-rsi/tasks.json",
           "--memory", "~/git/open-dream-rsi/.dream_rsi"]
    timeout: 300
active_provider: custom_spark-27b7
providers:
  custom_spark-27b7:
    enabled: true
    model: local-inference-lab/Qwen3.8-Flash-Next-NVFP4
    configured: true
GOOSE_TELEMETRY_ENABLED: false
"""


def write_cfg(root: Path) -> Path:
    p = root / "config.yaml"
    p.write_text(GOOSE_CONFIG, encoding="utf-8")
    return p


class GooseConfigEditorTests(unittest.TestCase):
    def _configure(self, cfg, **kw):
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = configure(cfg, "/opt/homebrew/bin/python3",
                           Path("/Users/tester/git/open-dream-rsi"),
                           kw.get("proxy_url", "http://127.0.0.1:8799/v1"),
                           check=kw.get("check", False))
        return rc, buf.getvalue()

    def test_replaces_stale_entry_preserving_foreign_lines(self):
        with TemporaryDirectory() as tmp:
            cfg = write_cfg(Path(tmp))
            rc, out = self._configure(cfg)
            self.assertEqual(rc, 0)
            self.assertIn('"action": "replaced"', out)
            text = cfg.read_text()
            # our block, canonical:
            self.assertIn('OPENAI_BASE_URL: "http://127.0.0.1:8799/v1"', text)
            self.assertIn('OPENAI_API_KEY: "proxy-internal"', text)
            self.assertIn('description: "Dream-RSI self-improvement loop.', text)
            self.assertNotIn("stale inline comment", text)
            self.assertNotIn("~/git", text)  # tildes gone, absolute paths
            # foreign content untouched:
            self.assertIn("display_name: Todo", text)
            self.assertIn("active_provider: custom_spark-27b7", text)
            self.assertIn("GOOSE_TELEMETRY_ENABLED: false", text)
            self.assertEqual(text.count("  open-dream-rsi:"), 1)  # one canonical entry
            self.assertEqual(text.count("name: open-dream-rsi"), 1)

    def test_idempotent_second_run(self):
        with TemporaryDirectory() as tmp:
            cfg = write_cfg(Path(tmp))
            self._configure(cfg)
            before = cfg.read_text()
            rc, out = self._configure(cfg)
            self.assertEqual(rc, 0)
            self.assertIn('"action": "unchanged"', out)
            self.assertEqual(cfg.read_text(), before)

    def test_adds_when_missing(self):
        with TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.yaml"
            cfg.write_text(GOOSE_CONFIG.replace("  open-dream-rsi:", "  other:"),
                           encoding="utf-8")
            rc, out = self._configure(cfg)
            self.assertEqual(rc, 0)
            self.assertIn('"action": "added"', out)
            self.assertIn("name: open-dream-rsi", cfg.read_text())

    def test_check_only_writes_nothing(self):
        with TemporaryDirectory() as tmp:
            cfg = write_cfg(Path(tmp))
            before = cfg.read_text()
            rc, out = self._configure(cfg, check=True)
            self.assertEqual(rc, 0)
            self.assertIn('"action": "check-only"', out)
            self.assertIn('"changed": true', out)
            self.assertEqual(cfg.read_text(), before)
            self.assertFalse(list(Path(tmp).glob("config.yaml.bak-*")))

    def test_no_proxy_omits_envs(self):
        with TemporaryDirectory() as tmp:
            cfg = write_cfg(Path(tmp))
            self._configure(cfg, proxy_url=None)
            text = cfg.read_text()
            self.assertNotIn("OPENAI_BASE_URL", text)
            self.assertIn("name: open-dream-rsi", text)

    def test_backup_created_and_missing_config_errors(self):
        with TemporaryDirectory() as tmp:
            cfg = write_cfg(Path(tmp))
            self._configure(cfg)
            self.assertTrue(list(Path(tmp).glob("config.yaml.bak-*")))
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = configure(Path(tmp, "nope.yaml"), "python3",
                               Path("/repo"), None)
            self.assertEqual(rc, 1)
            self.assertIn("not found", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
