"""Adapter tests for the DeepSeek Harness plugin (no DSH runtime required).

The adapter is pure configuration: two Cordis overlays that insert the ODR
MCP server behind ``@deepseek-ai/dsh-mcp-client``. These tests pin the
overlay contract — entry shape, server naming, the env re-pass (DSH spawns
stdio children with a scrubbed environment), the long tool-call timeout —
and verify that the exact stdio command the overlay configures speaks the
MCP protocol. DSH-side loading is verified manually with
``dsh --patch <overlay> --dump-config`` (needs the npm package).
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_PLUGIN = Path(__file__).resolve().parent
_REPO = _PLUGIN.parent.parent
STDIO_OVERLAY = _PLUGIN / "odr.cordis.yml"
HTTP_OVERLAY = _PLUGIN / "odr-http.cordis.yml"

SERVERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def _servername(text: str) -> str:
    match = re.search(r"serverName:\s*(\S+)", text)
    assert match is not None, "serverName missing from overlay"
    return match.group(1)


def _overlay_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    assert "!!js" in text, "cordis overlays rely on !!js tag expressions"
    return text


class TestStdioOverlayContract(unittest.TestCase):
    def setUp(self):
        self.text = _overlay_text(STDIO_OVERLAY)

    def test_single_insert_entry_with_mcp_client(self):
        self.assertIn("- insert:", self.text)
        self.assertIn("id: mcp-open-dream-rsi", self.text)
        self.assertIn("name: '@deepseek-ai/dsh-mcp-client'", self.text)

    def test_servername_valid_and_namespaced(self):
        self.assertRegex(_servername(self.text), SERVERNAME_RE)

    def test_stdio_transport_launches_odr_mcp_module(self):
        self.assertIn("transport: stdio", self.text)
        self.assertIn("command: python3", self.text)
        for fragment in ["- 'open_dream_rsi'", "- 'mcp'"]:
            self.assertIn(fragment, self.text)
        # Absolute-path placeholders for both mounts — DSH never expands `~`.
        self.assertIn("'--tasks'", self.text)
        self.assertIn("'--memory'", self.text)
        self.assertIn("/ABSOLUTE/PATH/", self.text)

    def test_env_repass_after_dsh_scrubbing(self):
        # DSH drops child-env names matching KEY|PASSWORD|SECRET|TOKEN and
        # all DSH_*; the overlay must re-pass the resolution vars explicitly.
        for var in ["OPENAI_BASE_URL", "OPENAI_API_KEY", "ODR_LLM_MODEL",
                    "ODR_LLM_PRESET", "PYTHONPATH"]:
            self.assertRegex(
                self.text, rf"{var}: !!js process\.env\.{var}")

    def test_tool_call_timeout_exceeds_default(self):
        match = re.search(r"toolCallTimeoutMs:\s*(\d+)", self.text)
        assert match is not None, "toolCallTimeoutMs missing from overlay"
        # Improvement cycles run for many minutes; the 60 s SDK default
        # would kill them mid-cycle.
        self.assertGreater(int(match.group(1)), 60000)

    def test_no_literal_secrets(self):
        self.assertNotRegex(self.text, r"sk-[A-Za-z0-9]{8,}")
        self.assertNotIn("Bearer ", self.text)


class TestBundleManifest(unittest.TestCase):
    """npm bundle form: package.json declares dsh.bundle + the layer file."""

    def setUp(self):
        self.pkg = json.loads(
            (_PLUGIN / "package.json").read_text(encoding="utf-8"))

    def test_declares_bundle_patch(self):
        self.assertEqual(self.pkg["name"], "dsh-open-dream-rsi")
        self.assertEqual(self.pkg["dsh"]["bundle"]["patch"], "./cordis.patch.yml")
        self.assertIn("cordis.patch.yml", self.pkg["files"])

    def test_layer_references_inbox_bridge_by_package_name(self):
        text = (_PLUGIN / "cordis.patch.yml").read_text(encoding="utf-8")
        self.assertIn("- insert:", text)
        self.assertIn("name: '@deepseek-ai/dsh-mcp-client'", text)
        self.assertIn("toolCallTimeoutMs: 1800000", text)
        # Bundle defaults resolve against the launch cwd — no placeholders.
        self.assertNotIn("/ABSOLUTE/PATH/", text)
        self.assertNotIn("/opt/data/", text)

    def test_no_literal_secrets(self):
        for text in [(_PLUGIN / "cordis.patch.yml").read_text(encoding="utf-8"),
                     (_PLUGIN / "package.json").read_text(encoding="utf-8")]:
            self.assertNotRegex(text, r"sk-[A-Za-z0-9]{8,}")


class TestHttpOverlayContract(unittest.TestCase):
    def setUp(self):
        self.text = _overlay_text(HTTP_OVERLAY)

    def test_streamable_http_entry(self):
        self.assertIn("transport: streamable-http", self.text)
        self.assertRegex(self.text, r"url: !!js process\.env\.ODR_MCP_URL")
        self.assertRegex(_servername(self.text), SERVERNAME_RE)

    def test_match_stdio_timeout(self):
        self.assertIn("toolCallTimeoutMs: 1800000", self.text)


class TestOverlayCommandSpeaksMCP(unittest.TestCase):
    """The exact stdio command the overlay configures must serve MCP tools."""

    def test_initialize_and_tools_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks = Path(tmp) / "tasks.json"
            tasks.write_text(
                json.dumps([{"task_id": "smoke", "category": "smoke",
                             "prompt": "noop", "max_attempts": 1}]),
                encoding="utf-8")
            memory = Path(tmp) / ".dream_rsi"
            env = dict(os.environ)
            env["PYTHONPATH"] = str(_REPO)
            env["ODR_MEMORY"] = str(memory)
            proc = subprocess.run(
                [sys.executable, "-m", "open_dream_rsi", "mcp",
                 "--tasks", str(tasks), "--memory", str(memory)],
                input='{"id":1,"method":"initialize","params":{}}\n'
                      '{"id":2,"method":"tools/list"}\n',
                capture_output=True, text=True, env=env, timeout=120,
                cwd=str(_REPO))
            lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
            self.assertTrue(lines, f"no protocol output; stderr={proc.stderr[-500:]}")
            tools = json.loads(lines[1])["result"]["tools"]
            names = {t["name"] for t in tools}
            self.assertEqual(
                names,
                {"odr_status", "odr_recipes", "odr_lessons",
                 "odr_add_task", "odr_run_once", "odr_dream"})


if __name__ == "__main__":
    unittest.main()
