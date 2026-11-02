"""Adapter tests for the DeepSeek Harness plugin (no DSH runtime required).

The adapter is pure configuration: two Cordis overlays that insert the ODR
MCP server behind ``@deepseek-ai/dsh-mcp-client``, plus the npm bundle's
``bridge.js`` — which both publishes the vendored server's location and
serves the dreamer's LLM calls through an in-process OpenAI-compatible
proxy backed by the harness's own dsh-llm service. These tests pin the
overlay contract (entry shape, server naming, the env re-pass — DSH spawns
stdio children with a scrubbed environment — the long tool-call timeout),
the bridge contract (child env points the child at the proxy unless the
harness has a real OPENAI_BASE_URL; the proxy answers /v1/models and
/v1/chat/completions from ctx.llm.stream), and verify that the exact stdio
command the overlay configures speaks the MCP protocol. DSH-side loading is
verified manually with ``dsh --patch <overlay> --dump-config`` (needs the
npm package).
"""
import json
import os
import re
import shutil
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
        # all DSH_*. Both adapters re-pass the LLM-resolution vars but DROP
        # unset/empty ones: an empty ODR_LLM_PRESET="" must never reach the
        # child (older ODR releases died with "Unknown preset ''").
        for var in ["OPENAI_BASE_URL", "OPENAI_API_KEY", "ODR_LLM_MODEL",
                    "ODR_LLM_PRESET", "PYTHONPATH"]:
            self.assertRegex(
                self.text, rf"{var}: !!js process\.env\.{var} \|\| undefined")
        self.assertNotIn("|| ''", self.text)

    def test_bundle_env_via_bridge_drops_empty(self):
        # The npm bundle routes the child env through the bridge's
        # childEnv(), which passes only vars that are actually set.
        patch_text = (_PLUGIN / "cordis.patch.yml").read_text(encoding="utf-8")
        self.assertIn("env: !!js ctx.odrBundle.childEnv()", patch_text)
        bridge = (_PLUGIN / "bridge.js").read_text(encoding="utf-8")
        for var in ["OPENAI_BASE_URL", "OPENAI_API_KEY", "ODR_LLM_MODEL",
                    "ODR_LLM_PRESET", "PYTHONPATH"]:
            self.assertIn(f'"{var}"', bridge)

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
                {"odr_status", "odr_recipes",
                 "odr_add_task", "odr_run_once", "odr_dream"})


class TestBridgeProxy(unittest.TestCase):
    """bridge.js is the dreamer's brain: an in-process OpenAI-compatible
    proxy backed by the harness's own dsh-llm service. Driven through node
    with a fake ctx.llm — no DSH runtime, no network beyond 127.0.0.1."""

    @classmethod
    def setUpClass(cls):
        cls.node = shutil.which("node")
        if cls.node is None:
            raise unittest.SkipTest("node not available")
        cls.bridge = str(_PLUGIN / "bridge.js")
        # Deterministic proxy mode: no OPENAI_*/ODR_* vars inherited.
        cls.env = {k: v for k, v in os.environ.items()
                   if not k.startswith(("OPENAI_", "ODR_"))}
        # The bridge path rides in via env (argv positionals are swallowed by
        # `node -e`).
        cls.env["ODR_BRIDGE"] = cls.bridge

    def _run(self, script: str) -> str:
        node = self.node
        assert node is not None  # setUpClass skips the class when node is absent
        proc = subprocess.run(
            [node, "--input-type=module", "-e", script, self.bridge],
            capture_output=True, text=True, timeout=60, env=self.env)
        self.assertEqual(proc.returncode, 0,
                         f"node failed: {proc.stderr[-800:]}")
        return proc.stdout

    # Shared prologue: import the bridge (path via ODR_BRIDGE env) and wait
    # for the proxy port.
    _BOOT = (
        "import { pathToFileURL } from 'node:url';\n"
        "const mod = await import(pathToFileURL(process.env.ODR_BRIDGE).href);\n"
        "for (let i = 0; i < 100 && !mod.state.port; i++)"
        " await new Promise(r => setTimeout(r, 10));\n"
    )

    def test_child_env_points_child_at_proxy_unless_direct_url_set(self):
        out = self._run(self._BOOT + """
delete process.env.OPENAI_BASE_URL;
process.env.ODR_LLM_PRESET = '';
let env = mod.childEnv();
console.log('PROXY_URL_OK=' + (env.OPENAI_BASE_URL === `http://127.0.0.1:${mod.state.port}/v1`));
console.log('DUMMY_KEY=' + env.OPENAI_API_KEY);
console.log('PRESET_DROPPED=' + !('ODR_LLM_PRESET' in env));
process.env.OPENAI_BASE_URL = 'http://real:1234/v1';
env = mod.childEnv();
console.log('DIRECT_OK=' + (env.OPENAI_BASE_URL === 'http://real:1234/v1'));
console.log('DIRECT_NO_DUMMY=' + (env.OPENAI_API_KEY !== 'dsh-bridge'));
delete process.env.OPENAI_BASE_URL;
""")
        self.assertIn("PROXY_URL_OK=true", out)
        self.assertIn("DUMMY_KEY=dsh-bridge", out)
        self.assertIn("PRESET_DROPPED=true", out)
        self.assertIn("DIRECT_OK=true", out)
        self.assertIn("DIRECT_NO_DUMMY=true", out)

    def test_proxy_serves_models_and_chat_from_ctx_llm(self):
        out = self._run(self._BOOT + """
mod.state.ctx = { llm: {
  listProviders: async () => [{ id: 'main' }],
  listModels: async () => [{ id: 'fake-model' }],
  stream: async function* (options) {
    console.log('CALL_OPTS=' + JSON.stringify({
      provider: options.provider, model: options.model,
      system: options.system ?? null, n: options.messages.length,
      temperature: options.temperature ?? null,
      maxTokens: options.maxTokens ?? null }));
    yield { type: 'text-delta', index: 0, text: 'Hello ' };
    yield { type: 'text-delta', index: 0, text: 'from fake llm' };
    yield { type: 'finish', finish: { kind: 'stop' } };
  },
}};
const base = `http://127.0.0.1:${mod.state.port}/v1`;
const models = await (await fetch(`${base}/models`)).json();
console.log('MODEL=' + models.data[0].id);
const res = await fetch(`${base}/chat/completions`, {
  method: 'POST', headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ model: 'ignored', messages: [
    { role: 'system', content: 'be brief' },
    { role: 'user', content: 'hi' } ], temperature: 0.5, max_tokens: 64 }) });
const body = await res.json();
console.log('STATUS=' + res.status);
console.log('CONTENT=' + body.choices[0].message.content);
console.log('FINISH=' + body.choices[0].finish_reason);
// The harness's own agent default route, once present, wins over the
// first-registered-provider fallback.
mod.state.model = null;
mod.state.ctx.agentDefaultModel = {
  currentSelection: () => ({ provider: 'agent-route', model: 'agent-model' }),
};
const res2 = await fetch(`${base}/chat/completions`, {
  method: 'POST', headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] }) });
console.log('AGENT_ROUTE=' + (await res2.json()).model);
delete mod.state.ctx.agentDefaultModel;
""")
        self.assertIn("MODEL=fake-model", out)
        self.assertIn("STATUS=200", out)
        self.assertIn("CONTENT=Hello from fake llm", out)
        self.assertIn("FINISH=stop", out)
        self.assertIn("AGENT_ROUTE=agent-model", out)
        opts = re.search(r"CALL_OPTS=(\{.*\})", out)
        assert opts is not None, f"stream was never called; stdout={out!r}"
        call = json.loads(opts.group(1))
        self.assertEqual(call["provider"], "main")
        self.assertEqual(call["model"], "fake-model")
        self.assertEqual(call["system"], "be brief")
        self.assertEqual(call["n"], 1)  # system folded out of messages
        self.assertEqual(call["temperature"], 0.5)
        self.assertEqual(call["maxTokens"], 64)

    def test_proxy_reports_provider_model_and_llm_failures(self):
        out = self._run(self._BOOT + """
const base = `http://127.0.0.1:${mod.state.port}/v1`;
const post = (llm) => { mod.state.ctx = { llm }; return fetch(
  `${base}/chat/completions`, { method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] }) }); };
let res = await post({ listProviders: async () => [],
  listModels: async () => [], stream: async function* () {} });
console.log('NO_PROVIDER=' + res.status + ':' +
  (await res.text()).includes('no LLM provider is registered'));
res = await post({ listProviders: async () => [{ id: 'main' }],
  listModels: async () => [], stream: async function* () {} });
console.log('NO_MODEL=' + res.status + ':' +
  (await res.text()).includes('cannot resolve a model'));
res = await post({ listProviders: async () => [{ id: 'main' }],
  listModels: async () => [{ id: 'm' }],
  stream: async function* () {
    yield { type: 'finish',
            finish: { kind: 'error', failure: { message: 'boom', code: 'X' } } };
  } });
console.log('LLM_ERROR=' + res.status + ':' +
  (await res.text()).includes('boom'));
""")
        self.assertIn("NO_PROVIDER=500:true", out)
        self.assertIn("NO_MODEL=500:true", out)
        self.assertIn("LLM_ERROR=502:true", out)


if __name__ == "__main__":
    unittest.main()

