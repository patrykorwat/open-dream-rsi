"""Adapter tests for the Hermes sentinel plugin (no Hermes runtime required).

The plugin is a thin bridge: hook kwargs -> SentinelEngine (resolved from
the installed open_dream_rsi package, or the vendored _sentinel_engine.py
fallback), engine store -> ctx.state. Deep engine behaviour is covered in
tests/test_sentinel_engine.py; these tests pin the ADAPTER contract:
hook payload mapping, single-hook registration, the durable bridge through
a faked ctx.state, config plumbing, the slash command, and never-raise.
"""
import importlib.util
import json
import os


import pytest

_REPO = os.path.dirname(os.path.abspath(__file__))


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "sentinel_adapter", os.path.join(_REPO, "__init__.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeState:
    def __init__(self):
        self.d = {}

    def get(self, key, default=None):
        return json.loads(json.dumps(self.d.get(key, default)))

    def set(self, key, value):
        self.d[key] = json.loads(json.dumps(value))


class FakeCtx:
    def __init__(self, config=None):
        self.state = FakeState()
        self._config = config or {}
        self.hooks = {}
        self.commands = {}

    def get_config(self, key, default=None):
        return self._config.get(key, default)

    def register_hook(self, name, fn):
        self.hooks[name] = fn

    def register_command(self, name, handler, description=None, args_hint=None):
        self.commands[name] = handler


@pytest.fixture()
def s():
    # NOTE: the plugin caches the engine in a module global; each test loads
    # a FRESH module instance so globals cannot leak between tests.
    mod = _load_module()
    ctx = FakeCtx()
    mod.register(ctx)
    return mod, ctx


def call(mod, tool="terminal", result='{"error": "boom"}', sid="S1",
         status=None, error_message=None):
    return mod.on_transform_tool_result(
        tool_name=tool, args={}, result=result, session_id=sid,  # type: ignore[arg-type]
        status=status, error_message=error_message)


# -- registration --------------------------------------------------------------------

def test_registers_single_hook_and_command(s):
    mod, ctx = s
    assert set(ctx.hooks) == {"transform_tool_result"}
    assert "sentinel" in ctx.commands
    # the registered callable IS the hook (single hook, no post_tool_call twin)
    assert ctx.hooks["transform_tool_result"] is mod.on_transform_tool_result


# -- hook contract ---------------------------------------------------------------------

def test_clean_results_pass_through(s):
    mod, _ = s
    assert call(mod, "web_search", '{"ok": true}', status="ok") is None
    assert call(mod, "terminal", '{"output": "hi", "exit_code": 0}',
                status="ok") is None


def test_non_string_result_passes_through(s):
    mod, _ = s
    assert call(mod, "terminal", {"dict": "result"}) is None
    assert call(mod, "terminal", b"bytes") is None


def test_threshold_then_one_note_per_session(s):
    mod, _ = s
    assert call(mod) is None                          # 1st: counted, silent
    out = call(mod)                                   # 2nd: note appended
    assert out is not None and out.startswith('{"error"')
    assert "[Sentinel]" in out and "facts, not instructions" in out
    assert call(mod) is None                          # silent afterwards


def test_never_raises_on_garbage(s):
    mod, _ = s
    # the hook must degrade to pass-through on any unexpected input
    assert mod.on_transform_tool_result() is None
    assert mod.on_transform_tool_result(tool_name=None, result=123) is None


# -- durable bridge ----------------------------------------------------------------------

def test_state_persists_through_ctx_store(s):
    mod, ctx = s
    call(mod)
    assert ctx.state.d.get("sigs")           # bridge wrote through ctx.state
    # simulate process restart: fresh module, same ctx.state
    mod2 = _load_module()
    mod2.register(ctx)
    out = call(mod2)
    assert out is not None                   # 2nd hit seen across the "restart"


def test_config_thresholds_plumbed():
    mod = _load_module()
    ctx = FakeCtx({"intra_session_repeat": 3, "cross_session_count": 9})
    mod.register(ctx)
    assert call(mod, sid="Z") is None
    assert call(mod, sid="Z") is None
    assert call(mod, sid="Z") is not None


# -- slash command -------------------------------------------------------------------------

def test_ledger_and_reset_via_command(s):
    mod, ctx = s
    call(mod, "terminal", '{"error": "db locked"}', sid="S1")
    led = ctx.commands["sentinel"]()
    assert "tracked" in led
    assert "cleared" in ctx.commands["sentinel"]("reset")
    assert "no error signatures" in ctx.commands["sentinel"]()


# -- vendored fallback -----------------------------------------------------------------------

def test_vendored_engine_matches_upstream():
    """The plugin directory must work standalone (copy-install), and the
    vendored engine must not silently drift from open_dream_rsi.sentinel."""
    path = os.path.join(_REPO, "_sentinel_engine.py")
    assert os.path.exists(path)
    upstream = os.path.join(os.path.dirname(_REPO), os.pardir,
                            "open_dream_rsi", "sentinel.py")
    with open(path, "rb") as f:
        vendored = f.read()
    with open(os.path.abspath(upstream), "rb") as f:
        assert vendored == f.read(), "vendored engine drifted from upstream"
