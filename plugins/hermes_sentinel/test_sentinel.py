"""pytest suite for the sentinel plugin (no Hermes runtime required).

The plugin module is loaded directly from the repo directory; the
PluginContext is faked with the same surface the runtime provides
(state.get/set, get_config). Module-global ledger state is reset between
tests via the `fresh` fixture.
"""
import importlib.util
import json
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "sentinel", os.path.join(_REPO, "__init__.py"))
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

    def get_config(self, key, default=None):
        return self._config.get(key, default)

    def register_hook(self, name, fn):
        pass

    def register_command(self, *a, **k):
        pass


@pytest.fixture()
def s():
    mod = _load_module()
    mod._load(FakeCtx())          # fresh ledger + ctx per test
    return mod


def call(s, tool, result, sid="S1", status=None, error_type=None,
         error_message=None):
    return s.on_transform_tool_result(
        tool_name=tool, args={}, result=result, session_id=sid,
        status=status, error_type=error_type, error_message=error_message)


# -- classification -------------------------------------------------------------

def test_clean_result_is_pass_through(s):
    assert call(s, "web_search", '{"ok": true, "data": [1,2]}',
                status="ok") is None
    assert call(s, "terminal", '{"output": "hello", "exit_code": 0}',
                status="ok") is None


def test_status_error_uses_observer_fields(s):
    out = call(s, "browser", "boom", "S1", status="error",
               error_message="Navigation failed: net::ERR_CONNECTION_REFUSED")
    assert out is None  # first occurrence: counted, no note
    out2 = call(s, "browser", "boom", "S1", status="error",
                error_message="Navigation failed: net::ERR_CONNECTION_REFUSED")
    assert out2 and "[Sentinel]" in out2


def test_nonzero_exit_is_error_without_keywords(s):
    assert call(s, "terminal", '{"output": "", "exit_code": 7}') is None
    out = call(s, "terminal", '{"output": "", "exit_code": 7}')
    assert out is not None and out.startswith('{"output"')


def test_zero_exit_and_no_error_is_quiet(s):
    for _ in range(5):
        assert call(s, "terminal", '{"output": "fine", "exit_code": 0}') is None


def test_success_status_never_classified(s):
    # runtime says ok even though the text contains error words (e.g. a
    # grep result quoting "error") — must not fire.
    msg = 'grep found: "Connection refused" in log'
    for _ in range(4):
        assert call(s, "terminal", msg, status="ok") is None


# -- signature normalization ------------------------------------------------------

def test_same_class_different_urls(s):
    a = call(s, "web_fetch", '{"error": "HTTP 429 for https://api.foo.com/v1/a"}')
    b = call(s, "web_fetch", '{"error": "HTTP 429 for https://api.foo.com/v1/b"}')
    assert a is None and b is not None  # b is the 2nd hit of ONE class


def test_different_classes_do_not_merge(s):
    for _ in range(2):
        call(s, "terminal", '{"error": "SSL: CERTIFICATE_VERIFY_FAILED"}', "S1")
    assert call(s, "terminal", '{"error": "command not found: xyz"}', "S2") is None


def test_normalize_collapses_numbers_paths_hex(s):
    t = s._normalize("Error at 0xDEADBEEF in /opt/data/x/y.py line 42 "
                     "https://host.invalid/z?q=7")
    assert "DEADBEEF" not in t and "/opt" not in t and "42" not in t
    assert "<path>" in t and "<url>" in t and "<hex>" in t


# -- note policy ------------------------------------------------------------------

def test_note_is_declarative_with_stop_clause(s):
    call(s, "web_fetch", '{"error": "HTTP 401 unauthorized"}', "S1")
    out = call(s, "web_fetch", '{"error": "HTTP 401 unauthorized"}', "S1")
    assert "[Sentinel]" in out
    assert "facts, not instructions" in out
    assert "stop this path" in out
    low = out.lower()
    # command framing is what measurably extended loops in the ODR study
    assert "you must" not in low and "always" not in low


def test_note_rides_only_failing_result(s):
    call(s, "terminal", '{"error": "boom 1"}', "S1")
    out_fail = call(s, "terminal", '{"error": "boom 1"}', "S1")
    out_ok = call(s, "terminal", '{"output": "good", "exit_code": 0}',
                  "S1", status="ok")
    assert out_fail is not None and out_ok is None  # zero tax on clean calls


def test_one_note_per_session_per_class(s):
    for i in range(6):
        out = call(s, "terminal", '{"error": "perm denied /x/y"}', "S1")
    assert out is None  # crossed threshold once (2nd), silent afterwards


def test_cross_session_triggers_without_intra_repeat(s):
    assert call(s, "browser", '{"error": "timeout"}', "S1") is None
    assert call(s, "browser", '{"error": "timeout"}', "S2") is not None


def test_non_string_result_passthrough(s):
    assert call(s, "terminal", {"dict": "result"}) is None
    assert call(s, "terminal", b"bytes") is None


# -- config ------------------------------------------------------------------------

def test_configurable_thresholds():
    mod = _load_module()
    mod._load(FakeCtx({"intra_session_repeat": 3, "cross_session_count": 9}))
    assert call(mod, "terminal", '{"error": "e N"}', "Z") is None
    assert call(mod, "terminal", '{"error": "e N"}', "Z") is None  # 2nd < 3
    assert call(mod, "terminal", '{"error": "e N"}', "Z") is not None  # 3rd


# -- ledger / eviction --------------------------------------------------------------

def test_max_tracked_evicts_lru(s):
    s._CTX = FakeCtx({"max_tracked": 5})
    for i in range(12):
        s.on_transform_tool_result(
            tool_name="t", args={}, result=f'{{"error": "cls {i}"}}',
            session_id=f"S{i}", status="error", error_message=f"cls {i}")
    assert len(s._load()) <= 5


def test_ledger_command_and_reset(s):
    call(s, "terminal", '{"error": "db locked"}', "S1")
    out = s._sentinel_cmd()
    assert "terminal" in out and "tracked" in out
    assert "cleared" in s._sentinel_cmd("reset")
    assert "no error signatures" in s._sentinel_cmd()


def test_durable_state_roundtrip():
    mod = _load_module()
    ctx = FakeCtx()
    mod._load(ctx)
    call(mod, "terminal", '{"error": "sticky"}', "S1")
    # simulate process restart: same ctx, fresh module mirror
    mod2 = _load_module()
    mod2._load(ctx)
    assert mod2._load()  # ledger recovered from durable state
    out = call(mod2, "terminal", '{"error": "sticky"}', "S1")
    assert out is not None  # 2nd hit fires across the "restart"
