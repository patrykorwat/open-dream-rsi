"""Engine-level tests for the host-agnostic error-class sentinel.

Adapter-specific behaviour (Hermes hook, Claude CLI contract) lives in
plugins/hermes_sentinel/test_sentinel.py and the CLI tests here.
"""
import json
import subprocess
import sys
from pathlib import Path

from open_dream_rsi.sentinel import (SentinelEngine, classify, normalize)

REPO = Path(__file__).resolve().parents[1]


def engine(tmp_path, **kw):
    return SentinelEngine(state_path=Path(tmp_path) / "s.json", **kw)


def observe(e, tool="terminal", result='{"error": "boom"}', sid="S1",
            status=None, error_message=None):
    return e.observe(tool, result, sid, status=status,
                      error_message=error_message)


# -- classification -------------------------------------------------------------

def test_clean_result_is_pass_through(tmp_path):
    e = engine(tmp_path)
    assert observe(e, "web_search", '{"ok": true, "data": [1,2]}',
                   status="ok") is None
    assert observe(e, "terminal", '{"output": "hello", "exit_code": 0}',
                   status="ok") is None


def test_status_error_uses_observer_fields(tmp_path):
    e = engine(tmp_path)
    assert observe(e, "browser", "boom", status="error",
                   error_message="net::ERR_CONNECTION_REFUSED") is None
    out = observe(e, "browser", "boom", status="error",
                  error_message="net::ERR_CONNECTION_REFUSED")
    assert out and "[Sentinel]" in out


def test_nonzero_exit_is_error_without_keywords(tmp_path):
    e = engine(tmp_path)
    assert observe(e, "terminal", '{"output": "", "exit_code": 7}') is None
    out = observe(e, "terminal", '{"output": "", "exit_code": 7}')
    assert out is not None


def test_exit_codes_are_different_classes(tmp_path):
    e = engine(tmp_path)
    assert observe(e, "t", '{"exit_code": 1}') is None
    # exit 1 (failed) and exit 127 (command not found) are different blockers:
    # the second occurrence of exit 127 must be its OWN first... check both
    # classes count independently and neither borrows the other's hits.
    assert observe(e, "t", '{"exit_code": 127}') is None       # would trigger if merged
    assert observe(e, "t", '{"exit_code": 127}') is not None   # 2nd x 127 -> note
    assert observe(e, "t", '{"exit_code": 1}') is not None     # 2nd x 1 -> own note


def test_zero_exit_and_no_error_is_quiet(tmp_path):
    e = engine(tmp_path)
    for _ in range(5):
        assert observe(e, "terminal", '{"output": "fine", "exit_code": 0}') is None


def test_success_status_never_classified(tmp_path):
    # runtime says ok even though text contains error words (grep quoting
    # "error") — must not fire.
    msg = 'grep found: "Connection refused" in log'
    e = engine(tmp_path)
    for _ in range(4):
        assert observe(e, "terminal", msg, status="ok") is None


def test_isError_flag_counts(tmp_path):
    # MCP-style error envelope (Claude tool_response uses isError)
    e = engine(tmp_path)
    assert observe(e, "mcp__x", '{"isError": true, "content": "denied"}') is None
    assert observe(e, "mcp__x", '{"isError": true, "content": "denied"}') is not None


# -- signature normalization ------------------------------------------------------

def test_same_class_different_urls(tmp_path):
    e = engine(tmp_path)
    assert observe(e, "web_fetch",
                   '{"error": "HTTP 429 for https://api.foo.com/v1/a"}') is None
    assert observe(e, "web_fetch",
                   '{"error": "HTTP 429 for https://api.foo.com/v1/b"}') is not None


def test_different_classes_do_not_merge(tmp_path):
    e = engine(tmp_path)
    for _ in range(2):
        observe(e, "terminal", '{"error": "SSL: CERTIFICATE_VERIFY_FAILED"}')
    assert observe(e, "terminal", '{"error": "command not found: xyz"}',
                   sid="S2") is None


def test_normalize_collapses_numbers_paths_hex():
    t = normalize("Error at 0xDEADBEEF in /opt/data/x/y.py line 42 "
                  "https://host.invalid/z?q=7")
    assert "DEADBEEF" not in t and "/opt" not in t and "42" not in t
    assert "<path>" in t and "<url>" in t and "<hex>" in t


def test_classify_returns_none_for_ok_payloads():
    assert classify("t", '{"output": "fine", "exit_code": 0}', None) is None
    assert classify("t", "plain text no error", "ok") is None


# -- note policy -------------------------------------------------------------------

def test_note_is_declarative_with_stop_clause(tmp_path):
    e = engine(tmp_path)
    observe(e, "web_fetch", '{"error": "HTTP 401 unauthorized"}')
    out = observe(e, "web_fetch", '{"error": "HTTP 401 unauthorized"}')
    assert out is not None
    assert "[Sentinel]" in out
    assert "facts, not instructions" in out
    assert "stop this path" in out
    low = out.lower()
    # command framing is what measurably extended loops in the ODR study
    assert "you must" not in low and "always" not in low


def test_one_note_per_session_per_class(tmp_path):
    e = engine(tmp_path)
    out = None
    for _ in range(6):
        out = observe(e, "terminal", '{"error": "perm denied /x/y"}')
    assert out is None  # crossed once (2nd), silent afterwards


def test_cross_session_triggers_without_intra_repeat(tmp_path):
    e = engine(tmp_path)
    assert observe(e, "browser", '{"error": "timeout"}', sid="S1") is None
    assert observe(e, "browser", '{"error": "timeout"}', sid="S2") is not None


def test_configurable_thresholds(tmp_path):
    e = engine(tmp_path, intra_session_repeat=3, cross_session_count=9)
    assert observe(e, "terminal", '{"error": "e N"}', sid="Z") is None
    assert observe(e, "terminal", '{"error": "e N"}', sid="Z") is None
    assert observe(e, "terminal", '{"error": "e N"}', sid="Z") is not None


# -- ledger / persistence -------------------------------------------------------------

def test_max_tracked_evicts_lru(tmp_path):
    e = engine(tmp_path, max_tracked=5)
    for i in range(12):
        e.observe("t", f'{{"error": "cls {i}"}}', f"S{i}",
                  status="error", error_message=f"cls {i}")
    assert len(e._sigs) <= 5


def test_ledger_text_and_reset(tmp_path):
    e = engine(tmp_path)
    observe(e, "terminal", '{"error": "db locked"}')
    assert "terminal" in e.ledger_text()
    e.reset()
    assert "no error signatures" in e.ledger_text()


def test_durable_state_roundtrip(tmp_path):
    p = Path(tmp_path) / "s.json"
    e1 = SentinelEngine(state_path=p)
    assert e1.observe("terminal", '{"error": "sticky"}', "S1") is None
    e2 = SentinelEngine(state_path=p)   # simulated process restart
    out = e2.observe("terminal", '{"error": "sticky"}', "S1")
    assert out is not None              # 2nd hit fires across the restart


# -- CLI contract (command-hook integration) ------------------------------------------

def _run(*args, stdin=""):
    return subprocess.run(
        [sys.executable, "-m", "open_dream_rsi", *args],
        input=stdin, capture_output=True, text=True, cwd=REPO, timeout=60)


def test_cli_plain_and_claude_formats(tmp_path):
    state = str(Path(tmp_path) / "s.json")
    ev = json.dumps({"tool_name": "Bash",
                     "tool_response": {"exit_code": 7},
                     "session_id": "t"})
    r1 = _run("sentinel", "check", "--state", state, stdin=ev)
    assert r1.returncode == 0 and r1.stdout.strip() == ""
    r2 = _run("sentinel", "check", "--state", state, "--format", "claude",
              stdin=ev)
    assert r2.returncode == 0
    payload = json.loads(r2.stdout)
    ctx = payload["hookSpecificOutput"]
    assert ctx["hookEventName"] == "PostToolUse"
    assert "[Sentinel]" in ctx["additionalContext"]


def test_cli_survives_garbage_stdin(tmp_path):
    r = _run("sentinel", "check", "--format", "claude",
             "--state", str(Path(tmp_path) / "s.json"), stdin="not json {{{")
    assert r.returncode == 0
    assert json.loads(r.stdout) == {}


def test_cli_posttoolusefailure_triggers(tmp_path):
    state = str(Path(tmp_path) / "s.json")
    ev = json.dumps({"hook_event_name": "PostToolUseFailure",
                     "tool_name": "Bash",
                     "tool_response": "something exploded weirdly",
                     "session_id": "t"})
    assert _run("sentinel", "check", "--state", state, stdin=ev).stdout.strip() == ""
    out = _run("sentinel", "check", "--state", state, stdin=ev).stdout
    assert "[Sentinel]" in out


def test_cli_ledger_and_reset(tmp_path):
    state = str(Path(tmp_path) / "s.json")
    ev = json.dumps({"tool": "x", "result": {"error": "cls"}, "session_id": "t"})
    _run("sentinel", "check", "--state", state, stdin=ev)
    led = _run("sentinel", "ledger", "--state", state)
    assert "tracked" in led.stdout and led.returncode == 0
    rst = _run("sentinel", "reset", "--state", state)
    assert "cleared" in rst.stdout


# -- finalize nudge (budget pressure channel) ---------------------------------------

def test_finalize_nudge_fires_once_at_budget(tmp_path):
    e = engine(tmp_path, finalize_budget=10, finalize_at=0.8)
    outs = [e.observe("t", '{"output": "fine", "exit_code": 0}', "S1")
            for _ in range(12)]
    fired = [o for o in outs if o and "Budget fact" in o]
    assert len(fired) == 1                      # dedupe: once per session
    assert "8 of 10" in fired[0]                # ceil(0.8*10)
    assert "facts, not instructions" in fired[0]


def test_finalize_nudge_disabled_by_default(tmp_path):
    e = engine(tmp_path)
    for _ in range(50):
        assert e.observe("t", '{"output": "fine", "exit_code": 0}', "S1") is None


def test_finalize_nudge_counts_clean_calls(tmp_path):
    # 73% of cut-off episodes in the goose study ended on a CLEAN call:
    # clean results must count toward the budget even when they stay silent.
    e = engine(tmp_path, finalize_budget=5, finalize_at=0.6)  # trigger: call 3
    assert e.observe("t", '{"ok": 1}', "S1", status="ok") is None
    assert e.observe("t", '{"ok": 2}', "S1", status="ok") is None
    out = e.observe("t", '{"ok": 3}', "S1", status="ok")
    assert out and "Budget fact" in out


def test_finalize_nudge_merges_with_recurrence_note(tmp_path):
    # trigger lands exactly on the call that crosses the repeat threshold:
    # both channels must ride the same result, nudge first.
    e = engine(tmp_path, finalize_budget=4, finalize_at=0.5)  # trigger: call 2
    e.observe("t", '{"error": "boom"}', "S1")
    out = e.observe("t", '{"error": "boom"}', "S1")
    assert out and "Budget fact" in out and "[Sentinel] Same error class" in out
    assert out.index("Budget fact") < out.index("Same error class")


def test_note_cap_is_per_session_and_opt_in(tmp_path):
    # The cap bounds ONE session's note volume (doom-loop defence at
    # episode scale). Production default 0 = unlimited: the 30-day replay
    # fixture shows a legit host session emitting 24 notes.
    def burst(e):
        fired = 0
        for i in range(12):                      # 12 distinct classes, one session
            cls = "cls " + chr(ord("a") + i)     # letters: survive normalize()
            for _ in range(2):                   # each class crosses intra=2 once
                out = e.observe("t", f'{{"error": "{cls}"}}', "S1")
                fired += 1 if out else 0
        return fired
    assert burst(engine(tmp_path, max_notes_per_session=8)) == 8
    assert burst(engine(tmp_path / "u", max_notes_per_session=0)) == 12
