"""Minimal MCP (stdio, newline-delimited JSON-RPC 2.0) server — stdlib only.

Exposes the spolki task sandbox (krs_lookup / fetch_company / fetch_page)
plus the SentinelEngine as tools, so any MCP-speaking agent (goose, Claude,
...) can be wired to ODR with zero host-specific code. This is the "OSI
arm" of the goose cross-agent benchmark: the sentinel is reachable ONLY as
a standard protocol tool the model must choose to call.

Reactive sentinel: the server also feeds every failing sandbox result into
SentinelEngine.observe(); when the engine emits a recurrence note, it is
appended to $SENTINEL_NOTE_FILE (JSONL) for a Stop-hook adapter to deliver.
This mirrors the Hermes plugin: observe on the wire, deliver where the host
actually lets you steer the model.

Serving a sandbox requires the bench fixtures; point them via env:
  SPOLKI_BENCH_DIR (default /opt/data/workspace/odr-bench),
  chunk dir from spolki_replay.CHUNKS.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from open_dream_rsi.sentinel import SentinelEngine  # noqa: E402

TOOL_NAMES = ("krs_lookup", "fetch_company", "fetch_page")

SERVER_INFO = {"name": "odr-sentinel-sandbox", "version": "0.1.0"}


class SandboxProvider:
    """Serves the three sandbox tools from the spolki replay fixtures."""

    def __init__(self):
        import hashlib
        sys.path.insert(0, os.environ.get("SPOLKI_BENCH_DIR",
                                          "/opt/data/workspace/odr-bench"))
        import spolki_replay as sr
        gold = json.loads(sr.GOLD.read_text())
        pages = json.loads(sr.PAGES.read_text())
        chunks_by_krs, page_text_by_krs = {}, {}
        for g in gold:
            chunk = json.loads((sr.CHUNKS / ("chunk_%02d.json" % g["chunk"]))
                               .read_text())
            for e in chunk:
                if e["krs"] == g["krs"]:
                    chunks_by_krs[g["krs"]] = e.get("wpisy", [])
            home = None
            for u in g["urls"]:
                pp = pages.get(hashlib.sha1(u.encode()).hexdigest()[:16], {})
                if "text" in pp and home is None:
                    home = pp
            if home is not None:
                page_text_by_krs[g["krs"]] = home
        self.sb = sr.Sandbox(chunks_by_krs, page_text_by_krs, pages)
        self.current_krs = ""

    def set_current(self, krs):
        self.current_krs = str(krs)
        self.sb.set_current(krs)

    def call(self, name, args):
        return self.sb.call(name, args)


def result_is_error(text: str) -> bool:
    try:
        d = json.loads(text)
        return isinstance(d, dict) and "blad" in d
    except Exception:
        return False


REFUSAL = {"blad": "budget_exhausted: tool-call budget spent; answer now "
                   "with the data you already have (further calls will be "
                   "refused)"}


class PermuteGate:
    """Choice removal, not persuasion: after ``trigger`` sandbox attempts
    the server REFUSES to serve (every further call is ``isError``).

    Measured on goose x TravelPlanner (paper sec. tp): three text channels
    (pull tool, Stop-hook note, finalize nudge) converted 0 cap-dying
    episodes; removing the option to keep exploring moved paired delivery
    from 23% to 94% on the preregistered eval split. Off by default
    (budget=0); arms opt in via SENTINEL_GATE_BUDGET / SENTINEL_GATE_AT.
    """

    def __init__(self, budget: int, at: float = 0.8):
        self.trigger = -(-budget * at // 1) if budget else 0
        self.attempts: dict = {}

    def blocks(self, session: str) -> bool:
        if not self.trigger:
            return False
        return self.attempts.get(session, 0) >= self.trigger

    def count(self, session: str) -> None:
        self.attempts[session] = self.attempts.get(session, 0) + 1


def main() -> None:
    note_file = os.environ.get("SENTINEL_NOTE_FILE", "/tmp/odr_sentinel_notes.jsonl")
    state_file = os.environ.get("SENTINEL_STATE_FILE", "/tmp/odr_sentinel_state.json")
    call_log = os.environ.get("ODR_CALL_LOG", "")
    sentinel_on = os.environ.get("SENTINEL_OFF") != "1"
    engine = SentinelEngine(state_path=Path(state_file)) if sentinel_on else None
    gate = PermuteGate(int(os.environ.get("SENTINEL_GATE_BUDGET", "0") or 0),
                       float(os.environ.get("SENTINEL_GATE_AT", "0.8")))
    sandbox = SandboxProvider() if os.environ.get("SPOLKI_SERVE") == "1" else None

    def emit(obj):
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def ok(req_id, result):
        emit({"jsonrpc": "2.0", "id": req_id, "result": result})

    def err(req_id, code, message):
        emit({"jsonrpc": "2.0", "id": req_id,
              "error": {"code": code, "message": message}})

    tools_list = []
    if sentinel_on and os.environ.get("SENTINEL_TOOL") != "0":
        tools_list.append({"name": "sentinel_check",
         "description": "Zapytaj sentinela ODR, czy dla bieżącego błędu/"
                        "narzędzia znana jest powtarzająca się klasa błędów. "
                        "Wywołuj PRZED ponowną próbą, która różni się tylko "
                        "parametrem.",
         "inputSchema": {"type": "object", "properties": {
             "tool": {"type": "string"},
             "error_excerpt": {"type": "string"}},
             "required": ["tool"]}})
    if sandbox is not None:
        tools_list += [
            {"name": "krs_lookup", "description":
             "Wpisy rejestrowe KRS dla spółki (organy, role, maski nazwisk).",
             "inputSchema": {"type": "object", "properties": {
                 "krs": {"type": "string"}}, "required": ["krs"]}},
            {"name": "fetch_company", "description":
             "Treść głównej pobranej strony władz spółki.",
             "inputSchema": {"type": "object", "properties": {
                 "krs": {"type": "string"}}, "required": ["krs"]}},
            {"name": "fetch_page", "description":
             "Treść strony WWW po pełnym URL (ta sama domena co strona spółki).",
             "inputSchema": {"type": "object", "properties": {
                 "url": {"type": "string"}}, "required": ["url"]}},
        ]

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = req.get("method")
        req_id = req.get("id")
        if method == "initialize":
            ok(req_id, {"protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": SERVER_INFO})
        elif method in ("notifications/initialized", "initialized"):
            continue
        elif method == "ping":
            ok(req_id, {})
        elif method == "tools/list":
            ok(req_id, {"tools": tools_list})
        elif method == "tools/call":
            params = req.get("params", {})
            name = params.get("name")
            args = params.get("arguments", {}) or {}
            if name == "sentinel_check" and engine is not None:
                note = engine.observe(args.get("tool", "unknown"),
                                      {"blad": args.get("error_excerpt", "")},
                                      session_id=os.environ.get(
                                          "SENTINEL_SESSION", "mcp-default"),
                                      status="error")  # caller asserts it failed
                ok(req_id, {"content": [{"type": "text",
                                         "text": note or "brak znanej klasy błędu dla tego narzędzia"}],
                            "isError": False})
            elif sandbox is not None and name in TOOL_NAMES:
                sid = os.environ.get("SENTINEL_SESSION", "mcp-default")
                gate.count(sid)  # count attempts INCLUDING gated ones
                if gate.blocks(sid):
                    text = json.dumps(REFUSAL)
                    if call_log:
                        with open(call_log, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"tool": name,
                                                "ok": False}) + "\n")
                    ok(req_id, {"content": [{"type": "text", "text": text}],
                                "isError": True})
                    continue
                if name == "fetch_company" or name == "krs_lookup":
                    if str(args.get("krs", "")):
                        sandbox.set_current(args.get("krs", ""))
                text = sandbox.call(name, args)
                if call_log:
                    with open(call_log, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"tool": name, "ok":
                                            not result_is_error(text)}) + "\n")
                if result_is_error(text) and engine is not None:
                    note = engine.observe(name, text, session_id=os.environ.get(
                        "SENTINEL_SESSION", "mcp-default"))
                    if note:
                        with open(note_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"ts": time.time(),
                                                "note": note},
                                               ensure_ascii=False) + "\n")
                ok(req_id, {"content": [{"type": "text", "text": text}],
                            "isError": result_is_error(text)})
            else:
                err(req_id, -32602, f"unknown tool: {name}")
        elif method == "shutdown":
            ok(req_id, {})
        elif req_id is not None:
            err(req_id, -32601, f"method not found: {method}")


if __name__ == "__main__":
    main()
