#!/usr/bin/env python3
"""TravelPlanner sandbox as a stdio MCP server (JSON-RPC 2.0, stdlib + bench venv).

Exposes the OFFICIAL TravelPlanner tools 1:1 against the official offline
database, so a benchmark agent (goose) can be arm-tested on a public,
deterministically-scored task:

  flight_search(origin, destination, date)   -> Flights.run
  attraction_search(city)                    -> Attractions.run
  accommodation_search(city)                 -> Accommodations.run
  restaurant_search(city)                    -> Restaurants.run
  distance_matrix(origin, destination, mode) -> GoogleDistanceMatrix.run
  city_search(state)                         -> Cities.run

The tool classes come from the official osunlp/TravelPlanner checkout
(pinned: e52c87f4ac348a3410c46dc3553c519db5ec5e23); the pickle frame cache
is a startup optimisation only (see Sandbox.load_frame), not a data fork.

Env:
  TP_SERVE=1        required to serve tools
  TP_REPO           official TravelPlanner checkout (default /tmp/tp);
                    DB = $TP_REPO/database, frame cache = $TP_REPO/df_cache
  TP_DB / TP_DF_CACHE  override the two paths individually
  SENTINEL_* / ODR_CALL_LOG  same semantics as open_dream_rsi.mcp_server
                             (reactive sentinel on error results).
  SENTINEL_GATE_BUDGET / SENTINEL_GATE_AT  permute-gate arm (see main()).
"""
import json
import os
import sys
import types as _types
from pathlib import Path

TP_REPO = os.environ.get("TP_REPO", "/tmp/tp")
DB = os.environ.get("TP_DB", TP_REPO + "/database")
sys.path.insert(0, TP_REPO)
if "gradio" not in sys.modules:  # utils.func imports it for the demo UI only
    sys.modules["gradio"] = _types.ModuleType("gradio")

from tools.flights.apis import Flights  # noqa: E402
from tools.accommodations.apis import Accommodations  # noqa: E402
from tools.restaurants.apis import Restaurants  # noqa: E402
from tools.googleDistanceMatrix.apis import GoogleDistanceMatrix  # noqa: E402
from tools.attractions.apis import Attractions  # noqa: E402
from tools.cities.apis import Cities  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from open_dream_rsi.sentinel import SentinelEngine  # noqa: E402

SERVER_INFO = {"name": "tp-sandbox", "version": "0.1.0"}


def _df_to_text(r):
    import pandas as pd
    if isinstance(r, pd.DataFrame):
        return r.to_string(index=False)
    return str(r)


DF_CACHE = os.environ.get("TP_DF_CACHE", TP_REPO + "/df_cache")


class Sandbox:
    def __init__(self):
        import contextlib, io, pickle
        # the official tool classes print load banners to stdout —
        # that would corrupt the JSON-RPC channel. GoogleDistanceMatrix
        # also reads its CSV from '../database/...' (cwd-relative, as in
        # the official agent scripts which chdir to evaluation/).
        os.chdir(os.path.join(TP_REPO, "evaluation"))

        def load_frame(name):
            # pickled frames built once by build_df_cache.py: the flights
            # CSV is 291 MB and pandas cold-load takes ~11-14s under our
            # swapped host — goose's MCP init timeout then SIGKILLs the
            # server mid-load ("process quit before initialization") and
            # the episode silently runs WITHOUT tools. Pickle startup <1s.
            p = Path(DF_CACHE) / f"{name}.pkl"
            if p.exists():
                import pandas as pd
                return pickle.load(open(p, "rb")), p
            return None, p

        with contextlib.redirect_stdout(io.StringIO()):
            fr, _ = load_frame("flights")
            self.flights = Flights(path=DB + "/flights/clean_Flights_2022.csv")
            if fr is not None:
                self.flights.data = fr
            else:
                self.flights.load_db()
            fr, _ = load_frame("accom")
            self.accom = Accommodations(path=DB + "/accommodations/clean_accommodations_2022.csv")
            if fr is not None:
                self.accom.data = fr
            fr, _ = load_frame("rest")
            self.rest = Restaurants(path=DB + "/restaurants/clean_restaurant_2022.csv")
            if fr is not None:
                self.rest.data = fr
            fr, _ = load_frame("gdm")
            self.gdm = GoogleDistanceMatrix()
            if fr is not None:
                self.gdm.data = fr
            fr, _ = load_frame("attr")
            self.attr = Attractions(path=DB + "/attractions/attractions.csv")
            if fr is not None:
                self.attr.data = fr
            self.cities = Cities(path=DB + "/background/citySet_with_states.txt")

    def call(self, name, args):
        try:
            if name == "flight_search":
                r = self.flights.run(args["origin"], args["destination"], args["date"])
            elif name == "attraction_search":
                r = self.attr.run(args["city"])
            elif name == "accommodation_search":
                r = self.accom.run(args["city"])
            elif name == "restaurant_search":
                r = self.rest.run(args["city"])
            elif name == "distance_matrix":
                r = self.gdm.run(args["origin"], args["destination"],
                                 args.get("mode", "self-driving"))
            elif name == "city_search":
                r = self.cities.run(args["state"])
            else:
                return json.dumps({"blad": f"unknown tool {name}"})
        except Exception as e:
            return json.dumps({"blad": f"{type(e).__name__}: {str(e)[:200]}"})
        text = _df_to_text(r)
        if text.startswith("There is no") or "no valid information" in text:
            return json.dumps({"blad": text[:200]})
        return text[:6000]


TOOLS = [
    ("flight_search", "Find flights between two cities on one date (YYYY-MM-DD).",
     {"origin": "string", "destination": "string", "date": "string"}),
    ("attraction_search", "List attractions in a city.", {"city": "string"}),
    ("accommodation_search", "List accommodations in a city with prices/room types/rules.",
     {"city": "string"}),
    ("restaurant_search", "List restaurants in a city with average cost and cuisines.",
     {"city": "string"}),
    ("distance_matrix", "Self-driving or taxi duration/distance/cost between two cities.",
     {"origin": "string", "destination": "string", "mode": "self-driving|taxi"}),
    ("city_search", "List cities in a US state.", {"state": "string"}),
]


def result_is_error(text):
    try:
        d = json.loads(text)
        return isinstance(d, dict) and "blad" in d
    except Exception:
        return False


def append_note(text, note):
    """Ride the nudge/note as ordinary result text. JSON results get the
    note under a 'sentinel' key (the consumer is an LLM, not a strict
    parser); text results get it appended below a divider."""
    if not note:
        return text
    try:
        d = json.loads(text)
        if isinstance(d, dict):
            d["sentinel"] = note.strip()
            return json.dumps(d, ensure_ascii=False)
    except Exception:
        pass
    return text + note


def main():
    note_file = os.environ.get("SENTINEL_NOTE_FILE", "/tmp/odr_sentinel_notes.jsonl")
    state_file = os.environ.get("SENTINEL_STATE_FILE", "/tmp/odr_sentinel_state.json")
    call_log = os.environ.get("ODR_CALL_LOG", "")
    sentinel_on = os.environ.get("SENTINEL_OFF") != "1"
    # finalize nudge (see open_dream_rsi.sentinel): rides the NEXT sandbox
    # result at ~finalize_at of the episode's tool-call budget, clean or
    # failing. budget=0 -> disabled (v1 behaviour: error observation only).
    engine = SentinelEngine(
        state_path=Path(state_file),
        finalize_budget=int(os.environ.get("SENTINEL_FINALIZE_BUDGET", "0")),
        finalize_at=float(os.environ.get("SENTINEL_FINALIZE_AT", "0.8")),
        max_notes_per_session=int(os.environ.get("SENTINEL_MAX_NOTES", "8")),
    ) if sentinel_on else None
    # permute-gate: past the trigger the sandbox REFUSES to serve data —
    # option removal, not persuasion. Three measured failures (pull tool,
    # Stop-hook note, budget nudge) showed no text in any context channel
    # makes this model stop exploring; only the absence of the option can.
    gate_budget = int(os.environ.get("SENTINEL_GATE_BUDGET", "0"))
    gate_at = float(os.environ.get("SENTINEL_GATE_AT", "0.8"))
    gate_trigger = int(-(-gate_budget * gate_at // 1)) if gate_budget else 0
    served: dict = {}   # sandbox attempts seen per session (incl. gated)
    sb = Sandbox() if os.environ.get("TP_SERVE") == "1" else None

    def emit(o):
        sys.stdout.write(json.dumps(o, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    ok = lambda i, r: emit({"jsonrpc": "2.0", "id": i, "result": r})
    err = lambda i, c, m: emit({"jsonrpc": "2.0", "id": i,
                                "error": {"code": c, "message": m}})

    tools_list = [
        {"name": n, "description": d,
         "inputSchema": {"type": "object", "properties":
                         {k: {"type": v.split("|")[0]} for k, v in props.items()},
                         "required": list(props)}}
        for n, d, props in TOOLS] if sb else []
    if sentinel_on and os.environ.get("SENTINEL_TOOL") != "0":
        tools_list.append({
            "name": "sentinel_check",
            "description": ("Ask the ODR sentinel whether the current tool/error "
                            "belongs to a known recurring error class. Call it "
                            "BEFORE retrying a call that only differs by a parameter."),
            "inputSchema": {"type": "object", "properties": {
                "tool": {"type": "string"},
                "error_excerpt": {"type": "string"}},
                "required": ["tool"]}})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        method, req_id = req.get("method"), req.get("id")
        if method == "initialize":
            ok(req_id, {"protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}}, "serverInfo": SERVER_INFO})
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
                                          "SENTINEL_SESSION", "tp-default"),
                                      status="error")
                ok(req_id, {"content": [{"type": "text",
                                         "text": note or "no known error class"}],
                            "isError": False})
            elif sb is not None and any(n == name for n, _, _ in TOOLS):
                sid = os.environ.get("SENTINEL_SESSION", "tp-default")
                if gate_trigger:
                    served[sid] = served.get(sid, 0) + 1
                    if served[sid] >= gate_trigger:
                        text = json.dumps({"blad": (
                            "budget_exhausted: the sandbox will not serve "
                            "further searches (%d of %d calls used). Write "
                            "your final JSON answer NOW from the data "
                            "already in this conversation; every missing "
                            "field gets the best value you have."
                            % (served[sid], gate_budget))})
                        if call_log:
                            with open(call_log, "a", encoding="utf-8") as f:
                                f.write(json.dumps({"tool": name,
                                                    "ok": False}) + "\n")
                        ok(req_id, {"content": [{"type": "text",
                                                 "text": text}],
                                    "isError": True})
                        continue
                text = sb.call(name, args)
                is_err = result_is_error(text)
                if call_log:
                    with open(call_log, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"tool": name,
                                            "ok": not is_err}) + "\n")
                note = None
                if engine is not None:
                    # feeds the budget counter on EVERY result (clean too);
                    # returns the finalize nudge, the recurrence note, or both
                    note = engine.observe(name, text, session_id=os.environ.get(
                        "SENTINEL_SESSION", "tp-default"))
                if note:
                    # split the merged channels by marker: the nudge rides the
                    # result as ordinary text (a blocked Stop burns the turn
                    # it tries to save); the recurrence note keeps the v1
                    # Stop-hook delivery path so v1 results stay comparable.
                    same = note.find("[Sentinel] Same error class")
                    if same >= 0:
                        with open(note_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"ts": __import__("time").time(),
                                                "note": note[same:].strip()},
                                               ensure_ascii=False) + "\n")
                        nudge = note[:same]
                    else:
                        nudge = note
                    text = append_note(text, nudge)
                ok(req_id, {"content": [{"type": "text", "text": text}],
                            "isError": is_err})
            else:
                err(req_id, -32602, f"unknown tool: {name}")
        elif method == "shutdown":
            ok(req_id, {})
        elif req_id is not None:
            err(req_id, -32601, f"method not found: {method}")


if __name__ == "__main__":
    main()
