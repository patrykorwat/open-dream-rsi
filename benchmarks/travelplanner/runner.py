#!/usr/bin/env python3
"""goose x TravelPlanner runner — the benchmark episode loop.

Public, deterministic, offline-scored: the official osunlp/TravelPlanner
task CSVs (committed under splits/) + official commonsense/hard
evaluators (score.py). Tools served to goose via sandbox_server.py against
the official offline database (see README.md for prerequisites).

Arms (sandbox behaviour is the ONLY variable; system prompt, call cap,
model and thinking-off proxy are shared):
  cold       sandbox MCP only, sentinel off
  osi_gate   permute-gate: past 80% of the cap the sandbox REFUSES to
             serve (option removal); all persuasion channels OFF
  osi_nudge  sandbox serves to the cap; a finalize nudge rides the next
             tool result at ~80% of the budget
  osi_full   nudge + sentinel_check pull tool + Stop-hook notes
             (the transform_tool_result analogue)

Runs cases task-interleaved with fixed arm order so per-arm vs cold is
paired (McNemar-ready). Fresh goose + fresh MCP process + fresh sentinel
state per episode. Endpoint-dead -> RuntimeError abort (invalid arms never
aggregate).

Usage:
  python3 runner.py --csv splits/validation_holdout.csv --arms cold,osi_gate
  python3 runner.py --csv splits/validation_holdout.csv --assemble

Results append to results/tp_results_<split>.jsonl; then assemble
submissions and score with the official evaluators (README.md).
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GOOSE = os.environ.get("ODR_GOOSE", "goose")
GOOSE_HOME = Path(os.environ.get("ODR_GOOSE_HOME", "/tmp/ghome_tp_bench"))
TP_PY = os.environ.get("TP_PY", str(HERE / ".venv-tp/bin/python"))
TP_SERVER = str(HERE / "sandbox_server.py")
STOP_HOOK = str(REPO / "plugins/goose/hooks/stop_hook.py")
MAX_TURNS = int(os.environ.get("TP_MAX_TURNS", "45"))
EPISODE_TIMEOUT = int(os.environ.get("TP_EPISODE_TIMEOUT", "600"))

SYSTEM = (
    "You are a travel-planning agent. Build a complete itinerary answering "
    "the user's query using ONLY data returned by the provided MCP tools "
    "(flight_search, attraction_search, accommodation_search, "
    "restaurant_search, distance_matrix, city_search). Never invent flight "
    "numbers, prices, names or times: every value must be copied verbatim "
    "from a tool result.\n"
    "Rules of the sandbox:\n"
    "- The trip is a closed circle: outbound flight/drive on Day 1 from the "
    "origin city, return leg on the last day back to the origin. Visit "
    "exactly the number of cities asked (transit cities cannot be visited).\n"
    "- Each non-final day needs an accommodation booked in that night's "
    "city; the last day's accommodation is '-'. Respect minimum_nights and "
    "maximum occupancy (book multiple rooms if needed).\n"
    "- Meals must be real restaurants of the current city (breakfast/lunch/"
    "dinner must be DIFFERENT restaurants each day; no restaurant repeated "
    "within a day). Attractions must be real attractions of the current "
    "city and must not repeat across the trip.\n"
    "- Intra-city travel is Taxi or Self-driving using distance_matrix "
    "costs; inter-city legs use a real flight (flight_search) when asked "
    "for flights. Keep the whole plan within the stated budget.\n"
    "Before retrying a tool call that only differs by a parameter, call "
    "sentinel_check (if available) to avoid a known recurring error class.\n"
    "Budget discipline (hard limit ~45 tool calls before you are cut off "
    "with NO answer and score 0): plan first, call tools second. For each "
    "city you need at most ONE flight_search per leg, ONE attraction_search, "
    "ONE accommodation_search, ONE restaurant_search; distance_matrix only "
    "if the query requires self-driving/taxi. Draft your complete JSON "
    "answer internally by call ~20; if a needed item is missing by call 30, "
    "fill it with your best available option and answer anyway. A complete "
    "answer with imperfect choices scores far above being cut off. "
    "When done, answer with EXACTLY one JSON array (no prose after it), one "
    "object per day, keys: day, current_city, transportation, breakfast, "
    "attraction, lunch, dinner, accommodation. Format each day like:\n"
    '{"day":1,"current_city":"from A to B",'
    '"transportation":"Flight Number: F1234567, from A to B, Departure '
    'Time: 08:00, Arrival Time: 10:00","breakfast":"Name, B",'
    '"attraction":"Name, B","lunch":"Name, B",'
    '"dinner":"Name, B","accommodation":"Name, B"}\n'
    "Day 1 transport: 'Flight Number: ..., from A to B, Departure Time: "
    "HH:MM, Arrival Time: HH:MM'. Ground days: transportation '-'. "
    "Multi attractions separated by ';', last one also ends with ';'. "
    "Every meal/attraction/accommodation entry is exactly 'Name, City' — "
    "NO costs, NO dollar signs, NO extra text after the city. "
    "On the return day transportation is the flight back and accommodation "
    "is '-'."
)


def base_env():
    """goose environment: the ONLY model wiring is provider/model/host.

    GOOSE_MODEL and ODR_UPSTREAM must match the endpoint under test
    (README.md); the thinking_proxy in front of it forces
    enable_thinking:false so goose and any other agent under test share
    the exact same model configuration."""
    return {
        **os.environ,
        "HOME": str(GOOSE_HOME),
        "GOOSE_PROVIDER": os.environ.get("TP_PROVIDER", "openai"),
        "GOOSE_MODEL": os.environ.get(
            "TP_MODEL", "local-inference-lab/Qwen3.8-Flash-Next-NVFP4"),
        "OPENAI_HOST": os.environ.get("TP_PROXY_HOST", "http://127.0.0.1:8801"),
        "OPENAI_BASE_PATH": "/v1/chat/completions",
        "OPENAI_API_KEY": "bench",
        "GOOSE_DIR": str(GOOSE_HOME / ".goose"),
        "GOOSE_DISABLE_UPDATE_CHECKS": "1",
    }


def load_cases(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows, start=1):
        r["idx"] = i
    return rows


def task_text(row):
    return (row["query"].strip() +
            "\nDates: " + str(eval(row["date"]) if row["date"].startswith("[")
                              else eval(row["date"])))


def extract_plan(out: str):
    dec = json.JSONDecoder()
    best = None
    for i, ch in enumerate(out):
        if ch != "[":
            continue
        try:
            cand, _ = dec.raw_decode(out, i)
        except json.JSONDecodeError:
            continue
        if (isinstance(cand, list) and cand and isinstance(cand[0], dict)
                and "day" in cand[0] or (isinstance(cand, list) and cand
                                         and "days" in cand[0])):
            # normalise 'days' key per official format
            for unit in cand:
                if "days" not in unit and "day" in unit:
                    unit["days"] = unit["day"]
                unit.setdefault("current_city", "-")
                for k in ("transportation", "breakfast", "attraction",
                          "lunch", "dinner", "accommodation"):
                    unit.setdefault(k, "-")
            best = cand  # keep scanning; last valid wins
    return best


def _count_lines(p: Path) -> int:
    try:
        return sum(1 for ln in p.read_text().splitlines() if ln.strip())
    except FileNotFoundError:
        return 0


def run_episode(row, arm, tmp):
    idx = row["idx"]
    ep_id = f"tp{idx}_{arm}"
    note_file = tmp / f"notes_{ep_id}.jsonl"
    state_file = tmp / f"state_{ep_id}.json"
    call_log = tmp / f"calls_{ep_id}.jsonl"
    for p in (note_file, state_file, call_log):
        p.unlink(missing_ok=True)

    env = base_env()
    # goose's default extension timeout is shorter than a cold TP-DB load
    # (pandas parses ~90 MB of CSVs at MCP-server startup; under swap
    # thrash it exceeded the default and goose silently ran WITHOUT the
    # extension — goose answered in prose, llm_calls<=4, zero tool calls).
    env["GOOSE_DEFAULT_EXTENSION_TIMEOUT"] = "180"
    mcp_env = {
        "TP_SERVE": "1",
        "SENTINEL_SESSION": ep_id,
        "SENTINEL_STATE_FILE": str(state_file),
        "SENTINEL_NOTE_FILE": str(note_file),
        "ODR_CALL_LOG": str(call_log),
        "PYTHONPATH": str(REPO),
    }
    if arm == "cold":
        mcp_env["SENTINEL_OFF"] = "1"
    if arm == "osi_nudge":
        # finalize nudge rides the tool result at ~80% of the harness call
        # cap (goose cuts at ~45); recurrence notes keep the Stop-hook path
        # so cold/nudge/full stay comparable episode-by-episode.
        mcp_env["SENTINEL_FINALIZE_BUDGET"] = str(MAX_TURNS)
    if arm == "osi_gate":
        # permute-gate: past 80% of the cap the sandbox REFUSES to serve
        # (option removal). Persuasion fully off: engine silenced, no Stop
        # plugin, no pull tool — one variable vs cold: the closing gate.
        mcp_env["SENTINEL_OFF"] = "1"
        mcp_env["SENTINEL_GATE_BUDGET"] = str(MAX_TURNS)
    mcp_cmd = (" ".join(f"{k}={v}" for k, v in mcp_env.items())
               + f" {TP_PY} {TP_SERVER}")
    cmd = [GOOSE, "run", "-q", "--no-profile",
           "--max-turns", str(MAX_TURNS),
           "--system", SYSTEM,
           "--with-extension", mcp_cmd,
           "-t", task_text(row)]

    if arm in ("osi_stop", "osi_full", "osi_nudge"):
        # goose loads EVERY user plugin dir: prune stale per-episode Stop
        # plugins from earlier episodes or they all fire on the same note file
        for stale in (GOOSE_HOME / ".agents/plugins").glob("odr_stop_*"):
            if stale.name != f"odr_stop_{ep_id}":
                shutil.rmtree(stale, ignore_errors=True)
        plugin = GOOSE_HOME / f".agents/plugins/odr_stop_{ep_id}"
        (plugin / "hooks").mkdir(parents=True, exist_ok=True)
        (plugin / "hooks/hooks.json").write_text(json.dumps(
            {"hooks": {"Stop": [{"hooks": [{
                "type": "command",
                "command": f"{sys.executable} {STOP_HOOK}",
                "timeout": 10}]}]}}))
        env.update({"SENTINEL_NOTE_FILE": str(note_file)})

    llm_file = Path(os.environ.get("ODR_LLM_CALLS_FILE",
                                   "/tmp/odr_llm_calls.jsonl"))
    before = _count_lines(llm_file)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           timeout=EPISODE_TIMEOUT)
        out = r.stdout or ""
        rc, err = r.returncode, (r.stderr or "")
    except subprocess.TimeoutExpired:
        out, rc, err = "", -1, "timeout"
    except Exception as e:
        raise RuntimeError(f"endpoint/harness dead in {ep_id}: {e}") from e
    if "Ran into this error" in out and "endpoint" in (out + err).lower():
        raise RuntimeError(f"endpoint dead in {ep_id}: {out[:300]}")

    plan = extract_plan(out)
    return {"idx": idx, "level": row["level"], "days": int(row["days"]),
            "arm": arm, "ep": ep_id,
            "plan": plan if plan is not None else [],
            "tool_calls": _count_lines(call_log),
            "llm_calls": _count_lines(llm_file) - before,
            "n_notes": _count_lines(note_file),
            "rc": rc, "elapsed": round(time.time() - t0, 1),
            "out_tail": "" if plan else out[-400:]}


def assemble(results_path, csv_path, out_dir):
    by_arm = {}
    for ln in results_path.read_text().splitlines():
        d = json.loads(ln)
        by_arm.setdefault(d["arm"], {})[d["idx"]] = d["plan"]
    rows = load_cases(csv_path)
    for arm, plans in by_arm.items():
        p = out_dir / f"{csv_path.stem}_{arm}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"idx": r["idx"], "query": r["query"],
                                    "plan": plans.get(r["idx"], [])},
                                   ensure_ascii=False) + "\n")
        print(arm, "->", p, f"({sum(1 for v in plans.values() if v)} non-empty)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="", help="1-based csv idx list")
    ap.add_argument("--arms", default="cold,osi_gate")
    ap.add_argument("--csv", default=str(HERE / "splits/validation_holdout.csv"))
    ap.add_argument("--assemble", action="store_true")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_path = HERE / "results" / f"tp_results_{csv_path.stem}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.assemble:
        assemble(out_path, csv_path, HERE / "results/submissions")
        return

    rows = load_cases(csv_path)
    sel = [int(x) for x in args.cases.split(",")] if args.cases else \
        [r["idx"] for r in rows]
    arms = args.arms.split(",")
    tmp = Path(os.environ.get("TP_TMP", "/tmp/tp_bench"))
    tmp.mkdir(exist_ok=True)
    GOOSE_HOME.mkdir(parents=True, exist_ok=True)
    (GOOSE_HOME / ".agents/plugins").mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        for ln in out_path.read_text().splitlines():
            d = json.loads(ln)
            done.add((d["idx"], d["arm"]))
    with out_path.open("a", encoding="utf-8") as f:
        for idx in sel:                       # task-interleaved, fixed order
            row = rows[idx - 1]
            for arm in arms:
                if (idx, arm) in done:
                    continue
                rec = run_episode(row, arm, tmp)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"idx={idx} {arm}: plan={'Y' if rec['plan'] else 'N'} "
                      f"calls={rec['tool_calls']} llm={rec['llm_calls']} "
                      f"notes={rec['n_notes']} {rec['elapsed']}s", flush=True)
    print("DONE", out_path)


if __name__ == "__main__":
    main()
