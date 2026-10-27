"""Success metrics for a Dream-RSI deployment.

`odr metrics` (and the MCP ``odr_metrics`` tool) answers the recurring
operator question "is the loop actually working?" straight from the memory
dir — events.jsonl, task_outcomes.json, lessons.json, recipes.json — with
no jq gymnastics:

* cycle health: how many cycles ran, API-call spend, budget-exhaustion rate
* solve rates: attempted/solved per category (window + all-time)
* error and rejection classes: llm_error causes, lesson_rejected reasons
* knowledge state: staging vs active lessons, curation throughput, recipes
* backlog and staleness: queued probes, hours since the last full dream
* trigger budget: the effective --budget of the most recent fire

Everything is derived from files already on disk; nothing is called,
nothing is guessed. Empty inputs degrade to zeros, never exceptions.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Reason buckets for lesson_rejected events (first-match wins).
_REJECT_CLASSES = (
    ("empty", "empty payload (curator judged: no lesson in evidence)"),
    ("timed out", "llm timed out (generation too long for the call timeout)"),
    ("llm error: timed out", "llm timed out (generation too long for the call timeout)"),
    ("llm error", "llm error (endpoint/transport)"),
    ("lesson validation failed", "validation failed (payload shape)"),
    ("no failure evidence", "no failure evidence"),
)


def _classify(text: str, classes) -> str:
    low = (text or "").lower()
    for needle, label in classes:
        if needle in low:
            return label
    return "other"


def collect(memory_root: "str | Path", hours: Optional[float] = None) -> Dict[str, Any]:
    """Gather success metrics from a memory dir.

    ``hours`` limits the events-derived sections to the last N hours
    (``None`` = all recorded history). Store files (lessons, recipes,
    outcomes) are always reported as they are now.
    """
    root = Path(memory_root)
    cutoff = (time.time() - hours * 3600.0) if hours else None

    cycles: List[Dict[str, Any]] = []
    task_attempted: Counter = Counter()
    task_solved: Counter = Counter()
    llm_errors: Counter = Counter()
    rejects: Counter = Counter()
    gate_verdicts = 0
    lessons_added = 0
    lessons_merged = 0
    corrections = 0
    fires = 0
    last_budget: Optional[int] = None
    tokens_prompt = tokens_completion = tokens_cycles = 0

    events = root / "events.jsonl"
    if events.exists():
        with events.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                ts = float(e.get("ts", 0) or 0)
                if cutoff is not None and ts < cutoff:
                    continue
                kind = e.get("kind")
                if kind == "cycle":
                    cycles.append(e)
                    t = e.get("tokens")
                    if isinstance(t, dict) and t.get("total_tokens"):
                        tokens_prompt += int(t.get("prompt_tokens", 0) or 0)
                        tokens_completion += int(t.get("completion_tokens", 0) or 0)
                        tokens_cycles += 1
                elif kind == "task":
                    cat = e.get("category") or "?"
                    task_attempted[cat] += 1
                    if e.get("solved"):
                        task_solved[cat] += 1
                elif kind == "llm_error":
                    llm_errors[_classify(str(e.get("error", "")), _REJECT_CLASSES)] += 1
                elif kind == "lesson_rejected":
                    rejects[_classify(str(e.get("reason", "")), _REJECT_CLASSES)] += 1
                elif kind == "lesson_gate":
                    gate_verdicts += 1
                elif kind == "lessons_curated":
                    lessons_added += int(e.get("added", 0) or 0)
                    lessons_merged += int(e.get("merged", 0) or 0)
                elif kind == "user_correction_found":
                    corrections += int(e.get("count", 1) or 1)

    log = root / "trigger.log"
    if log.exists():
        with log.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if " fire:" not in line:
                    continue
                raw_ts = line.split(" ", 1)[0]
                try:
                    ts = float(raw_ts)
                except ValueError:
                    ts = 0.0
                if cutoff is not None and ts < cutoff:
                    continue
                fires += 1
                if "--budget" in line:
                    try:
                        last_budget = int(line.rsplit("--budget", 1)[1].split()[0])
                    except (ValueError, IndexError):
                        pass

    outcomes: Dict[str, Dict[str, int]] = {}
    outcomes_path = root / "task_outcomes.json"
    if outcomes_path.exists():
        try:
            raw = json.loads(outcomes_path.read_text(encoding="utf-8"))
        except ValueError:
            raw = {}
        for cat, records in raw.items():
            if isinstance(records, list):
                outcomes[cat] = {
                    "attempted": len(records),
                    "solved": sum(1 for r in records if isinstance(r, dict) and r.get("solved")),
                }

    lessons: Dict[str, Dict[str, int]] = {}
    lesson_usage: List[Dict[str, Any]] = []
    lessons_path = root / "lessons.json"
    if lessons_path.exists():
        try:
            raw = json.loads(lessons_path.read_text(encoding="utf-8"))
        except ValueError:
            raw = {}
        for cat, items in raw.items():
            if isinstance(items, list):
                st = Counter(str(i.get("status", "?")) for i in items if isinstance(i, dict))
                lessons[cat] = {"active": st.get("active", 0), "staging": st.get("staging", 0),
                                "total": len(items)}
                for i in items:
                    if isinstance(i, dict):
                        lesson_usage.append({
                            "category": cat,
                            "trigger": str(i.get("trigger", "")),
                            "uses": int(i.get("uses", 0) or 0),
                            "wins": int(i.get("wins", 0) or 0),
                            "status": str(i.get("status", "?")),
                        })

    recipes: Dict[str, float] = {}
    recipes_path = root / "recipes.json"
    if recipes_path.exists():
        try:
            raw = json.loads(recipes_path.read_text(encoding="utf-8"))
        except ValueError:
            raw = {}
        recipes = {cat: float(v.get("score", 0.0))
                   for cat, v in raw.items() if isinstance(v, dict)}

    queue_len = 0
    tasks_path = root / "tasks.json"
    if tasks_path.exists():
        try:
            raw = json.loads(tasks_path.read_text(encoding="utf-8"))
            queue_len = len(raw) if isinstance(raw, list) else 0
        except ValueError:
            pass

    state_path = root / "dream_state.json"
    hours_since_full: Optional[float] = None
    if state_path.exists():
        try:
            last_full = float(json.loads(state_path.read_text(encoding="utf-8"))
                              .get("last_full_dream", 0) or 0)
            if last_full:
                hours_since_full = max(0.0, (time.time() - last_full) / 3600.0)
        except (ValueError, OSError):
            pass

    api_calls = sum(int(c.get("api_calls", 0) or 0) for c in cycles)
    budget_exhausted = sum(1 for c in cycles
                           if c.get("stopped_reason") == "api_budget_exhausted")
    return {
        "window_hours": hours,
        "cycles": {
            "total": len(cycles),
            "api_calls": api_calls,
            "budget_exhausted": budget_exhausted,
            "solved": sum(int(c.get("tasks_solved", 0) or 0) for c in cycles),
            "attempted": sum(int(c.get("tasks_attempted", 0) or 0) for c in cycles),
            "stopped_reasons": dict(Counter(str(c.get("stopped_reason")) for c in cycles)),
        },
        "trigger": {"fires": fires, "last_budget": last_budget},
        "tasks": {cat: {"attempted": task_attempted[cat], "solved": task_solved[cat]}
                  for cat in sorted(set(task_attempted) | set(task_solved))},
        "llm_errors": dict(llm_errors),
        "lesson_rejections": dict(rejects),
        "curation": {"gate_verdicts": gate_verdicts,
                     "lessons_added": lessons_added,
                     "lessons_merged": lessons_merged,
                     "user_corrections": corrections},
        "tokens": {"prompt": tokens_prompt, "completion": tokens_completion,
                   "cycles_with_usage": tokens_cycles},
        "lessons": lessons,
        "lesson_usage": lesson_usage,
        "recipes": recipes,
        "outcomes_all_time": outcomes,
        "backlog": {"queued_tasks": queue_len,
                    "hours_since_full_dream": hours_since_full},
    }


def format_metrics(data: Dict[str, Any]) -> str:
    """Render :func:`collect` output as a compact operator report (ASCII)."""
    w = data.get("window_hours")
    head = f"ODR metrics — memory window: {'last ' + str(w) + 'h' if w else 'all history'}"
    out = [head, "=" * len(head)]

    cyc = data["cycles"]
    out.append(f"\ncycles: {cyc['total']}  api_calls: {cyc['api_calls']}  "
               f"budget_exhausted: {cyc['budget_exhausted']}/{cyc['total']}")
    tok = data.get("tokens") or {}
    if tok.get("cycles_with_usage"):
        out.append(f"  tokens: prompt {tok['prompt']:,}  completion {tok['completion']:,}  "
                   f"total {tok['prompt'] + tok['completion']:,} "
                   f"({tok['cycles_with_usage']} instrumented cycles)")
    else:
        out.append("  tokens: not instrumented (needs v0.3.13+ endpoints "
                   "reporting OpenAI usage blocks)")
    if cyc["attempted"]:
        out.append(f"  tasks attempted {cyc['attempted']}, solved {cyc['solved']} "
                   f"({100.0 * cyc['solved'] / max(1, cyc['attempted']):.1f}%)")
    trig = data["trigger"]
    if trig["fires"] or trig["last_budget"]:
        out.append(f"trigger fires: {trig['fires']}  last budget: {trig['last_budget']}")

    out.append("\nsolve rate (events, this window):")
    tasks = data["tasks"]
    if tasks:
        for cat, s in tasks.items():
            rate = 100.0 * s["solved"] / max(1, s["attempted"])
            flag = "  <-- no solves" if s["attempted"] >= 5 and s["solved"] == 0 else ""
            out.append(f"  {cat}: {s['solved']}/{s['attempted']} ({rate:.1f}%){flag}")
    else:
        out.append("  (no task events in window)")

    if data["llm_errors"]:
        out.append("\nllm errors:")
        for cause, n in sorted(data["llm_errors"].items(), key=lambda x: -x[1]):
            out.append(f"  {n:5d}  {cause}")
    if data["lesson_rejections"]:
        out.append("\nlesson rejections:")
        for cause, n in sorted(data["lesson_rejections"].items(), key=lambda x: -x[1]):
            out.append(f"  {n:5d}  {cause}")

    cur = data["curation"]
    out.append(f"\ncuration: gate verdicts {cur['gate_verdicts']}, "
               f"lessons added {cur['lessons_added']}, merged {cur['lessons_merged']}, "
               f"user corrections {cur['user_corrections']}")

    lessons = data["lessons"]
    out.append("\nlessons (active/staging):")
    if lessons:
        for cat, s in sorted(lessons.items()):
            out.append(f"  {cat}: {s['active']}/{s['staging']}")
    else:
        out.append("  (knowledge base empty)")
    usage = data.get("lesson_usage") or []
    if usage:
        out.append("\nlesson usage (uses -> wins, credit rate; uses/wins come from "
                   "record_lesson_usage: surfaced into proposals / task solved):")
        def _dead(u):
            return u["wins"] == 0 and u["uses"] >= 4  # mirrors curator.PRUNE_MIN_USES
        ranked = sorted(usage, key=lambda u: (0 if _dead(u) else 1,
                                              -u["uses"], -u["wins"], u["trigger"]))
        shown = ranked[:12]
        for u in shown:
            rate = f"{100.0 * u['wins'] / u['uses']:.0f}%" if u["uses"] else "-"
            flag = "  <-- dead (>=4 uses, 0 wins)" if _dead(u) else ""
            out.append(f"  {u['category']} | {u['trigger']}: {u['uses']} -> "
                       f"{u['wins']} ({rate}){flag}")
        if len(ranked) > len(shown):
            out.append(f"  ...{len(ranked) - len(shown)} more")
    recipes = data["recipes"]
    if recipes:
        out.append("recipes: " + ", ".join(f"{c}={s:g}" for c, s in sorted(recipes.items())))

    outcomes = data["outcomes_all_time"]
    zero = [c for c, s in outcomes.items()
            if s["attempted"] >= 10 and s["solved"] == 0]
    if outcomes:
        out.append("\nall-time outcomes: " + ", ".join(
            f"{c} {s['solved']}/{s['attempted']}" for c, s in sorted(outcomes.items())))
    if zero:
        out.append("  headroom flags (>=10 attempts, 0 solves): " + ", ".join(sorted(zero)))

    backlog = data["backlog"]
    out.append(f"\nbacklog: {backlog['queued_tasks']} queued task(s)")
    if backlog["hours_since_full_dream"] is not None:
        out.append(f"last full dream: {backlog['hours_since_full_dream']:.1f}h ago")
    return "\n".join(out)
