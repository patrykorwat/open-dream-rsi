"""MCP (Model Context Protocol) stdio server — plug Open Dream-RSI into any harness.

Exposes the self-improvement loop as tools that OpenCode, Goose, Claude Code,
Cursor or any other MCP-speaking harness can call over stdio JSON-RPC 2.0
(newline-delimited, per the 2025 MCP spec). Zero external dependencies, as
with the rest of the package.

Run manually (an MCP client normally spawns it for you)::

    python -m open_dream_rsi.mcp [--memory .dream_rsi] [--tasks tasks.json]

Tools exposed:

* ``odr_status``     — what the loop has learned (policies, recipes, lessons).
* ``odr_recipes``    — best verified solution for a category (warm start).
* ``odr_lessons``    — curated failure lessons for a category or search query.
* ``odr_add_task``   — queue a new task so the dreamer starts working on it.
* ``odr_run_once``   — run a single improvement cycle now (bounded by budget).

Environment (all optional): ``ODR_MEMORY`` (memory dir), ``ODR_TASKS`` (task
file), plus the usual ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` /
``ODR_LLM_MODEL`` / ``ODR_LLM_PRESET`` handled by :mod:`open_dream_rsi.llm`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "open-dream-rsi"

# ---------------------------------------------------------------------------
# Tool implementations (pure functions over memory; no arbitrary exec)
# ---------------------------------------------------------------------------


def _memory_root(overrides: Dict[str, Any]) -> Path:
    return Path(overrides.get("memory") or os.environ.get("ODR_MEMORY", ".dream_rsi"))


def _tasks_path(overrides: Dict[str, Any]) -> Path:
    return Path(overrides.get("tasks_file") or os.environ.get("ODR_TASKS", "tasks.json"))


def tool_odr_status(args: Dict[str, Any]) -> Dict[str, Any]:
    """Summarise learned policies, recipes, lessons and recent events."""
    from open_dream_rsi.memory import DreamMemory

    memory = DreamMemory(_memory_root(args))
    policies = memory._load(memory.root / "policies.json", {})
    codes = memory._load(memory.root / "policy_codes.json", {})
    recipes = memory._load(memory.root / "recipes.json", {})
    events_path = memory.root / "events.jsonl"
    events: List[str] = []
    if events_path.exists():
        events = events_path.read_text(encoding="utf-8").splitlines()[-5:]
    return {
        "memory_root": str(memory.root.resolve()),
        "categories": sorted(set(policies) | set(recipes) | set(codes)),
        "policies": {k: v.get("params") for k, v in policies.items()},
        "promoted_policy_codes": {k: {"score": v.get("score"), "steps": v.get("steps")}
                                  for k, v in codes.items()},
        "recipes": {k: {"score": v.get("score")} for k, v in recipes.items()},
        "recent_events": [e[:200] for e in events],
    }


def tool_odr_recipes(args: Dict[str, Any]) -> Dict[str, Any]:
    """Return the best verified solution (recipe) for a task category."""
    from open_dream_rsi.memory import DreamMemory

    category = args.get("category", "")
    if not category:
        raise ValueError("'category' is required")
    memory = DreamMemory(_memory_root(args))
    recipe = memory.get_recipe(category)
    if recipe is None:
        return {"category": category, "found": False}
    return {"category": category, "found": True, "code": recipe}


def tool_odr_lessons(args: Dict[str, Any]) -> Dict[str, Any]:
    """Curated failure lessons for a category; optional text search query."""
    from open_dream_rsi.core.curator import format_lessons, select_lessons
    from open_dream_rsi.memory import DreamMemory

    category = args.get("category", "")
    query = args.get("query", "")
    memory = DreamMemory(_memory_root(args))
    if category:
        lessons = memory.get_lessons(category)
    else:  # search across all categories
        all_lessons = memory._load(memory.root / "lessons.json", {})
        lessons = [dict(l) for entries in all_lessons.values() for l in entries]
    selected = select_lessons(lessons, query or category)
    return {"count": len(selected), "lessons": format_lessons(selected) or "(none)"}


def tool_odr_add_task(args: Dict[str, Any]) -> Dict[str, Any]:
    """Append a task to the task file so the next loop cycle picks it up."""
    task = {
        "task_id": args.get("task_id"),
        "category": args.get("category"),
        "prompt": args.get("prompt"),
        "tests": args.get("tests"),
        "max_attempts": int(args.get("max_attempts", 4)),
    }
    for field_name in ("task_id", "category", "prompt", "tests"):
        if not task[field_name]:
            raise ValueError(f"'{field_name}' is required")
    if not isinstance(task["tests"], list) or not task["tests"]:
        raise ValueError("'tests' must be a non-empty list of {call, expected}")

    path = _tasks_path(args)
    existing: List[Dict[str, Any]] = []
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    if any(t.get("task_id") == task["task_id"] for t in existing):
        return {"added": False, "reason": f"task_id '{task['task_id']}' already queued"}
    existing.append(task)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return {"added": True, "task_id": task["task_id"], "tasks_total": len(existing),
            "tasks_file": str(path)}


def tool_odr_run_once(args: Dict[str, Any]) -> Dict[str, Any]:
    """Run one full improvement cycle (online attempts + offline dreaming)."""
    from open_dream_rsi.cli import _build_client
    from open_dream_rsi.loop import AutoRSIRuntime, Task
    from open_dream_rsi.memory import DreamMemory

    path = _tasks_path(args)
    if not path.exists():
        raise ValueError(f"task file not found: {path} (use odr_add_task first)")
    tasks = [Task(**t) for t in json.loads(path.read_text(encoding="utf-8"))]
    provider = args.get("provider") or os.environ.get("ODR_LLM_PRESET", "openai")
    if provider == "mock":  # key-free smoke test of the whole pipeline
        from open_dream_rsi.llm import StubClient

        client = StubClient()
    else:
        client = _build_client(provider, args.get("model"))
    runtime = AutoRSIRuntime(
        client=client,
        memory=DreamMemory(_memory_root(args)),
        tasks=tasks,
        api_call_budget=int(args.get("budget", 10)),
        # Thought-conditioned branching ships ON (library default): attempts
        # record their PLAN line, proposals see the tried-idea ledger, and
        # expansion leaves dead idea families. Harnesses can opt out.
        enable_thoughts=bool(args.get("thoughts", True)),
    )
    return runtime.run_once().to_dict()


# ---------------------------------------------------------------------------
# Tool registry + JSON-RPC plumbing
# ---------------------------------------------------------------------------

TOOLS: Dict[str, Dict[str, Any]] = {
    "odr_status": {
        "description": "Summarise what the Dream-RSI loop has learned: dreamed "
                       "policies, promoted policy programs, recipe scores, recent events.",
        "inputSchema": {"type": "object", "properties": {
            "memory": {"type": "string", "description": "Memory dir (default $ODR_MEMORY or .dream_rsi)"}}},
        "fn": tool_odr_status,
    },
    "odr_recipes": {
        "description": "Fetch the loop's best verified solution (warm-start recipe) "
                       "for a task category — reuse it instead of solving from scratch.",
        "inputSchema": {"type": "object", "properties": {
            "category": {"type": "string"},
            "memory": {"type": "string"}}, "required": ["category"]},
        "fn": tool_odr_recipes,
    },
    "odr_lessons": {
        "description": "Curated failure lessons distilled by the knowledge curator "
                       "for a category (or searched by query across all categories).",
        "inputSchema": {"type": "object", "properties": {
            "category": {"type": "string"}, "query": {"type": "string"},
            "memory": {"type": "string"}}},
        "fn": tool_odr_lessons,
    },
    "odr_add_task": {
        "description": "Queue a self-contained Python task (prompt + test cases) for the "
                       "self-improvement loop. tests = [{'call': 'add(2, 3)', 'expected': 5}].",
        "inputSchema": {"type": "object", "properties": {
            "task_id": {"type": "string"}, "category": {"type": "string"},
            "prompt": {"type": "string"},
            "tests": {"type": "array", "items": {"type": "object"}},
            "max_attempts": {"type": "integer"},
            "tasks_file": {"type": "string"}},
            "required": ["task_id", "category", "prompt", "tests"]},
        "fn": tool_odr_add_task,
    },
    "odr_run_once": {
        "description": "Run one Dream-RSI cycle over the queued tasks now (online LLM "
                       "attempts + offline dreaming, thought-conditioned branching "
                       "on by default). Returns a solve/budget report.",
        "inputSchema": {"type": "object", "properties": {
            "tasks_file": {"type": "string"}, "memory": {"type": "string"},
            "provider": {"type": "string", "enum": ["openai", "cursor", "local", "mock"]},
            "model": {"type": "string"},
            "thoughts": {"type": "boolean",
                         "description": "thought-conditioned branching (default true)"},
            "budget": {"type": "integer", "description": "Max API calls (default 10)"}}},
        "fn": tool_odr_run_once,
    },
}


def _tool_specs() -> List[Dict[str, Any]]:
    return [{"name": name, "description": spec["description"],
             "inputSchema": spec["inputSchema"]} for name, spec in TOOLS.items()]


def _result(ok: bool, payload: Any) -> Dict[str, Any]:
    text = json.dumps(payload, indent=2, ensure_ascii=False) if not isinstance(payload, str) else payload
    return {"content": [{"type": "text", "text": text}], "isError": not ok}


def handle_request(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One JSON-RPC request -> response dict (or None for notifications)."""
    method = request.get("method", "")
    req_id = request.get("id")

    if method == "initialize":
        return {"jsonrpc": "2.0", "id": req_id, "result": {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": _server_version()},
        }}
    if method in ("notifications/initialized", "initialized"):
        return None  # notification — no response
    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}
    if req_id is None:
        return None  # unknown notification

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": _tool_specs()}}
    if method == "tools/call":
        params = request.get("params", {})
        name = params.get("name", "")
        spec = TOOLS.get(name)
        if spec is None:
            return {"jsonrpc": "2.0", "id": req_id, "result": _result(
                True, {"error": f"unknown tool '{name}'"}), "error": {
                "code": -32602, "message": f"unknown tool '{name}'"}}
        try:
            payload = spec["fn"](params.get("arguments") or {})
            return {"jsonrpc": "2.0", "id": req_id, "result": _result(True, payload)}
        except Exception as exc:  # tool errors are reported as isError results
            return {"jsonrpc": "2.0", "id": req_id,
                    "result": _result(False, {"error": f"{type(exc).__name__}: {exc}"})}

    return {"jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32601, "message": f"method not found: {method}"}}


def _server_version() -> str:
    from open_dream_rsi import __version__

    return __version__


def serve(stdin=None, stdout=None) -> None:
    """Newline-delimited JSON-RPC loop over stdin/stdout (MCP stdio transport).

    Uses ``readline()`` rather than ``for line in stdin``: file iteration
    read-ahead-buffers and would delay the first response until the buffer
    fills or stdin closes (clients time out during the handshake).
    """
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    while True:
        line = stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = handle_request(request)
        if response is not None:
            stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            stdout.flush()


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="odr-mcp", description="Open Dream-RSI MCP stdio server")
    parser.add_argument("--memory", default=None, help="memory dir (default $ODR_MEMORY)")
    parser.add_argument("--tasks", default=None, help="task file (default $ODR_TASKS or tasks.json)")
    args = parser.parse_args(argv)
    if args.memory:
        os.environ["ODR_MEMORY"] = args.memory
    if args.tasks:
        os.environ["ODR_TASKS"] = args.tasks
    # Protocol stays on the real stdin/stdout; anything the library prints is
    # diverted to stderr so it cannot pollute the JSON-RPC channel.
    protocol_out = sys.stdout
    sys.stdout = sys.stderr
    try:
        serve(stdin=sys.stdin, stdout=protocol_out)
    finally:
        sys.stdout = protocol_out
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
