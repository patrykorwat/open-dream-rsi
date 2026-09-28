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
    if args.get("criteria"):
        task["criteria"] = str(args["criteria"])
    for field_name in ("task_id", "category", "prompt"):
        if not task[field_name]:
            raise ValueError(f"'{field_name}' is required")
    if not task.get("tests") and not task.get("criteria"):
        raise ValueError("either 'tests' (non-empty list of {call, expected}) "
                         "or 'criteria' (success description for the "
                         "completion judge) is required")
    if task.get("tests") and (not isinstance(task["tests"], list)
                              or not task["tests"]):
        raise ValueError("'tests' must be a non-empty list of {call, expected}")
    task.setdefault("tests", [])

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


def _default_provider() -> str:
    """Zero-configuration provider chain for harness-spawned servers.

    Precedence: ODR_LLM_PRESET > OPENAI_* env (proxy/direct) > goose's own
    model (desktop/CLI config on this machine) > local vLLM/Ollama default.
    A goose/Claude/Codex/Hermes-spawned MCP server often gets a scrubbed
    environment — the goose resolver reads config files instead of env, so
    the dreamer still borrows the host's brain without any setup.
    """
    if os.environ.get("ODR_LLM_PRESET"):
        return os.environ["ODR_LLM_PRESET"]
    if os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_KEY"):
        return "openai"
    try:
        from open_dream_rsi.utils.goose import goose_available

        if goose_available():
            return "goose"
    except Exception:
        pass
    return "local"


def _is_official_endpoint(base_url: str) -> bool:
    return any(h in base_url for h in ("api.openai.com", "api2.cursor.sh"))


def tool_odr_run_once(args: Dict[str, Any]) -> Dict[str, Any]:
    """Run one full improvement cycle (online attempts + offline dreaming).

    Works with no arguments at all: provider/model resolve automatically, an
    empty task queue is a benign no-op, and self-hosted endpoints default to
    disabled hidden reasoning (which otherwise eats the completion budget —
    the client self-heals if an endpoint rejects the flag).
    """
    from open_dream_rsi.cli import _build_client
    from open_dream_rsi.loop import AutoRSIRuntime, Task
    from open_dream_rsi.memory import DreamMemory

    path = _tasks_path(args)
    if not path.exists():  # first run: an empty queue is a no-op, not an error
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]", encoding="utf-8")
    tasks = [Task(**t) for t in json.loads(path.read_text(encoding="utf-8"))]
    provider = args.get("provider") or _default_provider()
    if provider == "mock" or not tasks:  # key-free smoke test / nothing to do
        from open_dream_rsi.llm import StubClient

        client = StubClient()
    else:
        client = _build_client(provider, args.get("model"))
        # Reasoning models spend the completion budget on hidden thinking and
        # the endpoint can take minutes per call — let harnesses tune both.
        if hasattr(client, "config"):
            client.config.timeout = float(args.get("timeout", 300))
            no_thinking = args.get("no_thinking")
            if no_thinking is None:  # auto: on for self-hosted, off for SaaS
                no_thinking = not _is_official_endpoint(client.config.base_url)
            if no_thinking:
                client.config.extra_payload.setdefault(
                    "chat_template_kwargs", {})["enable_thinking"] = False
    runtime = AutoRSIRuntime(
        client=client,
        memory=DreamMemory(_memory_root(args)),
        tasks=tasks,
        api_call_budget=int(args.get("budget", 10)),
        max_tokens=int(args.get("max_tokens", 4096)),
        # Thought-conditioned branching ships ON (library default): attempts
        # record their PLAN line, proposals see the tried-idea ledger, and
        # expansion leaves dead idea families. Harnesses can opt out.
        enable_thoughts=bool(args.get("thoughts", True)),
    )
    report = runtime.run_once().to_dict()
    if not tasks:
        report["note"] = ("task queue is empty — queue a task with "
                          "odr_add_task first, then call odr_run_once again")
    return report


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
        "description": "Queue a self-contained Python task for the "
                       "self-improvement loop. Prefer tests = "
                       "[{'call': 'add(2, 3)', 'expected': 5}]; a task with "
                       "no tests may instead give 'criteria' (success "
                       "description) — an LLM judge then verdicts completion.",
        "inputSchema": {"type": "object", "properties": {
            "task_id": {"type": "string"}, "category": {"type": "string"},
            "prompt": {"type": "string"},
            "tests": {"type": "array", "items": {"type": "object"}},
            "criteria": {"type": "string",
                         "description": "success criterion for test-less "
                                        "tasks (completion-judged)"},
            "max_attempts": {"type": "integer"},
            "tasks_file": {"type": "string"}},
            "required": ["task_id", "category", "prompt"]},
        "fn": tool_odr_add_task,
    },
    "odr_run_once": {
        "description": "Run one Dream-RSI cycle over the queued tasks now (online LLM "
                       "attempts + offline dreaming). No arguments needed: the LLM "
                       "endpoint auto-resolves (env vars, then the local goose config, "
                       "then localhost vLLM/Ollama) and reasoning models run with "
                       "hidden thinking disabled on self-hosted endpoints. Returns a "
                       "solve/budget report.",
        "inputSchema": {
            "type": "object", "properties": {
            "tasks_file": {"type": "string"}, "memory": {"type": "string"},
            "provider": {"type": "string", "enum": ["openai", "cursor", "local", "goose", "mock"],
                         "description": "defaults to auto-detection; only override to force"},
            "model": {"type": "string"},
            "thoughts": {"type": "boolean",
                         "description": "thought-conditioned branching (default true)"},
            "max_tokens": {"type": "integer",
                           "description": "completion budget per LLM call (default 4096; "
                                          "raise for reasoning models)"},
            "timeout": {"type": "number",
                        "description": "seconds per LLM call (default 300; slow local "
                                       "reasoning models need this)"},
            "no_thinking": {"type": "boolean",
                            "description": "disable hidden reasoning via "
                                           "chat_template_kwargs (vLLM Qwen builds; "
                                           "much faster, often better code)"},
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


# ---------------------------------------------------------------------------
# Streamable HTTP transport (Claude Cowork / claude.ai connectors / web)
# ---------------------------------------------------------------------------


def make_http_handler():
    """Minimal stateless Streamable-HTTP MCP endpoint (POST /mcp).

    Each POST carries one JSON-RPC message and gets one JSON response —
    legal per the MCP spec's stateless mode and enough for tool-calling
    connectors (initialize -> tools/list -> tools/call). Bind to a public
    interface only behind your own tunnel/auth.
    """
    from http.server import BaseHTTPRequestHandler

    class HttpHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _json(self, code: int, payload) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):  # noqa: N802
            path = self.path.split("?")[0]
            if path in ("/health", "/mcp/health"):
                self._json(200, {"status": "ok", "server": SERVER_NAME,
                                 "tools": list(TOOLS)})
            else:
                self._json(405, {"error": "POST JSON-RPC to /mcp"})

        def do_POST(self):  # noqa: N802
            if self.path.split("?")[0] not in ("/mcp", "/"):
                self._json(404, {"error": "post JSON-RPC to /mcp"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                request = json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, json.JSONDecodeError):
                self._json(400, {"error": "invalid JSON"})
                return
            response = handle_request(request)
            if response is None:  # notification
                self.send_response(202)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self._json(200, response)

        def log_message(self, format, *args):  # noqa: A002
            pass

    return HttpHandler


def serve_http(host: str = "127.0.0.1", port: int = 8800):
    from http.server import ThreadingHTTPServer

    return ThreadingHTTPServer((host, port), make_http_handler())


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="odr-mcp", description="Open Dream-RSI MCP server (stdio or HTTP)")
    parser.add_argument("--memory", default=None, help="memory dir (default $ODR_MEMORY)")
    parser.add_argument("--tasks", default=None, help="task file (default $ODR_TASKS or tasks.json)")
    parser.add_argument("--http", action="store_true",
                        help="serve stateless Streamable-HTTP MCP (POST /mcp) instead "
                             "of stdio — for claude.ai/Cowork connectors and web clients")
    parser.add_argument("--host", default="127.0.0.1",
                        help="HTTP bind address (default loopback)")
    parser.add_argument("--port", type=int, default=8800, help="HTTP port (default 8800)")
    args = parser.parse_args(argv)
    if args.memory:
        os.environ["ODR_MEMORY"] = args.memory
    if args.tasks:
        os.environ["ODR_TASKS"] = args.tasks
    if args.http:
        httpd = serve_http(args.host, args.port)
        print(f"[odr mcp] HTTP on http://{args.host}:{args.port}/mcp", file=sys.stderr)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[odr mcp] stopped.", file=sys.stderr)
        return 0
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
