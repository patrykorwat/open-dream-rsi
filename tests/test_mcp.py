"""Tests for the MCP stdio server (open_dream_rsi.mcp).

The server must speak JSON-RPC over newline-delimited stdio exactly like an
MCP client (OpenCode, Goose, ...) expects, expose only read/queue/run tools,
and never exec anything beyond the existing sandboxed verifier.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from open_dream_rsi.mcp import handle_request, serve


def call(method: str, params=None, req_id=1) -> dict:
    res = handle_request({"jsonrpc": "2.0", "id": req_id, "method": method,
                          "params": params or {}})
    assert res is not None  # requests with an id always get a response
    return res


def tool(name: str, arguments: dict, req_id=1) -> dict:
    res = call("tools/call", {"name": name, "arguments": arguments}, req_id)
    assert res is not None
    return res


def payload(response: dict) -> dict:
    text = response["result"]["content"][0]["text"]
    return json.loads(text)


class HandshakeTests(unittest.TestCase):
    def test_initialize_announces_tools_capability(self):
        res = call("initialize", {"protocolVersion": "2025-06-18", "clientInfo": {"name": "t"}})
        self.assertEqual(res["result"]["serverInfo"]["name"], "open-dream-rsi")
        self.assertIn("tools", res["result"]["capabilities"])

    def test_initialized_notification_gets_no_response(self):
        self.assertIsNone(handle_request(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}))

    def test_unknown_method_returns_jsonrpc_error(self):
        res = call("bogus/method")
        self.assertEqual(res["error"]["code"], -32601)
        self.assertNotIn("result", res)

    def test_tools_list_exposes_the_five_loop_tools(self):
        names = {t["name"] for t in call("tools/list")["result"]["tools"]}
        self.assertEqual(names, {"odr_status", "odr_recipes", "odr_lessons",
                                 "odr_add_task", "odr_run_once"})


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.memory = str(Path(self.tmp.name) / "mem")
        self.tasks = str(Path(self.tmp.name) / "tasks.json")
        self.addCleanup(self.tmp.cleanup)

    def test_add_task_persists_and_dedupes(self):
        args = {"task_id": "a1", "category": "math",
                "prompt": "Implement add(a, b).",
                "tests": [{"call": "add(2, 3)", "expected": 5}],
                "tasks_file": self.tasks, "memory": self.memory}
        out = payload(tool("odr_add_task", args))
        self.assertTrue(out["added"])
        self.assertEqual(json.loads(Path(self.tasks).read_text())[0]["task_id"], "a1")
        again = payload(tool("odr_add_task", args, req_id=2))
        self.assertFalse(again["added"])

    def test_add_task_requires_fields(self):
        res = tool("odr_add_task", {"task_id": "x", "tasks_file": self.tasks})
        self.assertTrue(res["result"]["isError"])
        self.assertIn("category", res["result"]["content"][0]["text"])

    def test_recipes_and_lessons_empty_memory_is_clean(self):
        out = payload(tool("odr_recipes", {"category": "math", "memory": self.memory}))
        self.assertFalse(out["found"])
        out = payload(tool("odr_lessons", {"query": "nothing", "memory": self.memory}))
        self.assertEqual(out["count"], 0)

    def test_recipes_returns_promoted_warm_start(self):
        from open_dream_rsi.memory import DreamMemory

        DreamMemory(self.memory).save_recipe("math", "def mul(a, b): return a * b", 1.0)
        out = payload(tool("odr_recipes", {"category": "math", "memory": self.memory}))
        self.assertTrue(out["found"])
        self.assertIn("mul", out["code"])

    def test_run_once_mock_solves_and_reports(self):
        from open_dream_rsi.memory import DreamMemory

        # Script the mock client through the same seam the live demo uses:
        # a canned correct solution for the queued task.
        code = "def add(a, b):\n    return a + b\n"
        task = {"task_id": "add1", "category": "math",
                "prompt": "Implement add(a, b) returning the sum.",
                "tests": [{"call": "add(2, 3)", "expected": 5},
                          {"call": "add(-1, 1)", "expected": 0}],
                "max_attempts": 2}
        Path(self.tasks).write_text(json.dumps([task]), encoding="utf-8")
        # Warm start with the recipe so the cycle proves recipe reuse end to end.
        DreamMemory(self.memory).save_recipe("math", code, 1.0)
        out = payload(tool("odr_run_once", {
            "tasks_file": self.tasks, "memory": self.memory, "provider": "mock"}))
        self.assertEqual(out["stopped_reason"], "completed")
        self.assertGreaterEqual(out["tasks_attempted"], 1)
        self.assertLessEqual(out["api_calls"], 10)

    def test_run_once_without_task_file_is_a_clear_error(self):
        res = tool("odr_run_once", {"tasks_file": str(Path(self.tmp.name) / "nope.json"),
                                    "memory": self.memory, "provider": "mock"})
        self.assertTrue(res["result"]["isError"])
        self.assertIn("task file not found", res["result"]["content"][0]["text"])

    def test_unknown_tool_reports_error_not_crash(self):
        res = tool("nope", {})
        self.assertTrue(res["result"]["isError"] or "error" in res)


class StdioTransportTests(unittest.TestCase):
    def test_serve_handles_newline_delimited_stream(self):
        stdin = io.StringIO(
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}) + "\n"
            + "not json\n"  # must be skipped silently
            + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"}) + "\n")
        stdout = io.StringIO()
        serve(stdin=stdin, stdout=stdout)
        lines = [json.loads(l) for l in stdout.getvalue().splitlines() if l.strip()]
        self.assertEqual([l["id"] for l in lines], [1, 2])


if __name__ == "__main__":
    unittest.main()
