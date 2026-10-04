"""Tests for the stdio MCP sentinel/sandbox server (no network, no goose).

Spawns open_dream_rsi.mcp_server as a subprocess with a tiny fake sandbox
(no bench fixtures needed: SPOLKI_SERVE unset -> only sentinel_check),
speaks JSON-RPC over pipes, asserts protocol + sentinel semantics.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def spawn(env_extra):
    env = {**os.environ, "PYTHONPATH": str(REPO), **env_extra}
    return subprocess.Popen(
        [sys.executable, "-m", "open_dream_rsi.mcp_server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        env=env, text=True, bufsize=1)


class McpServerTest(unittest.TestCase):
    def _rpc(self, proc, reqs):
        for r in reqs:
            proc.stdin.write(json.dumps(r) + "\n")
        proc.stdin.flush()
        replies = {}
        for _ in range(sum(1 for r in reqs if "id" in r)):
            line = proc.stdout.readline()
            self.assertTrue(line, "server closed stdout")
            d = json.loads(line)
            replies[d.get("id")] = d
        return replies

    def _serve(self, env, reqs):
        with tempfile.TemporaryDirectory() as td:
            proc = spawn({**env, "SENTINEL_STATE_FILE": td + "/s.json",
                          "SENTINEL_NOTE_FILE": td + "/n.jsonl"})
            try:
                return self._rpc(proc, reqs), td
            finally:
                proc.terminate()
                proc.wait(timeout=5)

    def test_handshake_and_tool_list(self):
        replies, _ = self._serve({"SENTINEL_SESSION": "t1"}, [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}])
        self.assertEqual(
            replies[1]["result"]["serverInfo"]["name"],
            "odr-sentinel-sandbox")
        names = [t["name"] for t in replies[2]["result"]["tools"]]
        self.assertEqual(names, ["sentinel_check"])

    def test_sentinel_note_on_second_failure(self):
        reqs = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        ]
        for i in (2, 3):
            reqs.append({"jsonrpc": "2.0", "id": i, "method": "tools/call",
                         "params": {"name": "sentinel_check", "arguments": {
                             "tool": "fetch_page",
                             "error_excerpt": "blad: strona poza witryną"}}})
        replies, _ = self._serve({"SENTINEL_SESSION": "t2"}, reqs)
        t1 = replies[2]["result"]["content"][0]["text"]
        t2 = replies[3]["result"]["content"][0]["text"]
        self.assertIn("brak znanej klasy", t1)   # first sighting: silent
        self.assertNotIn("brak znanej klasy", t2)  # repeat: note emitted

    def test_cold_arm_no_sentinel(self):
        replies, _ = self._serve({"SENTINEL_SESSION": "t3",
                                  "SENTINEL_OFF": "1"}, [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}])
        names = [t["name"] for t in replies[2]["result"]["tools"]]
        self.assertEqual(names, [])

    def test_pull_tool_hidden_for_stop_arm(self):
        replies, _ = self._serve({"SENTINEL_SESSION": "t4",
                                  "SENTINEL_TOOL": "0"}, [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}])
        names = [t["name"] for t in replies[2]["result"]["tools"]]
        self.assertEqual(names, [])

    def test_unknown_method_errors(self):
        replies, _ = self._serve({"SENTINEL_SESSION": "t5"}, [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 9, "method": "bogus/method"}])
        self.assertEqual(replies[9]["error"]["code"], -32601)


if __name__ == "__main__":
    unittest.main()
