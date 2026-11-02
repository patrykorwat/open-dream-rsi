"""Tests for the automated dreaming seam (dream.py): cadence tiers,
evidence ingest, single-instance lock, skills publish.

Run:  python -m unittest discover -s tests
"""

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from open_dream_rsi.dream import _acquire, _release, dream_once


def _hermes_db(path: Path, n_sessions: int = 3, fail: bool = True) -> None:
    """Minimal Hermes-shaped state.db with failure-shaped sessions."""
    con = sqlite3.connect(path)
    con.execute("create table sessions (id text primary key, title text,"
                " display_name text, cwd text, started_at real,"
                " archived int default 0, hidden int default 0)")
    con.execute("create table messages (id integer primary key, session_id"
                " text, role text, content text, tool_name text,"
                " timestamp real, active int default 1)")
    mid = 0
    for i in range(n_sessions):
        sid = f"s{i}"
        con.execute("insert into sessions values (?,?,?,?,?,0,0)",
                    (sid, f"session {i}", "", "/tmp/proj", 1737136400 + i))
        mid += 1
        con.execute("insert into messages values (?,?,?,?,NULL,?,1)",
                    (mid, sid, "user", "fix this broken thing",
                     1737136401 + i))
        if fail:
            mid += 1
            con.execute("insert into messages values (?,?,?,?,NULL,?,1)",
                        (mid, sid, "assistant",
                         "Error: KeyError 'median' in sort_and_pick",
                         1737136402 + i))
    con.commit()
    con.close()


class _NoLLM:
    """Client whose chat always raises — a configured-but-dead endpoint:
    distill surfaces an 'llm error' and the session is DEFERRED (stays
    unseen) instead of being consumed."""

    def chat(self, *a, **k):
        raise RuntimeError("no llm in tests")


class _JunkLLM:
    """Client that replies with non-JSON: the policy validator rejects the
    content, so the cycle records an llm_error — the contrast to _NoLLM."""

    def chat(self, *a, **k):
        return "not a json array"


class TestDreamOnce(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.mem = self.tmp / "mem"
        self.mem.mkdir()
        self.db = self.tmp / "state.db"
        _hermes_db(self.db, n_sessions=3)

    def test_offline_maintenance_run(self):
        report = dream_once(self.mem, client=None)
        self.assertFalse(report["skipped"])
        self.assertEqual(report["tier"], "full")  # first run: age > max_age

    def test_lock_collapses_concurrent_runs(self):
        lock = _acquire(self.mem)
        self.assertIsNotNone(lock)
        try:
            report = dream_once(self.mem, client=None)
            self.assertTrue(report["skipped"])
        finally:
            _release(lock)
        report = dream_once(self.mem, client=None)
        self.assertFalse(report["skipped"])

    def _write_lock(self, pid: int, age_seconds: float = 0.0) -> Path:
        lock = self.mem / "dream.lock"
        self.mem.mkdir(parents=True, exist_ok=True)
        lock.write_text(str(pid))
        past = time.time() - age_seconds
        os.utime(lock, (past, past))
        return lock

    def test_live_owner_is_never_stale_regardless_of_age(self):
        # A full cycle legitimately outlives LOCK_TIMEOUT_S: the next fire
        # must NOT steal the lock of a live owner (the age-first order once
        # produced two concurrent dreams on one memory dir).
        self._write_lock(os.getpid(), age_seconds=10 * 3600)
        self.assertIsNone(_acquire(self.mem))

    def test_dead_owner_pid_is_reclaimed_even_when_fresh(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()  # reaped: this PID is guaranteed gone
        self._write_lock(proc.pid, age_seconds=0.0)
        lock = _acquire(self.mem)
        self.assertIsNotNone(lock)
        _release(lock)

    def test_release_never_removes_a_foreign_lock(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        lock = self._write_lock(proc.pid)
        _release(lock)  # our PID does not own it: must survive
        self.assertTrue(lock.exists())
        lock.unlink()

class CursorIngestTests(unittest.TestCase):
    """dream --source cursor: the Cursor store ingests like a Hermes one
    (no MCP involved)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.mem = self.tmp / "mem"
        self.mem.mkdir()

    def _cursor_store(self, fail: bool = True) -> str:
        user = self.tmp / "User"
        (user / "globalStorage").mkdir(parents=True)
        g = sqlite3.connect(user / "globalStorage" / "state.vscdb")
        g.execute("create table cursorDiskKV (key TEXT UNIQUE, value BLOB)")
        header = {"composerId": "c-1", "name": "scraper fix",
                  "createdAt": 1737136403732,
                  "fullConversationHeadersOnly": [{"bubbleId": "b1", "type": 1},
                                                  {"bubbleId": "b2", "type": 2}]}
        g.execute("insert into cursorDiskKV values ('composerData:c-1', ?)",
                  (json.dumps(header),))
        g.execute("insert into cursorDiskKV values ('bubbleId:c-1:b1', ?)",
                  (json.dumps({"type": 1, "text": "the site blocks requests",
                               "createdAt": 1737136404000}),))
        g.execute("insert into cursorDiskKV values ('bubbleId:c-1:b2', ?)",
                  (json.dumps({"type": 2, "text": "Error: 429 Too Many Requests",
                               "createdAt": 1737136405000}),))
        g.commit()
        g.close()
        return str(user)

class ProbeQueueCapTests(unittest.TestCase):
    """The queue backstop (dream._cap_probe_queue): retires OLD, already-
    attempted probes only — an unprocessed probe is never dropped."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.queue = self.tmp / "tasks.json"

    def _write(self, ids):
        self.queue.write_text(json.dumps(
            [{"task_id": i, "category": "c", "prompt": "p", "criteria": "d"}
             for i in ids]))

class QueueCliTests(unittest.TestCase):
    """odr task add/list — the no-MCP queue seam (loop.append_task)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.queue = self.tmp / "tasks.json"

    def _add(self, **kw):
        from open_dream_rsi.loop import append_task
        return append_task(self.queue, {"task_id": kw.get("id", "t1"),
                                        "category": kw.get("cat", "demo"),
                                        "prompt": kw.get("prompt", "do it"),
                                        "criteria": kw.get("criteria", "done")})

    def test_append_task_writes_and_normalises(self):
        out = self._add()
        self.assertTrue(out["added"])
        task = json.loads(self.queue.read_text())[0]
        self.assertEqual(task["tests"], [])
        self.assertEqual(task["max_attempts"], 4)
        self.assertEqual(task["criteria"], "done")

    def test_append_task_refuses_duplicates(self):
        self.assertTrue(self._add()["added"])
        out = self._add()
        self.assertFalse(out["added"])
        self.assertIn("already queued", out["reason"])
        self.assertEqual(len(json.loads(self.queue.read_text())), 1)

    def test_append_task_validates_required_fields(self):
        from open_dream_rsi.loop import append_task
        with self.assertRaises(ValueError):
            append_task(self.queue, {"task_id": "x", "category": "c"})
        # neither tests nor criteria -> rejected
        with self.assertRaises(ValueError):
            append_task(self.queue, {"task_id": "x", "category": "c",
                                     "prompt": "p"})


class _SolveLLM:
    """Returns a correct add(a, b) implementation — solves the wake probe."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, model=None, temperature=0.7, max_tokens=1024):
        self.calls += 1
        return "```python\ndef add(a, b):\n    return a + b\n```"


class WakeTests(unittest.TestCase):
    """The wake self-test: an idle FULL dream still exercises the endpoint."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.mem = self.tmp / "mem"
        self.mem.mkdir()

    def test_full_dream_with_empty_queue_seeds_wake_probe(self):
        report = dream_once(self.mem, client=_SolveLLM(), max_age_seconds=0)
        self.assertTrue(report["wake"])
        cycle = report["cycle"]
        self.assertEqual(cycle["tasks_attempted"], 1)
        self.assertEqual(cycle["api_calls"], 1)
        # tasks.json is never touched by the wake seed
        self.assertFalse((self.mem / "tasks.json").exists())
        events = [json.loads(line) for line
                  in (self.mem / "events.jsonl").read_text().splitlines()
                  if line.strip()]
        solved = [e for e in events
                  if e.get("kind") == "task" and e.get("solved")]
        self.assertEqual(len(solved), 1)
        self.assertEqual(solved[0]["category"], "wake")

    def test_wake_flag_off_disables_the_seed(self):
        report = dream_once(self.mem, client=_SolveLLM(),
                            max_age_seconds=0, wake=False)
        self.assertNotIn("wake", report)
        self.assertEqual(report["cycle"]["api_calls"], 0)

    def test_wake_env_opt_out(self):
        old = os.environ.get("ODR_WAKE")
        os.environ["ODR_WAKE"] = "0"
        try:
            report = dream_once(self.mem, client=_SolveLLM(),
                                max_age_seconds=0)
        finally:
            if old is None:
                os.environ.pop("ODR_WAKE", None)
            else:
                os.environ["ODR_WAKE"] = old
        self.assertNotIn("wake", report)

    def test_wake_never_replaces_a_nonempty_queue(self):
        from open_dream_rsi.loop import append_task
        append_task(self.mem / "tasks.json",
                    {"task_id": "manual-1", "category": "demo",
                     "prompt": "p", "criteria": "c"})
        report = dream_once(self.mem, client=_SolveLLM(), max_age_seconds=0)
        self.assertNotIn("wake", report)
        queue = json.loads((self.mem / "tasks.json").read_text())
        self.assertEqual([t["task_id"] for t in queue], ["manual-1"])

    def test_wake_fires_only_on_a_full_tier(self):
        dream_once(self.mem, client=_SolveLLM(), max_age_seconds=0)
        # second immediate call: maintenance tier (fresh last_full_dream,
        # no new evidence; a positive max-age keeps the tier cheap) —
        # wake must not fire off-tier
        report = dream_once(self.mem, client=_JunkLLM(), max_age_seconds=3600)
        self.assertEqual(report["tier"], "maintenance")
        self.assertNotIn("wake", report)
        self.assertEqual(report["cycle"]["api_calls"], 0)

    def test_wake_never_fires_without_a_real_client(self):
        # client=None -> StubClient path (offline/mock): zero-spend promise
        report = dream_once(self.mem, client=None, max_age_seconds=0)
        self.assertNotIn("wake", report)
        self.assertEqual(report["cycle"]["api_calls"], 0)


if __name__ == "__main__":
    unittest.main()
