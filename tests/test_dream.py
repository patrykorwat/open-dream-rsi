"""Tests for the automated dreaming seam (dream.py): cadence tiers,
evidence ingest, single-instance lock, skills publish.

Run:  python -m unittest discover -s tests
"""

import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from open_dream_rsi.dream import _acquire, _release, dream_once, ingest_sessions


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
    """Client that fails every call — forces distill to reject cleanly and
    keeps tests free of LLM traffic."""

    def chat(self, *a, **k):
        raise RuntimeError("no llm in tests")


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

    def test_skills_published_from_active_only(self):
        kb = self.mem / "lessons.json"
        kb.write_text(json.dumps({
            "median": [{"trigger": "median even",
                        "text": "Sort first, return the mean of the two "
                                "middle values, then answer.",
                        "status": "active", "wins": 3, "uses": 5,
                        "evidence": [], "updated_at": time.time()}]}))
        out = self.tmp / "skills"
        report = dream_once(self.mem, client=None, skills_out=out)
        self.assertEqual(report["skills"], 1)
        self.assertTrue((out / "median" / "SKILL.md").exists())

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

    def test_probe_tasks_queued_from_failure_sessions(self):
        from open_dream_rsi.memory import DreamMemory
        mem = DreamMemory(self.mem)
        ingest_sessions(mem, self.db, client=_NoLLM())
        tp = self.mem / "tasks.json"
        self.assertTrue(tp.exists())
        tasks = json.loads(tp.read_text())
        probes = [t for t in tasks if str(t["task_id"]).startswith("probe:")]
        self.assertEqual(len(probes), 3)
        self.assertEqual(probes[0]["max_attempts"], 1)
        self.assertEqual(probes[0]["prompt"], "fix this broken thing")
        # idempotent: a second ingest adds no duplicates
        ingest_sessions(mem, self.db, client=_NoLLM())
        self.assertEqual(len(json.loads(tp.read_text())), 3)

    def test_ingest_skips_seen_sessions(self):
        from open_dream_rsi.memory import DreamMemory
        mem = DreamMemory(self.mem)
        # a live-shaped client (distillation may fail; sessions are still
        # accounted as seen). client=None means "no endpoint" -> sessions
        # stay unseen until a cycle can actually distil them.
        r1 = ingest_sessions(mem, self.db, client=_NoLLM())
        self.assertEqual(r1["new"], 3)
        r2 = ingest_sessions(mem, self.db, client=_NoLLM())
        self.assertEqual(r2["new"], 0)
        fresh = Path(tempfile.mkdtemp()) / "mem2"
        fresh.mkdir()
        r3 = ingest_sessions(DreamMemory(fresh), self.db, client=None)
        self.assertEqual(r3["new"], 0)  # offline never marks evidence seen

    def test_maintenance_tier_when_no_new_evidence(self):
        # first full run establishes last_full_dream and accounts the
        # sessions; an immediate second call must be maintenance (budget 0)
        dream_once(self.mem, sessions_db=self.db, client=_NoLLM())
        report = dream_once(self.mem, sessions_db=self.db, client=_NoLLM())
        self.assertEqual(report["tier"], "maintenance")


if __name__ == "__main__":
    unittest.main()
