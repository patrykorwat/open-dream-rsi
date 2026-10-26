"""Tests for the success-metrics module (odr metrics / odr_metrics MCP tool)."""
import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from open_dream_rsi.metrics import collect, format_metrics


def _write_events(path, events):
    with open(path, "w", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")


class MetricsCollectTest(unittest.TestCase):
    def test_collect_aggregates_events_and_files(self):
        now = time.time()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_events(root / "events.jsonl", [
                {"ts": now - 60, "kind": "cycle", "api_calls": 5,
                 "tasks_attempted": 2, "tasks_solved": 1,
                 "stopped_reason": "api_budget_exhausted",
                 "tokens": {"prompt_tokens": 1200, "completion_tokens": 300,
                            "total_tokens": 1500}},
                {"ts": now - 50, "kind": "task", "category": "cat", "solved": True},
                {"ts": now - 40, "kind": "task", "category": "cat", "solved": False},
                {"ts": now - 30, "kind": "llm_error", "error": "timed out"},
                {"ts": now - 20, "kind": "lesson_rejected", "reason": "empty payload"},
                {"ts": now - 10, "kind": "lessons_curated", "added": 2, "merged": 1},
                {"ts": now - 5, "kind": "lesson_gate"},
            ])
            (root / "lessons.json").write_text(json.dumps(
                {"cat": [{"status": "active"}, {"status": "staging"}]}))
            (root / "recipes.json").write_text(json.dumps(
                {"cat": {"code": "x", "score": 0.9}}))
            (root / "task_outcomes.json").write_text(json.dumps(
                {"cat": [{"solved": True}, {"solved": False}, {"solved": False}]}))
            (root / "tasks.json").write_text(json.dumps([{}, {}]))
            (root / "dream_state.json").write_text(json.dumps(
                {"last_full_dream": now - 3600}))
            (root / "trigger.log").write_text(
                f"{now - 20} fire: odr dream --budget 300\n"
                f"{now - 400000} fire: odr dream --budget 20\n")

            m = collect(root, hours=24)
            self.assertEqual(m["cycles"]["total"], 1)
            self.assertEqual(m["cycles"]["api_calls"], 5)
            self.assertEqual(m["cycles"]["budget_exhausted"], 1)
            self.assertEqual(m["tasks"]["cat"], {"attempted": 2, "solved": 1})
            self.assertIn("llm timed out", " ".join(m["llm_errors"]))
            self.assertIn("empty payload", " ".join(m["lesson_rejections"]))
            self.assertEqual(m["curation"]["lessons_added"], 2)
            self.assertEqual(m["curation"]["gate_verdicts"], 1)
            self.assertEqual(m["lessons"]["cat"], {"active": 1, "staging": 1, "total": 2})
            self.assertEqual(m["recipes"], {"cat": 0.9})
            self.assertEqual(m["outcomes_all_time"]["cat"], {"attempted": 3, "solved": 1})
            self.assertEqual(m["backlog"]["queued_tasks"], 2)
            self.assertIsNotNone(m["backlog"]["hours_since_full_dream"])
            self.assertEqual(m["trigger"]["fires"], 1)  # old fire outside the window
            self.assertEqual(m["trigger"]["last_budget"], 300)
            self.assertEqual(m["tokens"], {"prompt": 1200, "completion": 300,
                                           "cycles_with_usage": 1})
            text = format_metrics(m)
            self.assertIn("tokens: prompt 1,200  completion 300", text)

    def test_empty_memory_dir_degrades_to_zeros(self):
        with TemporaryDirectory() as tmp:
            m = collect(tmp)
            self.assertEqual(m["cycles"]["total"], 0)
            self.assertEqual(m["tasks"], {})
            self.assertEqual(m["lessons"], {})
            self.assertIsNone(m["backlog"]["hours_since_full_dream"])
            text = format_metrics(m)  # must render, not raise
            self.assertIn("cycles: 0", text)
            self.assertIn("not instrumented", text)  # no token usage recorded

    def test_format_flags_zero_solve_categories(self):
        data = {"window_hours": 24,
                "cycles": {"total": 1, "api_calls": 1, "budget_exhausted": 0,
                           "solved": 0, "attempted": 1, "stopped_reasons": {}},
                "trigger": {"fires": 0, "last_budget": None},
                "tasks": {"hard": {"attempted": 7, "solved": 0}},
                "llm_errors": {}, "lesson_rejections": {},
                "curation": {"gate_verdicts": 0, "lessons_added": 0,
                             "lessons_merged": 0, "user_corrections": 0},
                "lessons": {}, "recipes": {},
                "outcomes_all_time": {"hard": {"attempted": 10, "solved": 0}},
                "backlog": {"queued_tasks": 0, "hours_since_full_dream": None}}
        text = format_metrics(data)
        self.assertIn("no solves", text)
        self.assertIn("headroom flags", text)


if __name__ == "__main__":
    unittest.main()
