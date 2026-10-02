"""Headroom precondition: categories the baseline solves cheaply get zero
curator/gate calls (live-replay v3-v9: seven injection mechanisms, zero
paired gains on a 19/20 @ 3.0 calls/solve cold baseline)."""

import tempfile
import unittest

from open_dream_rsi.core.curator import headroom_verdict
from open_dream_rsi.loop import AutoRSIRuntime, DreamMemory, Task

LESSON_JSON = ('```json\n[{"trigger": "median even", "text": "sort first, '
               'then answer the question."}]\n```')


def rows(solved_flags, calls=2):
    return [{"solved": s, "calls": calls} for s in solved_flags]


class TestHeadroomVerdict(unittest.TestCase):
    def test_empty_history_has_headroom(self):
        self.assertTrue(headroom_verdict([])["has_headroom"])

    def test_cheap_cold_baseline_has_no_headroom(self):
        v = headroom_verdict(rows([True] * 19 + [False], calls=3))
        self.assertFalse(v["has_headroom"])
        self.assertIn("perturbation tax", v["reason"])

    def test_low_solve_rate_has_headroom(self):
        v = headroom_verdict(rows([True] * 6 + [False] * 4))
        self.assertTrue(v["has_headroom"])
        self.assertIn("solve_rate", v["reason"])

    def test_expensive_solves_have_headroom(self):
        v = headroom_verdict(rows([True] * 10, calls=8))
        self.assertTrue(v["has_headroom"])
        self.assertIn("calls/solve", v["reason"])

    def test_only_recent_window_counts(self):
        hist = rows([False] * 20, calls=8) + rows([True] * 10, calls=2)
        self.assertFalse(headroom_verdict(hist)["has_headroom"])

    def test_all_fail_counts_low_solve_rate(self):
        self.assertTrue(headroom_verdict(rows([False] * 5))["has_headroom"])

    def test_thresholds_configurable(self):
        hist = rows([True] * 10, calls=4)
        self.assertFalse(headroom_verdict(hist)["has_headroom"])
        self.assertTrue(headroom_verdict(hist, max_calls_per_solve=3.5)["has_headroom"])


class RecordingClient:
    """Records (system, user) of every call; always FAILS the verifier, and
    plays the curator when asked (a staging lesson would then be gated)."""

    def __init__(self):
        self.calls = []

    def chat(self, messages, model=None, temperature=0.7, max_tokens=1024):
        system = messages[0]["content"] or ""
        user = messages[-1]["content"] or ""
        self.calls.append((system, user))
        if "knowledge curator" in system:
            return LESSON_JSON
        return "```python\ndef median(xs):\n    return 0\n```"

    def curator_calls(self):
        return [c for c in self.calls if "knowledge curator" in c[0]]

    def lesson_leaks(self):
        return [c for c in self.calls if "sort first" in c[1]]


class TestLoopHeadroom(unittest.TestCase):
    def _tasks(self, n):
        tests = [{"call": "median([3,1,2])", "expected": 2}]
        return [Task(task_id=f"t{i}", category="c", prompt="compute median",
                     tests=tests, max_attempts=1) for i in range(n)]

    def _run(self, mem, n, budget=60):
        client = RecordingClient()
        rt = AutoRSIRuntime(client=client, memory=mem, tasks=self._tasks(n),
                            api_call_budget=budget, max_tokens=128,
                            enable_policy_code=False, enable_thoughts=False)
        rt.run_once()
        return client

    def test_curator_zero_calls_on_headroomless_category(self):
        # The category solved its family cheaply last cycle (solve-rate 1.0,
        # 1 call/solve) -> failures this cycle must NOT buy curator/gate calls.
        mem = DreamMemory(tempfile.mkdtemp())
        for _ in range(10):
            mem.record_task_outcome("c", solved=True, calls=1)
        client = self._run(mem, 3)
        self.assertEqual(client.curator_calls(), [])
        self.assertEqual(client.lesson_leaks(), [])
        self.assertEqual(mem.get_lessons("c"), [])

    def test_fresh_category_still_gets_curation(self):
        # empty history = headroom: the section-4 path must stay alive.
        # The client never solves even WITH the lesson, so the paired-replay
        # gate must reject it: the curator was ASKED (a headroomless category
        # would skip it entirely) and nothing ACTIVE remains in the KB.
        # (client.lesson_leaks() also counts the gate's own with-lesson probe
        # prompts, which legitimately carry the lesson text.)
        mem = DreamMemory(tempfile.mkdtemp())
        client = self._run(mem, 1)
        self.assertTrue(client.curator_calls())
        self.assertEqual([l for l in mem.get_lessons("c")
                          if l.get("status") == "active"], [])

    def test_outcomes_recorded_per_task(self):
        mem = DreamMemory(tempfile.mkdtemp())
        self._run(mem, 2, budget=10)
        hist = mem.get_task_outcomes("c")
        self.assertEqual(len(hist), 2)
        self.assertTrue(all(o["solved"] is False for o in hist))


if __name__ == "__main__":
    unittest.main()
