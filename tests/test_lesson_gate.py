"""Lesson promotion gate: staging-by-default, paired replay, stop clause."""

import tempfile
import unittest

from open_dream_rsi.core.curator import (
    curate_lessons,
    format_lessons,
    has_stop_clause,
    lesson_gate_verdict,
    lesson_key,
    select_lessons,
)
from open_dream_rsi.loop import AutoRSIRuntime, DreamMemory, Task


GOOD_ITEM = {"trigger": "median even",
             "text": "Sort the list first, return the mean of the two "
                     "middle values for even lengths, then answer."}


class TestStaging(unittest.TestCase):
    def test_new_lessons_stage_by_default(self):
        r = curate_lessons([], [GOOD_ITEM])
        self.assertEqual(r.entries[0]["status"], "staging")

    def test_staging_can_be_disabled_for_seeded_knowledge(self):
        r = curate_lessons([], [GOOD_ITEM], staging=False)
        self.assertEqual(r.entries[0]["status"], "active")

    def test_select_hides_staging(self):
        r = curate_lessons([], [GOOD_ITEM])
        self.assertEqual(select_lessons(r.entries, "median even task"), [])
        picked = select_lessons(r.entries, "median even task",
                                include_staging=True)
        self.assertEqual(len(picked), 1)

    def test_legacy_entries_without_status_stay_visible(self):
        legacy = {"trigger": "median", "text": "handle even lengths carefully"}
        self.assertEqual(len(select_lessons([legacy], "median")), 1)

    def test_promoted_entry_becomes_visible(self):
        r = curate_lessons([], [GOOD_ITEM])
        r.entries[0]["status"] = "active"
        self.assertEqual(len(select_lessons(r.entries, "median even")), 1)


class TestStopClause(unittest.TestCase):
    def test_stop_clauses_detected(self):
        self.assertTrue(has_stop_clause("check the edge case, then answer"))
        self.assertTrue(has_stop_clause("fetch at most one subpage"))
        self.assertTrue(has_stop_clause("sprawdź podstronę i odpowiedz natychmiast"))

    def test_open_ended_rejected(self):
        self.assertFalse(has_stop_clause("keep searching other pages until "
                                         "all members are found"))


class TestGateVerdict(unittest.TestCase):
    def test_net_gain_promotes(self):
        key = lesson_key(GOOD_ITEM)
        v = lesson_gate_verdict({key: [(True, False), (True, False),
                                      (False, False)]})
        self.assertEqual(v.promoted, [key])

    def test_any_regression_rejects(self):
        key = lesson_key(GOOD_ITEM)
        v = lesson_gate_verdict({key: [(True, False), (False, True)]})
        self.assertEqual(v.rejected, [key])

    def test_no_effect_rejects(self):
        key = lesson_key(GOOD_ITEM)
        v = lesson_gate_verdict({key: [(True, True), (False, False)]})
        self.assertEqual(v.rejected, [key])


class TrackingClient:
    """Solves when `solve_marker` is in the proposal prompt; records prompts.
    Acts as the knowledge curator when asked (returns a fixed lesson)."""

    def __init__(self, solve_marker, lesson_json=None):
        self.solve_marker = solve_marker
        self.lesson_json = lesson_json
        self.prompts = []

    def chat(self, messages, model=None, temperature=0.7, max_tokens=1024):
        user = messages[-1]["content"]
        if "knowledge curator" in (messages[0]["content"] or ""):
            return self.lesson_json or "[]"
        self.prompts.append(user)
        if self.solve_marker in user:
            return "```python\ndef median(xs):\n    return sorted(xs)[len(xs)//2]\n```"
        return "```python\ndef median(xs):\n    return 0\n```"


class TestLoopGate(unittest.TestCase):
    def _tasks(self):
        tests = [{"call": "median([3,1,2])", "expected": 2}]
        return [Task(task_id="t1", category="c", prompt="compute median",
                     tests=tests, max_attempts=2),
                Task(task_id="t2", category="c", prompt="compute median",
                     tests=tests, max_attempts=2)]

    def test_harmful_lesson_never_leaks_into_proposals(self):
        """Staging pre-seeded (never gated) must stay invisible to the solver."""
        client = TrackingClient("middle values")
        rt = AutoRSIRuntime(client=client,
                            memory=DreamMemory(tempfile.mkdtemp()),
                            tasks=self._tasks(), api_call_budget=20,
                            max_tokens=256, enable_policy_code=False,
                            enable_knowledge=False, enable_thoughts=False)
        r = curate_lessons([], [GOOD_ITEM])
        rt.memory.replace_lessons("c", r.entries)
        rt.run_once()
        self.assertTrue(all("middle values" not in p for p in client.prompts))

    def test_open_ended_lesson_killed_even_on_good_sample(self):
        """Client only solves WITH the lesson, but it lacks a stop clause
        -> the paired sample is positive yet the gate must still kill it."""
        bad = {"trigger": "median even",
               "text": "keep searching alternative implementations until all "
                       "even-length edge cases are covered by the code body"}
        client = TrackingClient("alternative implementations",
                                lesson_json="```json\n[%s]\n```" % (
                                    '{"trigger": "median even", "text": '
                                    '"keep searching alternative '
                                    'implementations until all even-length '
                                    'edge cases are covered by the code '
                                    'body"}'))
        rt = AutoRSIRuntime(client=client,
                            memory=DreamMemory(tempfile.mkdtemp()),
                            tasks=self._tasks(), api_call_budget=40,
                            max_tokens=256, enable_policy_code=False,
                            enable_thoughts=False)
        rt.run_once()   # failures -> curator -> staging lesson -> gate runs
        keys = [lesson_key(l) for l in rt.memory.get_lessons("c")]
        self.assertNotIn(lesson_key(bad), keys)

    def test_good_lesson_with_stop_clause_gets_promoted(self):
        client = TrackingClient("middle values",
                                lesson_json="```json\n[%s]\n```" % (
                                    '{"trigger": "median even", "text": '
                                    '"Sort the list first, return the mean '
                                    'of the two middle values for even '
                                    'lengths, then answer."}'))
        rt = AutoRSIRuntime(client=client,
                            memory=DreamMemory(tempfile.mkdtemp()),
                            tasks=self._tasks(), api_call_budget=40,
                            max_tokens=256, enable_policy_code=False,
                            enable_thoughts=False)
        rt.run_once()
        entries = rt.memory.get_lessons("c")
        keys = [lesson_key(l) for l in entries]
        self.assertIn(lesson_key(GOOD_ITEM), keys)
        self.assertEqual(
            [l["status"] for l in entries if lesson_key(l) == lesson_key(GOOD_ITEM)],
            ["active"])


if __name__ == "__main__":
    unittest.main()
