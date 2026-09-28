"""Tests for the LLM completion judge (test-less task verdicts).

Run:  python -m unittest tests.test_judge
"""

import unittest

from open_dream_rsi.core.judge import LLMJudge, parse_verdict
from open_dream_rsi.loop import AutoRSIRuntime, Task
from open_dream_rsi.memory import DreamMemory


class TestParseVerdict(unittest.TestCase):
    def test_strict_object(self):
        v = parse_verdict('{"solved": true, "score": 0.9, "reason": "ok"}')
        self.assertEqual(v["solved"], True)
        self.assertAlmostEqual(v["score"], 0.9)

    def test_surrounding_prose_stripped(self):
        v = parse_verdict('Here is my verdict:\n{"solved": false, "score": 0.2,'
                          ' "reason": "missing edge case"}\nDone.')
        self.assertEqual(v["solved"], False)

    def test_score_clamped(self):
        v = parse_verdict('{"solved": true, "score": 42, "reason": "x"}')
        self.assertEqual(v["score"], 1.0)

    def test_fail_closed_on_garbage(self):
        for bad in ["", "yes solved!", "{}", '{"solved": "true"}',
                    '{"solved": true}', '{"solved": true, "score": null}',
                    '{"solved": true, "score": 1, "reason": 5}',
                    '[1, 2]']:
            self.assertIsNone(parse_verdict(bad), bad)


class ScriptedJudge:
    """LLM stand-in: answers the judge prompt with a fixed verdict string."""

    def __init__(self, code_response, verdict):
        self.code_response = code_response
        self.verdict = verdict
        self.judge_calls = 0

    def chat(self, messages, model=None, temperature=0.7, max_tokens=1024):
        system = messages[0]["content"]
        if "task-completion judge" in system:
            self.judge_calls += 1
            return self.verdict
        return self.code_response


def no_test_task(**kw):
    kw.setdefault("task_id", "t1")
    kw.setdefault("category", "misc")
    kw.setdefault("prompt", "Write a function is_even(n).")
    kw.setdefault("criteria", "is_even(2) is True, is_even(3) is False")
    kw.setdefault("tests", [])
    return Task(**kw)


class TestJudgeUnit(unittest.TestCase):
    def test_transport_error_fails_closed(self):
        class Boom:
            def chat(self, *a, **k):
                raise ConnectionError("endpoint down")
        j = LLMJudge(Boom())
        v = j.judge("p", "c", "code", "evidence")
        self.assertFalse(v["solved"])
        self.assertIn("judge unavailable", v["reason"])

    def test_garbage_verdict_fails_closed(self):
        j = LLMJudge(ScriptedJudge("", "I think it works!"))
        v = j.judge("p", "c", "code", "evidence")
        self.assertFalse(v["solved"])
        self.assertIn("fail-closed", v["reason"])


class TestLoopWithJudge(unittest.TestCase):
    CODE = ("def is_even(n):\n    return n % 2 == 0\n")

    def _run(self, verdict, enable_judge=True, budget=6):
        import tempfile
        client = ScriptedJudge(f"```python\n{self.CODE}```", verdict)
        rt = AutoRSIRuntime(client=client,
                            memory=DreamMemory(tempfile.mkdtemp()),
                            tasks=[no_test_task()],
                            api_call_budget=budget, enable_judge=enable_judge,
                            enable_policy_code=False, enable_knowledge=False)
        return rt.run_once(), client

    def test_judge_solves_test_less_task(self):
        report, client = self._run('{"solved": true, "score": 1.0, "reason": "meets"}')
        self.assertEqual(report.tasks_solved, 1)
        self.assertEqual(client.judge_calls, 1)

    def test_negative_verdict_not_solved(self):
        report, client = self._run('{"solved": false, "score": 0.4, "reason": "no docstring"}')
        self.assertEqual(report.tasks_solved, 0)
        # budget 6, 2 calls per attempt (proposal + judge) -> 3 attempts
        self.assertEqual(client.judge_calls, 3)

    def test_non_executable_code_skips_judge(self):
        class SyntaxErr(ScriptedJudge):
            pass
        client = SyntaxErr("```python\ndef broken(:\n```", "irrelevant")
        import tempfile
        rt = AutoRSIRuntime(client=client, memory=DreamMemory(tempfile.mkdtemp()),
                            tasks=[no_test_task()], api_call_budget=6,
                            enable_policy_code=False, enable_knowledge=False)
        report = rt.run_once()
        self.assertEqual(report.tasks_solved, 0)
        self.assertEqual(client.judge_calls, 0)  # hard sandbox fail, no LLM used

    def test_budget_exhaustion_skips_judge(self):
        # budget 1: the single proposal call exhausts it; judge never runs
        report, client = self._run('{"solved": true, "score": 1.0, "reason": "x"}',
                                    budget=1)
        self.assertEqual(report.tasks_solved, 0)
        self.assertEqual(client.judge_calls, 0)

    def test_judge_never_overrides_failing_tests(self):
        # tests present -> verifier path only, judge never consulted even
        # though it would have said "solved"
        import tempfile
        client = ScriptedJudge(
            "```python\ndef is_even(n):\n    return False\n```",
            '{"solved": true, "score": 1.0, "reason": "wrongly generous"}')
        task = no_test_task(tests=[{"call": "is_even(2)", "expected": True}])
        rt = AutoRSIRuntime(client=client, memory=DreamMemory(tempfile.mkdtemp()),
                            tasks=[task], api_call_budget=6,
                            enable_policy_code=False, enable_knowledge=False)
        report = rt.run_once()
        self.assertEqual(report.tasks_solved, 0)
        self.assertEqual(client.judge_calls, 0)

    def test_disabled_judge_verdicts_unsolved(self):
        report, client = self._run('{"solved": true, "score": 1.0, "reason": "x"}',
                                   enable_judge=False)
        self.assertEqual(report.tasks_solved, 0)
        self.assertEqual(client.judge_calls, 0)


if __name__ == "__main__":
    unittest.main()
