"""Tests for shared-fixture 'setup' blocks in verifier tests and prompts."""

import unittest

from open_dream_rsi.tools import CodeVerifier
from open_dream_rsi.loop import AutoRSIRuntime, Task, DreamMemory


class TestSetupFixtures(unittest.TestCase):
    def test_setup_executed_once_and_shared(self):
        tests = [
            {"setup": "FIX = [1, 2]", "call": "sum(FIX)", "expected": 3},
            {"setup": "FIX = [1, 2]", "call": "len(FIX)", "expected": 2},
        ]
        res = CodeVerifier().run("def noop(): pass", tests)
        self.assertTrue(res.solved)

    def test_setup_failure_is_hard_error(self):
        tests = [{"setup": "x = 1/0", "call": "len(x)", "expected": 1}]
        res = CodeVerifier().run("pass", tests)
        self.assertFalse(res.ok)
        self.assertIn("fixture setup failed", str(res.detail))

    def test_setup_can_shadow_candidate_wrong_name(self):
        # setup runs AFTER candidate exec; a fixture name collision wins
        tests = [{"setup": "add = 5", "call": "add", "expected": 5}]
        res = CodeVerifier().run("def add(a, b): return a+b", tests)
        self.assertTrue(res.solved)

    def test_tests_without_setup_unchanged(self):
        res = CodeVerifier().run("def f(): return 1", [{"call": "f()", "expected": 1}])
        self.assertTrue(res.solved)


class FixtureOnlyClient:
    def __init__(self):
        self.prompts = []

    def chat(self, messages, model=None, temperature=0.7, max_tokens=1024):
        self.prompts.append(messages[-1]["content"])
        return "```python\ndef f(s):\n    return s\n```"


class TestPromptHoisting(unittest.TestCase):
    def test_shared_setup_renders_once(self):
        big = "X" * 500
        tests = [
            {"setup": f"FIX = {big!r}", "call": "f(FIX[:3])", "expected": "XXX"},
            {"setup": f"FIX = {big!r}", "call": "len(FIX)", "expected": 500},
        ]
        client = FixtureOnlyClient()
        task = Task(task_id="t", category="c", prompt="wrap", tests=tests)
        rt = AutoRSIRuntime(client=client, memory=DreamMemory(
            __import__("tempfile").mkdtemp()), tasks=[task],
            api_call_budget=1, enable_policy_code=False, enable_knowledge=False)
        rt.run_once()
        prompt = client.prompts[0]
        self.assertEqual(prompt.count(big), 1)          # hoisted, rendered once
        self.assertIn("Fixtures (already defined", prompt)
        self.assertIn("f(FIX[:3])", prompt)

    def test_no_setup_prompts_identical_to_before(self):
        tests = [{"call": "f('a')", "expected": "a"}]
        client = FixtureOnlyClient()
        task = Task(task_id="t", category="c", prompt="wrap", tests=tests)
        rt = AutoRSIRuntime(client=client, memory=DreamMemory(
            __import__("tempfile").mkdtemp()), tasks=[task],
            api_call_budget=1, enable_policy_code=False, enable_knowledge=False)
        rt.run_once()
        self.assertNotIn("Fixtures (already defined", client.prompts[0])


if __name__ == "__main__":
    unittest.main()
