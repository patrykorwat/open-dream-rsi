"""End-to-end tests for the autonomous RSI loop (no network, no API key).

Note: the working loop fixes code WITHIN a cycle (verifier feedback feeds the
next attempt), so a task can go from buggy to solved in one cycle; cross-cycle
memory is about warm starts, policy persistence and cheap re-verification.

Run:  python -m unittest discover -s tests
"""

import unittest

from open_dream_rsi.loop import AutoRSIRuntime, Task
from open_dream_rsi.memory import DreamMemory

BUGGY = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"


class ScriptedLLM:
    """Deterministic stand-in for an LLM: reacts to recipe/feedback in the prompt."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, model=None, temperature=0.7, max_tokens=1024):
        self.calls += 1
        prompt = messages[-1]["content"]
        if FIXED.strip() in prompt:                 # learned recipe visible -> warm start
            code = FIXED
        elif "Previous failure feedback:\n(none)" in prompt:
            code = BUGGY                             # first ever attempt: wrong solution
        else:
            code = FIXED                             # after verifier feedback: the fix
        return f"```python\n{code}```"


def make_task(task_id="add1", category="math"):
    return Task(task_id=task_id, category=category, prompt="Implement add(a, b).",
                tests=[{"call": "add(2, 3)", "expected": 5},
                       {"call": "add(-1, 1)", "expected": 0}],
                max_attempts=3)


class LoopTest(unittest.TestCase):
    def setUp(self):
        self.memory = DreamMemory(root=self._tmp_name())
        self.llm = ScriptedLLM()
        self.runtime = AutoRSIRuntime(
            client=self.llm, memory=self.memory, tasks=[make_task()],
            api_call_budget=10, dream_iterations=20,
            enable_policy_code=False,  # policy generation has its own test module
        )

    @staticmethod
    def _tmp_name():
        import tempfile, os
        d = tempfile.mkdtemp(prefix="odr-test-")
        return f"{d}/mem"

    def test_feedback_fixes_within_cycle_and_memory_persists(self):
        r1 = self.runtime.run_once()
        self.assertEqual(r1.tasks_solved, 1)        # buggy -> feedback -> fixed
        recipe = self.memory.get_recipe("math")
        self.assertIsNotNone(recipe)
        self.assertIn("def add", recipe or "")
        policy = self.memory.get_policy("math")
        self.assertIsNotNone(policy)                # policy dreamed every cycle

        reopened = DreamMemory(root=self.memory.root)
        self.assertIsNotNone(reopened.get_policy("math"))  # survives restart

    def test_warm_start_is_cheap(self):
        self.runtime.run_once()
        calls_before = self.llm.calls
        r2 = self.runtime.run_once()
        self.assertEqual(r2.tasks_solved, 1)
        self.assertLessEqual(self.llm.calls - calls_before, 1)  # recipe reuse, no re-derivation

    def test_budget_guard_stops_next_task(self):
        self.runtime.api_call_budget = 1
        self.runtime._tasks = [make_task("t1"), make_task("t2")]
        report = self.runtime.run_once()
        self.assertLessEqual(report.api_calls, 1)
        self.assertEqual(report.tasks_attempted, 1)  # second task never started
        self.assertEqual(report.stopped_reason, "api_budget_exhausted")
        self.assertGreater(report.dream_iterations, 0)  # dreaming stayed free

    def test_verifier_sandbox_blocks_env_access(self):
        leak = ("import os\n"
                "def add(a, b):\n"
                "    return os.environ.get('OPENAI_API_KEY') or 'nokey'\n")
        result = self.runtime.verifier.run(leak, [{"call": "add(1, 1)", "expected": "nokey"}])
        self.assertTrue(result.solved)  # keys are not visible inside the sandbox

    def test_verifier_rejects_timeout(self):
        evil = "import time\ndef add(a, b):\n    time.sleep(30)\n    return a + b\n"
        self.runtime.verifier.timeout = 1.0
        result = self.runtime.verifier.run(evil, [{"call": "add(1, 1)", "expected": 2}])
        self.assertFalse(result.ok)


if __name__ == "__main__":
    unittest.main()
