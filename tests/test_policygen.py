"""Tests for the section-3 policy generator: gate, sandbox, replay, loop wiring.

Run:  python -m unittest discover -s tests
"""

import json
import unittest
from pathlib import Path

from open_dream_rsi.core.policygen import (
    PolicyGenerator,
    PolicySandbox,
    PolicyValidationError,
    evaluate_policy,
    extract_python_block,
    greedy_replay_score,
    replay_world,
    validate_policy_source,
)
from open_dream_rsi.core.tree import DiscoveryTree
from open_dream_rsi.loop import AutoRSIRuntime, Task
from open_dream_rsi.memory import DreamMemory

BEST_POLICY = """
def choose_action(frontier, step):
    if not frontier:
        return None
    ranked = sorted(frontier, key=lambda n: (n["score"], n["children"]), reverse=True)
    return ranked[0]["node_id"]
"""

GOOD_POLICY = BEST_POLICY  # sandbox tests only need a valid contract-compliant policy


def build_tree():
    """A small loop-shaped tree: root + scored leaves in insertion order."""
    tree = DiscoveryTree()
    root = tree.add_node("root", action="seed", result={}, score=0.0)
    scores = [0.1, 0.4, 0.2, 0.6, 0.3]
    prev = root.node_id
    for i, s in enumerate(scores):
        node = tree.add_node(f"n{i}", action=f"write_code_v{i}", result={},
                             score=s, parent_id=prev)
        prev = node.node_id
    return tree


class ValidationTest(unittest.TestCase):
    def test_accepts_contract_compliant_code(self):
        validate_policy_source(GOOD_POLICY)  # must not raise

    def test_rejects_missing_entry_point(self):
        with self.assertRaises(PolicyValidationError):
            validate_policy_source("def other(frontier, step):\n    return None\n")

    def test_rejects_import_and_dangerous_names(self):
        for src in (
            "import os\ndef choose_action(f, s):\n    return None\n",
            "def choose_action(f, s):\n    return open('x').read()\n",
            "def choose_action(f, s):\n    return f.__class__\n",
            "def choose_action(f, s):\n    x = eval('1')\n    return None\n",
        ):
            with self.assertRaises(PolicyValidationError):
                validate_policy_source(src)

    def test_extract_python_block(self):
        self.assertEqual(extract_python_block("```python\nx=1\n```"), "x=1")
        self.assertEqual(extract_python_block("plain"), "plain")
        self.assertIsNone(extract_python_block("```   ```"))


class SandboxTest(unittest.TestCase):
    def setUp(self):
        self.sandbox = PolicySandbox(timeout=5.0)
        self.frontier = [{"node_id": "a", "action": "x", "score": 0.2, "children": 0},
                         {"node_id": "b", "action": "y", "score": 0.9, "children": 0}]

    def test_good_policy_choice(self):
        run = self.sandbox.choose(GOOD_POLICY, self.frontier, 0)
        self.assertTrue(run.ok)
        self.assertEqual(run.choice, "b")

    def test_invalid_code_never_executes(self):
        run = self.sandbox.choose("import os\ndef choose_action(f, s):\n    return None\n",
                                  self.frontier, 0)
        self.assertFalse(run.ok)
        self.assertIn("validation", run.error or "")

    def test_crashing_policy_reports_error(self):
        run = self.sandbox.choose("def choose_action(f, s):\n    return 1 / 0\n",
                                  self.frontier, 0)
        self.assertFalse(run.ok)

    def test_hanging_policy_times_out(self):
        hang = ("def choose_action(f, s):\n"
                "    total = 0\n"
                "    for i in range(10**12):\n"
                "        total += i\n"
                "    return f[0]['node_id'] if f else None\n")
        sandbox = PolicySandbox(timeout=1.0)
        run = sandbox.choose(hang, self.frontier, 0)
        self.assertFalse(run.ok)
        self.assertIn("timed out", run.error or "")

    def test_dict_return_is_accepted(self):
        run = self.sandbox.choose(
            "def choose_action(f, s):\n    return {'node_id': 'a'} if f else None\n",
            self.frontier, 0)
        self.assertEqual(run.choice, "a")


class ReplayTest(unittest.TestCase):
    def setUp(self):
        self.tree = build_tree()
        self.world = replay_world(self.tree)
        self.sandbox = PolicySandbox()

    def test_replay_world_shape(self):
        self.assertEqual(len(self.world), len(self.tree.nodes) - 1)
        for frontier, chosen in self.world:
            self.assertIn(chosen, {n["node_id"] for n in frontier})

    def test_greedy_baseline_scores_reasonably(self):
        score = greedy_replay_score(self.world)
        self.assertGreater(score, 0.0)

    def test_evaluate_policy_beats_or_matches_greedy(self):
        score, err = evaluate_policy(self.sandbox, GOOD_POLICY, self.world)
        self.assertEqual(err, "")
        self.assertGreaterEqual(score, greedy_replay_score(self.world) - 1e-9)

    def test_empty_world_is_rejected(self):
        score, err = evaluate_policy(self.sandbox, GOOD_POLICY, [])
        self.assertEqual(score, float("-inf"))
        self.assertIn("empty", err)


class ScriptedPolicyClient:
    """Returns the given policy for every policy-generation call."""

    def __init__(self, source):
        self.source = source
        self.policy_calls = 0

    def chat(self, messages, model=None, temperature=0.7, max_tokens=None):
        self.policy_calls += 1
        return f"```python\n{self.source}\n```"


class GeneratorGateTest(unittest.TestCase):
    def setUp(self):
        self.tree = build_tree()

    def _generate(self, source):
        gen = PolicyGenerator(ScriptedPolicyClient(source), max_tokens=None)
        world_baseline = greedy_replay_score(replay_world(self.tree))
        return gen.generate("math", None, world_baseline, self.tree)

    def test_beating_incumbent_is_promoted(self):
        result = self._generate(GOOD_POLICY)
        self.assertIsNotNone(result.source)
        self.assertEqual(result.error, "")

    def test_worse_than_incumbent_is_rejected(self):
        worst = ("def choose_action(frontier, step):\n"
                 "    if not frontier:\n        return None\n"
                 "    return min(frontier, key=lambda n: n['score'])['node_id']\n")
        result = self._generate(worst)
        self.assertIsNone(result.source)
        self.assertIn("replay score", result.error)

    def test_invalid_code_never_promotes(self):
        result = self._generate("def broken(:\n    pass\n")
        self.assertIsNone(result.source)


class ScriptedLLM:
    """Task solver + policy writer in one deterministic client."""

    def __init__(self, policy_source=GOOD_POLICY):
        self.calls = 0
        self.policy_source = policy_source

    def chat(self, messages, model=None, temperature=0.7, max_tokens=1024):
        self.calls += 1
        system = messages[0]["content"] if messages else ""
        if "exploration policy" in system:
            return f"```python\n{self.policy_source}\n```"
        prompt = messages[-1]["content"]
        code = "def add(a, b):\n    return a - b\n" if "feedback:\n(none)" in prompt \
            else "def add(a, b):\n    return a + b\n"
        return f"```python\n{code}```"


class LoopPolicyIntegrationTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.memory = DreamMemory(root=f"{tempfile.mkdtemp(prefix='odr-pg-')}/mem")
        self.llm = ScriptedLLM()
        self.runtime = AutoRSIRuntime(
            client=self.llm, memory=self.memory,
            tasks=[Task(task_id="add1", category="math",
                        prompt="Implement add(a, b).",
                        tests=[{"call": "add(2, 3)", "expected": 5}],
                        max_attempts=3)],
            api_call_budget=20, dream_iterations=10,
        )

    def test_cycle_promotes_policy_and_persists_it(self):
        self.runtime.run_once()
        entry = self.memory.get_policy_code("math")
        self.assertIsNotNone(entry)
        code = (entry or {}).get("code", "")
        self.assertIn("def choose_action", code)
        self.assertGreater((entry or {}).get("score", 0.0), 0.0)
        on_disk = json.loads((Path(self.memory.root) / "policy_codes.json").read_text())
        self.assertIn("math", on_disk)

    def test_promoted_policy_steers_expansion(self):
        self.runtime.run_once()
        tree = build_tree()
        chosen = self.runtime._next_expansion(tree, "math")
        self.assertIn(chosen, {n.node_id for n in tree.nodes.values()})

    def test_broken_policy_falls_back_to_greedy(self):
        self.memory.save_policy_code("math", "def choose_action(f, s):\n    return 1/0\n", 1.0)
        tree = build_tree()
        chosen = self.runtime._next_expansion(tree, "math")
        best = max(self.runtime._frontier(tree), key=lambda n: n["score"])["node_id"]
        self.assertEqual(chosen, best)  # crash -> greedy baseline, loop survives

    def test_disable_flag_skips_policy_calls(self):
        self.runtime.enable_policy_code = False
        self.runtime.run_once()
        self.assertIsNone(self.memory.get_policy_code("math"))


if __name__ == "__main__":
    unittest.main()
