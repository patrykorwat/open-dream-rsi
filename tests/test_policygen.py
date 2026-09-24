"""Tests for the section-3 policy generator: gate, sandbox, rollout, loop wiring.

Run:  python -m unittest discover -s tests
"""

import json
import unittest
from pathlib import Path

from open_dream_rsi.core.policygen import (
    PolicyGenerator,
    PolicySandbox,
    PolicyValidationError,
    _simulate_step_rewards,
    evaluate_policy,
    extract_python_block,
    greedy_rollout_score,
    rollout_score,
    rollout_world_payload,
    validate_policy_source,
)
from open_dream_rsi.core.tree import DiscoveryTree
from open_dream_rsi.loop import AutoRSIRuntime, Task
from open_dream_rsi.memory import DreamMemory

BEST_POLICY = """
def choose_action(frontier, step):
    if not frontier:
        return None
    ranked = sorted(frontier, key=lambda n: (n["children"] == 0, n["outcome"],
                                             n["score"]), reverse=True)
    return ranked[0]["node_id"]
"""

GOOD_POLICY = BEST_POLICY  # sandbox tests only need a valid contract-compliant policy


def build_tree():
    """Loop-shaped tree: seed -> decoy chain (.667) with a .333 branch that
    leads to the fix (1.0) one expansion deeper — the decoy-trap world."""
    tree = DiscoveryTree()
    seed = tree.add_node("seed", action="warm_start", result={}, score=0.0)
    decoy = tree.add_node("d0", action="write_code:decoy", result={
        "errors": ["t2 -> got 2 != 3"]}, score=0.667, parent_id=seed.node_id)
    prev = decoy.node_id
    for i in range(3):  # decoy polish chain — re-expanding never improves
        node = tree.add_node(f"d{i+1}", action=f"write_code:decoy{i}", result={
            "errors": ["t2 -> got 2 != 3"]}, score=0.667, parent_id=prev)
        prev = node.node_id
    prom = tree.add_node("p0", action="write_code:promising", result={
        "errors": ["t0 -> got [2] != [1,2]", "t1 -> got [] != [3]"]},
        score=0.333, parent_id=seed.node_id)
    tree.add_node("fix", action="write_code:fixed", result={}, score=1.0,
                  parent_id=prom.node_id)
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
        self.frontier = [{"node_id": "a", "action": "x", "score": 0.2,
                          "parent_id": None, "children": 0, "outcome": 0.2,
                          "errors": []},
                         {"node_id": "b", "action": "y", "score": 0.9,
                          "parent_id": None, "children": 0, "outcome": 0.9,
                          "errors": []}]

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


class RolloutTest(unittest.TestCase):
    def setUp(self):
        self.tree = build_tree()
        self.sandbox = PolicySandbox()

    def test_empty_tree_rejected(self):
        score, err = rollout_score(self.sandbox, GOOD_POLICY, DiscoveryTree())
        self.assertEqual(score, float("-inf"))
        self.assertIn("empty", err)

    def test_greedy_falls_into_the_decoy_trap(self):
        # greedy sees the .667 decoy forever; the fix (.333 child) stays hidden
        score = greedy_rollout_score(self.tree)
        self.assertLess(score, 0.75)

    def test_explorer_policy_beats_greedy_on_the_trap(self):
        # prefers unexpanded nodes ranked by back-propagated outcome:
        # opens the seed's low-score branch that leads to the fix
        explorer = """
def choose_action(frontier, step):
    if not frontier:
        return None
    fresh = [n for n in frontier if n['children'] == 0]
    pool = fresh or frontier
    return max(pool, key=lambda n: n['outcome'])['node_id']
"""
        score, err = rollout_score(self.sandbox, explorer, self.tree)
        self.assertEqual(err, "")
        self.assertGreater(score, greedy_rollout_score(self.tree))

    def test_crashing_policy_scores_minus_inf(self):
        score, err = rollout_score(self.sandbox,
                                   "def choose_action(f, s):\n    return 1/0\n",
                                   self.tree)
        self.assertEqual(score, float("-inf"))

    def test_evaluate_policy_alias(self):
        score, err = evaluate_policy(self.sandbox, GOOD_POLICY, self.tree)
        self.assertEqual(err, "")
        self.assertGreaterEqual(score, greedy_rollout_score(self.tree) - 1e-9)

    # -- issue #1 regression: prefix-only rollout --------------------------------

    def test_replay_never_leaks_future_scores(self):
        """root -> child(0.1) -> future(1.0): before the path reaches 'future',
        the policy must not see its score, and camping on root must not get
        paid for it (issue #1 reproduction)."""
        tree = DiscoveryTree()
        tree.add_node("root", "seed", {}, 0.0)
        tree.add_node("child", "attempt", {}, 0.1, "root")
        tree.add_node("future", "attempt", {}, 1.0, "child")
        world = rollout_world_payload(tree, horizon=3)
        # the serialized world carries no full-tree futures
        self.assertEqual({n["node_id"]: n["outcome"] for n in world["nodes"]},
                         {"root": 0.0, "child": 0.1, "future": 1.0})
        seen = []
        def camp_root(visible, step):
            seen.append({n["node_id"]: n["outcome"] for n in visible})
            return "root"
        rewards, picks, invalid, err = _simulate_step_rewards(world, camp_root)
        self.assertEqual(err, None)
        # step 0: only root is visible and its outcome is its OWN score —
        # neither the unrevealed child nor the future grandchild
        self.assertEqual(seen[0], {"root": 0.0})
        # camping pays the recorded child ladder (0.1) then the last recorded
        # child — never the 1.0 the real path only found under 'child'
        self.assertEqual(rewards, [0.1, 0.1, 0.1])

    def test_exhausted_branch_pays_last_recorded_child_not_best_descendant(self):
        """Seed of the trap tree: after its recorded ladder (decoy .667,
        promising .333) is exhausted, camping it pays the LAST recorded
        child's score — never the back-propagated best descendant (1.0)."""
        tree = build_tree()
        world = rollout_world_payload(tree, horizon=12)
        rewards, picks, invalid, err = _simulate_step_rewards(
            world, lambda visible, step: "seed")
        self.assertEqual(err, None)
        self.assertEqual(rewards[0], 0.667)  # first recorded child (decoy)
        self.assertEqual(rewards[1], 0.333)  # second recorded child (promising)
        self.assertTrue(all(abs(r - 0.333) < 1e-9 for r in rewards[2:]),
                        f"exhausted ladder leaked best-descendant credit: {rewards}")

    def test_outcome_becomes_visible_only_after_revealing(self):
        """The as-of-now outcome rises exactly when the counterfactual path
        opens the branch that contains the better score."""
        tree = build_tree()
        world = rollout_world_payload(tree, horizon=8)
        outcomes_at = []
        def open_seed_twice(visible, step):
            vis = {n["node_id"]: n["outcome"] for n in visible}
            outcomes_at.append(vis)
            return "seed"
        _simulate_step_rewards(world, open_seed_twice)
        self.assertEqual(outcomes_at[0].get("seed"), 0.0)          # nothing revealed
        self.assertAlmostEqual(outcomes_at[1]["seed"], 0.667, places=3)  # decoy revealed
        self.assertAlmostEqual(outcomes_at[2]["seed"], 0.667, places=3)  # promising revealed; fix still hidden


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

    def _generate(self, source, incumbent_score=None):
        gen = PolicyGenerator(ScriptedPolicyClient(source), max_tokens=None)
        baseline = (greedy_rollout_score(self.tree)
                    if incumbent_score is None else incumbent_score)
        return gen.generate("math", None, baseline, self.tree)

    def test_beating_incumbent_is_promoted(self):
        explorer = """
def choose_action(frontier, step):
    if not frontier:
        return None
    fresh = [n for n in frontier if n['children'] == 0]
    pool = fresh or frontier
    return max(pool, key=lambda n: n['outcome'])['node_id']
"""
        result = self._generate(explorer)
        self.assertIsNotNone(result.source)
        self.assertEqual(result.error, "")

    def test_worse_than_incumbent_is_rejected(self):
        # incumbent set above anything reachable on this tree (a strong
        # remembered policy) — anything short of it must not promote
        worst = ("def choose_action(frontier, step):\n"
                 "    if not frontier:\n        return None\n"
                 "    return min(frontier, key=lambda n: n['outcome'])['node_id']\n")
        result = self._generate(worst, incumbent_score=0.99)
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
            api_call_budget=20, dream_iterations=10, rng_seed=42,
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
