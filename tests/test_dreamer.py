"""Tests for the offline DreamEngine optimiser."""

import unittest

from open_dream_rsi.core.dreamer import (
    DEPTH_MAX,
    DEPTH_MIN,
    TEMP_MAX,
    TEMP_MIN,
    DreamEngine,
)
from open_dream_rsi.core.simulator import ReplaySimulator
from open_dream_rsi.core.tree import DiscoveryTree


def tree_with(n=10):
    t = DiscoveryTree()
    t.add_node("root", action="explore", result="r", score=1.0)
    for i in range(n):
        t.add_node(f"n{i}", action="explore", result=i, score=(i % 5) + 1.0, parent_id="root")
    return t


class DreamEngineTest(unittest.TestCase):
    def test_long_run_does_not_runaway(self):
        """Regression: raw score*temperature used to drift to the ceiling forever."""
        engine = DreamEngine(simulator=ReplaySimulator(tree_with()))
        for _ in range(30):                      # 30 successive dream rounds
            engine.run_offline_optimization(iterations=60)
        p = engine.policy_parameters
        self.assertLessEqual(p["temperature"], TEMP_MAX)
        self.assertGreaterEqual(p["temperature"], TEMP_MIN)
        self.assertLessEqual(p["exploration_depth"], DEPTH_MAX)
        self.assertGreaterEqual(p["exploration_depth"], DEPTH_MIN)
        # interior optimum expected: not glued to the ceiling after 1800 its
        self.assertLess(p["temperature"], 1.2, f"temperature drifted to {p['temperature']}")

    def test_policy_deterministically_improves(self):
        tree = tree_with()
        engine = DreamEngine(simulator=ReplaySimulator(tree))
        start = dict(engine.policy_parameters)
        s0 = engine._evaluate_policy_in_dream(start)
        best = engine.run_offline_optimization(iterations=120)
        self.assertGreaterEqual(engine._evaluate_policy_in_dream(best), s0)

    # -- issue #1 regression: both parameters must move the objective ----------

    def test_exploration_depth_causally_changes_score(self):
        """The objective simulates the policy's choices, so the re-polish cap
        (exploration_depth) must change the score on a branching tree — it
        was a dead parameter when the objective was a fixed function of T."""
        engine = DreamEngine(simulator=ReplaySimulator(tree_with()))
        low = engine._evaluate_policy_in_dream(
            {"temperature": 0.7, "exploration_depth": DEPTH_MIN})
        high = engine._evaluate_policy_in_dream(
            {"temperature": 0.7, "exploration_depth": DEPTH_MAX})
        self.assertNotAlmostEqual(low, high, places=3)

    def test_temperature_causally_changes_score(self):
        engine = DreamEngine(simulator=ReplaySimulator(tree_with()))
        cold = engine._evaluate_policy_in_dream(
            {"temperature": TEMP_MIN, "exploration_depth": 3.0})
        hot = engine._evaluate_policy_in_dream(
            {"temperature": TEMP_MAX, "exploration_depth": 3.0})
        self.assertNotAlmostEqual(cold, hot, places=3)

    def test_score_depends_on_the_recorded_tree_not_only_on_parameters(self):
        """A fixed function of (T, depth) — the old objective — scores every
        tree with the same n nodes identically. The rollout objective must
        rank the trees differently when scores differ."""
        good = DiscoveryTree()
        good.add_node("root", action="explore", result="r", score=1.0)
        bad = DiscoveryTree()
        bad.add_node("root", action="explore", result="r", score=1.0)
        for i in range(8):
            good.add_node(f"g{i}", action="explore", result=i, score=5.0, parent_id="root")
            bad.add_node(f"b{i}", action="explore", result=i, score=0.1, parent_id="root")
        engine = DreamEngine(simulator=ReplaySimulator(good))
        params = {"temperature": 0.7, "exploration_depth": 3.0}
        s_good = engine._evaluate_policy_in_dream(params)
        engine.simulator = ReplaySimulator(bad)
        s_bad = engine._evaluate_policy_in_dream(params)
        self.assertGreater(s_good, s_bad)

    def test_dream_run_is_deterministic(self):
        r1 = DreamEngine(simulator=ReplaySimulator(tree_with())).run_offline_optimization(60)
        r2 = DreamEngine(simulator=ReplaySimulator(tree_with())).run_offline_optimization(60)
        self.assertEqual(r1, r2)


if __name__ == "__main__":
    unittest.main()
