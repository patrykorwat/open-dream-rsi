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


if __name__ == "__main__":
    unittest.main()
