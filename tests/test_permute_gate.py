"""Permute-gate unit tests: choice-removal semantics, off by default.

The gate is the mechanism that moved TravelPlanner delivery 23%->94%
(paper sec. tp); its trigger math and refuse-forever behaviour are pinned
at the class level, and the off-by-default rule is pinned too (arms opt in
via SENTINEL_GATE_BUDGET only).
"""
import unittest

from open_dream_rsi.mcp_server import PermuteGate, REFUSAL, result_is_error


class PermuteGateTest(unittest.TestCase):
    def test_off_by_default(self):
        g = PermuteGate(0)
        for _ in range(100):
            g.count("s")
        self.assertFalse(g.blocks("s"))

    def test_trigger_is_ceil_at_times_budget(self):
        self.assertEqual(PermuteGate(45, 0.8).trigger, 36)
        self.assertEqual(PermuteGate(5, 0.8).trigger, 4)
        self.assertEqual(PermuteGate(7, 0.8).trigger, 6)  # ceil(5.6)

    def test_serves_then_refuses_permanently(self):
        g = PermuteGate(5, 0.8)  # trigger 4
        for _ in range(3):
            g.count("a")
            self.assertFalse(g.blocks("a"), "must serve before trigger")
        for _ in range(10):
            g.count("a")
            self.assertTrue(g.blocks("a"), "must refuse at/after trigger")

    def test_sessions_isolated(self):
        g = PermuteGate(2, 0.8)  # trigger 2
        g.count("a")
        g.count("a")
        self.assertTrue(g.blocks("a"))
        self.assertFalse(g.blocks("b"), "gate must be per-session")

    def test_refusal_is_error_shaped(self):
        text = '{"blad": "budget_exhausted: ..."}'
        self.assertTrue(result_is_error(text))
        self.assertIn("budget_exhausted", REFUSAL["blad"])
        self.assertIn("answer now", REFUSAL["blad"].lower())


if __name__ == "__main__":
    unittest.main()
