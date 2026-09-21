"""Decoy-trap policy benchmark: the headline "the policy works" evidence.

These tests are the *guard rails* for the README figure: if the trap suite
stops separating the arms (greedy must never solve, the replay-gated
LLM-written policy must clearly beat both baselines), the figure's claim is
invalid and the benchmark has rotted.

Run:  python -m unittest discover -s tests
"""

import unittest

from open_dream_rsi.bench_policy import (
    ARMS,
    TRAP_SUITES,
    TrapSolver,
    run_policy_arm,
    to_svg,
)
from open_dream_rsi.tools import CodeVerifier


class SuiteCalibrationTest(unittest.TestCase):
    """The four code families must have exactly the intended verifier scores."""

    def test_family_scores(self):
        v = CodeVerifier()
        for s in TRAP_SUITES:
            with self.subTest(category=s.category):
                self.assertAlmostEqual(v.run(s.decoy, s.tests).score, 2 / 3, places=3)
                self.assertAlmostEqual(v.run(s.promising, s.tests).score, 1 / 3, places=3)
                self.assertEqual(v.run(s.fix, s.tests).score, 1.0)
                self.assertEqual(v.run(s.low, s.tests).score, 0.0)

    def test_decoy_is_plausible_not_trivial(self):
        # a 0.667 decoy that passes the SAME test as the low family is boring;
        # decoys must pass a test the promising family fails
        v = CodeVerifier()
        for s in TRAP_SUITES:
            d = v.run(s.decoy, s.tests)
            p = v.run(s.promising, s.tests)
            self.assertNotEqual(sorted(str(x) for x in d.detail),
                                sorted(str(x) for x in p.detail),
                                f"{s.category}: decoy and promising fail identically")


class TrapSolverTest(unittest.TestCase):
    def test_fix_only_via_promising_branch(self):
        solver = TrapSolver()
        suite = TRAP_SUITES[0]
        def ask(branch: str) -> str:
            reply = solver.chat([
                {"role": "system", "content": "code-improvement agent"},
                {"role": "user", "content": (
                    f"Task [{suite.category}]: do it\n\n"
                    f"Your best previous solution (may be absent):\n(none yet)\n\n"
                    f"Branch you are expanding from (code already tried on this branch, may be absent):\n{branch}\n\n"
                    f"Policy hints:\n{{}}\n\nPrevious failure feedback:\n(none)\n")},
            ])
            return reply.split("```")[1].replace("python", "").strip()
        # seed ladder: decoy, promising, decoy
        self.assertEqual(ask("(fresh branch)"), suite.decoy.strip())
        self.assertEqual(ask("(fresh branch)"), suite.promising.strip())
        # promising branch converges to the fix
        self.assertEqual(ask(suite.promising), suite.fix.strip())
        # decoy polishing decays to the low family — never the fix
        self.assertEqual(ask(suite.decoy), suite.decoy.strip())
        self.assertEqual(ask(suite.decoy), suite.low.strip())

    def test_recipe_warm_start_returns_fix(self):
        solver = TrapSolver()
        suite = TRAP_SUITES[1]
        reply = solver.chat([
            {"role": "system", "content": "code-improvement agent"},
            {"role": "user", "content": f"Task [{suite.category}]:\n{suite.fix}\n"},
        ])
        self.assertIn("```", reply)
        self.assertIn("def ", reply)
        self.assertEqual(CodeVerifier().run(
            reply.split("```")[1].replace("python", "").strip(), suite.tests).score, 1.0)

    def test_policy_call_returns_candidate_program(self):
        solver = TrapSolver()
        reply = solver.chat([
            {"role": "system", "content": "You are writing an exploration policy."},
            {"role": "user", "content": "ignore"},
        ])
        self.assertIn("def choose_action", reply)
        self.assertEqual(solver.policy_calls, 1)


class ArmsTest(unittest.TestCase):
    """Behavioural contract behind the README figure (few cycles for runtime)."""

    CYCLES = 6
    BUDGET = 24
    SEEDS = (7, 11, 23)

    def test_greedy_arm_never_escapes_the_trap(self):
        r = run_policy_arm("greedy", ARMS["greedy"], self.CYCLES, self.BUDGET,
                           seeds=self.SEEDS)
        self.assertEqual(r.solves_total, 0)
        self.assertGreater(r.api_calls_total, 0)   # it burns budget, fruitlessly
        self.assertEqual(r.policy_calls, 0)

    def test_replay_gated_policy_beats_both_baselines(self):
        slots = self.CYCLES * len(TRAP_SUITES) * len(self.SEEDS)
        eps = run_policy_arm("epsilon_greedy", ARMS["epsilon_greedy"],
                             self.CYCLES, self.BUDGET, seeds=self.SEEDS)
        pol = run_policy_arm("evolved_policy", ARMS["evolved_policy"],
                             self.CYCLES, self.BUDGET, seeds=self.SEEDS)
        # epsilon alone only solves a minority of the traps...
        self.assertLess(eps.solves_total, slots)
        # ...while promoted LLM-written policies solve EVERY category by the
        # final cycle (persistent recipes) at FEWER total calls than epsilon
        # wasted chasing luck
        self.assertEqual(pol.mean_solve_rate_by_cycle[-1], 100.0)
        self.assertGreater(pol.solves_total, 1.5 * eps.solves_total)
        self.assertLess(pol.api_calls_total, eps.api_calls_total)
        self.assertGreater(pol.policy_calls, 0)     # policies really were requested
        # solve rate must be non-decreasing (memory monotonicity)
        self.assertEqual(pol.mean_solve_rate_by_cycle,
                         sorted(pol.mean_solve_rate_by_cycle))

    def test_svg_renders_with_three_arms(self):
        from open_dream_rsi.bench_policy import run_policy_benchmark
        s = run_policy_benchmark(cycles=2, budget=24, seeds=(7,))
        svg = to_svg(s)
        self.assertTrue(svg.startswith("<svg"))
        for arm in ARMS:
            self.assertIn(arm, svg)


if __name__ == "__main__":
    unittest.main()
