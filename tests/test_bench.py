"""Benchmark: dreaming loop must beat the cold baseline on API cost."""

import unittest

from open_dream_rsi.bench import run_benchmark


class BenchTest(unittest.TestCase):
    def test_dream_arm_beats_cold_baseline(self):
        s = run_benchmark(cycles=4, provider="mock", dream_iterations=40)
        cold = s["arms"][0]
        dream = s["arms"][1]
        # scripted model makes both arms reach full solve capability
        self.assertEqual(dream["solves_total"], 4 * s["tasks"])
        self.assertEqual(cold["solves_total"], 4 * s["tasks"])
        # dreaming + persistent memory => strictly fewer API calls
        self.assertLess(dream["api_calls_total"], cold["api_calls_total"])
        self.assertGreater(s["api_call_saving"], 0.0)


if __name__ == "__main__":
    unittest.main()
