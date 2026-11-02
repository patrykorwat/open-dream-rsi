"""Anytime-valid evolution stopping (arXiv:2610.04756 mapped to policy
evolution): the wealth process must be anytime-valid (every factor
positive), alarm on dead categories, never alarm on categories that keep
promoting, and freeze regeneration only after the alarm."""
import unittest

from open_dream_rsi.evolution import EvolutionStopper


class StopperTest(unittest.TestCase):
    def test_dead_category_alarms(self):
        st = EvolutionStopper()
        alarm = None
        for _ in range(20):
            d = st.observe("dead", promoted=False)
            if d["frozen"]:
                alarm = d
                break
        self.assertIsNotNone(alarm)
        # p0 floored at 0.5 -> f(miss) = 1.5; threshold 20 crossed in
        # 8 betting rounds after the 3-round calibration (1.5^8 = 25.6)
        self.assertEqual(alarm["alarm_round"], 11)
        self.assertAlmostEqual(alarm["wealth"], 1.5 ** 8, places=6)
        self.assertAlmostEqual(alarm["p0"], 0.5)

    def test_perfect_calibration_cap(self):
        # regression: all-ones calibration window must not set p0 = 1.0
        # (ZeroDivisionError on the first miss)
        st = EvolutionStopper()
        for _ in range(3):
            st.observe("hot", promoted=True)
        d = st.observe("hot", promoted=False)  # first miss after calibration
        self.assertAlmostEqual(d["p0"], 0.99)
        self.assertGreaterEqual(d["wealth"], 20.0)  # f(0) = 50.5 -> instant
        self.assertTrue(d["frozen"])

    def test_live_category_never_alarms(self):
        st = EvolutionStopper()
        frozen = False
        for r in range(30):
            d = st.observe("live", promoted=(r % 2 == 0))
            frozen = frozen or d["frozen"]
        self.assertFalse(frozen)

    def test_frozen_short_circuits_and_reset_reopens(self):
        st = EvolutionStopper()
        for _ in range(12):
            st.observe("dead", promoted=False)
        self.assertTrue(st.is_frozen("dead"))
        rounds_before = st._state["dead"]["rounds"]
        st.observe("dead", promoted=False)
        self.assertEqual(st._state["dead"]["rounds"], rounds_before)  # frozen: no count
        st.reset("dead")
        self.assertFalse(st.is_frozen("dead"))

    def test_min_rounds_respected(self):
        # a single huge miss before min_rounds must NOT alarm
        st = EvolutionStopper(min_rounds=4)
        for _ in range(3):
            st.observe("x", promoted=True)
        d = st.observe("x", promoted=False)   # wealth 50.5 >= 20 at round 4
        self.assertTrue(d["frozen"])          # round 4 >= min_rounds 4: OK
        st2 = EvolutionStopper(min_rounds=6)
        for _ in range(3):
            st2.observe("y", promoted=True)
        d2 = st2.observe("y", promoted=False)  # round 4 < min_rounds 6
        self.assertFalse(d2["frozen"])


class LoopWiringTest(unittest.TestCase):
    """enable_evolution_stop: regeneration calls stop after the alarm."""

    def test_regeneration_stops_after_alarm(self):
        import tempfile
        from open_dream_rsi.loop import AutoRSIRuntime, Task
        from open_dream_rsi.memory import DreamMemory

        class BrokenPolicyClient:
            """Valid task attempts never happen; policy gens always reject."""
            def __init__(self):
                self.policy_calls = 0

            def chat(self, messages, model=None, temperature=0.7, max_tokens=None):
                system = messages[0]["content"]
                if "exploration policy" in system:
                    self.policy_calls += 1
                    return "```python\ndef broken(:\n```"  # syntax error
                return "def nope():\n    pass"  # fails the task tests

        task = Task(task_id="dedupe_1", category="dedupe",
                    prompt="Implement dedupe(xs): remove duplicates, keep order.",
                    tests=[{"call": "dedupe([1, 1])", "expected": [1]}],
                    max_attempts=2)
        mem = DreamMemory(tempfile.mkdtemp())
        rt = AutoRSIRuntime(client=BrokenPolicyClient(), memory=mem, tasks=[task],
                            api_call_budget=24, dream_iterations=2, rng_seed=1,
                            fair_queue=False, enable_policy_code=True,
                            explore_epsilon=0.0,
                            enable_thoughts=False, enable_judge=False,
                            enable_evolution_stop=True)
        import json
        events_path = mem.root / "events.jsonl"
        alarm_seen = False
        for cycle in range(14):
            rt.api_calls_used = 0
            rt.run_once()
            events = [json.loads(l) for l in events_path.read_text().splitlines()]
            if any(e.get("kind") == "evolution_alarm" for e in events):
                alarm_seen = True
                last_alarm = max(i for i, e in enumerate(events)
                                 if e.get("kind") == "evolution_alarm")
                # after the alarm, no more policy-generation calls
                later = events[last_alarm:]
                self.assertFalse(
                    any(e.get("kind") == "llm_call"
                        and e.get("sub_kind") == "policy_gen" for e in later),
                    "policy generation continued after the alarm")
        self.assertTrue(alarm_seen, "no evolution_alarm event recorded")


if __name__ == "__main__":
    unittest.main()
