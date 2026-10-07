"""Structured Sentinel observation -> DiscoveryTree world metadata.

Covers commits 1+2 of the Sentinel-powered Dream-RSI integration:

* ``observe_structured`` mirrors ``observe`` on the SAME engine and the
  SAME ledger (identical counting/thresholds/persistence), and the text
  contract ``observe()`` keeps byte-identical behavior (vendored pin).
* ``to_world_dict()`` is the ONLY view allowed into a TreeNode, and it is
  replay-safe BY CONSTRUCTION: the cross-session counter (knowledge from
  other — possibly future — episodes) can never reach node state. This is
  the prefix-only replay invariant: a historical world must not contain
  facts from a future session (same contamination class as the
  lesson-gate first-failure snapshot rule).
* TreeNode persists/round-trips ``sentinel`` and ``termination_reason``
  through DreamMemory's trees/*.json; old trees without the keys load
  unchanged; the default runtime (no engine) produces None fields, so
  archived trees are byte-identical to pre-feature runs.
"""
import json
import tempfile
import unittest

from open_dream_rsi.core.tree import DiscoveryTree, TerminationReason, TreeNode
from open_dream_rsi.loop import AutoRSIRuntime, DreamMemory, Task
from open_dream_rsi.sentinel import SentinelEngine, SentinelObservation

FAIL = '{"error": "db locked"}'
OK = '{"ok": true}'


def engine(tmp, **kw):
    from pathlib import Path
    return SentinelEngine(state_path=Path(tempfile.mkdtemp()) / "s.json", **kw)


class TestObserveStructured(unittest.TestCase):
    def test_clean_result_is_data_not_prose(self):
        e = engine("x")
        o = e.observe_structured("web", OK, "S1")
        self.assertEqual(o.signature, "")
        self.assertIsNone(o.error_class)
        self.assertEqual(o.in_session_count, 0)
        self.assertEqual(o.cross_session_count, 0)
        self.assertFalse(o.recurring)

    def test_counts_track_the_same_ledger_as_observe(self):
        # Interleaving the two APIs must share one ledger: same signature,
        # counts continue across channels, note thresholds fire once.
        e = engine("x")
        o1 = e.observe_structured("t", FAIL, "S1")
        self.assertEqual(o1.in_session_count, 1)
        self.assertFalse(o1.recurring)          # intra=2 not yet crossed
        t2 = e.observe("t", FAIL, "S1")         # second hit via TEXT api
        self.assertIsNotNone(t2)                # note fires here
        o3 = e.observe_structured("t", FAIL, "S1")
        self.assertEqual(o3.in_session_count, 3)
        self.assertEqual(o3.cross_session_count, 3)
        self.assertFalse(o3.recurring)          # threshold already crossed
        self.assertEqual(o1.signature, o3.signature)

    def test_budget_facts_and_nudge_flag(self):
        e = engine("x", finalize_budget=10, finalize_at=0.8)
        for _ in range(7):
            e.observe_structured("t", OK, "S9")
        o = e.observe_structured("t", OK, "S9")
        self.assertTrue(o.finalize_nudge)       # 8th call crosses ceil(0.8*10)
        self.assertEqual(o.budget_used, 8)
        self.assertEqual(o.budget_total, 10)
        again = e.observe_structured("t", OK, "S9")
        self.assertFalse(again.finalize_nudge)  # at most once per session
        self.assertEqual(again.budget_used, 9)

    def test_note_cap_suppresses_text_not_facts(self):
        # The cap is an annotation-volume rule; the structured facts and the
        # durable count must survive it (the world is not a text channel).
        e = engine("x", max_notes_per_session=1)
        self.assertIsNone(e.observe("t", FAIL, "S1"))          # 1st hit: below intra=2
        self.assertIsNotNone(e.observe("t", FAIL, "S1"))       # note #1 -> cap reached
        self.assertIsNone(e.observe("t", '{"error": "other"}', "S1"))  # capped
        o = e.observe_structured("t", '{"error": "third"}', "S1")
        self.assertGreaterEqual(o.in_session_count, 1)
        self.assertTrue(o.signature)

    def test_blocked_passthrough(self):
        e = engine("x")
        o = e.observe_structured("t", FAIL, "S1", blocked=True)
        self.assertTrue(o.to_world_dict()["blocked"])


class TestWorldDictEpistemicSplit(unittest.TestCase):
    """THE leakage invariant: policygen/replay sees repeat_world, NEVER
    cross_session_count — even when the live ledger knows 17 sessions."""

    def test_cross_session_never_reaches_world_dict(self):
        e = engine("x")
        # 17 other sessions hit the same class (cross_session_count = 18 now)
        for i in range(17):
            e.observe("fetch", FAIL, f"S{i}")
        obs = e.observe_structured("fetch", FAIL, "S-live")
        self.assertEqual(obs.cross_session_count, 18)
        self.assertGreaterEqual(obs.cross_session_count, 17)
        w = obs.to_world_dict()
        self.assertEqual(w["repeat_world"], 1)          # episode-local only
        self.assertNotIn("cross_session_count", json.dumps(w))
        self.assertNotIn("sessions", json.dumps(w))
        # full view keeps it (live host / curator may use it)
        self.assertEqual(obs.to_dict()["cross_session_count"], 18)

    def test_world_dict_shape(self):
        obs = SentinelObservation(signature="abc", error_class="t|e",
                                  in_session_count=2, cross_session_count=9,
                                  recurring=True, finalize_nudge=False,
                                  budget_used=41, budget_total=50)
        w = obs.to_world_dict()
        self.assertEqual(w["signature"], "abc")
        self.assertEqual(w["repeat_world"], 2)
        self.assertTrue(w["recurring"])
        self.assertFalse(w["blocked"])
        self.assertAlmostEqual(w["budget_fraction"], 0.82)
        # frozen dataclass: observations are values, not mutable state
        with self.assertRaises(Exception):
            obs.in_session_count = 99  # type: ignore[misc]


class TestTreeNodeWorldMetadata(unittest.TestCase):
    def test_round_trip_through_dream_memory(self):
        mem = DreamMemory(tempfile.mkdtemp())
        tree = DiscoveryTree()
        tree.add_node("r", action="seed", result=None, score=0.0)
        tree.add_node("n1", action="fetch", result="x", score=0.4,
                      parent_id="r",
                      sentinel={"signature": "sig1", "repeat_world": 3,
                                "recurring": True, "blocked": False,
                                "budget_fraction": 0.82},
                      termination_reason=TerminationReason.BUDGET_GATE.value)
        mem.archive_tree("t1", tree)
        back = mem.load_tree("t1")
        self.assertIsNotNone(back)
        n = back.nodes["n1"]
        self.assertEqual(n.sentinel["repeat_world"], 3)
        self.assertEqual(n.termination_reason, "budget_gate")

    def test_legacy_trees_without_new_keys_load(self):
        mem = DreamMemory(tempfile.mkdtemp())
        path = mem.root / "trees" / "old.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "root_id": "r",
            "nodes": [{"node_id": "r", "action": "a", "result": None,
                       "score": 1.0, "parent_id": None, "children": [],
                       "thought": ""}],   # pre-feature serialized shape
        }), encoding="utf-8")
        back = mem.load_tree("old")
        self.assertIsNotNone(back)
        self.assertIsNone(back.nodes["r"].sentinel)
        self.assertIsNone(back.nodes["r"].termination_reason)

    def test_termination_reason_is_a_closed_vocabulary(self):
        values = {m.value for m in TerminationReason}
        self.assertEqual(values, {"completed", "policy_stop", "tool_failure",
                                  "budget_gate", "sentinel_block", "unknown"})
        # str enum: JSON-serializable and comparable to its bare value
        self.assertEqual(TerminationReason.BUDGET_GATE, "budget_gate")
        mem = DreamMemory(tempfile.mkdtemp())
        t = DiscoveryTree()
        t.add_node("r", "a", None, 0.0,
                   termination_reason=TerminationReason.SENTINEL_BLOCK)
        mem.archive_tree("t2", t)  # enum member must serialize
        self.assertEqual(json.loads(
            (mem.root / "trees" / "t2.json").read_text()
        )["nodes"][0]["termination_reason"], "sentinel_block")


class RecordingClient:
    """Fails the verifier twice, then succeeds — exercises the structured
    attach path and both termination labels in one run."""

    def __init__(self):
        self.n = 0

    def chat(self, messages, model=None, temperature=0.7, max_tokens=1024):
        self.n += 1
        if self.n < 3:
            return "```python\ndef median(xs):\n    return 0\n```"
        return ("```python\nfrom statistics import median as _m\n"
                "def median(xs):\n    return _m(xs)\n```")


class TestRuntimeWiring(unittest.TestCase):
    def _task(self):
        return [Task(task_id="t1", category="c", prompt="compute median",
                     tests=[{"call": "median([3,1,2])", "expected": 2}],
                     max_attempts=3)]

    def test_default_runtime_leaves_nodes_clean(self):
        mem = DreamMemory(tempfile.mkdtemp())
        rt = AutoRSIRuntime(client=RecordingClient(), memory=mem,
                            tasks=self._task(), api_call_budget=30,
                            max_tokens=128, enable_policy_code=False,
                            enable_thoughts=False)
        rt.run_once()
        tree = mem.load_tree("t1")
        for n in tree.nodes.values():
            self.assertIsNone(n.sentinel)
            self.assertIsNone(n.termination_reason)

    def test_wired_runtime_records_facts_and_stop_cause(self):
        mem = DreamMemory(tempfile.mkdtemp())
        sent = SentinelEngine(state_path=mem.root / "sentinel.json")
        rt = AutoRSIRuntime(client=RecordingClient(), memory=mem,
                            tasks=self._task(), api_call_budget=30,
                            max_tokens=128, enable_policy_code=False,
                            enable_thoughts=False, sentinel_engine=sent)
        rt.run_once()
        tree = mem.load_tree("t1")
        interior = [n for n in tree.nodes.values()
                    if n.action.startswith("write_code") and not n.children]
        self.assertTrue(interior)
        last = interior[-1]
        self.assertEqual(last.termination_reason, "completed")  # client solves
        failing = [n for n in tree.nodes.values()
                   if isinstance(n.result, dict) and n.result.get("errors")
                   and n.sentinel]
        self.assertTrue(failing)
        # same failing class on consecutive attempts -> repeat_world grows,
        # and the world view carries NO cross-session knowledge
        reps = [n.sentinel["repeat_world"] for n in failing]
        self.assertGreater(max(reps), 1)
        for n in tree.nodes.values():
            if n.sentinel:
                self.assertNotIn("cross_session_count", json.dumps(n.sentinel))
                self.assertNotIn("sessions", json.dumps(n.sentinel))


if __name__ == "__main__":
    unittest.main()
