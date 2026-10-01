"""Offline gate-replay validator: joins arms, applies the REAL gate rule.

The validator must (1) reproduce lesson_gate_verdict decisions on synthetic
pairs, (2) tolerate the record shapes real replay harnesses emit (top-level
list, single-key object, explicit records_key), and (3) report join stats
honestly instead of silently comparing different task sets.
"""
import json
import os
import tempfile
import unittest

from open_dream_rsi.gate_replay import (
    arm_pairs,
    filter_rows,
    gate_replay,
    load_arm_rows,
    to_markdown,
)


def rows(pairs, key="task_id"):
    return [{key: f"t{i}", "solved": w} for i, (w, _) in enumerate(pairs)]


class LoadArmRowsTests(unittest.TestCase):
    def _write(self, obj):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(path, "w") as f:
            json.dump(obj, f)
        self.addCleanup(os.unlink, path)
        return path

    def test_top_level_list(self):
        p = self._write([{"task_id": "a", "solved": True}])
        self.assertEqual(load_arm_rows(p), [{"task_id": "a", "solved": True}])

    def test_single_list_auto_detected(self):
        p = self._write({"warm": [{"task_id": "a", "solved": False}],
                        "meta": "ignored"})
        self.assertEqual(len(load_arm_rows(p)), 1)

    def test_ambiguous_without_key_raises(self):
        p = self._write({"warm": [{"task_id": "a"}], "cold": [{"task_id": "b"}]})
        with self.assertRaises(ValueError):
            load_arm_rows(p)

    def test_explicit_records_key(self):
        p = self._write({"warm": [{"task_id": "a"}], "cold": [{"task_id": "b"}]})
        self.assertEqual(load_arm_rows(p, "cold"), [{"task_id": "b"}])

    def test_missing_records_key_raises(self):
        p = self._write({"warm": []})
        with self.assertRaises(ValueError):
            load_arm_rows(p, "nope")


class ArmPairsTests(unittest.TestCase):
    def test_pair_order_is_with_without(self):
        base = [{"task_id": "a", "solved": False}]
        arm = [{"task_id": "a", "solved": True}]
        pairs, stats = arm_pairs(base, arm)
        self.assertEqual(pairs, [(True, False)])  # (with, without)
        self.assertEqual(stats["joined"], 1)

    def test_unmatched_rows_counted_not_silently_dropped(self):
        base = [{"task_id": k, "solved": True} for k in "abc"]
        arm = [{"task_id": k, "solved": True} for k in "ab"] + [
            {"task_id": "zz", "solved": True}]
        pairs, stats = arm_pairs(base, arm)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(stats["skipped_unmatched"], 2)

    def test_duplicate_keys_dropped_and_counted(self):
        base = [{"task_id": "a", "solved": True}, {"task_id": "a", "solved": False}]
        arm = [{"task_id": "a", "solved": False}]
        pairs, stats = arm_pairs(base, arm)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(stats["duplicate_keys"], 1)

    def test_custom_key_and_solved_field(self):
        base = [{"krs": "001", "ok": 0}]
        arm = [{"krs": "001", "ok": 1}]
        pairs, _ = arm_pairs(base, arm, key="krs", solved_field="ok")
        self.assertEqual(pairs, [(True, False)])

    def test_filter_rows_splits_multi_arm_file(self):
        rows = [{"label": "cold", "task_id": "a", "solved": True},
                {"label": "warm", "task_id": "a", "solved": False},
                {"label": "warm", "task_id": "b", "solved": True}]
        cold = filter_rows(rows, ["label=cold"])
        warm = filter_rows(rows, ["label=warm"])
        self.assertEqual(len(cold), 1)
        self.assertEqual(len(warm), 2)
        self.assertEqual(filter_rows(rows), rows)
        with self.assertRaises(ValueError):
            filter_rows(rows, ["nope"])


class GateReplayTests(unittest.TestCase):
    def test_harmful_arm_rejected_matches_gate_rule(self):
        # baseline solves 3/4; arm loses every solve and gains nothing -> reject
        base = [{"task_id": f"t{i}", "solved": i != 3} for i in range(4)]
        arm = [{"task_id": f"t{i}", "solved": False} for i in range(4)]
        report = gate_replay(base, {"lessons": arm})
        d = report["arms"]["lessons"]
        self.assertFalse(d["promoted"])
        self.assertEqual(d["regressions"], 3)
        self.assertLessEqual(d["net"], 0)

    def test_helpful_arm_promoted(self):
        base = [{"task_id": f"t{i}", "solved": i > 0} for i in range(4)]  # t0 fails
        arm = [{"task_id": f"t{i}", "solved": True} for i in range(4)]
        report = gate_replay(base, {"lessons": arm})
        self.assertTrue(report["arms"]["lessons"]["promoted"])

    def test_regressions_kill_a_net_positive_arm(self):
        # two fail->solve, one solve->fail: net +1 but asymmetry rejects
        base = [{"task_id": "t0", "solved": False},
                {"task_id": "t1", "solved": False},
                {"task_id": "t2", "solved": True},
                {"task_id": "t3", "solved": True}]
        arm = [{"task_id": "t0", "solved": True},
               {"task_id": "t1", "solved": True},
               {"task_id": "t2", "solved": False},
               {"task_id": "t3", "solved": True}]
        report = gate_replay(base, {"lessons": arm})
        d = report["arms"]["lessons"]
        self.assertEqual(d["gains"], 2)
        self.assertEqual(d["regressions"], 1)
        self.assertFalse(d["promoted"])

    def test_stop_clause_reported_per_lesson(self):
        base = [{"task_id": "t0", "solved": True}]
        arm = [{"task_id": "t0", "solved": True}]
        texts = ["fetch more pages to be sure",
                 "check the BIP subpage once, then answer immediately"]
        report = gate_replay(base, {"a": arm}, lesson_texts=texts)
        flags = [s["has_stop_clause"] for s in report["stop_clauses"]]
        self.assertEqual(flags, [False, True])

    def test_markdown_renders_verdicts(self):
        base = [{"task_id": "t0", "solved": True}]
        arm = [{"task_id": "t0", "solved": False}]
        md = to_markdown(gate_replay(base, {"bad": arm}))
        self.assertIn("REJECT", md)
        self.assertIn("| bad |", md)


class CliTests(unittest.TestCase):
    def test_cli_end_to_end_json(self):
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as d:
            base_p = os.path.join(d, "cold.json")
            arm_p = os.path.join(d, "warm.json")
            json.dump([{"krs": "a", "solved": True},
                       {"krs": "b", "solved": True}], open(base_p, "w"))
            json.dump({"warm": [{"krs": "a", "solved": False},
                                {"krs": "b", "solved": False}]},
                      open(arm_p, "w"))
            proc = subprocess.run(
                [sys.executable, "-m", "open_dream_rsi", "gate-replay",
                 "--baseline", f"cold={base_p}",
                 "--compare", f"lessons={arm_p}:warm",
                 "--key", "krs", "--format", "md"],
                capture_output=True, text=True,
                env={**os.environ, "PYTHONPATH":
                     os.path.dirname(os.path.dirname(os.path.abspath(__file__)))})
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("REJECT", proc.stdout)
            self.assertIn("lessons", proc.stderr)  # summary line on stderr


if __name__ == "__main__":
    unittest.main()
