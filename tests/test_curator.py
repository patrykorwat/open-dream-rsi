"""Tests for the section-4 knowledge curator: gate, KB lifecycle, retrieval,
usage-pruning, loop wiring and the knowledge-arm benchmark contract.

Run:  python -m unittest discover -s tests
"""

import json
import tempfile
import unittest

from open_dream_rsi.core.curator import (
    MAX_LESSON_CHARS,
    KnowledgeCurator,
    LessonValidationError,
    curate_lessons,
    evidence_snippets,
    extract_json_array,
    format_lessons,
    lesson_key,
    normalize,
    select_lessons,
    validate_lesson_items,
)
from open_dream_rsi.loop import AutoRSIRuntime, Task
from open_dream_rsi.memory import DreamMemory

GOOD_ITEM = {
    "trigger": "dedupe order stability",
    "text": "Re-slicing a list that already passed partial tests preserves the same "
            "bug; rebuild the ordering rule from the first principles of the spec.",
}


def make_items(*overrides):
    base = {"trigger": GOOD_ITEM["trigger"], "text": GOOD_ITEM["text"]}
    return [{**base, **o} for o in overrides] or [base]


class ValidationTest(unittest.TestCase):
    def test_accepts_contract_payload(self):
        out = validate_lesson_items([GOOD_ITEM], max_lessons=3)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["text"], GOOD_ITEM["text"])

    def test_rejects_non_array_and_non_object(self):
        for bad in ({"trigger": "x"}, "text", [42]):
            with self.assertRaises(LessonValidationError):
                validate_lesson_items(bad, max_lessons=3)

    def test_rejects_bad_trigger_width(self):
        for trig in ("", "one two three four five"):
            with self.assertRaises(LessonValidationError):
                validate_lesson_items([{"trigger": trig, "text": GOOD_ITEM["text"]}], 3)

    def test_rejects_too_short_or_generic_text(self):
        with self.assertRaises(LessonValidationError):
            validate_lesson_items([{"trigger": "x", "text": "be careful"}], 3)
        long = {"trigger": "x", "text": "a" * (MAX_LESSON_CHARS + 1)}
        with self.assertRaises(LessonValidationError):
            validate_lesson_items([long], 3)

    def test_dedups_within_payload_and_caps_at_max(self):
        items = [dict(GOOD_ITEM), dict(GOOD_ITEM)] + make_items(
            {"trigger": "extra lesson one", "text": "z" * 30},
            {"trigger": "extra lesson two", "text": "y" * 30})
        out = validate_lesson_items(items, max_lessons=2)
        self.assertEqual(len(out), 2)

    def test_extract_json_array(self):
        self.assertEqual(extract_json_array('```json\n[{"a": 1}]\n```'), [{"a": 1}])
        self.assertEqual(extract_json_array('noise [{"a": 1}] noise'), [{"a": 1}])
        self.assertIsNone(extract_json_array("no array here"))
        self.assertIsNone(extract_json_array('{"a": 1}'))  # object, not array

    def test_normalize_and_lesson_key(self):
        self.assertEqual(normalize("  Foo-BAR! baz "), "foo bar baz")
        a = lesson_key({"trigger": "Foo bar", "text": "Same text here, twice!! " + "x" * 40})
        b = lesson_key({"trigger": "foo, BAR", "text": "same text here, twice?? " + "x" * 40})
        self.assertEqual(a, b)


class CurationLifecycleTest(unittest.TestCase):
    def test_add_merge_and_stable_identities(self):
        r1 = curate_lessons([], [GOOD_ITEM], evidence=["score=0.667 t2 fails"])
        self.assertEqual(len(r1.added), 1)
        self.assertEqual(r1.merged, 0)
        self.assertEqual(r1.entries[0]["wins"], 0)
        # same lesson again -> MERGE (evidence grows), never a duplicate
        r2 = curate_lessons(r1.entries, [GOOD_ITEM], evidence=["score=0.000 t0 fails"])
        self.assertEqual(len(r2.added), 0)
        self.assertEqual(r2.merged, 1)
        self.assertEqual(len(r2.entries), 1)
        self.assertEqual(len(r2.entries[0]["evidence"]), 2)

    def test_dead_lessons_are_pruned(self):
        dead = {**GOOD_ITEM, "uses": 4, "wins": 0}
        healthy = {"trigger": "other thing", "text": "b" * 40, "uses": 1, "wins": 2}
        r = curate_lessons([dead, healthy], [])
        self.assertEqual(len(r.dropped), 1)
        self.assertEqual(r.entries[0]["trigger"], "other thing")

    def test_kb_capped_at_strongest(self):
        entries = [{"trigger": f"t{i}", "text": f"{'u' * 40}{i}", "uses": i, "wins": 1}
                   for i in range(10)]
        r = curate_lessons(entries, [])
        self.assertEqual(len(r.entries), 8)
        self.assertTrue(all(e["uses"] >= 2 for e in r.entries))  # weak ones evicted

    def test_empty_curate_is_no_change(self):
        r = curate_lessons([], [])
        self.assertFalse(r.changed)


class RetrievalTest(unittest.TestCase):
    def setUp(self):
        self.lessons = [
            {"trigger": "median parity", "text": "even length lists need the mean "
                                                 "of the two middles, not the upper one.",
             "wins": 0, "uses": 0, "updated_at": 1.0},
            {"trigger": "dedupe ordering", "text": "dict.fromkeys preserves insertion "
                                                   "order while removing duplicates.",
             "wins": 3, "uses": 5, "updated_at": 2.0},
            {"trigger": "string padding", "text": "c" * 50, "wins": 9, "uses": 9,
             "updated_at": 3.0},
        ]

    def test_trigger_overlap_beats_wins(self):
        picked = select_lessons(self.lessons, "Implement median(xs) even length case")
        self.assertEqual(picked[0]["trigger"], "median parity")

    def test_no_overlap_falls_back_to_wins(self):
        picked = select_lessons(self.lessons, "unrelated unrelated")
        self.assertEqual(picked[0]["trigger"], "string padding")

    def test_char_budget_and_caps(self):
        many = [{"trigger": f"t{i}", "text": "d" * 100, "wins": i, "uses": 0,
                 "updated_at": float(i)} for i in range(10)]
        picked = select_lessons(many, "task", max_lessons=4, max_chars=250)
        self.assertLessEqual(len(picked), 4)
        self.assertLessEqual(sum(len(l["text"]) for l in picked), 250)

    def test_format_lessons(self):
        text = format_lessons([{"trigger": "a b", "text": "body"}])
        self.assertIn("[a b] body", text)


class ScriptedCuratorClient:
    """Returns queued replies for curator calls; counts them."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.curator_calls = 0

    def chat(self, messages, model=None, temperature=0.7, max_tokens=None):
        assert "knowledge curator" in messages[0]["content"]
        self.curator_calls += 1
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


class DistillTest(unittest.TestCase):
    def test_valid_payload(self):
        client = ScriptedCuratorClient(["```json\n[{\"trigger\": \"a b\", \"text\": \""
                                        + "z" * 40 + "\"}]\n```"])
        lessons, err = KnowledgeCurator(client).distill(
            "cat", "task", [{"action": "x", "score": 0.3, "errors": ["e"]}], [])
        self.assertEqual(err, "")
        self.assertEqual(len(lessons), 1)
        self.assertEqual(client.curator_calls, 1)

    def test_repair_round_on_invalid_payload(self):
        bad = "not json at all"
        good = "```json\n[{\"trigger\": \"a b\", \"text\": \"" + "z" * 40 + "\"}]\n```"
        client = ScriptedCuratorClient([bad, good])
        calls = [0]
        lessons, err = KnowledgeCurator(client, max_repair=1).distill(
            "cat", "task", [{"action": "x", "score": 0.3, "errors": ["e"]}], [],
            api_calls=calls)
        self.assertEqual(err, "")
        self.assertEqual(client.curator_calls, 2)
        self.assertEqual(calls[0], 2)

    def test_all_invalid_returns_error(self):
        client = ScriptedCuratorClient(["garbage"])
        lessons, err = KnowledgeCurator(client, max_repair=0).distill(
            "cat", "task", [{"action": "x", "score": 0.3, "errors": ["e"]}], [])
        self.assertEqual(lessons, [])
        self.assertIn("JSON", err)

    def test_no_evidence_no_call(self):
        client = ScriptedCuratorClient(["```json\n[]\n```"])
        lessons, err = KnowledgeCurator(client).distill("cat", "task", [], [])
        self.assertEqual(lessons, [])
        self.assertIn("no failure evidence", err)
        self.assertEqual(client.curator_calls, 0)

    def test_evidence_snippets_render(self):
        out = evidence_snippets([{"action": "write_code:abcd", "score": 0.5,
                                  "errors": ["t0 -> 1 != 2"]}])
        self.assertIn("score=0.500", out[0])
        self.assertIn("t0 -> 1 != 2", out[0])


# ---------------------------------------------------------------------------
# Loop wiring + benchmark contracts
# ---------------------------------------------------------------------------


class TrapLoop:
    """Drives one trap category end-to-end with the knowledge arm on."""

    @staticmethod
    def run(category="median", cycles=5, budget=24, seed=5):
        from open_dream_rsi.bench_policy import TrapSolver, TRAP_LESSON_TEXT, _trap_tasks

        solver = TrapSolver()
        memory = DreamMemory(tempfile.mkdtemp(prefix="odr-cur-") + "/mem")
        tasks = [t for t in _trap_tasks() if t.category == category]
        runtime = AutoRSIRuntime(
            client=solver, memory=memory, tasks=tasks, api_call_budget=budget,
            dream_iterations=20, rng_seed=seed,
            enable_policy_code=False, enable_knowledge=True, explore_epsilon=0.3,
        )
        reports = []
        for _ in range(cycles):
            runtime.api_calls_used = 0
            reports.append(runtime.run_once())
        return solver, memory, reports


class LoopKnowledgeTest(unittest.TestCase):
    def test_curator_populates_kb_and_lesson_unlocks_the_trap(self):
        solver, memory, reports = TrapLoop.run("median", cycles=5)
        lessons = memory.get_lessons("median")
        self.assertGreaterEqual(len(lessons), 1)
        self.assertIn("decoy trap", lessons[0]["trigger"])
        self.assertEqual(solver.policy_calls, 0)       # knowledge-only arm
        self.assertGreater(solver.curator_calls, 0)
        self.assertTrue(any(r.tasks_solved for r in reports))  # lesson broke the lock-in

    def test_usage_and_wins_accumulate_after_solve(self):
        solver, memory, reports = TrapLoop.run("median", cycles=5)
        lessons = [l for l in memory.get_lessons("median") if l.get("wins", 0) > 0]
        self.assertGreaterEqual(len(lessons), 1)       # KB is credit-scored
        self.assertTrue(all(l["uses"] >= l["wins"] for l in lessons))

    def test_evidence_dedup_bounds_curator_calls(self):
        solver, memory, reports = TrapLoop.run("median", cycles=6, budget=16)
        # deterministic solver repeats ladder codes; the digest must stop the
        # curator from re-buying calls on identical evidence every cycle
        self.assertLess(solver.curator_calls, 6 * 5)

    def test_disable_flag_skips_curator(self):
        from open_dream_rsi.bench_policy import TrapSolver, _trap_tasks
        solver = TrapSolver()
        memory = DreamMemory(tempfile.mkdtemp(prefix="odr-cur-") + "/mem")
        tasks = [t for t in _trap_tasks() if t.category == "median"]
        runtime = AutoRSIRuntime(client=solver, memory=memory, tasks=tasks,
                                 api_call_budget=24, dream_iterations=20, rng_seed=5,
                                 enable_policy_code=False, enable_knowledge=False)
        runtime.run_once()
        self.assertEqual(solver.curator_calls, 0)
        self.assertEqual(memory.get_lessons("median"), [])

    def test_budget_guard_blocks_curation(self):
        from open_dream_rsi.bench_policy import TrapSolver, _trap_tasks
        solver = TrapSolver()
        memory = DreamMemory(tempfile.mkdtemp(prefix="odr-cur-") + "/mem")
        tasks = [t for t in _trap_tasks() if t.category == "median"]
        runtime = AutoRSIRuntime(client=solver, memory=memory, tasks=tasks,
                                 api_call_budget=1, dream_iterations=5, rng_seed=5,
                                 enable_policy_code=False, enable_knowledge=True)
        runtime.run_once()  # one attempt only — curation never afforded
        self.assertEqual(solver.curator_calls, 0)

    def test_lessons_survive_restart(self):
        solver, memory, _ = TrapLoop.run("median", cycles=3)
        reopened = DreamMemory(memory.root)
        self.assertEqual(len(reopened.get_lessons("median")),
                         len(memory.get_lessons("median")))


class KnowledgeArmContractTest(unittest.TestCase):
    """README-figure guard: the knowledge arm must actually help epsilon."""

    CYCLES = 6
    BUDGET = 24
    SEEDS = (7, 11, 23)

    def test_knowledge_arm_dominates_epsilon_at_zero_policy_calls(self):
        from open_dream_rsi.bench_policy import ARMS, run_policy_arm
        eps = run_policy_arm("epsilon_greedy", ARMS["epsilon_greedy"],
                             self.CYCLES, self.BUDGET, seeds=self.SEEDS)
        kn = run_policy_arm("knowledge_curator", ARMS["knowledge_curator"],
                            self.CYCLES, self.BUDGET, seeds=self.SEEDS)
        self.assertEqual(kn.policy_calls, 0)           # improvement is knowledge-only
        self.assertGreater(kn.curator_calls, 0)
        self.assertGreater(kn.solves_total, 2 * eps.solves_total)
        self.assertEqual(kn.mean_solve_rate_by_cycle[-1], 100.0)
        # knowledge replaces luck: epsilon spends fewer total calls once the
        # trap is remembered (luck is reserved for genuinely unknown branches)
        self.assertLess(kn.api_calls_total, eps.api_calls_total)
        self.assertEqual(kn.mean_solve_rate_by_cycle,
                         sorted(kn.mean_solve_rate_by_cycle))

    def test_existing_arms_untouched_by_knowledge(self):
        from open_dream_rsi.bench_policy import ARMS, run_policy_arm
        r = run_policy_arm("greedy", ARMS["greedy"], 2, 24, seeds=(7,))
        self.assertEqual(r.solves_total, 0)
        self.assertEqual(r.curator_calls, 0)


class DashboardLessonsTest(unittest.TestCase):
    def test_snapshot_exposes_lessons(self):
        from open_dream_rsi.dashboard import DashboardState
        solver, memory, _ = TrapLoop.run("median", cycles=3)
        from open_dream_rsi.bench_policy import _trap_tasks
        state = DashboardState(_trap_tasks())
        snap = state.snapshot(memory, budget=24, interval=1.0)
        self.assertIn("median", snap["lessons"])
        self.assertIn("text", snap["lessons"]["median"][0])


if __name__ == "__main__":
    unittest.main()
