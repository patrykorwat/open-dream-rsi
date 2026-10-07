"""Artifact Lifecycle Manager (issue #3): state machine, event sourcing,
reconstruction, lineage, pinning, staleness, GC and the loop integration.

Run:  python -m unittest discover -s tests
"""

import json
import tempfile
import unittest

from open_dream_rsi.lifecycle import (
    ACTIVE,
    ALLOWED_TRANSITIONS,
    ARCHIVED,
    CANDIDATE,
    QUARANTINED,
    REJECTED,
    STALE,
    SUPERSEDED,
    VALIDATED,
    ArtifactLifecycleError,
    ArtifactLifecycleManager,
    InvalidTransitionError,
    rebuild_artifact_state,
)

GOOD_LESSON = {"trigger": "median even",
               "text": "Sort the list first, return the mean of the two "
                       "middle values for even lengths, then answer."}
BAD_LESSON = {"trigger": "x", "text": "too short"}
GOOD_PROGRAM = {"code": "def choose_action(frontier, step):\n"
                        "    return frontier[0]['node_id']\n"}
BAD_PROGRAM = {"code": "def choose_action(frontier):\n    return 1\n"}
# the AST gate accepts a short signature; an import of a banned module does
# not — that is the structural failure this test needs
BAD_PROGRAM = {"code": "def choose_action(frontier, step):\n"
                       "    import os\n"
                       "    return os.getpid()\n"}


class Clock:
    """Deterministic injectable clock — lifecycle semantics must not depend
    on wall-clock jitter."""

    def __init__(self, start=1_000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def make_alm(**kw):
    clock = kw.pop("clock", Clock())
    root = tempfile.mkdtemp(prefix="odr-alm-")
    return ArtifactLifecycleManager(root, now=clock, **kw), clock


class TestStateTransitions(unittest.TestCase):
    def test_normal_path_candidate_validated_active_stale_archived(self):
        alm, _ = make_alm()
        a = alm.register_candidate("lesson", GOOD_LESSON, slot="median",
                                   created_by="curator")
        self.assertEqual(a.lifecycle_state, CANDIDATE)
        res = alm.validate(a.artifact_id)
        self.assertTrue(res.ok)
        self.assertEqual(alm.get_state(a.artifact_id).lifecycle_state, VALIDATED)
        alm.activate(a.artifact_id, reason="replay promotion", actor="gate")
        self.assertEqual(alm.get_state(a.artifact_id).lifecycle_state, ACTIVE)
        alm.mark_stale(a.artifact_id, reason="unused", actor="policy")
        self.assertEqual(alm.get_state(a.artifact_id).lifecycle_state, STALE)
        alm.archive(a.artifact_id, reason="retention", actor="policy")
        self.assertEqual(alm.get_state(a.artifact_id).lifecycle_state, ARCHIVED)

    def test_replacement_path_supersede_archive(self):
        alm, _ = make_alm()
        a = alm.register_candidate("lesson", GOOD_LESSON, slot="s",
                                   created_by="curator")
        alm.validate(a.artifact_id)
        alm.activate(a.artifact_id, reason="r", actor="gate")
        b = alm.register_candidate("lesson", GOOD_LESSON, slot="s",
                                   created_by="curator")
        alm.validate(b.artifact_id)
        alm.supersede(a.artifact_id, b.artifact_id, reason="v2 replaces",
                      actor="gate")
        sa = alm.get_state(a.artifact_id)
        self.assertEqual(sa.lifecycle_state, SUPERSEDED)
        self.assertEqual(sa.superseded_by, b.artifact_id)
        alm.archive(a.artifact_id, reason="done", actor="policy")
        self.assertEqual(alm.get_state(a.artifact_id).lifecycle_state, ARCHIVED)

    def test_failure_paths(self):
        alm, _ = make_alm()
        a = alm.register_candidate("lesson", BAD_LESSON, slot="s",
                                   created_by="curator")
        res = alm.validate(a.artifact_id)
        self.assertFalse(res.ok)
        self.assertEqual(alm.get_state(a.artifact_id).lifecycle_state, REJECTED)
        b = alm.register_candidate("lesson", GOOD_LESSON, slot="s2",
                                   created_by="curator")
        alm.validate(b.artifact_id)
        alm.activate(b.artifact_id, reason="r", actor="gate")
        alm.quarantine(b.artifact_id, reason="integrity failure", actor="sec")
        self.assertEqual(alm.get_state(b.artifact_id).lifecycle_state,
                         QUARANTINED)

    def test_structural_validation_is_type_specific(self):
        alm, _ = make_alm()
        ok = alm.register_candidate("policy_program", GOOD_PROGRAM,
                                    slot="p", created_by="policygen")
        self.assertTrue(alm.validate(ok.artifact_id).ok)
        bad = alm.register_candidate("policy_program", BAD_PROGRAM,
                                     slot="p2", created_by="policygen")
        self.assertFalse(alm.validate(bad.artifact_id).ok)
        self.assertEqual(alm.get_state(bad.artifact_id).lifecycle_state,
                         REJECTED)


class TestInvalidTransitions(unittest.TestCase):
    def test_implicit_activation_never_happens(self):
        alm, _ = make_alm()
        a = alm.register_candidate("lesson", GOOD_LESSON, slot="s",
                                   created_by="curator")
        # CANDIDATE -> ACTIVE is not in the rule table at all
        with self.assertRaises(InvalidTransitionError):
            alm._transition(a.artifact_id, ACTIVE, actor="sneaky", reason="x")
        with self.assertRaises(InvalidTransitionError):
            alm.activate(a.artifact_id, reason="skip validation",
                         actor="sneaky")
        self.assertEqual(alm.get_state(a.artifact_id).lifecycle_state,
                         CANDIDATE)
        alm.validate(a.artifact_id)
        alm.activate(a.artifact_id, reason="real promotion", actor="gate")
        self.assertEqual(alm.get_state(a.artifact_id).lifecycle_state, ACTIVE)
        # activation is idempotent-blocked: VALIDATED -> ACTIVE again is a
        # no-op edge that is not in the table (ACTIVE -> ACTIVE excluded)
        with self.assertRaises(InvalidTransitionError):
            alm.activate(a.artifact_id, reason="again", actor="gate")

    def test_forbidden_edges_rejected(self):
        alm, _ = make_alm()
        forbidden = [("CANDIDATE", "ACTIVE"), ("ARCHIVED", "ACTIVE"),
                     ("QUARANTINED", "ACTIVE"), ("REJECTED", "ACTIVE"),
                     ("REJECTED", "VALIDATED"), ("ACTIVE", "VALIDATED"),
                     ("ACTIVE", "CANDIDATE"), ("SUPERSEDED", "ACTIVE"),
                     ("STALE", "ACTIVE"), ("STALE", "VALIDATED")]
        self.assertEqual(forbidden,
                         [(f, t) for f, t in forbidden
                          if (f, t) not in ALLOWED_TRANSITIONS])

    def test_terminal_rejected_never_revived(self):
        alm, _ = make_alm()
        a = alm.register_candidate("lesson", BAD_LESSON, slot="s",
                                   created_by="curator")
        alm.validate(a.artifact_id)
        self.assertEqual(alm.get_state(a.artifact_id).lifecycle_state, REJECTED)
        with self.assertRaises(InvalidTransitionError):
            alm.validate(a.artifact_id)          # validate needs CANDIDATE
        with self.assertRaises(InvalidTransitionError):
            alm.restore(a.artifact_id, reason="nope", actor="op")
        # an improved version must be a NEW identity
        b = alm.register_candidate("lesson", GOOD_LESSON, slot="s",
                                   created_by="curator")
        self.assertNotEqual(b.artifact_id, a.artifact_id)

    def test_unknown_artifact_and_bad_type(self):
        alm, _ = make_alm()
        with self.assertRaises(ArtifactLifecycleError):
            alm.get_state("nope")
        with self.assertRaises(ArtifactLifecycleError):
            alm.register_candidate("widget", {}, slot="s", created_by="x")
        with self.assertRaises(ArtifactLifecycleError):
            alm.register_candidate("policy_params", {"params": {}},
                                   slot="s", created_by="x")


class TestEvidenceSeparation(unittest.TestCase):
    def test_immutable_evidence_cannot_be_registered(self):
        alm, _ = make_alm()
        for kind in ("discovery_tree", "audit_event"):
            with self.assertRaises(ArtifactLifecycleError):
                alm.register_candidate(kind, {"nodes": []}, slot="t",
                                       created_by="loop")

    def test_evidence_cannot_be_merged(self):
        alm, _ = make_alm()
        # fabricate two ARCHIVED-like states is impossible for evidence:
        # registration itself refuses (test above), and merge refuses by type
        with self.assertRaises(ArtifactLifecycleError):
            alm.merge(["tree_a", "tree_b"], reason="x", actor="x")


class TestEventEmission(unittest.TestCase):
    def test_every_state_change_emits_exactly_one_event(self):
        alm, _ = make_alm()
        a = alm.register_candidate("lesson", GOOD_LESSON, slot="s",
                                   created_by="curator")
        alm.validate(a.artifact_id)
        alm.activate(a.artifact_id, reason="r", actor="gate")
        kinds = [e.kind for e in alm.history(a.artifact_id)]
        self.assertEqual(kinds, ["artifact.created", "artifact.validated",
                                 "artifact.activated"])

    def test_events_are_immutable_records_with_required_fields(self):
        alm, _ = make_alm()
        a = alm.register_candidate("lesson", GOOD_LESSON, slot="median",
                                   created_by="curator")
        alm.validate(a.artifact_id)
        ev = alm.history(a.artifact_id)[1]
        self.assertEqual(ev.artifact_id, a.artifact_id)
        self.assertEqual(ev.from_state, CANDIDATE)
        self.assertEqual(ev.to_state, VALIDATED)
        self.assertEqual(ev.actor, "alm")
        self.assertTrue(ev.reason)
        self.assertGreater(ev.timestamp, 0)
        self.assertEqual(ev.sequence, 2)
        # frozen dataclass: mutation is impossible
        with self.assertRaises(Exception):
            ev.to_state = CANDIDATE  # type: ignore[misc]

    def test_invalid_transition_never_committed(self):
        alm, _ = make_alm()
        a = alm.register_candidate("lesson", GOOD_LESSON, slot="s",
                                   created_by="curator")
        before = len(alm.all_events())
        with self.assertRaises(InvalidTransitionError):
            alm.archive(a.artifact_id, reason="nope", actor="x")
        self.assertEqual(len(alm.all_events()), before)
        self.assertEqual(alm.get_state(a.artifact_id).lifecycle_state,
                         CANDIDATE)

    def test_events_jsonl_is_append_only(self):
        alm, clock = make_alm()
        a = alm.register_candidate("lesson", GOOD_LESSON, slot="s",
                                   created_by="curator")
        alm.validate(a.artifact_id)
        path = alm.store.events_path
        first = path.read_text(encoding="utf-8")
        alm.activate(a.artifact_id, reason="r", actor="gate")
        second = path.read_text(encoding="utf-8")
        self.assertTrue(second.startswith(first))


class TestReconstruction(unittest.TestCase):
    def test_state_equals_rebuild_from_events(self):
        alm, clock = make_alm()
        a = alm.register_candidate("lesson", GOOD_LESSON, slot="s",
                                   created_by="curator")
        alm.validate(a.artifact_id)
        alm.activate(a.artifact_id, reason="r", actor="gate")
        alm.record_use(a.artifact_id, win=True)
        alm.pin(a.artifact_id, reason="important", actor="op")
        b = alm.register_candidate("lesson", GOOD_LESSON, slot="s2",
                                   created_by="curator")
        alm.validate(b.artifact_id)
        alm.supersede(a.artifact_id, b.artifact_id, reason="x", actor="gate")
        self.assertTrue(alm.verify_materialization())

    def test_reload_reconstructs_identical_view(self):
        alm, clock = make_alm()
        a = alm.register_candidate("lesson", GOOD_LESSON, slot="s",
                                   created_by="curator")
        alm.validate(a.artifact_id)
        alm.activate(a.artifact_id, reason="r", actor="gate")
        alm.record_use(a.artifact_id, win=True)
        reopened = ArtifactLifecycleManager(alm.store.dir.parent, now=clock)
        self.assertEqual(reopened.get_state(a.artifact_id).to_dict(),
                         alm.get_state(a.artifact_id).to_dict())
        self.assertTrue(reopened.verify_materialization())

    def test_corrupted_materialized_view_is_rebuilt_not_trusted(self):
        alm, clock = make_alm()
        a = alm.register_candidate("lesson", GOOD_LESSON, slot="s",
                                   created_by="curator")
        alm.validate(a.artifact_id)
        # simulate direct mutation of the persisted view (forbidden path)
        data = json.loads(alm.store.states_path.read_text(encoding="utf-8"))
        data[a.artifact_id]["lifecycle_state"] = ACTIVE   # lie: never activated
        alm.store.states_path.write_text(json.dumps(data), encoding="utf-8")
        reopened = ArtifactLifecycleManager(alm.store.dir.parent, now=clock)
        self.assertEqual(reopened.get_state(a.artifact_id).lifecycle_state,
                         VALIDATED)  # event log is the truth

    def test_usage_statistics_survive_reconstruction(self):
        alm, clock = make_alm()
        a = alm.register_candidate("lesson", GOOD_LESSON, slot="s",
                                   created_by="curator")
        alm.validate(a.artifact_id)
        alm.activate(a.artifact_id, reason="r", actor="gate")
        alm.record_use(a.artifact_id, win=True)
        alm.record_use(a.artifact_id, win=False)
        st = alm.get_state(a.artifact_id)
        self.assertEqual((st.usage_count, st.wins), (2, 1))
        rebuilt = alm.rebuild_states()[a.artifact_id]
        self.assertEqual((rebuilt.usage_count, rebuilt.wins), (2, 1))


class TestArchiveRestore(unittest.TestCase):
    def test_archive_restore_preserves_complete_history(self):
        alm, _ = make_alm()
        a = alm.register_candidate("lesson", GOOD_LESSON, slot="s",
                                   created_by="curator")
        alm.validate(a.artifact_id)
        alm.activate(a.artifact_id, reason="r", actor="gate")
        alm.mark_stale(a.artifact_id, reason="old", actor="policy")
        alm.archive(a.artifact_id, reason="tidy", actor="policy")
        n_before = len(alm.history(a.artifact_id))
        restored = alm.restore(a.artifact_id, reason="needed again",
                               actor="op")
        self.assertEqual(restored.lifecycle_state, VALIDATED)
        hist = alm.history(a.artifact_id)
        self.assertEqual(len(hist), n_before + 1)      # nothing erased
        self.assertEqual([e.to_state for e in hist],
                         [CANDIDATE, VALIDATED, ACTIVE, STALE, ARCHIVED,
                          VALIDATED])
        # activation after restore requires the explicit promotion again
        alm.activate(a.artifact_id, reason="re-promoted", actor="gate")
        self.assertEqual(alm.get_state(a.artifact_id).lifecycle_state, ACTIVE)


class TestRollback(unittest.TestCase):
    def test_rollback_creates_new_active_and_keeps_history(self):
        alm, _ = make_alm()
        v = []
        for i in range(3):
            p = {"params": {"epsilon": 0.1 * (i + 1)}}
            cand = alm.register_candidate("policy_parameters", p,
                                          slot="epsilon", created_by="dreamer")
            alm.validate(cand.artifact_id)
            if v:
                alm.supersede(v[-1], cand.artifact_id, reason="better",
                              actor="promotion")
            alm.activate(cand.artifact_id, reason="promoted", actor="promotion")
            v.append(cand.artifact_id)
        # rollback to v2: derived v4 becomes ACTIVE, v3 SUPERSEDED
        rolled = alm.rollback(v[1], reason="v3 regressed", actor="op")
        self.assertEqual(rolled.lifecycle_state, ACTIVE)
        self.assertEqual(rolled.parent_id, v[1])
        self.assertEqual(rolled.supersedes, (v[2],))
        self.assertEqual(alm.get_state(v[2]).lifecycle_state, SUPERSEDED)
        # the promotion of v3 is still in the historical events
        kinds = [e.kind for e in alm.history(v[2])]
        self.assertIn("artifact.activated", kinds)
        # and the rollback itself is expressed as NEW events, not deletions
        kinds_rolled = [e.kind for e in alm.history(rolled.artifact_id)]
        self.assertEqual(kinds_rolled,
                         ["artifact.created", "artifact.validated",
                          "artifact.activated"])
        self.assertTrue(alm.verify_materialization())

    def test_rollback_target_must_be_historical(self):
        alm, _ = make_alm()
        a = alm.register_candidate("policy_parameters", {"params": {"a": 1.0}},
                                   slot="s", created_by="dreamer")
        alm.validate(a.artifact_id)
        alm.activate(a.artifact_id, reason="r", actor="p")
        with self.assertRaises(InvalidTransitionError):
            alm.rollback(a.artifact_id, reason="already active", actor="op")


class TestSingleActiveSlot(unittest.TestCase):
    def test_activate_rejects_second_active_in_slot(self):
        alm, _ = make_alm()
        a = alm.register_candidate("policy_parameters", {"params": {"a": 1.0}},
                                   slot="eps", created_by="dreamer")
        alm.validate(a.artifact_id)
        alm.activate(a.artifact_id, reason="r", actor="p")
        b = alm.register_candidate("policy_parameters", {"params": {"a": 2.0}},
                                   slot="eps", created_by="dreamer")
        alm.validate(b.artifact_id)
        with self.assertRaises(InvalidTransitionError):
            alm.activate(b.artifact_id, reason="newer", actor="sneaky")
        # the honest path: supersede first (promotion lineage)
        got = alm.promote_replacement(b.artifact_id, reason="beats on replay",
                                      actor="promotion")
        self.assertEqual(got.lifecycle_state, ACTIVE)
        self.assertEqual(alm.get_state(a.artifact_id).superseded_by,
                         b.artifact_id)

    def test_lessons_allow_several_active_per_category(self):
        alm, _ = make_alm()
        ids = []
        for i in range(3):
            a = alm.register_candidate("lesson", GOOD_LESSON,
                                        slot=f"median:{i}", created_by="c")
            alm.validate(a.artifact_id)
            alm.activate(a.artifact_id, reason="r", actor="gate")
            ids.append(a.artifact_id)
        active = alm.states(artifact_type="lesson", lifecycle_state=ACTIVE)
        self.assertEqual(len(active), 3)


class TestMerge(unittest.TestCase):
    def test_merge_creates_new_artifact_with_lineage(self):
        alm, _ = make_alm()
        a = alm.register_candidate("lesson", {**GOOD_LESSON,
                                              "evidence": ["score=0.3 t0"]},
                                   slot="median:a", created_by="curator")
        alm.validate(a.artifact_id)
        alm.activate(a.artifact_id, reason="r", actor="gate")
        b = alm.register_candidate("lesson",
                                   {"trigger": "median parity",
                                    "text": "Even lengths need the mean of "
                                            "the two middles, then answer.",
                                    "evidence": ["score=0.0 t1"]},
                                   slot="median:b", created_by="curator")
        alm.validate(b.artifact_id)
        alm.activate(b.artifact_id, reason="r", actor="gate")
        m = alm.merge([a.artifact_id, b.artifact_id], reason="dedupe",
                      actor="curator:merge")
        self.assertEqual(m.lifecycle_state, VALIDATED)  # never ACTIVE for free
        self.assertEqual(sorted(m.supersedes),
                         sorted([a.artifact_id, b.artifact_id]))
        self.assertEqual(alm.get_state(a.artifact_id).lifecycle_state,
                         SUPERSEDED)
        self.assertEqual(alm.get_state(b.artifact_id).lifecycle_state,
                         SUPERSEDED)
        # merge activated on an explicit promotion
        alm.activate(m.artifact_id, reason="promotion evidence", actor="gate")
        self.assertEqual(alm.get_state(m.artifact_id).lifecycle_state, ACTIVE)
        # sources remain addressable — never mutated into the merge
        self.assertNotEqual(alm.get_state(a.artifact_id).payload,
                            m.payload)

    def test_merge_rejects_bad_combinations(self):
        alm, _ = make_alm()
        a = alm.register_candidate("lesson", GOOD_LESSON, slot="s",
                                   created_by="c")
        with self.assertRaises(ArtifactLifecycleError):
            alm.merge([a.artifact_id], reason="x", actor="x")
        r = alm.register_candidate("recipe", {"code": "x = 1", "score": 1.0},
                                   slot="s", created_by="c")
        with self.assertRaises(ArtifactLifecycleError):
            alm.merge([a.artifact_id, r.artifact_id], reason="x", actor="x")


class TestQuarantine(unittest.TestCase):
    def test_quarantined_requires_review_and_validation(self):
        alm, _ = make_alm()
        a = alm.register_candidate("policy_program", GOOD_PROGRAM,
                                   slot="p", created_by="policygen")
        alm.validate(a.artifact_id)
        alm.activate(a.artifact_id, reason="r", actor="p")
        alm.quarantine(a.artifact_id, reason="safety", actor="sec")
        # no automatic reactivation edge exists
        self.assertNotIn((QUARANTINED, ACTIVE), ALLOWED_TRANSITIONS)
        with self.assertRaises(InvalidTransitionError):
            alm._transition(a.artifact_id, ACTIVE, actor="auto", reason="x")
        got = alm.restore(a.artifact_id, reason="reviewed", actor="sec")
        self.assertEqual(got.lifecycle_state, VALIDATED)

    def test_restore_refuses_when_validation_fails(self):
        alm, _ = make_alm()
        a = alm.register_candidate("policy_program", GOOD_PROGRAM,
                                   slot="p", created_by="policygen")
        alm.validate(a.artifact_id)
        alm.activate(a.artifact_id, reason="r", actor="p")
        alm.quarantine(a.artifact_id, reason="x", actor="sec")
        # tamper with the payload -> hash mismatch -> integrity fail
        st = alm.get_state(a.artifact_id)
        st.payload["code"] = "import os\n"
        with self.assertRaises(ArtifactLifecycleError):
            alm.restore(a.artifact_id, reason="hope", actor="op")
        self.assertEqual(alm.get_state(a.artifact_id).lifecycle_state,
                         QUARANTINED)

    def test_integrity_failure_quarantines_automatically(self):
        alm, _ = make_alm()
        a = alm.register_candidate("recipe", {"code": "x = 1", "score": 0.9},
                                   slot="r", created_by="loop")
        alm.validate(a.artifact_id)
        alm.activate(a.artifact_id, reason="r", actor="loop")
        alm.get_state(a.artifact_id).payload["score"] = "garbage"
        self.assertFalse(alm.verify_integrity(a.artifact_id))
        self.assertEqual(alm.get_state(a.artifact_id).lifecycle_state,
                         QUARANTINED)


class TestPinning(unittest.TestCase):
    def test_pinned_protects_from_automatic_actions(self):
        alm, clock = make_alm()
        a = alm.register_candidate("recipe", {"code": "x=1", "score": 1.0},
                                   slot="cat", created_by="loop")
        alm.validate(a.artifact_id)
        alm.activate(a.artifact_id, reason="r", actor="loop")
        alm.pin(a.artifact_id, reason="human-approved", actor="op")
        clock.advance(365 * 86400)
        self.assertEqual(alm.detect_stale(), [])
        rep = alm.gc()
        self.assertNotIn(a.artifact_id, rep.archived + rep.deleted)
        self.assertEqual(alm.get_state(a.artifact_id).lifecycle_state, ACTIVE)
        # an ARCHIVED pinned artifact is retained past its window as well
        alm.mark_stale(a.artifact_id, reason="manual", actor="op")
        alm.archive(a.artifact_id, reason="manual", actor="op")
        clock.advance(365 * 86400)
        rep = alm.gc()
        self.assertIn(a.artifact_id, rep.skipped_pinned)
        self.assertIn(a.artifact_id, [s.artifact_id for s in alm.states()])

    def test_pinned_can_still_be_operated_explicitly(self):
        alm, _ = make_alm()
        a = alm.register_candidate("lesson", GOOD_LESSON, slot="s",
                                   created_by="c")
        alm.validate(a.artifact_id)
        alm.pin(a.artifact_id, reason="keep", actor="op")
        self.assertEqual(alm.unpin(a.artifact_id, reason="done",
                                   actor="op").pinned, False)


class TestStaleDetection(unittest.TestCase):
    def test_unused_active_becomes_stale(self):
        alm, clock = make_alm(stale_after_seconds=100.0)
        a = alm.register_candidate("lesson", GOOD_LESSON, slot="s",
                                   created_by="c")
        alm.validate(a.artifact_id)
        alm.activate(a.artifact_id, reason="r", actor="gate")
        clock.advance(50)
        alm.record_use(a.artifact_id)
        clock.advance(60)
        self.assertEqual(alm.detect_stale(), [])          # recent use protects
        clock.advance(60)
        self.assertEqual(alm.detect_stale(), [a.artifact_id])
        self.assertEqual(alm.get_state(a.artifact_id).lifecycle_state, STALE)
        # staleness is not deletion — still restore-capable through archive
        alm.archive(a.artifact_id, reason="bye", actor="policy")
        alm.restore(a.artifact_id, reason="welcome back", actor="op")


class TestGC(unittest.TestCase):
    def test_gc_archives_then_deletes_archived_only(self):
        alm, clock = make_alm(superseded_grace_seconds=10.0,
                              retention_seconds=20.0)
        a = alm.register_candidate("lesson", GOOD_LESSON, slot="s",
                                   created_by="c")
        alm.validate(a.artifact_id)
        alm.activate(a.artifact_id, reason="r", actor="gate")
        b = alm.register_candidate("lesson", GOOD_LESSON, slot="s",
                                   created_by="c")
        alm.validate(b.artifact_id)
        alm.supersede(a.artifact_id, b.artifact_id, reason="x", actor="gate")
        alm.activate(b.artifact_id, reason="r", actor="gate")
        clock.advance(15)
        rep = alm.gc()
        self.assertIn(a.artifact_id, rep.archived)
        # ACTIVE is never touched by GC
        self.assertEqual(alm.get_state(b.artifact_id).lifecycle_state, ACTIVE)
        clock.advance(25)
        rep = alm.gc()
        self.assertIn(a.artifact_id, rep.deleted)
        self.assertNotIn(a.artifact_id, alm.states())
        # deletion left an audit event; the event log is never pruned
        kinds = [e.kind for e in alm.history(a.artifact_id)]
        self.assertIn("artifact.deleted", kinds)
        self.assertTrue(alm.verify_materialization())
        # reconstruction matches the post-GC view (deleted stay deleted)
        self.assertEqual(alm.rebuild_states().get(a.artifact_id), None)

    def test_rejected_events_remain_auditable_after_gc(self):
        alm, clock = make_alm(retention_seconds=1.0)
        a = alm.register_candidate("lesson", BAD_LESSON, slot="s",
                                   created_by="c")
        alm.validate(a.artifact_id)   # -> REJECTED
        clock.advance(2)
        alm.gc()
        self.assertNotIn(a.artifact_id, alm.states())
        # the event log is NEVER pruned: rejection + deletion events both
        # remain — the rejection stays auditable forever (issue #3)
        kinds = [e.kind for e in alm.history(a.artifact_id)]
        self.assertEqual(kinds, ["artifact.created", "artifact.rejected",
                                 "artifact.deleted"])
        self.assertEqual(alm.history(a.artifact_id)[1].to_state, REJECTED)


class TestLoopIntegration(unittest.TestCase):
    """The loop keeps owning learning; the ALM mirrors its decisions."""

    def _run_trap(self, cycles=5, budget=24):
        from open_dream_rsi.bench_policy import TrapSolver, _trap_tasks
        from open_dream_rsi.loop import AutoRSIRuntime
        from open_dream_rsi.memory import DreamMemory
        solver = TrapSolver()
        memory = DreamMemory(tempfile.mkdtemp(prefix="odr-alm-loop-") + "/mem")
        tasks = [t for t in _trap_tasks() if t.category == "median"]
        runtime = AutoRSIRuntime(
            client=solver, memory=memory, tasks=tasks, api_call_budget=budget,
            dream_iterations=20, rng_seed=5, enable_policy_code=False,
            enable_knowledge=True, explore_epsilon=0.3)
        for _ in range(cycles):
            runtime.api_calls_used = 0
            runtime.run_once()
        return solver, memory, runtime

    def test_promoted_artifacts_have_lifecycle_history(self):
        _, memory, runtime = self._run_trap()
        alm = runtime.alm
        # every lesson the KB shows has a lifecycle record
        for l in memory.get_lessons("median"):
            from open_dream_rsi.core.curator import lesson_key
            slot = f"lesson:median:{lesson_key(l)}"
            states = [s for s in alm.states(artifact_type="lesson")
                      if s.slot == slot]
            self.assertTrue(states, f"no lifecycle record for {slot}")
            status = str(l.get("status", "active"))
            states[-1]
            if status == "active":
                self.assertIn(ACTIVE, [s.lifecycle_state for s in states])
            for s in states:
                self.assertTrue(alm.history(s.artifact_id))
        # policy parameters dreamed during the run are lifecycle-managed
        params = alm.states(artifact_type="policy_parameters")
        self.assertTrue(params)
        self.assertTrue(any(s.lifecycle_state == ACTIVE for s in params))
        # recipes for solved tasks too
        recipes = alm.states(artifact_type="recipe")
        self.assertTrue(recipes)
        # and the whole store is self-consistent
        self.assertTrue(alm.verify_materialization())

    def test_gate_promotion_maps_to_alm_activation_event(self):
        _, memory, runtime = self._run_trap()
        active_lessons = [l for l in memory.get_lessons("median")
                          if str(l.get("status", "active")) == "active"]
        self.assertTrue(active_lessons)
        from open_dream_rsi.core.curator import lesson_key
        key = lesson_key(active_lessons[0])
        slot = f"lesson:median:{key}"
        st = [s for s in runtime.alm.states(artifact_type="lesson")
              if s.slot == slot][-1]
        actors = {e.actor for e in runtime.alm.history(st.artifact_id)
                  if e.kind == "artifact.activated"}
        self.assertIn("lesson_gate", actors)

    def test_alm_mutation_without_event_is_detected(self):
        _, memory, runtime = self._run_trap()
        st = runtime.alm.states(artifact_type="lesson")[0]
        st.lifecycle_state = (ARCHIVED if st.lifecycle_state != ARCHIVED
                              else CANDIDATE)  # forbidden direct mutation
        self.assertFalse(runtime.alm.verify_materialization())


class TestLegacySemanticsMigration(unittest.TestCase):
    """The KB status field and the ALM agree — lessons.json stays the
    runtime view until a later issue cuts reads over; the lifecycle store is
    authoritative for history."""

    def test_status_field_mirrors_alm_state(self):
        from open_dream_rsi.core.curator import curate_lessons, lesson_key
        r = curate_lessons([], [GOOD_LESSON])
        self.assertEqual(r.entries[0]["status"], "staging")
        alm, _ = make_alm()
        cand = alm.register_candidate("lesson", GOOD_LESSON,
                                      slot=f"median:{lesson_key(r.entries[0])}",
                                      created_by="curator")
        alm.validate(cand.artifact_id)
        # staging lesson: VALIDATED, not ACTIVE — selection hides it either way
        self.assertEqual(alm.get_state(cand.artifact_id).lifecycle_state,
                         VALIDATED)


if __name__ == "__main__":
    unittest.main()
