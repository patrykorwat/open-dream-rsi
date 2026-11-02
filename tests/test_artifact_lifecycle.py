"""Artifact Lifecycle Manager (issue #3): state machine, event sourcing,
reconstruction, lineage, pinning, staleness, GC and the loop integration.

Run:  python -m unittest discover -s tests
"""

import tempfile
import unittest

from open_dream_rsi.lifecycle import (
    ACTIVE,
    ALLOWED_TRANSITIONS,
    QUARANTINED,
    REJECTED,
    SUPERSEDED,
    VALIDATED,
    ArtifactLifecycleError,
    ArtifactLifecycleManager,
    InvalidTransitionError,
)

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
class TestLoopIntegration(unittest.TestCase):
    """The loop keeps owning learning; the ALM mirrors its decisions."""



if __name__ == "__main__":
    unittest.main()
