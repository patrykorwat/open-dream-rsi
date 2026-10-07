"""Artifact Lifecycle Manager (ALM) — issue #3.

The Dream-RSI learning pipeline (Sentinel -> Evolving World -> Replay/Dreamer
-> Promotion) owns *learning*: it decides whether an artifact deserves to be
active. This module owns the opposite half — **lifecycle**: validation,
activation, staleness, supersession, archival, quarantine, merge lineage,
rollback, retention/GC and auditability of artifacts *once they exist*.

    Dream-RSI owns learning.
    ALM owns artifact lifecycle.
    The event log owns transition history.
    Discovery Trees own historical world evidence.

Two planes, never mixed
-----------------------
``ArtifactState``     current materialized state — "what is true NOW".
``ArtifactTransitionEvent``  immutable append-only history — "how it got here".

The event log (``<memory>/artifacts/events.jsonl``) is the authoritative
history; ``artifacts/states.json`` is a materialized view for fast runtime
reads and is always reconstructible from the event stream
(:meth:`ArtifactLifecycleManager.rebuild_states`). A state mutation without
a corresponding event is invalid — every mutation below goes through
:meth:`_transition`, which appends the event FIRST and only then exposes the
new state. An invalid transition is never committed.

Lifecycle-managed artifacts vs immutable evidence
-------------------------------------------------
``policy_parameters``, ``policy_program``, ``recipe``, ``lesson`` are
runtime-usable artifacts with a lifecycle. ``discovery_tree`` and
``audit_event`` are **evidence**: replayable history that the ALM must never
activate, supersede, merge or rewrite — registering them is rejected by
construction. Replay keeps operating on historical trees without mutation;
the audit log stays append-only (GC appends a ``artifact.deleted`` event,
it never removes history).

Determinism
-----------
Every transition is governed by an explicit rule table
(:data:`ALLOWED_TRANSITIONS`), never by "newer/similar/plausible". LLM
assistance may *propose* semantic merges upstream, but the transition here
is rule-governed and recorded as an immutable event. The clock is
injectable (``now=``) so tests and replay are deterministic.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# -- vocabulary -------------------------------------------------------------

#: Artifact types that carry a lifecycle (runtime-usable objects).
ARTIFACT_TYPES = ("policy_parameters", "policy_program", "recipe", "lesson")

#: Immutable evidence — retention/archival/integrity only, NEVER lifecycle
#: transitions (issue #3: "Discovery Trees must remain replayable historical
#: evidence"; "Audit events must remain append-only").
EVIDENCE_TYPES = ("discovery_tree", "audit_event")

CANDIDATE = "CANDIDATE"
VALIDATED = "VALIDATED"
ACTIVE = "ACTIVE"
STALE = "STALE"
SUPERSEDED = "SUPERSEDED"
QUARANTINED = "QUARANTINED"
ARCHIVED = "ARCHIVED"
REJECTED = "REJECTED"

LIFECYCLE_STATES = (CANDIDATE, VALIDATED, ACTIVE, STALE, SUPERSEDED,
                    QUARANTINED, ARCHIVED, REJECTED)

#: The state machine. Every edge is an EXPLICIT operation; anything absent
#: here is rejected. Notably absent (must never happen implicitly):
#: CANDIDATE->ACTIVE, VALIDATED->ACTIVE (activation requires the explicit
#: :meth:`ArtifactLifecycleManager.activate` promotion call), ARCHIVED->ACTIVE,
#: QUARANTINED->ACTIVE (restore re-enters through VALIDATED + validation).
ALLOWED_TRANSITIONS: Dict[Tuple[str, str], str] = {
    (CANDIDATE, VALIDATED): "artifact.validated",
    (CANDIDATE, REJECTED): "artifact.rejected",
    (VALIDATED, ACTIVE): "artifact.activated",
    (VALIDATED, REJECTED): "artifact.rejected",   # failed promotion requirements
    (ACTIVE, STALE): "artifact.stale",
    (ACTIVE, SUPERSEDED): "artifact.superseded",
    (ACTIVE, QUARANTINED): "artifact.quarantined",
    (VALIDATED, QUARANTINED): "artifact.quarantined",
    (STALE, ARCHIVED): "artifact.archived",
    (STALE, QUARANTINED): "artifact.quarantined",
    (SUPERSEDED, ARCHIVED): "artifact.archived",
    (ARCHIVED, VALIDATED): "artifact.restored",   # restore + validation
    (QUARANTINED, VALIDATED): "artifact.restored",  # review + validation
}

#: Types where at most ONE artifact may be ACTIVE per logical slot at a time
#: ("Normally only one active version per logical policy"). Lessons and
#: recipes legitimately have several active entries per category.
SINGLE_ACTIVE_TYPES = ("policy_parameters", "policy_program")

#: Automatic lifecycle actions skip these states (terminal or pending).
_NON_AUTOMATIC_STATES = (REJECTED, ARCHIVED, SUPERSEDED, QUARANTINED)

#: Defaults for the deterministic retention/staleness policy.
DEFAULT_STALE_AFTER_SECONDS = 7 * 86400.0
DEFAULT_SUPERSEDED_GRACE_SECONDS = 3 * 86400.0
DEFAULT_RETENTION_SECONDS = 30 * 86400.0


class ArtifactLifecycleError(RuntimeError):
    """Any lifecycle rule violation (unknown artifact, bad type, ...)."""


class InvalidTransitionError(ArtifactLifecycleError):
    """The requested state transition is not in ALLOWED_TRANSITIONS."""


# -- models ------------------------------------------------------------------

@dataclass(frozen=True)
class ArtifactIdentity:
    """Stable, immutable identity + lineage of one artifact version."""
    artifact_id: str
    artifact_type: str
    version: int
    created_at: float
    created_by: str
    parent_id: Optional[str]
    supersedes: Tuple[str, ...]


@dataclass
class ArtifactState:
    """The CURRENT materialized state of one artifact — never its history."""
    artifact_id: str
    artifact_type: str
    slot: str
    lifecycle_state: str
    version: int
    created_at: float
    updated_at: float
    created_by: str
    parent_id: Optional[str] = None
    supersedes: Tuple[str, ...] = ()
    superseded_by: Optional[str] = None
    last_used_at: Optional[float] = None
    last_validated_at: Optional[float] = None
    usage_count: int = 0
    wins: int = 0
    pinned: bool = False
    payload: Dict[str, Any] = field(default_factory=dict)
    payload_sha256: str = ""

    def identity(self) -> ArtifactIdentity:
        return ArtifactIdentity(self.artifact_id, self.artifact_type,
                                self.version, self.created_at, self.created_by,
                                self.parent_id, tuple(self.supersedes))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["supersedes"] = list(self.supersedes)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArtifactState":
        d = dict(d)
        d["supersedes"] = tuple(d.get("supersedes") or ())
        known = {f for f in cls.__dataclass_fields__}  # tolerate future fields
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(frozen=True)
class ArtifactTransitionEvent:
    """One immutable record of one lifecycle transition (append-only)."""
    event_id: str
    artifact_id: str
    artifact_type: str
    sequence: int
    timestamp: float
    from_state: str
    to_state: str
    actor: str
    reason: str
    kind: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArtifactTransitionEvent":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ValidationResult:
    """Outcome of running the artifact-type validator."""
    artifact_id: str
    ok: bool
    errors: List[str] = field(default_factory=list)


@dataclass
class GCReport:
    """What a garbage-collection pass did (never touches the event log)."""
    archived: List[str] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)
    skipped_pinned: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -- type validators (structural validation only — behavior belongs to replay)

def _err_list() -> List[str]:
    return []


def _validate_lesson_payload(payload: Dict[str, Any]) -> List[str]:
    """Structural lesson check (schema/length). Structural validity does NOT
    imply behavioral usefulness — activation still needs promotion evidence."""
    errors: List[str] = []
    from open_dream_rsi.core.curator import validate_lesson_items
    item = {"trigger": payload.get("trigger", ""), "text": payload.get("text", "")}
    try:
        validate_lesson_items([item], max_lessons=1)
    except Exception as exc:
        errors.append(f"lesson: {exc}")
    return errors


def _validate_policy_program_payload(payload: Dict[str, Any]) -> List[str]:
    """Policy programs must pass syntax/AST validation before VALIDATED."""
    errors: List[str] = []
    code = payload.get("code")
    if not isinstance(code, str) or not code.strip():
        return ["policy_program: missing 'code'"]
    try:
        from open_dream_rsi.core.policygen import validate_policy_source
        validate_policy_source(code)
    except Exception as exc:
        errors.append(f"policy_program: {exc}")
    return errors


def _validate_policy_parameters_payload(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    params = payload.get("params", payload)
    if not isinstance(params, dict) or not params:
        return ["policy_parameters: 'params' must be a non-empty mapping"]
    for k, v in params.items():
        if not isinstance(k, str) or isinstance(v, bool) or not isinstance(v, (int, float)):
            errors.append(f"policy_parameters: {k!r} is not numeric")
        elif not (-1e9 <= float(v) <= 1e9):
            errors.append(f"policy_parameters: {k!r} out of range")
    return errors


def _validate_recipe_payload(payload: Dict[str, Any]) -> List[str]:
    """Recipes must preserve verification provenance."""
    errors: List[str] = []
    if not isinstance(payload.get("code"), str) or not payload["code"].strip():
        errors.append("recipe: missing 'code'")
    if "score" not in payload:
        errors.append("recipe: missing verification provenance ('score')")
    return errors


VALIDATORS: Dict[str, Callable[[Dict[str, Any]], List[str]]] = {
    "lesson": _validate_lesson_payload,
    "policy_program": _validate_policy_program_payload,
    "policy_parameters": _validate_policy_parameters_payload,
    "recipe": _validate_recipe_payload,
}


def _payload_sha(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# -- append-only event store ---------------------------------------------------

class ArtifactEventStore:
    """JSONL append-only store of transition events + a materialized view.

    The event log is the authoritative history and is NEVER rewritten or
    truncated — not even by GC. ``states.json`` is a cache that must always
    equal :meth:`rebuild_states` (asserted by ``verify_materialization``).
    """

    def __init__(self, root: "str | Path"):
        self.dir = Path(root) / "artifacts"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.dir / "events.jsonl"
        self.states_path = self.dir / "states.json"

    # -- events --
    def append(self, event: ArtifactTransitionEvent) -> None:
        with open(self.events_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")

    def read_events(self) -> List[ArtifactTransitionEvent]:
        if not self.events_path.exists():
            return []
        out: List[ArtifactTransitionEvent] = []
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            out.append(ArtifactTransitionEvent.from_dict(json.loads(line)))
        return out

    # -- materialized view --
    def save_states(self, states: Dict[str, ArtifactState]) -> None:
        payload = {aid: st.to_dict() for aid, st in states.items()}
        tmp = self.states_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(self.states_path)  # atomic

    def load_states(self) -> Dict[str, ArtifactState]:
        if not self.states_path.exists():
            return {}
        try:
            data = json.loads(self.states_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return {aid: ArtifactState.from_dict(d) for aid, d in data.items()}


# -- the manager -----------------------------------------------------------------

class ArtifactLifecycleManager:
    """Rule-governed lifecycle for Dream-RSI artifacts (see module docstring).

    State-changing methods are the ONLY supported mechanism for lifecycle
    transitions; direct mutation of a persisted ``ArtifactState.lifecycle_state``
    is treated as corruption (caught by ``verify_materialization``).
    """

    def __init__(self, memory_or_root: Any,
                 *,
                 stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
                 superseded_grace_seconds: float = DEFAULT_SUPERSEDED_GRACE_SECONDS,
                 retention_seconds: float = DEFAULT_RETENTION_SECONDS,
                 now: Optional[Callable[[], float]] = None):
        # NB: PurePath.root exists (the "/" anchor) — never getattr "root"
        # blindly; only a non-path object (DreamMemory) carries the store root.
        if isinstance(memory_or_root, (str, Path)):
            root = memory_or_root
        else:
            root = memory_or_root.root
        self.store = ArtifactEventStore(root)
        self.stale_after_seconds = stale_after_seconds
        self.superseded_grace_seconds = superseded_grace_seconds
        self.retention_seconds = retention_seconds
        self._now = now or time.time
        self._events = self.store.read_events()
        self._seq = max((e.sequence for e in self._events), default=0)
        # Trust the materialized view only when it matches the log; a stale
        # or hand-edited view is rebuilt, never trusted (event log = truth).
        loaded = self.store.load_states()
        rebuilt = rebuild_artifact_state(self._events)
        if not loaded or _states_equal(loaded, rebuilt):
            self._states = loaded or rebuilt
        else:
            self._states = rebuilt
            self.store.save_states(self._states)

    # -- reads ---------------------------------------------------------------

    def get_state(self, artifact_id: str) -> ArtifactState:
        st = self._states.get(artifact_id)
        if st is None:
            raise ArtifactLifecycleError(f"unknown artifact: {artifact_id}")
        return st

    def states(self, *, artifact_type: Optional[str] = None,
               slot: Optional[str] = None,
               lifecycle_state: Optional[str] = None) -> List[ArtifactState]:
        out = []
        for st in self._states.values():
            if artifact_type and st.artifact_type != artifact_type:
                continue
            if slot and st.slot != slot:
                continue
            if lifecycle_state and st.lifecycle_state != lifecycle_state:
                continue
            out.append(st)
        return sorted(out, key=lambda s: (s.slot, s.version))

    def history(self, artifact_id: str) -> List[ArtifactTransitionEvent]:
        return [e for e in self._events if e.artifact_id == artifact_id]

    def all_events(self) -> List[ArtifactTransitionEvent]:
        return list(self._events)

    # -- authoring (Dream-RSI learning pipeline creates candidates) ----------

    def register_candidate(self, artifact_type: str, payload: Dict[str, Any],
                           *, slot: str, created_by: str,
                           artifact_id: Optional[str] = None,
                           parent_id: Optional[str] = None,
                           supersedes: Tuple[str, ...] = ()) -> ArtifactState:
        """Persist a NEW artifact as CANDIDATE (never runtime-eligible).

        Candidates influence no runtime behavior until an explicit
        promotion. Creation belongs upstream (Dream-RSI pipeline); this is
        the persistence seam. A new VERSION is always a new identity — an
        existing identity is never re-registered or mutated in place.
        """
        if artifact_type in EVIDENCE_TYPES:
            raise ArtifactLifecycleError(
                f"{artifact_type} is immutable evidence, not a lifecycle "
                "artifact — the ALM must never activate/supersede/merge it")
        if artifact_type not in ARTIFACT_TYPES:
            raise ArtifactLifecycleError(
                f"unknown artifact type {artifact_type!r}; lifecycle-managed "
                f"types are {ARTIFACT_TYPES}")
        if artifact_id is None:
            artifact_id = self._auto_id(artifact_type, slot)
        elif artifact_id in self._states:
            raise ArtifactLifecycleError(
                f"identity {artifact_id} already exists — a new version must "
                "create a new identity, never mutate historical identity")
        elif self._event_exists(artifact_id):
            raise ArtifactLifecycleError(
                f"identity {artifact_id} exists in the event log (deleted or "
                "historical) — identity is immutable, use a new id/version")
        version = self._next_version(slot)
        state = ArtifactState(
            artifact_id=artifact_id, artifact_type=artifact_type, slot=slot,
            lifecycle_state=CANDIDATE, version=version,
            created_at=0.0, updated_at=0.0, created_by=created_by,
            parent_id=parent_id, supersedes=tuple(supersedes),
            payload=dict(payload), payload_sha256=_payload_sha(payload))
        self._states[artifact_id] = state
        self._commit(artifact_id, from_state="", to_state=CANDIDATE,
                     kind="artifact.created", actor=created_by,
                     reason="candidate registered by learning pipeline",
                     metadata={"slot": slot, "version": version,
                               "payload": dict(payload),
                               "payload_sha256": state.payload_sha256,
                               "parent_id": parent_id,
                               "supersedes": list(supersedes)})
        return self.get_state(artifact_id)

    # -- structural validation (ALM + artifact-specific validators) ----------

    def validate(self, artifact_id: str, *, actor: str = "alm") -> ValidationResult:
        """Run the type validator. Pass -> VALIDATED; fail -> REJECTED.

        VALIDATED does NOT mean behaviorally superior — only replay/promotion
        can say that (activation stays a separate explicit step).
        """
        st = self.get_state(artifact_id)
        if st.lifecycle_state != CANDIDATE:
            raise InvalidTransitionError(
                f"validate() requires CANDIDATE, {artifact_id} is "
                f"{st.lifecycle_state}")
        validator = VALIDATORS.get(st.artifact_type)
        errors = validator(st.payload) if validator else []
        result = ValidationResult(artifact_id, not errors, errors)
        if errors:
            self._transition(artifact_id, REJECTED, actor=actor,
                             reason="structural validation failed: "
                                    + "; ".join(errors)[:300])
        else:
            self._transition(artifact_id, VALIDATED, actor=actor,
                             reason="structural validation passed")
        return result

    # -- lifecycle transitions (the ONLY mutation API) ------------------------

    def activate(self, artifact_id: str, *, reason: str, actor: str) -> ArtifactState:
        """Explicit promotion: VALIDATED -> ACTIVE.

        The ALM never activates an artifact because it is newer, similar or
        plausible: this call IS the promotion decision (the Dream-RSI
        promotion pipeline invokes it with its evidence as ``reason``).
        For single-active types the previous ACTIVE in the slot must be
        superseded first (use :meth:`promote_replacement` for the pair).
        """
        st = self.get_state(artifact_id)
        if st.artifact_type in SINGLE_ACTIVE_TYPES:
            current = self.active_in_slot(st.slot, st.artifact_type)
            if current and current.artifact_id != artifact_id:
                raise InvalidTransitionError(
                    f"slot {st.slot!r} already has ACTIVE {current.artifact_id}"
                    " — supersede it explicitly first (promotion lineage)")
        self._transition(artifact_id, ACTIVE, actor=actor, reason=reason)
        return self.get_state(artifact_id)

    def promote_replacement(self, artifact_id: str, *, reason: str,
                            actor: str) -> ArtifactState:
        """Atomic promotion pair: supersede the slot's current ACTIVE (if
        any) with explicit lineage, then activate the new artifact."""
        st = self.get_state(artifact_id)
        current = self.active_in_slot(st.slot, st.artifact_type)
        if current and current.artifact_id != artifact_id:
            self.supersede(current.artifact_id, artifact_id,
                           reason=reason, actor=actor)
        self._transition(artifact_id, ACTIVE, actor=actor, reason=reason)
        return self.get_state(artifact_id)

    def mark_stale(self, artifact_id: str, *, reason: str,
                   actor: str) -> ArtifactState:
        """ACTIVE -> STALE. Staleness is not deletion: the artifact stays
        valid, it is simply no longer considered current."""
        self._transition(artifact_id, STALE, actor=actor, reason=reason)
        return self.get_state(artifact_id)

    def supersede(self, artifact_id: str, replacement_id: str, *,
                  reason: str, actor: str) -> ArtifactState:
        """ACTIVE -> SUPERSEDED with explicit replacement lineage. The
        superseded artifact remains historically addressable."""
        rep = self.get_state(replacement_id)
        if rep.artifact_type != self.get_state(artifact_id).artifact_type:
            raise ArtifactLifecycleError(
                "supersession requires the same artifact type")
        self._transition(artifact_id, SUPERSEDED, actor=actor, reason=reason,
                         metadata={"replacement_id": replacement_id})
        return self.get_state(artifact_id)

    def archive(self, artifact_id: str, *, reason: str,
                actor: str) -> ArtifactState:
        """STALE/SUPERSEDED -> ARCHIVED: retained for history/rollback,
        ineligible for runtime, still recoverable via restore()."""
        self._transition(artifact_id, ARCHIVED, actor=actor, reason=reason)
        return self.get_state(artifact_id)

    def restore(self, artifact_id: str, *, reason: str,
                actor: str) -> ArtifactState:
        """ARCHIVED/QUARANTINED -> VALIDATED, re-validation required.

        Archive/restore preserves complete history: nothing is rewritten —
        restore is one new event on top of the existing ones. Quarantined
        artifacts CANNOT be automatically reactivated: this call is the
        explicit review, and it must be followed by a passing validator
        (failure returns the artifact to QUARANTINED/ARCHIVED untouched).
        """
        st = self.get_state(artifact_id)
        if st.lifecycle_state not in (ARCHIVED, QUARANTINED):
            raise InvalidTransitionError(
                f"restore() requires ARCHIVED or QUARANTINED, {artifact_id} "
                f"is {st.lifecycle_state}")
        validator = VALIDATORS.get(st.artifact_type)
        errors = validator(st.payload) if validator else []
        if errors:
            raise ArtifactLifecycleError(
                f"restore refused — validation failed: {'; '.join(errors)[:300]}")
        self._transition(artifact_id, VALIDATED, actor=actor, reason=reason,
                         metadata={"restored_from": st.lifecycle_state})
        return self.get_state(artifact_id)

    def quarantine(self, artifact_id: str, *, reason: str,
                   actor: str) -> ArtifactState:
        """ACTIVE/VALIDATED -> QUARANTINED (integrity/safety/schema/trust).
        No automatic path back to ACTIVE exists in ALLOWED_TRANSITIONS."""
        self._transition(artifact_id, QUARANTINED, actor=actor, reason=reason)
        return self.get_state(artifact_id)

    def reject(self, artifact_id: str, *, reason: str,
               actor: str) -> ArtifactState:
        """CANDIDATE/VALIDATED -> REJECTED (terminal). A later improved
        version must be created as a NEW candidate, never by reviving this
        identity; the rejection event remains auditable forever."""
        self._transition(artifact_id, REJECTED, actor=actor, reason=reason)
        return self.get_state(artifact_id)

    # -- usage, pinning, integrity ---------------------------------------------

    def record_use(self, artifact_id: str, *, win: bool = False) -> None:
        """Runtime touch: usage_count/last_used_at feed stale detection and
        retrieval ranking. Also an event (state mutation without event is
        invalid — statistics included)."""
        st = self.get_state(artifact_id)
        if st.lifecycle_state != ACTIVE:
            return  # usage counts only make sense for runtime-eligible state
        self._commit(artifact_id, from_state=st.lifecycle_state,
                     to_state=st.lifecycle_state, kind="artifact.used",
                     actor="runtime", reason="surfaced into a proposal",
                     metadata={"win": bool(win)})

    def pin(self, artifact_id: str, *, reason: str, actor: str) -> ArtifactState:
        """Pin = protected from AUTOMATIC lifecycle actions (stale detection,
        GC, capacity eviction). Explicit human operations still work."""
        st = self.get_state(artifact_id)
        if st.pinned:
            return st
        self._commit(artifact_id, from_state=st.lifecycle_state,
                     to_state=st.lifecycle_state, kind="artifact.pinned",
                     actor=actor, reason=reason, metadata={},
                     extra_apply=lambda t: setattr(t, "pinned", True))
        return self.get_state(artifact_id)

    def unpin(self, artifact_id: str, *, reason: str, actor: str) -> ArtifactState:
        st = self.get_state(artifact_id)
        if not st.pinned:
            return st
        self._commit(artifact_id, from_state=st.lifecycle_state,
                     to_state=st.lifecycle_state, kind="artifact.unpinned",
                     actor=actor, reason=reason, metadata={},
                     extra_apply=lambda t: setattr(t, "pinned", False))
        return self.get_state(artifact_id)

    def verify_integrity(self, artifact_id: str, *, actor: str = "alm") -> bool:
        """Payload hash + validator check. Failure quarantines automatically
        (integrity failure is exactly the quarantine trigger)."""
        st = self.get_state(artifact_id)
        ok = _payload_sha(st.payload) == st.payload_sha256
        if ok:
            validator = VALIDATORS.get(st.artifact_type)
            ok = not (validator and validator(st.payload))
        if not ok and st.lifecycle_state in (ACTIVE, VALIDATED, STALE):
            self.quarantine(artifact_id, actor=actor,
                            reason="integrity verification failed "
                                   "(payload hash/validator mismatch)")
        return ok

    def annotate(self, artifact_id: str, payload: Dict[str, Any], *,
                 reason: str, actor: str) -> ArtifactState:
        """Refresh the payload of a NON-CANDIDATE artifact without changing
        its lifecycle state (evidence merge on an existing lesson). A new
        artifact VERSION is still created via register_candidate/merge; this
        is only the evidence-annotation channel the curator needs, recorded
        as its own event so reconstruction stays exact."""
        st = self.get_state(artifact_id)
        new_payload = dict(payload)
        self._commit(artifact_id, from_state=st.lifecycle_state,
                     to_state=st.lifecycle_state, kind="artifact.annotated",
                     actor=actor, reason=reason,
                     metadata={"payload": new_payload,
                               "payload_sha256": _payload_sha(new_payload)},
                     extra_apply=lambda t: _apply_payload(t, new_payload))
        return self.get_state(artifact_id)

    def retire(self, artifact_id: str, *, reason: str,
               actor: str) -> ArtifactState:
        """Take a lesson out of the runtime-active view through the state
        machine — ACTIVE -> STALE -> ARCHIVED, VALIDATED -> REJECTED. The
        record itself remains addressable and auditable (never deleted)."""
        st = self.get_state(artifact_id)
        if st.lifecycle_state == ACTIVE:
            self.mark_stale(artifact_id, actor=actor, reason=reason)
            return self.archive(artifact_id, actor=actor, reason=reason)
        if st.lifecycle_state in (STALE, SUPERSEDED):
            return self.archive(artifact_id, actor=actor, reason=reason)
        if st.lifecycle_state in (CANDIDATE, VALIDATED):
            return self.reject(artifact_id, actor=actor, reason=reason)
        return st  # already ARCHIVED/REJECTED/QUARANTINED — nothing to do

    def active_in_slot(self, slot: str,
                       artifact_type: Optional[str] = None) -> Optional[ArtifactState]:
        for st in self._states.values():
            if st.slot == slot and st.lifecycle_state == ACTIVE \
                    and (artifact_type is None or st.artifact_type == artifact_type):
                return st
        return None

    def known(self, artifact_id: str) -> bool:
        """True if this identity exists now or ever existed in the log."""
        return artifact_id in self._states or self._event_exists(artifact_id)

    def sync_active(self, artifact_type: str, slot: str,
                    payload: Dict[str, Any], *, reason: str,
                    actor: str) -> Optional[ArtifactState]:
        """Mirror a learning-pipeline promotion into the lifecycle store.

        The Dream-RSI pipeline (dreamer / replay gate) has ALREADY decided
        this payload is the new best for ``slot`` (the caller only invokes
        this where the legacy save actually changed something). The ALM
        records the decision as explicit lifecycle transitions: register ->
        structural validation -> supersede previous ACTIVE -> activate.
        Returns None when the payload is identical to the current ACTIVE
        (no churn, no events) or when structural validation rejects it.
        """
        current = self.active_in_slot(slot)
        new_sha = _payload_sha(payload)
        if current is not None and current.payload_sha256 == new_sha \
                and current.artifact_type == artifact_type:
            return current
        cand = self.register_candidate(artifact_type, payload, slot=slot,
                                       created_by=actor)
        result = self.validate(cand.artifact_id, actor=actor)
        if not result.ok:
            return None
        if current is not None and current.artifact_type == artifact_type:
            self.supersede(current.artifact_id, cand.artifact_id,
                           reason=reason, actor=actor)
        elif current is not None:
            # slot changed artifact type — the previous ACTIVE must leave
            # the slot explicitly before a different type occupies it
            self.quarantine(current.artifact_id, actor=actor,
                            reason="slot re-typed to "
                                   f"{artifact_type}; cross-type supersession "
                                   "is not allowed")
        self._transition(cand.artifact_id, ACTIVE, actor=actor, reason=reason)
        return self.get_state(cand.artifact_id)

    # -- automated policies (deterministic, pin-protected) ---------------------

    def detect_stale(self, *, now: Optional[float] = None,
                     actor: str = "alm:staleness_policy") -> List[str]:
        """ACTIVE artifacts unused for ``stale_after_seconds`` -> STALE.
        Pinned artifacts are never marked stale automatically."""
        now = now if now is not None else self._now()
        touched: List[str] = []
        for st in list(self._states.values()):
            if st.lifecycle_state != ACTIVE or st.pinned:
                continue
            ref = st.last_used_at if st.last_used_at else st.created_at
            if now - ref > self.stale_after_seconds:
                self.mark_stale(st.artifact_id, actor=actor,
                                reason=f"unused for {now - ref:.0f}s "
                                       f"> {self.stale_after_seconds:.0f}s")
                touched.append(st.artifact_id)
        return touched

    def merge(self, artifact_ids: List[str], *, reason: str, actor: str,
              payload: Optional[Dict[str, Any]] = None,
              slot: Optional[str] = None,
              combiner: Optional[Callable[[List[ArtifactState]], Dict[str, Any]]] = None,
              created_by: Optional[str] = None) -> ArtifactState:
        """Merge two or more same-type artifacts into a NEW artifact.

        A merge NEVER mutates its sources: ACTIVE sources go
        ACTIVE -> SUPERSEDED with the merge artifact as the explicit
        replacement, and the merge artifact carries ``supersedes`` lineage
        to every source. The result is VALIDATED (never ACTIVE for free) —
        activation still requires an explicit promotion decision. Sources
        from different slots merge into ``slot`` (default: the first
        source's slot).
        """
        if len(artifact_ids) < 2:
            raise ArtifactLifecycleError("merge requires at least two artifacts")
        sources = [self.get_state(a) for a in artifact_ids]
        kinds = {s.artifact_type for s in sources}
        if len(kinds) != 1:
            raise ArtifactLifecycleError("merge requires one artifact type")
        art_type = sources[0].artifact_type
        if art_type in EVIDENCE_TYPES:
            raise ArtifactLifecycleError("immutable evidence must never be merged")
        for s in sources:
            if s.lifecycle_state not in (ACTIVE, VALIDATED, STALE):
                raise InvalidTransitionError(
                    f"merge source {s.artifact_id} is {s.lifecycle_state}; "
                    "merge requires usable artifacts")
        if payload is None:
            if combiner is None:
                combiner = _default_combiners.get(art_type)
            if combiner is None:
                raise ArtifactLifecycleError(
                    "merge needs an explicit payload or a combiner for this type")
            payload = combiner(sources)
        slot = sources[0].slot
        merged = self.register_candidate(
            art_type, payload, slot=slot,
            created_by=created_by or actor,
            parent_id=sources[0].artifact_id,
            supersedes=tuple(s.artifact_id for s in sources))
        validator = VALIDATORS.get(art_type)
        errors = validator(payload) if validator else []
        if errors:
            self.reject(merged.artifact_id, actor=actor,
                        reason="merged payload failed validation: "
                               + "; ".join(errors)[:300])
            raise ArtifactLifecycleError(
                f"merged payload invalid: {'; '.join(errors)[:300]}")
        self._transition(merged.artifact_id, VALIDATED, actor=actor,
                         reason="merged payload validated")
        for s in sources:
            if s.lifecycle_state == ACTIVE:
                self.supersede(s.artifact_id, merged.artifact_id,
                               reason=f"merged into {merged.artifact_id}",
                               actor=actor)
        return self.get_state(merged.artifact_id)

    def rollback(self, artifact_id: str, *, reason: str,
                 actor: str) -> ArtifactState:
        """Explicit rollback to a previous version — a NEW derived artifact
        becomes ACTIVE; the historical events of the version being rolled
        back (including its promotion) are never erased.

        Example: policy:v3 ACTIVE, rollback(v2) -> policy:v4 with
        parent=v2, supersedes=[v3]; v3 goes SUPERSEDED; v4 becomes ACTIVE.
        """
        target = self.get_state(artifact_id)
        if target.lifecycle_state not in (SUPERSEDED, ARCHIVED, STALE):
            raise InvalidTransitionError(
                f"rollback target {artifact_id} is {target.lifecycle_state}; "
                "rollback targets are historical (SUPERSEDED/ARCHIVED/STALE)")
        current = self.active_in_slot(target.slot)
        if current and current.artifact_id == artifact_id:
            raise InvalidTransitionError("rollback target is already ACTIVE")
        derived = self.register_candidate(
            target.artifact_type, dict(target.payload), slot=target.slot,
            created_by=actor,
            parent_id=target.artifact_id,
            supersedes=(current.artifact_id,) if current else ())
        self.validate(derived.artifact_id, actor=actor)
        if self.get_state(derived.artifact_id).lifecycle_state != VALIDATED:
            raise ArtifactLifecycleError(
                "rollback refused — derived payload failed validation")
        if current:
            self.supersede(current.artifact_id, derived.artifact_id,
                           reason=f"rolled back to {artifact_id}: {reason}"[:300],
                           actor=actor)
        self._transition(derived.artifact_id, ACTIVE, actor=actor,
                         reason=f"rollback promotion of {artifact_id}: {reason}"[:300])
        return self.get_state(derived.artifact_id)

    def gc(self, *, now: Optional[float] = None,
           actor: str = "alm:retention_policy") -> GCReport:
        """Retention/GC — physical deletion is SEPARATE from lifecycle state.

        ARCHIVED/REJECTED older than ``retention_seconds`` and not pinned are
        physically removed from the materialized store. SUPERSEDED/STALE
        past their grace window are auto-ARCHIVED first. The event log is
        NEVER pruned: every deletion appends an ``artifact.deleted`` event,
        so required audit history survives (and reconstruction still works:
        deleted artifacts simply do not reappear, matching the view)."""
        now = now if now is not None else self._now()
        report = GCReport()
        for st in list(self._states.values()):
            if st.pinned:
                if st.lifecycle_state in (STALE, SUPERSEDED, ARCHIVED, REJECTED):
                    report.skipped_pinned.append(st.artifact_id)
                continue
            if st.lifecycle_state in (SUPERSEDED, STALE) \
                    and now - st.updated_at > self.superseded_grace_seconds:
                self.archive(st.artifact_id, actor=actor,
                             reason="retention: superseded/stale past grace window")
                report.archived.append(st.artifact_id)
        for st in list(self._states.values()):
            st = self._states.get(st.artifact_id)
            if st is None or st.pinned:
                continue
            if st.lifecycle_state in (ARCHIVED, REJECTED) \
                    and now - st.updated_at > self.retention_seconds:
                self._commit(st.artifact_id, from_state=st.lifecycle_state,
                             to_state=DELETED_SENTINEL, kind="artifact.deleted",
                             actor=actor,
                             reason="retention policy: physical deletion",
                             metadata={"state_at_deletion": st.lifecycle_state},
                             extra_apply=None, delete=True)
                report.deleted.append(st.artifact_id)
        return report

    # -- auditability ----------------------------------------------------------

    def rebuild_states(self) -> Dict[str, ArtifactState]:
        """Rebuild the full current view from the append-only event stream."""
        return rebuild_artifact_state(self._events)

    def verify_materialization(self) -> bool:
        """True iff the persisted materialized view equals the reconstructed
        one — the invariant tests pin: state == rebuild(events)."""
        return _states_equal(self._states, self.rebuild_states())

    # -- internals ---------------------------------------------------------------

    def _auto_id(self, artifact_type: str, slot: str) -> str:
        n = sum(1 for e in self._events if e.kind == "artifact.created") + 1
        return f"{artifact_type}_{n:04d}"

    def _next_version(self, slot: str) -> int:
        versions = [e.metadata.get("version", 0) for e in self._events
                    if e.kind == "artifact.created"
                    and e.metadata.get("slot") == slot]
        return (max(versions) if versions else 0) + 1

    def _event_exists(self, artifact_id: str) -> bool:
        return any(e.artifact_id == artifact_id for e in self._events)

    def _transition(self, artifact_id: str, to_state: str, *, actor: str,
                    reason: str, metadata: Optional[Dict[str, Any]] = None,
                    extra: Optional[Dict[str, Any]] = None) -> ArtifactTransitionEvent:
        """Validate current state + rule table, APPEND the event, only then
        mutate the materialized view (all-or-nothing per transition)."""
        st = self.get_state(artifact_id)
        key = (st.lifecycle_state, to_state)
        if key not in ALLOWED_TRANSITIONS:
            raise InvalidTransitionError(
                f"transition {st.lifecycle_state} -> {to_state} is not allowed "
                f"for {artifact_id}")
        return self._commit(
            artifact_id, from_state=st.lifecycle_state, to_state=to_state,
            kind=ALLOWED_TRANSITIONS[key], actor=actor, reason=reason,
            metadata=dict(metadata or {}),
            extra_apply=_applier(to_state, extra or {}))

    def _commit(self, artifact_id: str, *, from_state: str, to_state: str,
                kind: str, actor: str, reason: str,
                metadata: Dict[str, Any],
                extra_apply: Optional[Callable[[ArtifactState], None]] = None,
                delete: bool = False) -> ArtifactTransitionEvent:
        now = self._now()
        self._seq += 1
        event = ArtifactTransitionEvent(
            event_id=f"evt_{self._seq:08d}",
            artifact_id=artifact_id,
            artifact_type=self._states[artifact_id].artifact_type,
            sequence=self._seq, timestamp=now,
            from_state=from_state, to_state=to_state,
            actor=actor, reason=reason[:500], kind=kind, metadata=metadata)
        # History first: the event is committed before the view is exposed.
        self.store.append(event)
        self._events.append(event)
        if delete:
            self._states.pop(artifact_id, None)
            self.store.save_states(self._states)
        else:
            target = self._states.get(artifact_id)
            if target is None:
                raise ArtifactLifecycleError(
                    f"internal error: no materialized state for {artifact_id} "
                    "at commit time (creation must pre-materialize)")
            if kind == "artifact.created":
                target.created_at = now
            elif from_state != to_state:
                target.lifecycle_state = to_state
            if kind == "artifact.superseded":
                target.superseded_by = metadata.get("replacement_id")
            if kind in ("artifact.validated", "artifact.restored"):
                target.last_validated_at = now
            if kind == "artifact.used":
                _apply_use(target, bool(metadata.get("win")), event.timestamp)
            if extra_apply is not None:
                extra_apply(target)
            target.updated_at = now
            self.store.save_states(self._states)
        return event


DELETED_SENTINEL = "DELETED"


def _applier(to_state: str, extra: Dict[str, Any]) -> Optional[Callable[[ArtifactState], None]]:
    if not extra:
        return None
    def apply(t: ArtifactState) -> None:
        for k, v in extra.items():
            if hasattr(t, k):
                setattr(t, k, v)
    return apply


def _apply_use(t: ArtifactState, win: bool, now: float) -> None:
    t.usage_count += 1
    t.last_used_at = now
    if win:
        t.wins += 1


def _apply_payload(t: ArtifactState, payload: Dict[str, Any]) -> None:
    t.payload = dict(payload)
    t.payload_sha256 = _payload_sha(t.payload)


# -- merge combiners (semantic proposals live upstream; these are defaults)

def _merge_lesson(sources: List[ArtifactState]) -> Dict[str, Any]:
    from open_dream_rsi.core.curator import normalize
    texts = [str(s.payload.get("text", "")) for s in sources if s.payload.get("text")]
    words: List[str] = []
    for s in sources:
        for w in normalize(str(s.payload.get("trigger", ""))).split():
            if w not in words:
                words.append(w)
    evidence: List[str] = []
    for s in sources:
        for e in (s.payload.get("evidence") or []):
            if e not in evidence:
                evidence.append(e)
    return {"trigger": " ".join(words[:4]),
            "text": " / ".join(dict.fromkeys(texts)),
            "evidence": evidence}


def _merge_recipe(sources: List[ArtifactState]) -> Dict[str, Any]:
    best = max(sources, key=lambda s: float(s.payload.get("score", 0.0)))
    payload = dict(best.payload)
    payload["merged_from"] = [s.artifact_id for s in sources]
    return payload


_default_combiners = {"lesson": _merge_lesson, "recipe": _merge_recipe}


# -- reconstruction: the event stream is the source of truth -------------------

def rebuild_artifact_state(events: List[ArtifactTransitionEvent],
                           ) -> Dict[str, ArtifactState]:
    """Replay the event stream into the authoritative current view.

    Tests pin that this always equals the materialized ``states.json``.
    """
    states: Dict[str, ArtifactState] = {}
    for e in events:
        if e.kind == "artifact.created":
            md = e.metadata
            states[e.artifact_id] = ArtifactState(
                artifact_id=e.artifact_id, artifact_type=e.artifact_type,
                slot=md.get("slot", ""), lifecycle_state=CANDIDATE,
                version=md.get("version", 1), created_at=e.timestamp,
                updated_at=e.timestamp, created_by=e.actor,
                parent_id=md.get("parent_id"),
                supersedes=tuple(md.get("supersedes") or ()),
                payload=dict(md.get("payload") or {}),
                payload_sha256=md.get("payload_sha256", ""))
        elif e.kind == "artifact.deleted":
            states.pop(e.artifact_id, None)
        elif e.kind == "artifact.used":
            st = states.get(e.artifact_id)
            if st is not None:
                st.usage_count += 1
                st.last_used_at = e.timestamp
                if e.metadata.get("win"):
                    st.wins += 1
                st.updated_at = e.timestamp
        elif e.kind == "artifact.pinned":
            st = states.get(e.artifact_id)
            if st is not None:
                st.pinned = True
                st.updated_at = e.timestamp
        elif e.kind == "artifact.unpinned":
            st = states.get(e.artifact_id)
            if st is not None:
                st.pinned = False
                st.updated_at = e.timestamp
        elif e.kind == "artifact.annotated":
            st = states.get(e.artifact_id)
            if st is not None:
                st.payload = dict(e.metadata.get("payload") or {})
                st.payload_sha256 = e.metadata.get("payload_sha256", "")
                st.updated_at = e.timestamp
        else:
            st = states.get(e.artifact_id)
            if st is None:
                continue
            if e.to_state in LIFECYCLE_STATES:
                st.lifecycle_state = e.to_state
            if e.kind == "artifact.superseded":
                st.superseded_by = e.metadata.get("replacement_id")
            if e.kind == "artifact.validated" or e.kind == "artifact.restored":
                st.last_validated_at = e.timestamp
            st.updated_at = e.timestamp
    return states


def _states_equal(a: Dict[str, ArtifactState], b: Dict[str, ArtifactState]) -> bool:
    if set(a) != set(b):
        return False
    return all(a[k].to_dict() == b[k].to_dict() for k in a)
