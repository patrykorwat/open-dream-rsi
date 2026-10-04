"""Replay test: the sentinel engine on the anonymized production audit fixture.

Fixtures/sentinel_audit.json is (class_id, session_id) event pairs in
arrival order, derived from a production agent session store (30 days,
4914 error events / 459 classes / 476 sessions), fully anonymized: no
message content, timestamps, tool names or real session ids. Class ids
are opaque; the fixture is order-preserving, so the engine replay must
reproduce the recorded note count exactly — it is a behavioral
regression test for the whole policy stack (signature stability,
thresholds, one-note-per-class-per-session), not a statistic.
"""
import json
from pathlib import Path

import pytest

from open_dream_rsi.sentinel import SentinelEngine

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sentinel_audit.json"


@pytest.fixture(scope="module")
def audit():
    if not FIXTURE.exists():
        pytest.skip("anonymized audit fixture not present")
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _payload_for(class_id: int) -> str:
    # stable, class-unique, keyword-bearing payload: normalizes to itself
    # (letters only), so one fixture class == one live error class
    return f"error class {class_id:06d}".translate(
        str.maketrans("0123456789", "abcdefghij"))


def test_meta_counts_are_self_consistent(audit):
    meta, events = audit["meta"], audit["events"]
    assert len(events) == meta["error_events"]
    assert len({e[0] for e in events}) == meta["distinct_classes"]
    assert len({e[1] for e in events}) == meta["sessions"]
    per = {}
    repeats = 0
    for c, s in events:
        k = (c, s)
        per[k] = per.get(k, 0) + 1
        if per[k] >= 2:
            repeats += 1
    assert repeats == meta["repeat_events_same_class_same_session"]
    assert 40 < meta["pct_repeats"] < 60            # ~half of failures repeat


def test_engine_replay_reproduces_recorded_note_count(audit, tmp_path):
    meta, events = audit["meta"], audit["events"]
    engine = SentinelEngine(state_path=tmp_path / "s.json")
    notes = sum(1 for c, s in events
                if engine.observe("t", _payload_for(c), s, status="error"))
    assert notes == meta["notes_emitted_by_engine"]
    # the annotation tax is bounded: ~4 notes per 10 error events, and every
    # note is the threshold crossing of a class the session already suffered
    assert notes / meta["error_events"] < 0.5


def test_engine_is_stateless_across_restarts_on_the_fixture(audit, tmp_path):
    """Restarting the process mid-stream must not change the outcome —
    durability is by state file, not by memory."""
    meta, events = audit["meta"], audit["events"]
    state = tmp_path / "s.json"
    notes = 0
    for i in range(0, len(events), 500):          # simulated restarts
        chunk = events[i:i + 500]
        engine = SentinelEngine(state_path=state)
        notes += sum(1 for c, s in chunk
                     if engine.observe("t", _payload_for(c), s, status="error"))
    assert notes == meta["notes_emitted_by_engine"]
