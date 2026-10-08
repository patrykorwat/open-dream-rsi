"""Automated dreaming: the single seam every trigger calls.

One command, no shell, no cron, no flock script:

    python -m open_dream_rsi dream --memory .dream_rsi \
        --sessions /opt/data/state.db --skills-out ~/.hermes/skills/odr-curated

This module owns the *cadence*, which is the part a harness should not
guess about. Two tiers, decided here rather than by whoever fires us:

  CHEAP  (every call, near-free — zero LLM unless new evidence arrived)
    * ingest new host sessions -> staging lessons
    * ALM maintenance: staleness, supersession, archive, GC
    * publish ACTIVE lessons as skills (newly-earned knowledge surfaces now)

  FULL   (the actual world dreamer — attempts + dreaming + replay gate)
    runs only when the evidence justifies the cost:
      >= min_new_evidence new failure-shaped sessions, OR
      >= max_age_seconds since the last full dream
    The gate is expensive (~4 LLM calls per candidate lesson per probe), so
    firing it per session would spend the whole budget proving lessons that
    were already proven. A quiet day therefore dreams once; a busy day
    dreams a few times; an idle day not at all.

Everything that decides WHETHER a lesson is good stays where it belongs —
the paired-replay promotion gate. This module only decides WHEN, so it is
portable to any harness (Hermes hook, MCP tool, or a human terminal).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_MIN_NEW_EVIDENCE = 3
DEFAULT_MAX_AGE_SECONDS = 6 * 3600
LOCK_TIMEOUT_S = 30 * 60


# -- single-instance lock -----------------------------------------------------

def _acquire(memory_root: Path) -> Optional[Path]:
    """Atomic lock; a crashed run's lock is reclaimed after LOCK_TIMEOUT_S."""
    lock = memory_root / "dream.lock"
    memory_root.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            if time.time() - lock.stat().st_mtime > LOCK_TIMEOUT_S:
                lock.unlink(missing_ok=True)
                return _acquire(memory_root)
        except OSError:
            pass
        return None
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return lock


def _release(lock: Optional[Path]) -> None:
    if lock is not None:
        try:
            lock.unlink(missing_ok=True)
        except OSError:
            pass


def _state_path(memory_root: Path) -> Path:
    return memory_root / "dream_state.json"


def _load_state(memory_root: Path) -> Dict[str, Any]:
    p = _state_path(memory_root)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(memory_root: Path, state: Dict[str, Any]) -> None:
    tmp = _state_path(memory_root).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    tmp.replace(_state_path(memory_root))


# -- cheap tier: evidence -----------------------------------------------------

def ingest_sessions(memory, sessions_db: "str | Path", *,
                    limit: int = 25, client=None,
                    category: Optional[str] = None) -> Dict[str, Any]:
    """Distil NEW (unseen) host sessions into staging lessons.

    Returns {'new': n_sessions, 'added': n_lessons}. Sessions already
    processed are skipped — transcripts stay in state.db, so skipping a
    cycle loses nothing.
    """
    from open_dream_rsi.core.curator import (curate_lessons,
                                             evidence_snippets,
                                             KnowledgeCurator)
    from open_dream_rsi.sessions import episode_failures, read_hermes_sessions

    root = Path(memory.root)
    state = _load_state(root)
    seen = set(state.get("seen_sessions", []))

    eps = read_hermes_sessions(sessions_db, limit=limit + len(seen))
    new = [e for e in eps if e.session_id not in seen][:limit]
    if not new:
        return {"new": 0, "added": 0}

    curator = KnowledgeCurator(client) if client is not None else None
    if curator is None:
        # No endpoint right now: leave everything unseen — the next cycle
        # with a live client distils it; evidence is never lost. Report
        # zero new so the caller stays in the cheap maintenance tier.
        return {"new": 0, "added": 0, "deferred": len(new)}
    added_total = 0
    for ep in new:
        seen.add(ep.session_id)
        failures = episode_failures(ep)
        if not failures:
            continue
        cat = category or (Path(ep.cwd).name if ep.cwd else ep.source)
        distilled, err = curator.distill(cat, ep.title or cat, failures,
                                         memory.get_lessons(cat))
        if err or not distilled:
            memory.log_event("lesson_rejected", source=ep.source,
                             session_id=ep.session_id,
                             reason=(err or "empty")[:300])
            continue
        result = curate_lessons(memory.get_lessons(cat), distilled,
                                evidence=evidence_snippets(failures))
        memory.replace_lessons(cat, result.entries)
        memory.log_event("lessons_curated", source=ep.source,
                         session_id=ep.session_id, category=cat,
                         added=len(result.added))
        added_total += len(result.added)
    state["seen_sessions"] = sorted(seen)[-2000:]
    _save_state(root, state)
    return {"new": len(new), "added": added_total}


# -- the seam ------------------------------------------------------------------

def dream_once(memory_root: "str | Path", *,
               sessions_db: "str | Path | None" = None,
               skills_out: "str | Path | None" = None,
               client=None, tasks: Optional[List[Any]] = None,
               budget: int = 20, max_tokens: int = 4096,
               category: Optional[str] = None,
               min_new_evidence: int = DEFAULT_MIN_NEW_EVIDENCE,
               max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS) -> Dict[str, Any]:
    """Run one automated cycle. A concurrent run returns {'skipped': True}."""
    from open_dream_rsi.memory import DreamMemory
    from open_dream_rsi.sessions import lessons_to_skills

    root = Path(memory_root)
    memory = DreamMemory(root)
    lock = _acquire(root)
    if lock is None:
        return {"skipped": True, "reason": "another dream is running"}
    report: Dict[str, Any] = {"skipped": False}
    try:
        # 1. evidence (cheap unless new sessions carry failures)
        if sessions_db:
            report["ingest"] = ingest_sessions(
                memory, sessions_db, client=client, category=category)

        # 2. decide the tier
        state = _load_state(root)
        new = report.get("ingest", {}).get("new", 0)
        age = time.time() - float(state.get("last_full_dream", 0.0))
        full = (new >= min_new_evidence or age >= max_age_seconds)
        report["tier"] = "full" if full else "maintenance"

        # 3. learning cycle (full: attempts + gate; maintenance: budget 0)
        if tasks is None:
            tp = root / "tasks.json"
            from open_dream_rsi.loop import Task
            tasks = ([Task(**t) for t in json.loads(tp.read_text())]
                     if tp.exists() else [])
        if client is None:
            from open_dream_rsi.llm import StubClient
            run_client = StubClient()
        else:
            run_client = client
        from open_dream_rsi.loop import AutoRSIRuntime
        runtime = AutoRSIRuntime(client=run_client, memory=memory,
                                 tasks=(tasks if full else []),
                                 api_call_budget=(budget if full else 0),
                                 max_tokens=max_tokens)
        report["cycle"] = runtime.run_once().to_dict()
        if full:
            state["last_full_dream"] = time.time()
            _save_state(root, state)

        # 4. publish earned knowledge (cheap; surfaces new ACTIVE lessons)
        if skills_out:
            report["skills"] = len(lessons_to_skills(memory, skills_out))
        return report
    finally:
        _release(lock)
