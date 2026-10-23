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
    """Atomic lock; a crashed run's lock is reclaimed after LOCK_TIMEOUT_S.

    Owner liveness is checked BEFORE the lock's age: a live owner is never
    stale. A full cycle legitimately runs longer than LOCK_TIMEOUT_S, and
    the age-first order let the next fire steal the lock mid-run — two
    dreams racing on one memory dir (double API spend, last-writer-wins on
    the stores). The age timeout only covers locks whose owner PID is gone
    or unreadable (crash before the PID write, corrupt file).
    """
    lock = memory_root / "dream.lock"
    memory_root.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        stale = False
        try:
            owner = int(lock.read_text().strip())
        except (ValueError, OSError):
            owner = None
        if owner is not None:
            try:
                os.kill(owner, 0)  # raises when the owner is gone
                return None  # alive -> never stale
            except ProcessLookupError:
                stale = True  # PID gone: crashed run
            except PermissionError:
                return None  # alive, just not ours
        if not stale:
            try:
                stale = time.time() - lock.stat().st_mtime > LOCK_TIMEOUT_S
            except OSError:
                stale = False
        if stale:
            lock.unlink(missing_ok=True)
            return _acquire(memory_root)
        return None
    except OSError:
        return None
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return lock


def _release(lock: Optional[Path]) -> None:
    """Remove the lock only if we still own it.

    A stolen lock must survive the dispossessed owner's cleanup: otherwise
    the moment the old run finished, the thief's exclusivity would break
    and a third fire could walk in.
    """
    if lock is None:
        return
    try:
        if int(lock.read_text().strip()) != os.getpid():
            return  # the lock was stolen while we ran — leave it
    except (ValueError, OSError):
        return  # unreadable/already gone: a fresh _acquire will judge it
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

def _probe_task(ep, category: str) -> Optional[Dict[str, Any]]:
    """Derive a replay-probe task from an episode: the user's own request,
    verbatim (bounded), as a test-less task.

    The promotion gate needs tasks to probe lessons against — and the
    honest probe is the very task whose failure produced the lesson. These
    land at the END of tasks.json (manual/queued tasks keep priority) with
    max_attempts=1 so probes can never starve real work.
    """
    prompt = ""
    for m in getattr(ep, "messages", []):
        if getattr(m, "role", "") == "user":
            prompt = str(getattr(m, "text", "") or "").strip()
            break
    if not prompt:
        return None
    return {"task_id": f"probe:{ep.session_id}", "category": category,
            "prompt": prompt[:4000], "tests": [],
            "criteria": "Solve the task above without repeating the "
                        "errors recorded in this session's history.",
            "max_attempts": 1}


def _append_probe_tasks(memory_root: Path, probes: List[Dict[str, Any]]) -> int:
    """Append unseen probe tasks to tasks.json (dedup by task_id)."""
    if not probes:
        return 0
    tp = memory_root / "tasks.json"
    try:
        existing = json.loads(tp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        existing = []
    known = {t.get("task_id") for t in existing if isinstance(t, dict)}
    added = 0
    for t in probes:
        if t["task_id"] not in known:
            existing.append(t)
            known.add(t["task_id"])
            added += 1
    if added:
        tmp = tp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(existing, indent=1), encoding="utf-8")
        tmp.replace(tp)
    return added


#: Queue backstop: how many probe tasks may sit in tasks.json once they have
#: been processed. The fair queue (AutoRSIRuntime.fair_queue) is the actual
#: starvation fix — this cap only retires OLD, already-attempted probes.
MAX_PROBE_TASKS = 60


def _cap_probe_queue(memory_root: Path) -> int:
    """Retire old probe tasks AFTER they were processed (backstop, not the
    fix). A probe that never got an attempt is NEVER dropped — unprocessed
    work always deserves its turn, and the fair queue guarantees it gets
    one before the tail of the backlog. Returns the number removed.
    """
    tp = memory_root / "tasks.json"
    try:
        tasks = json.loads(tp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    probes = [(i, t) for i, t in enumerate(tasks)
              if str(t.get("task_id", "")).startswith("probe:")]
    if len(probes) <= MAX_PROBE_TASKS:
        return 0
    try:
        attempts = json.loads((memory_root / "task_attempts.json")
                              .read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        attempts = {}
    # probe ids embed a sortable timestamp: oldest first
    probes.sort(key=lambda it: str(it[1].get("task_id", "")))
    stale = probes[:-MAX_PROBE_TASKS]
    drop = {i for i, t in stale
            if int(attempts.get(str(t.get("task_id")), 0)) > 0}
    if not drop:
        return 0
    out = [t for i, t in enumerate(tasks) if i not in drop]
    tmp = tp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1), encoding="utf-8")
    tmp.replace(tp)
    return len(drop)


# -- wake self-test ------------------------------------------------------------

#: Task id of the endpoint self-test probe. Fixed ON PURPOSE: dedup by id
#: means the probe never accumulates in the queue — the same single task is
#: simply re-attempted (max_attempts=1) by every full dream that finds the
#: queue otherwise empty.
WAKE_TASK_ID = "wake:selftest"

_WAKE_PROMPT = (
    "Endpoint self-test (odr wake): write a Python function add(a, b) that "
    "returns the sum of its two arguments."
)
_WAKE_TESTS = ({"call": "add(2, 3)", "expected": 5},
               {"call": "add(-1, 1)", "expected": 0})


def wake_task(category: Optional[str] = None) -> Dict[str, Any]:
    """The wake self-test as a task dict: solvable by any competent model,
    verified by TESTS (never the judge), one attempt per full dream —
    i.e. one real API call against the configured endpoint."""
    return {"task_id": WAKE_TASK_ID, "category": category or "wake",
            "prompt": _WAKE_PROMPT,
            "tests": [dict(t) for t in _WAKE_TESTS],
            "max_attempts": 1}


def wake_enabled(flag: Optional[bool] = None) -> bool:
    """Resolve the wake switch: an explicit flag wins, otherwise the
    ``ODR_WAKE`` env var decides (default ON — an idle deployment should
    still exercise its endpoint once per full dream)."""
    if flag is not None:
        return bool(flag)
    return os.environ.get("ODR_WAKE", "1").lower() not in ("0", "false", "no")


def ingest_sessions(memory, sessions_db: "str | Path", *,
                    source: str = "hermes",
                    limit: int = 25, client=None,
                    category: Optional[str] = None) -> Dict[str, Any]:
    """Distil NEW (unseen) host sessions into staging lessons AND queue
    their source tasks as replay probes (the gate's counterparties).

    ``source="hermes"`` reads a Hermes ``state.db``; ``source="cursor"``
    reads the Cursor ``User`` dir (``state.vscdb``) directly — no MCP
    server involved, so this also works where MCP is blocked by policy.
    ``source="claude"`` reads Claude Code JSONL transcripts under
    ``~/.claude/projects`` (or the given projects dir).

    Returns {'new': n_sessions, 'added': n_lessons, 'probes': n_tasks,
             'deferred': n_deferred (endpoint failed; session left unseen)}.
    Sessions already processed are skipped — transcripts stay in the
    source store, so skipping a cycle loses nothing.
    """
    from open_dream_rsi.core.curator import (curate_lessons,
                                             evidence_snippets,
                                             KnowledgeCurator)
    from open_dream_rsi.sessions import episode_failures, read_hermes_sessions
    from open_dream_rsi.corrections import (
        dedupe_corrections,
        distillable,
        user_corrections,
    )

    root = Path(memory.root)
    state = _load_state(root)
    seen = set(state.get("seen_sessions", []))

    if source not in ("hermes", "cursor", "claude"):
        raise ValueError(f"unknown session source {source!r} "
                         "(expected 'hermes', 'cursor' or 'claude')")
    if source == "cursor":
        from open_dream_rsi.sessions import read_cursor_sessions
        eps = read_cursor_sessions(user_dir=sessions_db,
                                   limit=limit + len(seen))
    elif source == "claude":
        from open_dream_rsi.sessions import read_claude_code_sessions
        eps = read_claude_code_sessions(projects_dir=sessions_db,
                                        limit=limit + len(seen))
    else:
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
    deferred = 0
    probes: List[Dict[str, Any]] = []
    # Issue #6: user corrections are a second evidence class. Collect per
    # episode, dedupe deterministically ACROSS the batch before any paid
    # curator call, then distil failures + surviving corrections together
    # — one pipeline, one gate, no second learning path.
    ep_evidence = [(ep, episode_failures(ep), user_corrections(ep))
                   for ep in new]
    all_corr = distillable(
        dedupe_corrections([c for _, _, cs in ep_evidence for c in cs]))
    kept = {(c.get("session_id"), c.get("msg_index")) for c in all_corr}
    for ep, failures, corr_raw in ep_evidence:
        seen.add(ep.session_id)
        corrections = [c for c in corr_raw
                       if (c.get("session_id"), c.get("msg_index")) in kept]
        if corrections:
            memory.log_event("user_correction_found", source=ep.source,
                             session_id=ep.session_id, count=len(corrections),
                             classes=[c.get("class", "") for c in corrections][:6])
        evidence = failures + corrections
        if not evidence:
            continue
        cat = category or (Path(ep.cwd).name if ep.cwd else ep.source)
        distilled, err = curator.distill(cat, ep.title or cat, evidence,
                                         memory.get_lessons(cat))
        if err and err.startswith("llm error"):
            # Transport failure (connect/HTTP/response shape): this session
            # was never actually evaluated, so do NOT consume it — leave it
            # unseen for the next cycle with a live endpoint. Consuming here
            # would silently burn the evidence (a dead endpoint eats the
            # backlog one session per fire, and the tier stays full).
            seen.discard(ep.session_id)
            deferred += 1
            memory.log_event("lesson_deferred", source=ep.source,
                             session_id=ep.session_id,
                             reason=err[:300])
            continue
        probe = _probe_task(ep, cat)
        if probe is not None:
            probes.append(probe)
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
    n_probes = _append_probe_tasks(root, probes)
    if n_probes:
        memory.log_event("probe_tasks_queued", count=n_probes)
    return {"new": len(new), "added": added_total, "probes": n_probes,
            "deferred": deferred}


# -- the seam ------------------------------------------------------------------

def dream_once(memory_root: "str | Path", *,
               sessions_db: "str | Path | None" = None,
               session_source: str = "hermes",
               skills_out: "str | Path | None" = None,
               client=None, tasks: Optional[List[Any]] = None,
               budget: int = 100, max_tokens: int = 100_000,
               category: Optional[str] = None,
               min_new_evidence: int = DEFAULT_MIN_NEW_EVIDENCE,
               max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
               wake: Optional[bool] = None) -> Dict[str, Any]:
    """Run one automated cycle. A concurrent run returns {'skipped': True}.

    ``wake`` (default: the ``ODR_WAKE`` env var, on) seeds the endpoint
    self-test probe when a FULL dream finds the task queue empty, so an
    idle deployment still exercises its LLM once per full dream and
    ``odr metrics`` shows real api_calls instead of a silent heartbeat.
    """
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
                memory, sessions_db, source=session_source,
                client=client, category=category)

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
            if (full and not tasks and wake_enabled(wake)
                    and client is not None):
                # Idle deployment: seed the endpoint self-test so a FULL
                # dream always exercises the configured LLM once —
                # ``odr metrics`` shows real api_calls instead of a silent
                # heartbeat. Never replaces real work (an empty queue is
                # the precondition) and the fixed task_id dedups it.
                tasks = [Task(**wake_task(category))]
                report["wake"] = True
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

        # 3b. queue backstop — AFTER processing, never before. Starvation is
        # fixed by the loop's fair queue (least-processed task first); this
        # cap only retires probes that already had their one attempt. A probe
        # never attempted is NEVER dropped — it still deserves its turn.
        report["queue_pruned"] = _cap_probe_queue(root)

        # 4. publish earned knowledge (cheap; surfaces new ACTIVE lessons)
        if skills_out:
            report["skills"] = len(lessons_to_skills(memory, skills_out))
        return report
    finally:
        _release(lock)
