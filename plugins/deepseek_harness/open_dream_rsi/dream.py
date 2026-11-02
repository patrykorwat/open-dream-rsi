"""Automated dreaming: the single seam every trigger calls.

One command, no shell, no cron, no flock script:

    python -m open_dream_rsi dream --memory .dream_rsi

This module owns the *cadence*, which is the part a harness should not
guess about. Two tiers, decided here rather than by whoever fires us:

  CHEAP  (every call, near-free — zero LLM)
    * runtime housekeeping on an empty cycle

  FULL   (the actual world dreamer — attempts + dreaming + replay gate)
    runs on the age cadence: >= max_age_seconds since the last full dream.
    The dreamer is expensive (LLM-written policies, replay-gated), so an
    idle deployment dreams on the age cadence only.

Everything that decides WHETHER a policy is good stays where it belongs —
the paired-replay promotion gate. This module only decides WHEN, so it is
portable to any harness (Hermes hook, MCP tool, or a human terminal).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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

# -- the seam ------------------------------------------------------------------

def dream_once(memory_root: "str | Path", *,
               client=None, tasks: Optional[List[Any]] = None,
               budget: int = 100, max_tokens: int = 100_000,
               category: Optional[str] = None,
               max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
               wake: Optional[bool] = None) -> Dict[str, Any]:
    """Run one automated cycle. A concurrent run returns {'skipped': True}.

    ``wake`` (default: the ``ODR_WAKE`` env var, on) seeds the endpoint
    self-test probe when a FULL dream finds the task queue empty, so an
    idle deployment still exercises its LLM once per full dream and
    ``odr metrics`` shows real api_calls instead of a silent heartbeat.
    """
    from open_dream_rsi.memory import DreamMemory

    root = Path(memory_root)
    memory = DreamMemory(root)
    lock = _acquire(root)
    if lock is None:
        return {"skipped": True, "reason": "another dream is running"}
    report: Dict[str, Any] = {"skipped": False}
    try:
        # 1. decide the tier
        state = _load_state(root)
        age = time.time() - float(state.get("last_full_dream", 0.0))
        full = age >= max_age_seconds
        report["tier"] = "full" if full else "maintenance"

        # 2. learning cycle (full: attempts + dreaming; maintenance: budget 0)
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

        return report
    finally:
        _release(lock)
