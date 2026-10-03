"""Host-agnostic error-class sentinel engine.

Why: curation in any agent host is epistemic — it reads outcomes after
episodes end and wakes on a schedule. Measured on a production session
store (256k messages / 30 days): a recurring error class re-appears
WITHIN one session after a median of ~1.5 minutes. No schedule wins that
race, so the reaction must sit in the tool-execution layer. The logic
below is host-independent; per-host adapters just normalize their native
event into ``observe()`` and forward the returned note into whatever
context channel the host provides:

* Hermes plugin   -> transform_tool_result return string   (plugins/hermes_sentinel/)
* Claude Code     -> PostToolUse/PostToolUseFailure hook   (plugins/claude_code/)
* any command-hook host -> ``python -m open_dream_rsi sentinel check``

Principles (from the 7-arm live replay study): the note is reactive
(failing calls only — clean calls pay zero tax), declarative (recurrence
facts + an explicit stop condition; imperative framing measurably extends
agent loops), and never blocks — the engine only annotates.

State: one JSON file, atomic replace, IO errors swallowed — this is an
annotation aid, never a correctness surface. Stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Tracked-class caps: an annotation aid, kept small on purpose.
MAX_SESSIONS_PER_SIG = 8
MAX_NOTIFIED_PER_SIG = 12
EXAMPLE_LEN = 140

# -- classification --------------------------------------------------------------

_ERR_RE = re.compile(
    r"(?i)\berror\b|traceback|exception|refused|denied|not found|no such|"
    r"failed|timed out|timeout|unauthorized|forbidden|permission|"
    r"\b(4\d\d|5\d\d)\b|exit code|non-zero|could not|unable to|missing"
)
_NUM_RE = re.compile(r"\d+")
_HEX_RE = re.compile(r"0x[0-9a-fA-F]+")
_PATH_RE = re.compile(r"(?:/|~/)[^\s'\"=,)\]}]+")
_URL_RE = re.compile(r"https?://[^\s'\"<>)]+")
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Collapse an error payload to its class: URLs/paths/numbers/hexes
    become tokens so different arguments hitting the same failure hash alike."""
    t = _URL_RE.sub("<url>", text)
    t = _HEX_RE.sub("<hex>", t)   # token must contain no digits: _NUM_RE runs after
    t = _PATH_RE.sub("<path>", t)
    t = _NUM_RE.sub("N", t)
    return _WS_RE.sub(" ", t).strip()[:160]


def classify(tool: str, result: Any,
             status: Optional[str], error_message: Optional[str] = None) -> Optional[str]:
    """Return the normalized error payload for this call, or None when it is
    not an error. ``status``: ``"error"``/``"ok"`` from hosts that report it
    (authoritative), ``None`` = not reported -> conservative fallback over
    the raw payload (error key / non-zero exit code / error keywords)."""
    if status == "error":
        payload = error_message or (result if isinstance(result, str) else "")
        return normalize(f"{tool}|{payload}") or f"{tool}|"
    if status is not None:
        return None  # the host says ok; do not second-guess it
    text = result if isinstance(result, str) else json.dumps(result, default=str)[:4000]
    try:
        data = json.loads(result) if isinstance(result, str) else result
    except Exception:
        data = None
    if isinstance(data, dict):
        if data.get("error") or data.get("error_message") or data.get("isError"):
            text = str(data.get("error") or data.get("error_message")
                       or data.get("stderr") or "tool error")[:2000]
            if not _ERR_RE.search(text or ""):
                return normalize(f"{tool}|{text}") or f"{tool}|"
        elif str(data.get("exit_code", data.get("exitCode", "0"))) not in ("0", "None"):
            # non-zero exit is an error regardless of wording. Built WITHOUT
            # normalize(): the code itself is the class (exit 1 vs exit 127
            # — failed vs command-not-found — are different blockers), and
            # the payload has no other variable content to fingerprint.
            code = str(data.get("exit_code", data.get("exitCode")))[:8]
            return f"{tool}|exit {code}"
        else:
            return None
    if not _ERR_RE.search(text or ""):
        return None
    return normalize(f"{tool}|{text}") or f"{tool}|"


def signature_key(tool: str, payload: str) -> str:
    return hashlib.sha1((tool + "|" + payload).encode()).hexdigest()[:12]


# -- note --------------------------------------------------------------------------

# Declarative, facts + explicit stop condition. The 7-arm replay showed
# imperative/command framing measurably EXTENDS agent loops; this template
# therefore reports recurrence counts and states when to stop, never orders
# the model to do anything specific.
NOTE_TEMPLATE = (
    "\n\n[Sentinel] Same error class ({tool}) is recurring: "
    "{in_session}x this session, {total}x across {sessions} sessions. "
    "Previous message: \"{example}\". "
    "These are facts, not instructions. Another attempt with different "
    "arguments that yields this same error means a permanent blocker: "
    "stop this path and report the blocker to the user."
)


def build_note(tool: str, rec: Dict[str, Any],
               in_session: int, sessions: int) -> str:
    return NOTE_TEMPLATE.format(
        tool=tool,
        in_session=in_session,
        total=rec.get("count", 0),
        sessions=sessions,
        example=str(rec.get("example", ""))[:110],
    )


def default_state_path() -> Path:
    p = os.environ.get("ODR_SENTINEL_STATE")
    if p:
        return Path(p)
    base = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    return Path(base) / "odr-sentinel" / "state.json"


class FileStore:
    """Default durable store: one JSON file, atomic replace. IO errors are
    swallowed — the in-memory ledger still works for this process."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> Dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def save(self, sigs: Dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(sigs), encoding="utf-8")
            os.replace(tmp, self.path)
        except Exception:
            pass  # annotation aid, never fatal


# -- engine --------------------------------------------------------------------------

class SentinelEngine:
    """Counts normalized error-class occurrences across sessions and emits a
    one-per-(class, session) declarative note at the repeat threshold.

    Thread-safe. Durable via a pluggable store (default: one JSON file);
    host adapters may inject e.g. a plugin-namespace key/value store with
    ``load()``/``save(dict)`` methods. Store failures are swallowed — this is
    an annotation aid, never a correctness surface. Stdlib only.
    """

    def __init__(self, state_path: Optional[Path] = None,
                 intra_session_repeat: int = 2,
                 cross_session_count: int = 2,
                 max_tracked: int = 500,
                 store: Any = None) -> None:
        self.store = store if store is not None else FileStore(
            state_path or default_state_path())
        self.intra = int(intra_session_repeat)
        self.cross = int(cross_session_count)
        self.max_tracked = int(max_tracked)
        self._lock = threading.Lock()
        with self._lock:
            self._sigs = self._read()

    # -- persistence

    def _read(self) -> Dict[str, Any]:
        try:
            data = self.store.load()
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write(self) -> None:
        try:
            self.store.save(self._sigs)
        except Exception:
            pass  # annotation aid, never fatal

    # -- core

    def observe(self, tool: str, result: Any, session_id: str,
                status: Optional[str] = None,
                error_message: Optional[str] = None) -> Optional[str]:
        """Feed one tool result. Returns the note to append to the failing
        result (once per class per session), or None to stay silent."""
        payload = classify(tool, result, status, error_message)
        if payload is None:
            return None
        with self._lock:
            rec = self._sigs.setdefault(
                signature_key(tool, payload),
                {"tool": tool, "example": payload[:EXAMPLE_LEN], "sessions": [],
                 "count": 0, "notified": [], "session_hits": {}, "last_ts": 0.0})
            rec["count"] = int(rec.get("count", 0)) + 1
            rec["last_ts"] = time.time()
            hits = rec.setdefault("session_hits", {})
            hits[session_id] = int(hits.get(session_id, 0)) + 1
            in_session = hits[session_id]
            sess = rec.setdefault("sessions", [])
            if session_id not in sess:
                sess.append(session_id)
                del sess[:-MAX_SESSIONS_PER_SIG]
                for dead in [k for k in hits if k not in sess]:
                    hits.pop(dead, None)
            n_sessions = len(sess)
            if len(self._sigs) > self.max_tracked:  # evict least-recently-seen
                order = sorted(self._sigs,
                               key=lambda k: self._sigs[k].get("last_ts", 0))
                for dead in order[:len(self._sigs) - self.max_tracked]:
                    self._sigs.pop(dead, None)
            notified = rec.setdefault("notified", [])
            silent = (session_id in notified
                      or not (in_session >= self.intra or n_sessions >= self.cross))
            if silent:
                self._write()
                return None
            notified.append(session_id)
            del notified[:-MAX_NOTIFIED_PER_SIG]
            self._write()
            return build_note(tool, rec, in_session, n_sessions)

    # -- introspection

    def ledger_text(self) -> str:
        with self._lock:
            top = sorted(self._sigs.items(),
                         key=lambda kv: -int(kv[1].get("count", 0)))[:8]
            n = len(self._sigs)
        lines = [f"Sentinel — {n} error classes tracked; top recurring:"]
        for _k, r in top:
            lines.append(f"  [{r.get('tool','?')}] x{r.get('count',0)} in "
                         f"{len(r.get('sessions', []))} sessions — "
                         f"{r.get('example','')[:90]}")
        if not top:
            lines.append("  (no error signatures recorded yet)")
        return "\n".join(lines)

    def reset(self) -> None:
        with self._lock:
            self._sigs = {}
            self._write()
