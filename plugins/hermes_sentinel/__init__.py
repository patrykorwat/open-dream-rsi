"""sentinel — in-session error-class sentinel for Hermes.

Why: the skill curator runs every ``curator.interval_hours`` (default 7
days) and skills are written after a task finishes. Measured on this
install's own session store (2026-10 audit, 256k messages / 30 days): a
recurring error class re-appears WITHIN one session after a median of ~1.5
minutes (848 within-session repeats; top class re-occurred across 204
sessions) — the slow curator never sees the signal while it is actionable.
This plugin closes that gap: it watches every tool result through the
official hook, tracks semantic error signatures (numbers/hex/paths/urls
normalized away, so "different args, same failure" is ONE signature), and
on the configured repeat threshold appends a short declarative note to the
failing result itself.

Design constraints from the ODR live-replay findings (v8/v9): the note
rides ONLY the failing tool result (reactive injection — zero prompt tax on
clean calls, unlike proactively injected lesson blocks that measurably cost
solve-rate), it is declarative with an explicit stop clause (never an
imperative "keep checking"), and it states recurrence facts, not commands.

Hook choice (measured live on v0.21.5): ``transform_tool_result`` is the
one hook with an "every tool result" contract on both dispatch paths whose
return value reaches the model — ``post_tool_call`` is an observer whose
return is discarded. All bookkeeping and the note decision therefore live
in ONE hook: no ordering hazard, no double counting.

Hook contract (keyword payloads; **kwargs for forward compat):
  transform_tool_result -> optional str return (first string wins;
  None = pass the original through unchanged).
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from typing import Any, Dict, Optional

_LOCK = threading.Lock()

# In-process mirror of the durable signature table (lazy-loaded from
# ctx.state). Cross-process write races are tolerable: worst case one lost
# increment on an annotation aid, never a correctness surface
# (atomic_json_write keeps the file valid).
_SIGS: Optional[Dict[str, Dict[str, Any]]] = None
_CTX = None  # PluginContext, kept for durable state + settings

# Tracked-class caps (kept small on purpose: this is an annotation aid, and
# the durable state namespace has a hard byte quota).
MAX_SESSIONS_PER_SIG = 8
MAX_NOTIFIED_PER_SIG = 12
EXAMPLE_LEN = 140

# -- signature -----------------------------------------------------------------

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


def _normalize(text: str) -> str:
    """Collapse an error payload to its class: URLs/paths/numbers/hexes
    become tokens so different arguments to the same failure hash alike."""
    t = _URL_RE.sub("<url>", text)
    t = _HEX_RE.sub("<hex>", t)   # token must contain no digits: _NUM_RE runs after
    t = _PATH_RE.sub("<path>", t)
    t = _NUM_RE.sub("N", t)
    return _WS_RE.sub(" ", t).strip()[:160]


def _classify(tool: str, result: Any,
              status: Optional[str], error_type: Optional[str],
              error_message: Optional[str]) -> Optional[str]:
    """Return the normalized error payload for this call, or None when it is
    not an error. Prefers the runtime's own observer fields (authoritative:
    they already exclude guardrail refusals and know exit codes)."""
    if status == "error":
        payload = error_message or (result if isinstance(result, str) else "")
        return _normalize(f"{tool}|{payload}") or f"{tool}|"
    if status is not None:
        return None  # the runtime says ok; do not second-guess it
    # status missing (hook fired without observer fields): conservative
    # fallback classifier over the raw payload.
    text = result if isinstance(result, str) else json.dumps(result, default=str)[:4000]
    try:
        data = json.loads(result) if isinstance(result, str) else result
    except Exception:
        data = None
    if isinstance(data, dict):
        if data.get("error") or data.get("error_message"):
            text = str(data.get("error") or data.get("error_message"))[:2000]
            if not _ERR_RE.search(text or ""):
                return _normalize(f"{tool}|{text}") or f"{tool}|"
        elif str(data.get("exit_code", "0")) not in ("0", "None"):
            # non-zero exit is an error regardless of wording
            return _normalize(f"{tool}|exit {data.get('exit_code')}") or f"{tool}|"
        else:
            return None
    if not _ERR_RE.search(text or ""):
        return None
    return _normalize(f"{tool}|{text}") or f"{tool}|"


def _key(tool: str, payload: str) -> str:
    return hashlib.sha1((tool + "|" + payload).encode()).hexdigest()[:12]


# -- state ---------------------------------------------------------------------

def _load(ctx=None) -> Dict[str, Any]:
    global _SIGS, _CTX
    if ctx is not None:
        _CTX = ctx
    if _SIGS is None:
        stored = _CTX.state.get("sigs", {}) if _CTX is not None else {}
        _SIGS = stored if isinstance(stored, dict) else {}
    return _SIGS


def _persist() -> None:
    if _CTX is None or _SIGS is None:
        return
    try:
        _CTX.state.set("sigs", _SIGS)
    except Exception:
        pass  # annotation aid, never fatal


def _cfg(name: str, default: int) -> int:
    try:
        return int(_CTX.get_config(name, default)) if _CTX is not None else default
    except Exception:
        return default


# -- note ------------------------------------------------------------------------

# Declarative, facts + explicit stop condition. ODR replay (7-arm study)
# showed imperative/command framing measurably EXTENDS agent loops; this
# template therefore reports recurrence counts and states when to stop, and
# never orders the model to do anything specific.
NOTE_TEMPLATE = (
    "\n\n[Sentinel] Same error class ({tool}) is recurring: "
    "{in_session}x this session, {total}x across {sessions} sessions. "
    "Previous message: \"{example}\". "
    "These are facts, not instructions. Another attempt with different "
    "arguments that yields this same error means a permanent blocker: "
    "stop this path and report the blocker to the user."
)


def _build_note(tool: str, rec: Dict[str, Any],
                in_session: int, sessions: int) -> str:
    return NOTE_TEMPLATE.format(
        tool=tool,
        in_session=in_session,
        total=rec.get("count", 0),
        sessions=sessions,
        example=str(rec.get("example", ""))[:110],
    )


# -- hook ------------------------------------------------------------------------

def on_transform_tool_result(**kw) -> Optional[str]:
    """Bookkeeping + one-time note per (signature, session).

    Every branch returns None (pass-through) except the single threshold
    crossing for a given session. Must never raise: any unexpected input
    degrades to pass-through."""
    try:
        tool = str(kw.get("tool_name") or "?")
        result = kw.get("result")
        if not isinstance(result, str):
            return None
        payload = _classify(tool, result,
                            kw.get("status"), kw.get("error_type"),
                            kw.get("error_message"))
        if payload is None:
            return None
        sid = str(kw.get("session_id") or kw.get("task_id") or "?")
        intra = _cfg("intra_session_repeat", 2)
        cross = _cfg("cross_session_count", 2)
        with _LOCK:
            sigs = _load()
            rec = sigs.setdefault(
                _key(tool, payload),
                {"tool": tool, "example": payload[:EXAMPLE_LEN], "sessions": [],
                 "count": 0, "notified": [], "session_hits": {}, "last_ts": 0.0})
            rec["count"] = int(rec.get("count", 0)) + 1
            rec["last_ts"] = time.time()
            hits = rec.setdefault("session_hits", {})
            hits[sid] = int(hits.get(sid, 0)) + 1
            seen_this_session = hits[sid]
            sess = rec.setdefault("sessions", [])
            if sid not in sess:
                sess.append(sid)
                del sess[:-MAX_SESSIONS_PER_SIG]
                for dead in [k for k in hits if k not in sess]:
                    hits.pop(dead, None)
            n_sessions = len(sess)
            cap = _cfg("max_tracked", 500)
            if len(sigs) > cap:  # evict least-recently-seen beyond the cap
                for dead in sorted(
                        sigs, key=lambda k: sigs[k].get("last_ts", 0)
                )[:len(sigs) - cap]:
                    sigs.pop(dead, None)
            notified = rec.setdefault("notified", [])
            if sid in notified:      # one note per session per signature
                _persist()
                return None
            if not (seen_this_session >= intra or n_sessions >= cross):
                _persist()
                return None
            notified.append(sid)
            del notified[:-MAX_NOTIFIED_PER_SIG]
            _persist()
            note = _build_note(tool, rec, seen_this_session, n_sessions)
        return result + note
    except Exception:
        return None


# -- slash command ---------------------------------------------------------------

def _sentinel_cmd(raw: str = "") -> str:
    if (raw or "").strip().lower() == "reset":
        with _LOCK:
            _load().clear()
            _persist()
        return "Sentinel: signature ledger cleared."
    with _LOCK:
        sigs = {k: dict(v) for k, v in _load().items()}
    top = sorted(sigs.items(), key=lambda kv: -int(kv[1].get("count", 0)))[:8]
    lines = [f"Sentinel — {len(sigs)} error classes tracked; top recurring:"]
    for _key_, r in top:
        lines.append(f"  [{r.get('tool','?')}] x{r.get('count',0)} in "
                     f"{len(r.get('sessions', []))} sessions — "
                     f"{r.get('example','')[:90]}")
    if not top:
        lines.append("  (no error signatures recorded yet)")
    return "\n".join(lines)


def register(ctx) -> None:
    _load(ctx)  # bind ctx and eager-load the durable table
    ctx.register_hook("transform_tool_result", on_transform_tool_result)
    try:
        ctx.register_command(
            "sentinel", _sentinel_cmd,
            description="recurring error-class ledger (sentinel plugin)",
            args_hint="[reset]")
    except Exception:
        pass
