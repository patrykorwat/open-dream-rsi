"""sentinel — in-session error-class sentinel plugin for Hermes.

Thin adapter over the host-agnostic engine (open_dream_rsi.sentinel). The
engine holds ALL logic (classification, normalized error-class signatures,
the durable ledger, the declarative note); this adapter only:
  * bridges durability into Hermes' per-plugin state (ctx.state),
  * reads thresholds from Hermes plugin config,
  * registers the hook + slash command.

Why the mechanism lives here and not in the skill curator: curation wakes
on a schedule (weekly default) while recurring error classes re-appear
WITHIN one session after a median gap of 4.3 min (measured on a production
session store; anonymized replay fixture: fixtures/sentinel_audit.json). Reaction must be synchronous with tool execution.

The note is reactive (failing results only — zero tax on clean calls) and
declarative (recurrence facts + explicit stop condition, never a command;
imperative framing measurably extends loops — see the 7-arm replay study).

Hook choice (measured live on v0.21.5): transform_tool_result is the one
hook whose return value reaches the model — post_tool_call's return is
discarded. One hook, no ordering hazard, no double counting.

Falls back to the vendored _sentinel_engine.py when open_dream_rsi is not
importable, so the plugin installs by copying this directory alone.
"""
from __future__ import annotations

import importlib.util
import os
import threading
from typing import Optional


# -- engine: installed package first, vendored copy as fallback -------------------

def _engine_class():
    try:
        from open_dream_rsi.sentinel import SentinelEngine
        return SentinelEngine
    except Exception:
        pass
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "_sentinel_engine.py")
    spec = importlib.util.spec_from_file_location("_sentinel_engine", path)
    if spec is None or spec.loader is None:  # pragma: no cover - pathological
        raise ImportError(f"cannot load vendored engine: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SentinelEngine


# -- state bridge: engine store -> Hermes per-plugin ctx.state ----------------------

class _CtxStore:
    def __init__(self, ctx) -> None:
        self._ctx = ctx

    def load(self):
        try:
            data = self._ctx.state.get("sigs", {})
        except Exception:
            data = {}
        return data if isinstance(data, dict) else {}

    def save(self, sigs) -> None:
        try:
            self._ctx.state.set("sigs", sigs)
        except Exception:
            pass  # annotation aid, never fatal


_ENGINE = None
_LOCK = threading.Lock()
_CTX_REF: list = []


def _cfg(ctx, key: str, default: int) -> int:
    try:
        return int(ctx.get_config(key, default))
    except Exception:
        return default


def _get_engine(ctx):
    global _ENGINE
    with _LOCK:
        if _ENGINE is None:
            _ENGINE = _engine_class()(
                store=_CtxStore(ctx),
                intra_session_repeat=_cfg(ctx, "intra_session_repeat", 2),
                cross_session_count=_cfg(ctx, "cross_session_count", 2),
                max_tracked=_cfg(ctx, "max_tracked", 500))
    return _ENGINE


# -- hook -----------------------------------------------------------------------------

def on_transform_tool_result(**kw) -> Optional[str]:
    """Returns result + note at the threshold crossing, otherwise None
    (pass-through). Must never raise: any unexpected input degrades to
    pass-through."""
    try:
        result = kw.get("result")
        if not isinstance(result, str):
            return None
        engine = _get_engine(_CTX_REF[0])
        note = engine.observe(
            tool=str(kw.get("tool_name") or "?"),
            result=result,
            session_id=str(kw.get("session_id") or kw.get("task_id") or "?"),
            status=kw.get("status"),
            error_message=kw.get("error_message"))
        return (result + note) if note else None
    except Exception:
        return None


# -- slash command ---------------------------------------------------------------------

def _sentinel_cmd(raw: str = "") -> str:
    try:
        engine = _get_engine(_CTX_REF[0])
        if (raw or "").strip().lower() == "reset":
            engine.reset()
            return "Sentinel: signature ledger cleared."
        return engine.ledger_text()
    except Exception as exc:  # pragma: no cover - defensive
        return f"Sentinel unavailable: {exc!r}"


def register(ctx) -> None:
    _CTX_REF.append(ctx)
    _get_engine(ctx)  # eager bind + load durable table
    ctx.register_hook("transform_tool_result", on_transform_tool_result)
    try:
        ctx.register_command(
            "sentinel", _sentinel_cmd,
            description="recurring error-class ledger (sentinel plugin)",
            args_hint="[reset]")
    except Exception:
        pass
