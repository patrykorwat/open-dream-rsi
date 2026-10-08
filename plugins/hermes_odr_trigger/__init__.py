"""odr-trigger — event-driven Dream-RSI cycle for Hermes (issue #3).

The architecture says learning happens per task: execution -> sentinel ->
world -> replay -> promotion. This plugin is that seam: the
`on_session_end` hook (fired by turn_finalizer at the end of every session)
runs one automated ODR cycle in a background thread.

Design rules:
  * NO shell scripts, NO flock files, NO cron — the plugin calls
    open_dream_rsi.dream.dream_once() directly, in-process. Single-instance
    locking and cadence (cheap maintenance per call; the full world dreamer
    only when evidence or staleness justifies it) live in that module, so
    the same behaviour holds for the MCP tool and the CLI.
  * never blocks the session: fire-and-forget daemon thread.
  * never decides WHAT or WHETHER a lesson is good — that is the
    promotion gate's job. This adapter decides WHEN only.
  * installs by copying this directory alone: if open_dream_rsi is not
    importable it registers nothing and says so once in the log.

Config (plugins.entries.odr-trigger.settings.*):
  memory       ODR memory dir           (default: $HERMES_HOME/dream_rsi)
  sessions_db  host state.db            (default: $HERMES_HOME/state.db)
  skills_out   where SKILL.md files go  (default: $HERMES_HOME/skills/odr-curated)
  provider     LLM endpoint preset      (default: local)
  model        model override           (default: env ODR_LLM_MODEL)
  budget       max API calls per FULL dream (default 20)
"""
from __future__ import annotations

import atexit
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict

_STATE = {"last_fire": 0.0, "warned": False}
_CTX = None  # set by register()


def _cfg(key: str, default):
    try:
        if _CTX is not None:
            return _CTX.get_config(key, default)
    except Exception:
        pass
    return default


def _resolve() -> Dict[str, Any]:
    home = Path(os.environ.get("HERMES_HOME", "/opt/data"))
    return {
        "memory": str(_cfg("memory", home / "dream_rsi")),
        "sessions_db": str(_cfg("sessions_db", home / "state.db")),
        "skills_out": str(_cfg("skills_out",
                               home / "skills" / "odr-curated")),
        "provider": _cfg("provider", "local"),
        "model": _cfg("model", os.environ.get("ODR_LLM_MODEL")),
        "base_url": _cfg("base_url",
                         os.environ.get("ODR_LLM_BASE_URL")
                         or os.environ.get("OPENAI_BASE_URL")),
        "budget": int(_cfg("budget", 20)),
        "budget": int(_cfg("budget", 20)),
        "min_interval_seconds": float(_cfg("min_interval_seconds", 60)),
    }


def _run(cfg: Dict[str, Any]) -> None:
    try:
        from open_dream_rsi.dream import dream_once

        client = None
        if cfg.get("provider") != "mock":
            try:
                from open_dream_rsi.cli import _build_client
                client = _build_client(cfg["provider"], cfg.get("model"))
                if hasattr(client, "config"):
                    if cfg.get("base_url"):
                        client.config.base_url = cfg["base_url"]
                    # self-hosted endpoints: skip hidden reasoning (measured:
                    # it eats the completion budget)
                    if "api." not in client.config.base_url:
                        client.config.extra_payload.setdefault(
                            "chat_template_kwargs", {})["enable_thinking"] = False
            except Exception:
                client = None  # no endpoint -> maintenance-only cycle
        dream_once(cfg["memory"], sessions_db=cfg["sessions_db"],
                   skills_out=cfg["skills_out"], client=client,
                   budget=cfg["budget"])
    except Exception as exc:  # never surface in the user's session
        try:
            if _CTX is not None:
                _CTX.logger.warning("odr-trigger: dream failed: %r", exc)
        except Exception:
            pass


def on_session_end(**kw) -> None:
    try:
        now = time.time()
        cfg = _resolve()
        if now - _STATE["last_fire"] < float(cfg["min_interval_seconds"]):
            return
        _STATE["last_fire"] = now
        # Non-daemon: a CLI host process exits seconds after the session
        # ends — a daemon thread would die mid-dream. atexit gives it a
        # bounded grace period, then lets shutdown proceed (the lock makes
        # an unfinished dream harmless: the next trigger reclaims it).
        t = threading.Thread(target=_run, args=(cfg,), daemon=False)
        t.start()
        atexit.register(t.join, float(cfg.get("shutdown_grace_seconds", 90)))
    except Exception as exc:
        try:
            if _CTX is not None:
                _CTX.logger.warning("odr-trigger: trigger failed: %r", exc)
        except Exception:
            pass


def _bootstrap_path() -> None:
    """Make open_dream_rsi importable without host env setup (mass install):
    ODR_ROOT, /opt/odr, or a sibling checkout next to this plugin."""
    candidates = [os.environ.get("ODR_ROOT", ""), "/opt/odr"]
    here = Path(__file__).resolve()
    candidates.append(str(here.parents[2]))  # .../open-dream-rsi/plugins/x/..
    for c in candidates:
        if c and (Path(c) / "open_dream_rsi").is_dir():
            if c not in __import__("sys").path:
                __import__("sys").path.insert(0, c)
            return


def register(ctx) -> None:
    global _CTX
    try:
        _bootstrap_path()
        import open_dream_rsi  # noqa: F401
    except ImportError:
        if not _STATE["warned"]:
            _STATE["warned"] = True
            try:
                ctx.logger.warning(
                    "odr-trigger: open_dream_rsi not importable — install "
                    "the package (PYTHONPATH=/opt/odr or pip install "
                    "open-dream-rsi); hook stays inert")
            except Exception:
                pass
        return
    _CTX = ctx
    ctx.register_hook("on_session_end", on_session_end)
