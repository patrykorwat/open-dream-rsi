"""odr-trigger — event-driven Dream-RSI cycle for Hermes (issue #3).

The architecture says learning happens per task: execution -> sentinel ->
world -> replay -> promotion. This plugin is that seam: the
`on_session_end` hook (fired by turn_finalizer at the end of every session)
runs one automated ODR cycle in a background thread.

Design rules:
  * NO shell scripts, NO flock files, NO cron — the plugin calls
    open_dream_rsi.dream.dream_once() directly, in-process. Single-instance
    locking and cadence (cheap maintenance per call; the full world dreamer
    on the age cadence) live in that module, so
    the same behaviour holds for the MCP tool and the CLI.
  * never blocks the session: fire-and-forget daemon thread.
  * never decides WHAT or WHETHER a policy is good — that is the
    promotion gate's job. This adapter decides WHEN only.
  * installs by copying this directory alone: if open_dream_rsi is not
    importable it registers nothing and says so once in the log.

Config (plugins.entries.odr-trigger.settings.*):
  memory       ODR memory dir           (default: $HERMES_HOME/dream_rsi)
  provider     LLM endpoint preset      (default: local)
  model        model override           (default: env ODR_LLM_MODEL)
  budget       max API calls per FULL dream (default 100)
  odr_root     checkout the CHILD imports open_dream_rsi from (default env
               ODR_ROOT, else /opt/data/workspace/odr-work)
"""
from __future__ import annotations

import os
import subprocess
import sys
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
        "provider": _cfg("provider", "local"),
        "model": _cfg("model_override", os.environ.get("ODR_LLM_MODEL")),
        "base_url": _cfg("base_url",
                         os.environ.get("ODR_LLM_BASE_URL")
                         or os.environ.get("OPENAI_BASE_URL")),
        "budget": int(_cfg("budget", 100)),
        "min_interval_seconds": float(_cfg("min_interval_seconds", 60)),
        # empty default -> _odr_root() fallback chain (ODR_ROOT env ->
        # /opt/odr -> sibling checkout -> installed package); deployments
        # that must dodge a stale /opt/odr set the setting explicitly
        "odr_root": str(_cfg("odr_root", os.environ.get("ODR_ROOT") or "")),
    }


def on_session_end(**kw) -> None:
    try:
        now = time.time()
        cfg = _resolve()
        if now - _STATE["last_fire"] < float(cfg["min_interval_seconds"]):
            return
        _STATE["last_fire"] = now
        # Detached subprocess, not a thread: CLI host processes exit via
        # os._exit seconds after the answer — any in-process thread dies
        # with them. The dream is one plain command (python -m
        # open_dream_rsi dream ...); its own lock collapses overlaps and
        # its own cadence logic decides the tier, so this stays honest.
        # --memory is a GLOBAL flag: it must precede the subcommand.
        cmd = [sys.executable, "-m", "open_dream_rsi",
               "--memory", cfg["memory"], "dream",
               "--provider", cfg["provider"],
               "--budget", str(cfg["budget"])]
        if cfg.get("model"):
            cmd += ["--model", cfg["model"]]
        env = {k: v for k, v in os.environ.items()
               if k.split("_")[0] in ("PATH", "HOME", "USER", "LANG",
                                      "LC_ALL", "TERM", "SHELL", "TMPDIR")
               or k.startswith("XDG_")
               or k.startswith("ODR_")}
        # OPENAI_* is deliberately NOT inherited from the host env: the
        # settings-driven OPENAI_BASE_URL below is the deployment's source
        # of truth, and a stray host value would silently retarget the
        # child to a different endpoint.
        # The child must import the checkout this deployment actually runs
        # (odr-work), not a stale root-owned copy: prefer the configured
        # odr_root for BOTH ODR_ROOT and PYTHONPATH — _odr_root() alone
        # would fall back to /opt/odr because the hook process has no
        # ODR_ROOT in its own environment.
        root_cfg = cfg.get("odr_root")
        root = root_cfg if (root_cfg
                            and Path(root_cfg, "open_dream_rsi").is_dir()
                            ) else _odr_root()
        if root_cfg and Path(root_cfg, "open_dream_rsi").is_dir():
            env["ODR_ROOT"] = root_cfg
        if root:
            env["PYTHONPATH"] = root
        env.setdefault("PYTHONUNBUFFERED", "1")
        if cfg.get("base_url"):
            env["OPENAI_BASE_URL"] = cfg["base_url"]
        if cfg.get("model"):
            env["ODR_LLM_MODEL"] = cfg["model"]
        # self-logging: if the child never runs, the log stays empty and
        # the error is visible instead of silent
        log = Path(cfg["memory"]) / "trigger.log"
        fh = None
        try:
            fh = log.open("a", encoding="utf-8")
            fh.write(f"{now} fire: {' '.join(cmd)}\n")
            fh.flush()
        except Exception:
            fh = None
        try:
            subprocess.Popen(cmd, stdout=(fh or subprocess.DEVNULL),
                             stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL,
                             start_new_session=True, env=env)
        except Exception as exc:
            if fh:
                fh.write(f"{now} popen-failed: {exc!r}\n")
            raise
        finally:
            if fh:
                fh.close()
    except Exception as exc:
        try:
            if _CTX is not None:
                _CTX.logger.warning("odr-trigger: trigger failed: %r", exc)
        except Exception:
            pass


_ROOT = None  # resolved once; the CHILD needs the path even when the
# hook process can already import the package (sys.path bootstrap does
# not cross the process boundary).


def _odr_root() -> str:
    """Locate an importable open_dream_rsi without host env setup (mass
    install): ODR_ROOT, /opt/odr, or a sibling checkout of this plugin."""
    global _ROOT
    if _ROOT is not None:
        return _ROOT
    candidates = [os.environ.get("ODR_ROOT", ""), "/opt/odr"]
    here = Path(__file__).resolve()
    candidates.append(str(here.parents[2]))  # .../open-dream-rsi/plugins/x/..
    for c in candidates:
        if c and (Path(c) / "open_dream_rsi").is_dir():
            _ROOT = c
            return c
    try:  # installed package: its own directory is the child's path too
        import open_dream_rsi
        _ROOT = str(Path(open_dream_rsi.__file__).resolve().parent.parent)
    except ImportError:
        _ROOT = ""
    return _ROOT


def _bootstrap_path() -> None:
    root = _odr_root()
    if root and root not in __import__("sys").path:
        __import__("sys").path.insert(0, root)


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
