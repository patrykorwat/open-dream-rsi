"""Shared process-isolation primitive for executing untrusted candidate code.

Both :class:`open_dream_rsi.tools.CodeVerifier` (task-solution code, which is
NOT statically gated) and :class:`open_dream_rsi.core.policygen.PolicySandbox`
(policy code, additionally gated by an AST validator) run candidates through
:func:`run_isolated` so there is exactly one isolation boundary:

* a ``python -I`` child process with a scrubbed environment (no API keys);
* a wall-clock timeout that kills the WHOLE process group, so a candidate
  cannot survive its own death by parking work in a background child;
* POSIX resource limits: address space, CPU seconds, process count;
* an optional jail prefix — set ``ODR_SANDBOX_CMD`` to a wrapper such as
  ``bwrap --unshare-all --die-with-parent --ro-bind / /`` or an ``nsjail``
  invocation and it is exec'd in front of the interpreter.

This is defence in depth, not a container. ``python -I`` alone does NOT
restrict filesystem or network access; deployments that ingest hostile task
sources should set ``ODR_SANDBOX_CMD`` (or run the whole loop inside a
container) so the OS enforces what the AST gate cannot.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
from typing import Dict, List, Optional

#: Environment handed to sandboxed children: no API keys, no user env.
SANDBOX_ENV: Dict[str, str] = {"PATH": "/usr/bin:/bin"}

#: Address-space cap for a sandboxed child (MB). Override with
#: ODR_SANDBOX_MEMORY_MB for candidates that legitimately need more.
DEFAULT_MEMORY_MB = 512

#: Thread-count cap for the sandboxed child (it must not fork/thread-bomb).
#: RLIMIT_NPROC counts ALL threads of the uid (shared with the parent
#: runtime), so the floor sits well above a busy interpreter's task count;
#: the live-uid measurement below raises it further when needed.
MAX_PROCS = 512

#: Env var naming a wrapper command prefixed to every sandbox spawn, e.g.
#: ODR_SANDBOX_CMD='bwrap --unshare-net --unshare-pid --die-with-parent --ro-bind / /'
JAIL_ENV_VAR = "ODR_SANDBOX_CMD"

_IS_POSIX = os.name == "posix"


def _memory_mb() -> int:
    raw = os.environ.get("ODR_SANDBOX_MEMORY_MB", "")
    try:
        return max(64, int(raw))
    except ValueError:
        return DEFAULT_MEMORY_MB


def _make_preexec(timeout: float):
    """preexec_fn: RLIMIT_AS / RLIMIT_CPU / RLIMIT_NPROC for the child."""
    import resource

    mem_bytes = _memory_mb() * 1024 * 1024
    cpu_soft = max(2, int(timeout) + 1)

    def _limits() -> None:  # runs in the forked child only
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_soft, cpu_soft + 1))
        try:
            # RLIMIT_NPROC counts THREADS of this uid (shared with the
            # parent), so the floor must sit ABOVE the uid's current task
            # count or every fork fails with EAGAIN. Count threads via
            # /proc, fall back to the static floor where /proc is absent.
            existing = 0
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                try:
                    with open(f"/proc/{entry}/status") as fh:
                        status = fh.read()
                    if f"Uid:\t{os.getuid()}\n" in status:
                        for line in status.splitlines():
                            if line.startswith("Threads:"):
                                existing += int(line.split()[1])
                except OSError:
                    continue
        except OSError:
            existing = 0
        cap = max(MAX_PROCS, existing + 64)
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (cap, cap))
        except (ValueError, OSError):
            pass  # platform without NPROC limits

    return _limits


def jail_prefix() -> List[str]:
    """The ODR_SANDBOX_CMD wrapper argv ([] when unset)."""
    raw = os.environ.get(JAIL_ENV_VAR, "").strip()
    return shlex.split(raw) if raw else []


def run_isolated(
    argv: List[str],
    *,
    timeout: float,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    """Run ``argv`` in an isolated child; raise TimeoutExpired past ``timeout``.

    On timeout the ENTIRE process group is SIGKILLed before the exception is
    re-raised — ``subprocess.run`` only kills the direct child, leaving any
    grandchildren the candidate spawned running unsupervised.

    Raises the same :class:`subprocess.TimeoutExpired` as ``subprocess.run``
    so callers keep their existing handling.
    """
    argv = [*jail_prefix(), *argv]
    env = SANDBOX_ENV if env is None else env
    kwargs = {}
    preexec = _make_preexec(timeout) if _IS_POSIX else None
    if _IS_POSIX:
        # own session -> own process group we can kill wholesale
        kwargs["start_new_session"] = True
        kwargs["preexec_fn"] = preexec
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=cwd,
        **kwargs,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        out, err = proc.communicate()
        raise
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        if _IS_POSIX:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


__all__ = ["run_isolated", "jail_prefix", "SANDBOX_ENV", "JAIL_ENV_VAR"]

# Interpreter path kept here so callers do not each hard-code it.
PYTHON = sys.executable
