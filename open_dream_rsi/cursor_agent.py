"""Dreamer brain via the Cursor Agent CLI (headless, no HTTP endpoint).

Cursor ships no public OpenAI-compatible chat API, so the zero-proxy route
to "the dreamer thinks with Cursor" is the ``cursor-agent`` CLI: it runs a
headless agent turn authenticated by the user's Cursor login — no proxy, no
API key, no endpoint to configure. Output text mode returns the final
answer only, which is all a chat client needs to return.

Failure semantics: every failure here raises RuntimeError, which the
loop records as an ``llm_error`` event — a machine without the CLI
simply runs endpoint-less (maintenance-only cycles) instead of crashing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Dict, List, Optional

#: Hard ceiling for one headless agent turn. A hang here must not
#: stall the dream cycle.
DEFAULT_TIMEOUT = 240.0

_IS_WIN = sys.platform.startswith("win")


class CursorAgentClient:
    """OpenAI-compatible-shaped client backed by ``cursor-agent -p``.

    Accepts the same ``chat(messages, model=..., temperature=..., max_tokens=...)``
    call shape as ``OpenAICompatibleClient`` so it drops in anywhere a client
    is expected (loop, judge). ``temperature``/``max_tokens`` have no
    CLI equivalent and are ignored on purpose — the agent applies its own
    defaults. ``model``: only passed through when set; unset means "whatever
    the user's Cursor is configured for" (the no-pinning default).
    """

    def __init__(self, model: Optional[str] = None,
                 executable: str = "cursor-agent",
                 timeout: Optional[float] = None) -> None:
        self.model = model or os.environ.get("ODR_LLM_MODEL") or None
        self.executable = executable
        self.timeout = float(
            timeout or os.environ.get("ODR_CURSOR_AGENT_TIMEOUT") or DEFAULT_TIMEOUT)

    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None,
             temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """One headless agent turn over the flattened conversation."""
        effective = model or self.model
        exe = shutil.which(self.executable)
        if exe is None:
            raise RuntimeError(
                "cursor-agent CLI not found on PATH — install the Cursor CLI "
                "(cursor.com/docs/cli) or pick another --provider")
        argv = [exe, "-p", "--force", "--output-format", "text"]
        if effective:
            argv += ["--model", effective]
        argv.append(_flatten(messages))
        if _IS_WIN and exe.lower().endswith((".cmd", ".bat")):
            # Windows npm shims are .cmd batch files: CreateProcess cannot
            # exec them directly, so ride cmd.exe /c (Python >= 3.12 quotes
            # batch-file arguments safely — the Store 3.13 runtime qualifies).
            argv = ["cmd.exe", "/c"] + argv
        try:
            proc = subprocess.run(
                argv, capture_output=True, timeout=self.timeout,
                text=True, encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"cursor-agent timed out after {self.timeout:g}s") from exc
        if proc.returncode != 0:
            raise RuntimeError(
                f"cursor-agent rc={proc.returncode}: "
                f"{(proc.stderr or proc.stdout or '')[:300]}")
        text = (proc.stdout or "").strip()
        if not text:
            raise RuntimeError("cursor-agent returned no output")
        return text


def _flatten(messages: List[Dict[str, str]]) -> str:
    """system+user messages -> one prompt string (role-labelled)."""
    parts = []
    for m in messages:
        role = str(m.get("role", "user"))
        content = str(m.get("content", ""))
        parts.append(f"[{role}]\n{content}" if role != "user" else content)
    return "\n\n".join(parts)
