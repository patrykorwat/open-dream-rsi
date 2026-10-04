"""Tools the agent can invoke online.

Execution of candidate code happens in an isolated subprocess
(:func:`open_dream_rsi.sandbox.run_isolated`: ``python -I``, scrubbed
environment, wall-clock timeout that kills the whole process group, and
POSIX address-space / CPU / process-count limits), so a misbehaving
candidate cannot leak the API keys held by the runtime or survive its own
timeout. Task-solution code is NOT statically gated — for hostile task
sources set ``ODR_SANDBOX_CMD`` (bubblewrap/nsjail) or containerise the
loop; see the sandbox module docstring.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from open_dream_rsi.sandbox import run_isolated

VERIFIER_TEMPLATE = """
import json, sys, traceback
candidate = {candidate!r}
tests = {tests!r}
ns = {{}}
report = {{"passed": 0, "total": len(tests), "errors": []}}
try:
    exec(compile(candidate, "candidate.py", "exec"), ns)
    report["defined"] = sorted(n for n in ns if not n.startswith("__"))
except Exception:
    report["error"] = traceback.format_exc(limit=3)
    print(json.dumps(report)); sys.exit(0)
# Shared fixtures: tests may carry a 'setup' source string (e.g.
# "FIX_A = '...'") defining names the call strings reference. Each unique
# setup is exec'd ONCE, so big inputs live in one place instead of being
# repeated inside every call expression.
_executed = set()
for t in tests:
    s = t.get("setup")
    if s and s not in _executed:
        _executed.add(s)
        try:
            exec(compile(s, "fixtures.py", "exec"), ns)
        except Exception:
            report["error"] = "fixture setup failed: " + traceback.format_exc(limit=3)
            print(json.dumps(report)); sys.exit(0)
for t in tests:
    try:
        got = eval(t["call"], ns)
        if got == t["expected"]:
            report["passed"] += 1
        else:
            report["errors"].append(f"{{t['call']}} -> {{got!r}} != {{t['expected']!r}}")
    except Exception as e:
        report["errors"].append(f"{{t['call']}} -> {{type(e).__name__}}: {{e}}")
print(json.dumps(report))
"""


@dataclass
class ToolResult:
    ok: bool
    score: float
    detail: Any
    solved: bool = False


class CodeVerifier:
    """Runs candidate code against test cases in a sandboxed subprocess.

    ``tests`` is a list of ``{"call": "add(2, 3)", "expected": 5}`` dicts.
    Score = fraction of passing tests; ``solved`` only when every test passes.
    """

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    def smoke_run(self, code: str) -> Dict[str, Any]:
        """Execute candidate code for evidence only (no tests evaluated).

        Returns {"ok", "error", "defined", "stdout"} — what the completion
        judge uses as sandbox evidence for test-less tasks. Same isolation
        boundary as :meth:`run`.
        """
        script = VERIFIER_TEMPLATE.format(candidate=code, tests=[])
        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(script)
            path = fh.name
        try:
            proc = run_isolated(
                [sys.executable, "-I", path],
                timeout=self.timeout,
                cwd=tempfile.gettempdir(),
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "execution timed out",
                    "defined": [], "stdout": ""}
        finally:
            Path(path).unlink(missing_ok=True)
        line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        try:
            report = json.loads(line)
        except (json.JSONDecodeError, IndexError):
            return {"ok": False, "error": f"verifier crash: {proc.stderr[:300]!r}",
                    "defined": [], "stdout": ""}
        defined = report.get("defined", [])
        return {"ok": "error" not in report, "error": report.get("error", ""),
                "defined": defined,
                "stdout": proc.stdout.replace(line, "").strip()[:500]}

    def run(self, code: str, tests: List[Dict[str, Any]]) -> ToolResult:
        script = VERIFIER_TEMPLATE.format(candidate=code, tests=tests)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(script)
            path = fh.name
        try:
            proc = run_isolated(
                [sys.executable, "-I", path],
                timeout=self.timeout,
                cwd=tempfile.gettempdir(),
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, 0.0, "execution timed out")
        finally:
            Path(path).unlink(missing_ok=True)

        line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        try:
            report = json.loads(line)
        except (json.JSONDecodeError, IndexError):
            return ToolResult(False, 0.0, f"verifier crash: {proc.stderr[:300]!r}")

        total = report.get("total") or len(tests)
        passed = report.get("passed", 0)
        score = passed / total if total else 0.0
        if "error" in report:  # candidate failed to even import
            return ToolResult(False, 0.0, report["error"])
        errors = report.get("errors", [])
        return ToolResult(ok=passed == total, score=score, detail=errors, solved=passed == total and total > 0)
