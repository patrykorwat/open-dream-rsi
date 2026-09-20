"""Tools the agent can invoke online.

Execution of candidate code happens in an isolated subprocess
(``python -I`` with a scrubbed environment and a wall-clock timeout), so a
misbehaving candidate cannot leak the API keys held by the runtime.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

VERIFIER_TEMPLATE = """
import json, sys, traceback
candidate = {candidate!r}
tests = {tests!r}
ns = {{}}
report = {{"passed": 0, "total": len(tests), "errors": []}}
try:
    exec(compile(candidate, "candidate.py", "exec"), ns)
except Exception:
    report["error"] = traceback.format_exc(limit=3)
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

    def run(self, code: str, tests: List[Dict[str, Any]]) -> ToolResult:
        script = VERIFIER_TEMPLATE.format(candidate=code, tests=tests)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(script)
            path = fh.name
        try:
            proc = subprocess.run(
                [sys.executable, "-I", path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={"PATH": "/usr/bin:/bin"},  # scrubbed — no API keys inside the sandbox
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
