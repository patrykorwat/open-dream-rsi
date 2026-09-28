"""LLM-based task-completion judge — automatic verdicts for test-less tasks.

The loop's primary verdict source is the sandboxed :class:`CodeVerifier`
running explicit ``{call, expected}`` tests. Real-world tasks queued from a
harness often arrive WITHOUT tests — only a prompt and maybe a success
criterion. Without a verdict source such tasks can never be marked solved,
so recipes/lessons never form around them.

:class:`LLMJudge` closes that gap under three hard rules:

1. **Tests dominate.** The judge is consulted ONLY when a task has no
   tests; it can never override a failing verifier.
2. **Evidence, not self-report.** The judge sees the candidate code plus
   sandbox evidence (does it import cleanly, what does it define, what
   error did it throw) — never only the model's claim.
3. **Fail-closed.** The verdict must be a strict JSON object
   ``{"solved": bool, "score": 0..1, "reason": str}``. Anything else
   parses to NOT solved. Judge calls are charged to the same API budget
   guard as proposals, dreaming, and curation.

The judge's ``reason`` feeds the existing feedback channel, so a rejected
attempt gets the judge's complaint on the next proposal — the same
corrective loop that verifier errors drive for test-bearing tasks.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from open_dream_rsi.core.agent import ChatClient

JUDGE_SYSTEM = (
    "You are a strict task-completion judge in an autonomous code-improvement "
    "loop. Decide ONLY from the provided evidence whether the candidate "
    "solution satisfies the task. A clean import or plausible-looking code is "
    "NOT sufficient: the task's behaviour must be demonstrated or the "
    "criteria explicitly met. Reply with exactly one JSON object and nothing "
    'else: {"solved": true|false, "score": 0.0-1.0, "reason": "<short>"}'
)

JUDGE_USER_TEMPLATE = """Task: {prompt}

Success criteria (as given by the task author; may be empty):
{criteria}

Candidate code:
```python
{code}
```

Sandbox execution evidence:
{evidence}

Verdict JSON:"""

_VERDICT_RE = re.compile(r"\{.*\}", re.S)


def parse_verdict(text: str) -> Optional[Dict[str, Any]]:
    """Strictly parse a judge reply into a verdict dict; None on anything else.

    Accepts only a JSON object with bool 'solved', numeric 'score' clamped
    to [0, 1] and a string 'reason'. Extra keys are ignored.
    """
    m = _VERDICT_RE.search(text or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("solved"), bool):
        return None
    score = obj.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        return None
    reason = obj.get("reason")
    if not isinstance(reason, str):
        return None
    return {"solved": obj["solved"], "score": max(0.0, min(1.0, float(score))),
            "reason": reason.strip()[:400]}


class LLMJudge:
    """Asks the frozen model for a strict verdict on a test-less attempt."""

    def __init__(self, client: ChatClient, max_tokens: int = 256):
        self.client = client
        self.max_tokens = max_tokens

    def judge(self, prompt: str, criteria: str, code: str,
              evidence: str) -> Dict[str, Any]:
        """Return {"solved", "score", "reason"}; fail-closed on any mess."""
        user = JUDGE_USER_TEMPLATE.format(
            prompt=prompt, criteria=criteria.strip() or "(none given)",
            code=code[:4000], evidence=evidence[:1500])
        try:
            raw = self.client.chat(
                [{"role": "system", "content": JUDGE_SYSTEM},
                 {"role": "user", "content": user}],
                max_tokens=self.max_tokens, temperature=0.0)
        except Exception as exc:  # transport error = no verdict, not a pass
            return {"solved": False, "score": 0.0,
                    "reason": f"judge unavailable: {exc}"}
        verdict = parse_verdict(raw)
        if verdict is None:
            return {"solved": False, "score": 0.0,
                    "reason": "judge returned no strict verdict (fail-closed)"}
        return verdict
