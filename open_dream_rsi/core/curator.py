"""Knowledge curator: turns failures into a curated, inspectable knowledge base.

Recipes and policy programs capture *what* worked; they say nothing about
*why* attempts fail. This module closes that gap the way Hermes manages its
own skills and memory: every cycle, the failures that verifier feedback
exposed are distilled by the LLM into short, deduplicated lesson records
(``lessons.json`` via :class:`~open_dream_rsi.memory.DreamMemory`), which are

* **retrieved** into the proposal prompt of future attempts (task trigger
  matching + win/usage ranking),
* **merged** instead of duplicated (normalised trigger+text is the key),
* **pruned** when they demonstrably stop helping (used often, never credited
  with a solve).

Unlike section-3 policy code, lessons are pure text — never executed — so
the gate is structural (schema + length caps + evidence required) plus the
retrieval feedback loop (usage/win counters), rather than sandboxed replay.

Lesson contract
---------------
A distilled payload is a JSON array of at most ``max_lessons`` objects::

    {"trigger": str, "text": str}

``trigger``: 1-4 lowercase keywords tying the lesson to a task/failure
family; ``text``: one actionable insight (<= 280 chars) grounded in the
failure evidence shown to the curator. Generic advice is rejected.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

#: Hard caps enforced by the structural gate — oversized or vague records
#: never enter the knowledge base (cheap gate: lessons are retrieved text,
#: so contamination is the failure mode to defend against).
MAX_LESSON_CHARS = 280
MIN_TRIGGER_WORDS = 1
MAX_TRIGGER_WORDS = 4
MAX_LESSONS_PER_CATEGORY = 8
#: Retrieval: how many lessons a single proposal prompt may carry and the
#: total character budget for them (keeps the solver prompt honest).
MAX_LESSONS_IN_PROMPT = 4
MAX_PROMPT_LESSON_CHARS = 700
#: Pruning: a lesson used this many times with zero recorded solves is
#: dead weight and gets evicted to keep the KB from rotting.
PRUNE_MIN_USES = 4
#: Evidence snippets stored per lesson (audit trail for `odr status`).
MAX_EVIDENCE_PER_LESSON = 5
#: Paired-replay promotion (lesson gate): a staging lesson activates only
#: if re-running recorded probe episodes WITH the lesson strictly beats
#: WITHOUT it (sign test: net wins > 0 and zero solve->fail regressions).
#: Live-model replay (see docs / paper §live) measured that trust-gated
#: lessons can turn a working solver into a budget-burning loop — form
#: gates validate shape, only behavioral replay validates effect.
LESSON_GATE_MIN_NET = 1

#: A lesson must carry its own stop condition: greedy models have no
#: internal counter that competes with an imperative to keep exploring.
_STOP_WORDS = (
    "stop", "answer", "respond", "return", "finish", "submit", "immediately",
    "at most", "no more", "once you have", "without further", "never fetch",
    "do not fetch", "skip", "max ", "then write", "reply", "terminate",
    "natychmiast", "odpowiedz", "zakończ", "maxymalnie", "nie pobieraj",
)


def has_stop_clause(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in _STOP_WORDS)

LESSON_CONTRACT = (
    "You are the knowledge curator of a Dream-RSI self-improvement loop. "
    "You receive the failed attempts of one task category: scores, verifier "
    "errors and the action that produced each. Distil AT MOST {max_lessons} "
    "durable lessons that would help a future attempt avoid these failures. "
    "Reply with ONLY one fenced ```json``` block containing an array of "
    "objects, each exactly: {{\"trigger\": \"...\", \"text\": \"...\"}}. "
    "'trigger' is 1-4 lowercase keywords tying the lesson to this failure "
    "family; 'text' is one actionable insight in at most 280 characters, "
    "grounded ONLY in the evidence shown. Phrase the insight as a "
    "DECLARATIVE fact about the task family, never as a command to the "
    "solver: measured on a greedy decoder, the imperative form of exact "
    "true facts lost 12 paired comparisons with 0 wins against their "
    "declarative form. EVERY lesson MUST end with an "
    "explicit stop condition (e.g. '...then answer immediately', '...at "
    "most one extra lookup') — never instruct open-ended searching: models "
    "with deterministic decoding treat 'keep checking' as a command to "
    "never finish. No generic advice ('write better code'), no "
    "task-specific constants, no restate-the-error entries; if the evidence "
    "supports no lesson, return an empty array []."
)


class LessonValidationError(ValueError):
    """A distilled lesson payload violated the lesson contract."""


@dataclass
class CurationResult:
    added: List[Dict[str, Any]]
    merged: int
    dropped: List[str]
    entries: List[Dict[str, Any]]

    @property
    def changed(self) -> bool:
        return bool(self.added) or self.merged > 0 or bool(self.dropped)


def normalize(text: str) -> str:
    """Canonical form for dedup keys: lowercase, collapsed punctuation."""
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


def lesson_key(entry: Dict[str, Any]) -> str:
    """Stable identity of a lesson: normalised trigger + text prefix."""
    return f"{normalize(str(entry.get('trigger', '')))}|{normalize(str(entry.get('text', '')))[:80]}"


def validate_lesson_items(items: Any, max_lessons: int) -> List[Dict[str, str]]:
    """Structural gate for a curator payload. Raises LessonValidationError."""
    if not isinstance(items, list):
        raise LessonValidationError("payload is not a JSON array")
    out: List[Dict[str, str]] = []
    seen = set()
    for it in items:
        if not isinstance(it, dict):
            raise LessonValidationError("lesson entry is not an object")
        trig = str(it.get("trigger", "")).strip()
        text = str(it.get("text", "")).strip()
        words = normalize(trig).split()
        if not (MIN_TRIGGER_WORDS <= len(words) <= MAX_TRIGGER_WORDS):
            raise LessonValidationError(
                f"trigger must be {MIN_TRIGGER_WORDS}-{MAX_TRIGGER_WORDS} words, got {trig!r}")
        if not (20 <= len(text) <= MAX_LESSON_CHARS):
            raise LessonValidationError(
                f"lesson text must be 20-{MAX_LESSON_CHARS} chars, got {len(text)}")
        key = lesson_key({"trigger": trig, "text": text})
        if key in seen:
            continue
        seen.add(key)
        out.append({"trigger": trig, "text": text})
        if len(out) >= max_lessons:
            break
    return out


def extract_json_array(reply: str) -> Optional[List[Any]]:
    """Pull a JSON array out of a reply (fenced ```json block or bare text)."""
    s = (reply or "").strip()
    if "```" in s:
        parts = s.split("```")
        body = parts[1] if len(parts) > 1 else ""
        if body.lower().lstrip().startswith("json"):
            body = body[body.index("\n") + 1:] if "\n" in body else ""
        s = body.strip()
    start, end = s.find("["), s.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(s[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


def curate_lessons(existing: List[Dict[str, Any]],
                   distilled: List[Dict[str, str]],
                   evidence: Optional[List[str]] = None,
                   staging: bool = True) -> CurationResult:
    """Merge validated lessons into the KB: dedupe, refresh evidence, prune.

    Same-key entries MERGE (evidence grows, timestamp refreshes) instead of
    duplicating. Dead lessons (used often, zero wins) are evicted; the
    strongest ``MAX_LESSONS_PER_CATEGORY`` survive by (wins, uses, recency).

    New lessons enter as ``status="staging"`` (default): they never reach a
    proposal prompt until the paired-replay gate promotes them (the loop's
    LessonGate) — measured on live models, freshly distilled lessons can be
    NET HARMFUL (they bias greedy solvers into never finishing), so form
    gates alone must not activate them. ``staging=False`` marks entries
    directly active: use it only for pre-vetted/human-seeded knowledge.
    """
    status = "staging" if staging else "active"
    by_key = {lesson_key(l): dict(l) for l in existing}
    added: List[Dict[str, Any]] = []
    merged = 0
    now = time.time()
    for item in distilled:
        key = lesson_key(item)
        if key in by_key:
            entry = by_key[key]
            ev = list(entry.get("evidence") or [])
            for snippet in (evidence or []):
                if snippet and snippet not in ev:
                    ev.append(snippet[:160])
            entry["evidence"] = ev[-MAX_EVIDENCE_PER_LESSON:]
            entry["updated_at"] = now
            merged += 1
        else:
            entry = {"trigger": item["trigger"], "text": item["text"],
                     "created_at": now, "updated_at": now,
                     "wins": 0, "uses": 0, "status": status,
                     "evidence": [e[:160] for e in (evidence or [])][:MAX_EVIDENCE_PER_LESSON]}
            by_key[key] = entry
            added.append(entry)
    dropped: List[str] = []
    for key, entry in by_key.items():
        if int(entry.get("uses", 0)) >= PRUNE_MIN_USES and int(entry.get("wins", 0)) == 0:
            dropped.append(key)
    survivors = [e for k, e in by_key.items() if k not in set(dropped)]
    if len(survivors) > MAX_LESSONS_PER_CATEGORY:
        survivors.sort(key=lambda e: (int(e.get("wins", 0)), int(e.get("uses", 0)),
                                      float(e.get("updated_at", 0))), reverse=True)
        for e in survivors[MAX_LESSONS_PER_CATEGORY:]:
            dropped.append(lesson_key(e))
        survivors = survivors[:MAX_LESSONS_PER_CATEGORY]
    return CurationResult(added=added, merged=merged, dropped=dropped,
                          entries=survivors)


def select_lessons(lessons: List[Dict[str, Any]], task_prompt: str,
                   max_lessons: int = MAX_LESSONS_IN_PROMPT,
                   max_chars: int = MAX_PROMPT_LESSON_CHARS,
                   include_staging: bool = False) -> List[Dict[str, Any]]:
    """Rank lessons for one proposal: task-trigger overlap, then wins/uses.

    Only ``status="active"`` lessons reach a proposal prompt (entries
    without a ``status`` field predate the gate and count as active).
    Staging candidates are invisible until the paired-replay gate promotes
    them — the live-model replay showed freshly distilled lessons are
    net-harmful by default, so activation must be earned, never assumed.
    ``include_staging=True`` is for the gate itself (and tests).
    """
    if not include_staging:
        lessons = [l for l in lessons
                   if str(l.get("status", "active")) == "active"]
    prompt_words = set(normalize(task_prompt).split())
    def rank(l: Dict[str, Any]) -> tuple:
        trig_words = set(normalize(str(l.get("trigger", ""))).split())
        overlap = len(trig_words & prompt_words)
        return (overlap, int(l.get("wins", 0)), int(l.get("uses", 0)),
                float(l.get("updated_at", 0)))
    ranked = sorted(lessons, key=rank, reverse=True)
    out: List[Dict[str, Any]] = []
    budget = max_chars
    for l in ranked:
        text = str(l.get("text", ""))
        if not text or len(text) > budget:
            continue
        out.append(l)
        budget -= len(text)
        if len(out) >= max_lessons or budget <= 0:
            break
    return out


def format_lessons(lessons: List[Dict[str, Any]]) -> str:
    """Render selected lessons for the proposal prompt (plain bullets)."""
    return "\n".join(f"  - [{l.get('trigger', '')}] {l.get('text', '')}"
                     for l in lessons)


@dataclass
class GateVerdict:
    promoted: List[str]
    rejected: List[str]
    detail: Dict[str, Any]


def lesson_gate_verdict(results: Dict[str, List[Tuple[bool, bool]]]) -> GateVerdict:
    """Decide staging lessons from paired replay outcomes.

    ``results`` maps lesson_key -> list of (solved_with, solved_without)
    trial pairs recorded by the loop's replay gate. Promotion rule (sign
    test, harm-asymmetric):

    * net = #(fail->solve) - #(solve->fail) >= LESSON_GATE_MIN_NET, and
    * zero solve->fail regressions, and
    * the lesson text carries a stop clause (``has_stop_clause``) — an
      unbounded exploration instruction must never activate on a greedy
      solver no matter what a small sample says.
    """
    promoted, rejected, detail = [], [], {}
    for key, pairs in results.items():
        gains = sum(1 for w, wo in pairs if w and not wo)
        regs = sum(1 for w, wo in pairs if wo and not w)
        net = gains - regs
        ok = net >= LESSON_GATE_MIN_NET and regs == 0
        detail[key] = {"gains": gains, "regressions": regs, "net": net,
                       "promoted": ok}
        (promoted if ok else rejected).append(key)
    return GateVerdict(promoted, rejected, detail)


def evidence_snippets(failures: List[Dict[str, Any]], max_items: int = 6) -> List[str]:
    """Render failed-attempt records for the curator prompt + KB audit."""
    lines: List[str] = []
    for f in failures[-max_items:]:
        errs = "; ".join(str(e)[:80] for e in (f.get("errors") or [])[:3])
        lines.append(f"score={f.get('score', 0.0):.3f} action={f.get('action', '?')[:40]}"
                     + (f" errors=[{errs}]" if errs else ""))
    return lines


class KnowledgeCurator:
    """Distils verified failure evidence into validated lesson records.

    One distillation costs one API call (plus one repair call when the
    payload fails the structural gate). Nothing enters the knowledge base
    without passing :func:`validate_lesson_items`.
    """

    def __init__(self, client: Any, max_lessons: int = 3,
                 max_repair: int = 1, max_tokens: Optional[int] = None):
        self.client = client
        self.max_lessons = max_lessons
        self.max_repair = max_repair
        self.max_tokens = max_tokens

    def distill(self, category: str, task_prompt: str,
                failures: List[Dict[str, Any]],
                existing: List[Dict[str, Any]],
                api_calls: Optional[List[int]] = None) -> Tuple[List[Dict[str, str]], str]:
        """Return ``(validated_lessons, error)`` — empty list + error on failure."""
        evidence = evidence_snippets(failures)
        if not evidence:
            return [], "no failure evidence"
        known = "\n".join(f"  - [{l.get('trigger', '')}] {l.get('text', '')}"
                          for l in existing[-MAX_LESSONS_PER_CATEGORY:]) or "(empty KB)"
        feedback = ""
        for attempt in range(1 + self.max_repair):
            user = (f"Category [{category}]: {task_prompt}\n\n"
                    f"Lessons already in the knowledge base (do not restate):\n{known}\n\n"
                    f"Failed attempts this cycle:\n" + "\n".join(evidence) + "\n\n"
                    f"Previous rejection reason: {feedback or '(none)'}\n\n"
                    f"Return the JSON array of at most {self.max_lessons} lessons.")
            kwargs: Dict[str, Any] = {"temperature": 0.5}  # distillation, not creativity
            if self.max_tokens:
                kwargs["max_tokens"] = self.max_tokens
            try:
                reply = self.client.chat(
                    [{"role": "system", "content": LESSON_CONTRACT.format(max_lessons=self.max_lessons)},
                     {"role": "user", "content": user}],
                    **kwargs)
            except Exception as exc:
                return [], f"llm error: {exc}"
            if api_calls is not None:
                api_calls[0] += 1
            items = extract_json_array(reply or "")
            if items is None:
                feedback = "reply contained no JSON array"
                continue
            try:
                validated = validate_lesson_items(items, self.max_lessons)
            except LessonValidationError as exc:
                feedback = f"lesson validation failed: {exc}"
                continue
            return validated, ""
        return [], feedback or "distillation failed"


# -- headroom precondition (live-replay v3-v9 finding) --------------------------
#
# A cold model that already solves a family cheaply leaves no room for
# guidance to pay for its prompt-perturbation tax: across seven published
# injection mechanisms (imperative/declarative lessons, exemplars, abstract
# workflows, evidence certificates, reactive-on-error, end-position framing)
# NOT ONE beat a contemporaneous cold baseline of 19/20 solves at 3.0
# calls/solve — every arm had zero paired gains (worst: 11/20, p=0.008).
# Curation effort belongs to categories with MEASURED headroom instead.

#: Minimum recent solve-rate below which a category has headroom for lessons.
HEADROOM_MIN_SOLVE_RATE = 0.7
#: Maximum baseline calls-per-solve above which a category has headroom.
HEADROOM_MAX_CALLS_PER_SOLVE = 5.0
#: How many recent per-task outcomes the verdict is computed over.
HEADROOM_WINDOW = 10


def headroom_verdict(outcomes: List[Dict[str, Any]],
                     min_solve_rate: float = HEADROOM_MIN_SOLVE_RATE,
                     max_calls_per_solve: float = HEADROOM_MAX_CALLS_PER_SOLVE,
                     ) -> Dict[str, Any]:
    """Decide whether a category has room for lessons to pay off.

    ``outcomes``: per-task records ``{"solved": bool, "calls": int}`` (the
    loop's own proposal-call counts; the most recent ``HEADROOM_WINDOW`` are
    used). Headroom exists when the recent solve-rate is below
    ``min_solve_rate`` OR the mean calls-per-solve exceeds
    ``max_calls_per_solve``. An empty history has headroom: a fresh category
    gets the curator's benefit of the doubt. The stats used are returned so
    the decision is auditable in ``events.jsonl``.
    """
    recent = list(outcomes)[-HEADROOM_WINDOW:]
    if not recent:
        return {"has_headroom": True, "n": 0, "solve_rate": None,
                "calls_per_solve": None, "reason": "no history yet"}
    n = len(recent)
    solved = [o for o in recent if o.get("solved")]
    solve_rate = len(solved) / n
    cps = (sum(int(o.get("calls", 0)) for o in solved) / len(solved)) \
        if solved else None
    if solve_rate < min_solve_rate:
        return {"has_headroom": True, "n": n, "solve_rate": round(solve_rate, 3),
                "calls_per_solve": None if cps is None else round(cps, 2),
                "reason": f"solve_rate {solve_rate:.2f} < {min_solve_rate}"}
    if cps is not None and cps > max_calls_per_solve:
        return {"has_headroom": True, "n": n, "solve_rate": round(solve_rate, 3),
                "calls_per_solve": round(cps, 2),
                "reason": f"calls/solve {cps:.2f} > {max_calls_per_solve}"}
    return {"has_headroom": False, "n": n, "solve_rate": round(solve_rate, 3),
            "calls_per_solve": None if cps is None else round(cps, 2),
            "reason": "baseline solves cheaply: lessons can only pay their "
                      "perturbation tax (live replay v3-v9, zero paired gains)"}
