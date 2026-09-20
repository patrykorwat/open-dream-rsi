"""Autonomous RSI runtime — the self-improvement loop that runs itself.

Hermes-style design: a supervisor daemon ("``odr loop``") owns the cycle

    wake -> pick tasks -> online attempt (LLM + sandbox) -> offline dream
         -> persist policy/recipe/tree -> sleep -> wake ...

Nothing waits for a human between cycles:

* **Persistent memory** (:class:`~open_dream_rsi.memory.DreamMemory`): each
  category keeps its best dreamed policy and best-known code recipe across
  restarts, so run N+1 starts where run N ended.
* **Self-scheduling**: ``run_forever`` sleeps ``interval_seconds`` between
  cycles; alternatively use ``--once`` under cron/systemd — memory makes the
  two modes interchangeable.
* **Budget guard**: every cycle stops early once the API-call budget runs
  out — dreaming is free, so improvement continues offline.
* **Event log**: every decision is appended to ``events.jsonl`` (audit trail).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from open_dream_rsi.core.agent import ChatClient
from open_dream_rsi.core.dreamer import DreamEngine
from open_dream_rsi.core.simulator import ReplaySimulator
from open_dream_rsi.core.tree import DiscoveryTree
from open_dream_rsi.memory import DreamMemory
from open_dream_rsi.tools import CodeVerifier

CANDIDATE_SYSTEM_PROMPT = (
    "You are a code-improvement agent in an autonomous recursive "
    "self-improvement loop. You receive a task, its test cases, your best "
    "previous solution and exploration-policy hints. Reply with ONLY the full "
    "Python solution in one fenced code block. No explanations."
)

REFINE_USER_TEMPLATE = """Task [{category}]: {prompt}

Test cases (call -> expected):
{tests}

Your best previous solution (may be absent):
{recipe}

Policy hints (higher temperature -> try something genuinely different):
{policy}

Previous failure feedback:
{feedback}

Return the complete corrected/optimized solution as one python code block."""


@dataclass
class Task:
    task_id: str
    category: str
    prompt: str
    tests: List[Dict[str, Any]]
    max_attempts: int = 4


@dataclass
class CycleReport:
    tasks_solved: int = 0
    tasks_attempted: int = 0
    api_calls: int = 0
    dream_iterations: int = 0
    improvements: List[str] = field(default_factory=list)
    stopped_reason: str = "completed"

    def to_dict(self) -> Dict[str, Any]:
        return vars(self)


class AutoRSIRuntime:
    """The supervisor that runs the Dream-RSI cycle automatically.

    Args:
        client: OpenAI-compatible LLM client (OpenAI, Cursor Models API, local).
        memory: persistent store; one directory = one self-improving instance.
        tasks:  task queue (a callable returning a fresh list is re-read each
                cycle, so you can grow the queue from outside).
        verifier: sandboxed code runner; defaults to :class:`CodeVerifier`.
        api_call_budget / dream_iterations / interval_seconds: loop guards.
    """

    def __init__(
        self,
        client: ChatClient,
        memory: DreamMemory,
        tasks: List[Task] | Callable[[], List[Task]],
        verifier: Optional[CodeVerifier] = None,
        api_call_budget: int = 20,
        dream_iterations: int = 60,
        interval_seconds: float = 300.0,
    ):
        self.client = client
        self.memory = memory
        self._tasks = tasks
        self.verifier = verifier or CodeVerifier()
        self.api_call_budget = api_call_budget
        self.dream_iterations = dream_iterations
        self.interval_seconds = interval_seconds
        self.api_calls_used = 0

    # -- one full cycle ----------------------------------------------------------

    def run_once(self) -> CycleReport:
        report = CycleReport()
        tasks = self._tasks() if callable(self._tasks) else list(self._tasks)
        for task in tasks:
            if self.api_calls_used >= self.api_call_budget:
                report.stopped_reason = "api_budget_exhausted"
                break
            solved, improved = self._work_on_task(task, report)
            report.tasks_attempted += 1
            if solved:
                report.tasks_solved += 1
            if improved:
                report.improvements.append(improved)

        self.memory.log_event("cycle", **report.to_dict())
        return report

    def run_forever(self, max_cycles: Optional[int] = None) -> None:
        """Daemon mode: wake -> cycle -> sleep -> repeat (Ctrl-C to stop)."""
        cycle = 0
        while max_cycles is None or cycle < max_cycles:
            cycle += 1
            self.api_calls_used = 0
            report = self.run_once()
            self.memory.log_event(
                "cycle_done", cycle=cycle, api_calls=self.api_calls_used
            )
            print(
                f"[odr] cycle {cycle}: {report.tasks_solved}/{report.tasks_attempted} solved, "
                f"{report.api_calls} API calls, {report.dream_iterations} dream its "
                f"({report.stopped_reason}); sleeping {self.interval_seconds:.0f}s",
                flush=True,
            )
            if max_cycles is None or cycle < max_cycles:
                time.sleep(self.interval_seconds)

    # -- task working loop -----------------------------------------------------------

    def _work_on_task(self, task: Task, report: CycleReport) -> tuple[bool, str]:
        category = task.category
        policy = self.memory.get_policy(category)
        recipe = self.memory.get_recipe(category)
        tree = self.memory.load_tree(task.task_id) or self._seed_tree(task, policy)

        feedback = ""
        best_node = max(tree.nodes.values(), key=lambda n: n.score) if tree.nodes else None
        if best_node and not best_node.children:
            feedback = "current best still failing/unfinished"

        solved = False
        improved = ""
        state_id = best_node.node_id if best_node else tree.root_id
        for _ in range(task.max_attempts):
            if self.api_calls_used >= self.api_call_budget:
                break
            code = self._propose_candidate(task, policy, recipe, feedback)
            self.api_calls_used += 1
            report.api_calls += 1
            if code is None:
                continue
            result = self.verifier.run(code, task.tests)
            node = tree.add_node(
                f"{task.task_id}/try-{len(tree.nodes)}",
                action="write_code",
                result={"code": code, "errors": result.detail},
                score=result.score,
                parent_id=state_id,
            )
            state_id = node.node_id
            if result.solved:
                solved = True
                if self.memory.save_recipe(category, code, score=result.score):
                    improved = f"{category}: new best solution (score {result.score})"
                break
            feedback = str(result.detail)[:600]

        # -- offline dreaming on whatever we collected (free, no API) -------------
        engine = DreamEngine(simulator=ReplaySimulator(tree))
        if policy:
            engine.policy_parameters = dict(policy)
        best_policy = engine.run_offline_optimization(iterations=self.dream_iterations)
        report.dream_iterations += self.dream_iterations
        if self.memory.get_policy(category) != best_policy:
            self.memory.save_policy(category, best_policy)
        self.memory.archive_tree(task.task_id, tree)
        self.memory.log_event(
            "task", task_id=task.task_id, category=category,
            solved=solved, nodes=len(tree.nodes), policy=best_policy,
        )
        return solved, improved

    # -- helpers --------------------------------------------------------------------

    def _seed_tree(self, task: Task, policy: Optional[Dict[str, float]]) -> DiscoveryTree:
        tree = DiscoveryTree()
        recipe = self.memory.get_recipe(task.category)
        seed_score = 0.05 if recipe else 0.0
        tree.add_node(f"{task.task_id}/seed", action="warm_start",
                      result={"recipe": bool(recipe)}, score=seed_score)
        return tree

    def _propose_candidate(
        self,
        task: Task,
        policy: Optional[Dict[str, float]],
        recipe: Optional[str],
        feedback: str,
    ) -> Optional[str]:
        temperature = float((policy or {}).get("temperature", 0.7))
        user = REFINE_USER_TEMPLATE.format(
            category=task.category,
            prompt=task.prompt,
            tests="\n".join(f"  {t['call']} -> {t['expected']!r}" for t in task.tests),
            recipe=recipe or "(none yet)",
            policy=json.dumps(policy or {}),
            feedback=feedback or "(none)",
        )
        try:
            reply = self.client.chat(
                [
                    {"role": "system", "content": CANDIDATE_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=1024,
            )
        except Exception as exc:  # provider error => skip this attempt, keep dreaming
            self.memory.log_event("llm_error", task_id=task.task_id, error=str(exc)[:300])
            return None
        return _extract_python_code(reply)


def _extract_python_code(reply: str) -> Optional[str]:
    """Pull the first fenced ```python (or plain ```) block out of a reply."""
    marker = "```"
    if marker not in reply:
        return reply.strip() or None
    parts = reply.split(marker)
    body = parts[1] if len(parts) > 1 else ""
    if body.lower().startswith("python"):
        body = body[body.index("\n") + 1:] if "\n" in body else ""
    return body.strip() or None
