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
from open_dream_rsi.core.policygen import (
    PolicyGenerator,
    PolicySandbox,
    greedy_replay_score,
    replay_world,
)
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
        on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        max_tokens: int = 2048,
        enable_policy_code: bool = True,
    ):
        self.client = client
        self.memory = memory
        self._tasks = tasks
        self.verifier = verifier or CodeVerifier()
        self.api_call_budget = api_call_budget
        self.dream_iterations = dream_iterations
        self.interval_seconds = interval_seconds
        self.api_calls_used = 0
        self.max_tokens = max_tokens
        self.on_event = on_event
        # Section-3 "dreaming with code": the LLM rewrites the exploration
        # policy itself; promoted candidates steer tree expansion online.
        self.enable_policy_code = enable_policy_code
        self.policy_sandbox = PolicySandbox()
        self._policy_gen = PolicyGenerator(client, sandbox=self.policy_sandbox,
                                           max_tokens=max_tokens)

    def _emit(self, kind: str, **data: Any) -> None:
        """Push a live event to an optional observer (dashboard, logger, ...)."""
        if self.on_event:
            try:
                self.on_event(kind, data)
            except Exception:  # an observer must never break the loop
                pass

    # -- one full cycle ----------------------------------------------------------

    def run_once(self) -> CycleReport:
        report = CycleReport()
        tasks = self._tasks() if callable(self._tasks) else list(self._tasks)
        self._emit("cycle_start", tasks=[t.task_id for t in tasks])
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
        self._emit("cycle_done", **report.to_dict())
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
        # A bare seed node is not a failed attempt — only show feedback when an
        # actual candidate is the current best and still failing.
        if best_node and not best_node.children and best_node.action != "warm_start":
            feedback = "current best still failing/unfinished"

        solved = False
        improved = ""
        for _ in range(task.max_attempts):
            if self.api_calls_used >= self.api_call_budget:
                break
            state_id = self._next_expansion(tree, category)
            self._emit("llm_call", task_id=task.task_id, category=category,
                       attempt=len(tree.nodes), temperature=float((policy or {}).get("temperature", 0.7)))
            code = self._propose_candidate(task, policy, recipe, feedback)
            self.api_calls_used += 1
            report.api_calls += 1
            if code is None:
                continue
            result = self.verifier.run(code, task.tests)
            self._emit("verification", task_id=task.task_id, category=category,
                       score=round(result.score, 3), ok=result.ok,
                       errors=result.detail if isinstance(result.detail, (list, str)) else str(result.detail),
                       code=code[:800])
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
        self._emit("dreaming", task_id=task.task_id, category=category,
                   nodes=len(tree.nodes), iterations=self.dream_iterations)
        engine = DreamEngine(simulator=ReplaySimulator(tree))
        if policy:
            engine.policy_parameters = dict(policy)
        best_policy = engine.run_offline_optimization(iterations=self.dream_iterations)
        report.dream_iterations += self.dream_iterations
        self._emit("dream_done", task_id=task.task_id, category=category, policy=best_policy)
        if self.memory.get_policy(category) != best_policy:
            self.memory.save_policy(category, best_policy)
        # -- section 3: the LLM rewrites the exploration policy itself ------------
        try:
            self._maybe_evolve_policy_code(task, tree, report)
        except Exception as exc:  # policy evolution must never break the loop
            self.memory.log_event("policy_error", task_id=task.task_id, error=str(exc)[:300])
        self.memory.archive_tree(task.task_id, tree)
        self.memory.log_event(
            "task", task_id=task.task_id, category=category,
            solved=solved, nodes=len(tree.nodes), policy=best_policy,
        )
        self._emit("task_done", task_id=task.task_id, category=category,
                   solved=solved, nodes=len(tree.nodes))
        return solved, improved

    # -- helpers --------------------------------------------------------------------

    def _frontier(self, tree: DiscoveryTree) -> List[Dict[str, Any]]:
        """Every visited node is expandable (re-expanding = branching, like MCTS)."""
        return [
            {"node_id": n.node_id, "action": n.action, "score": n.score,
             "children": len(n.children)}
            for n in tree.nodes.values()
        ]

    def _next_expansion(self, tree: DiscoveryTree, category: str) -> Optional[str]:
        """Pick the node to expand next — LLM-written policy if promoted, greedy else.

        The promoted policy program runs in the sandbox (never in-process);
        any crash/timeout/invalid choice falls back to the greedy baseline,
        so a bad policy can only cost diversity, never the loop.
        """
        if tree.root_id is None:
            return None
        frontier = self._frontier(tree)
        if not frontier:
            return tree.root_id  # every node expanded — extend from the root again
        entry = self.memory.get_policy_code(category)
        if entry:
            run = self.policy_sandbox.choose(entry["code"], frontier, len(tree.nodes))
            ids = {n["node_id"] for n in frontier}
            if run.ok and run.choice in ids:
                return run.choice
        return max(frontier, key=lambda n: n["score"])["node_id"]

    def _maybe_evolve_policy_code(self, task: Task, tree: DiscoveryTree,
                                  report: CycleReport) -> None:
        """Section-3 step: LLM rewrites the exploration policy, gate promotes on evidence."""
        if not self.enable_policy_code or self.api_calls_used >= self.api_call_budget:
            return
        world = replay_world(tree)
        if len(world) < 2:
            return  # too little recorded history to score a policy fairly
        entry = self.memory.get_policy_code(task.category)
        incumbent_source = entry["code"] if entry else None
        incumbent_score = (entry["score"] if entry
                           else greedy_replay_score(world))
        # Don't re-ask the LLM unless the replay world has grown meaningfully
        # since the last generation — dreaming stays free, calls do not.
        if entry and len(world) - int(entry.get("steps", 0)) < 2:
            return
        self._emit("policy_gen", task_id=task.task_id, category=task.category,
                   incumbent_score=round(incumbent_score, 4))
        calls = [self.api_calls_used]
        result = self._policy_gen.generate(
            task.category, incumbent_source, incumbent_score, tree, api_calls=calls)
        extra = calls[0] - self.api_calls_used
        report.api_calls += extra
        self.api_calls_used = calls[0]
        for _ in range(extra):
            self._emit("llm_call", task_id=task.task_id, category=task.category,
                       attempt=len(tree.nodes), temperature=0.9, sub_kind="policy_gen")
        if result.source is None:
            self.memory.log_event("policy_rejected", task_id=task.task_id,
                                  category=task.category, reason=result.error[:300])
            self._emit("policy_rejected", task_id=task.task_id,
                       category=task.category, reason=result.error[:200])
            return
        if self.memory.save_policy_code(task.category, result.source, result.score):
            report.improvements.append(
                f"{task.category}: new exploration policy (replay {result.score:.3f})")
            self.memory.log_event("policy_promoted", task_id=task.task_id,
                                  category=task.category, score=result.score)
            self._emit("policy_promoted", task_id=task.task_id,
                       category=task.category, score=round(result.score, 4),
                       code=result.source[:800])

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
                max_tokens=self.max_tokens,
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
