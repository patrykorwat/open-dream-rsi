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

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from open_dream_rsi.core.agent import ChatClient
from open_dream_rsi.core.dreamer import DreamEngine
from open_dream_rsi.evolution import EvolutionStopper
from open_dream_rsi.core.policygen import (
    PolicyGenerator,
    PolicySandbox,
    frontier_entry,
    greedy_replay_score,
    outcome_map,
    rollout_score,
)
from open_dream_rsi.core.judge import LLMJudge
from open_dream_rsi.core.simulator import ReplaySimulator
from open_dream_rsi.core.tree import DiscoveryTree, TerminationReason
from open_dream_rsi.lifecycle import (
    ArtifactLifecycleError,
    ArtifactLifecycleManager,
)
from open_dream_rsi.memory import DreamMemory
from open_dream_rsi.tools import CodeVerifier, ToolResult

CANDIDATE_SYSTEM_PROMPT = (
    "You are a code-improvement agent in an autonomous recursive "
    "self-improvement loop. You receive a task, its test cases, your best "
    "previous solution and exploration-policy hints. Reply with ONLY the full "
    "Python solution in one fenced code block. No explanations."
)

REFINE_USER_TEMPLATE = """Task [{category}]: {prompt}

Test cases (call -> expected):
{tests}

Best previous solution in this category (from a DIFFERENT task — it may not
exist; treat it as a style/conventions reference only, its task-specific
logic usually does NOT transfer to the task above):
{recipe}

Branch you are expanding from (code already tried on this branch, may be absent):
{branch}
{ideas}
Policy hints (higher temperature -> try something genuinely different):
{policy}

Previous failure feedback:
{feedback}

Return the complete corrected/optimized solution as one python code block."""

#: Appended to the system prompt when thoughts are recorded: the model must
#: state its plan in one line before the code, and that line becomes the
#: semantic label of the branch (see TreeNode.thought).
PLAN_SYSTEM_SUFFIX = (
    " Before the code block, write exactly one line 'PLAN: <one sentence>' "
    "stating the distinct idea you are trying (not what you tried before)."
)

IDEAS_SECTION_TEMPLATE = """
Ideas already tried in this branch (idea -> outcome). Do not repeat a dead
idea verbatim; if one of these was structurally promising, build ON it:
{lines}
"""


@dataclass
class Task:
    task_id: str
    category: str
    prompt: str
    tests: List[Dict[str, Any]]
    max_attempts: int = 4
    #: Optional success criterion for test-less tasks: consulted by the
    #: completion judge when tests are empty (tests always dominate).
    criteria: str = ""


def append_task(tasks_file: "str | Path", task: Dict[str, Any]) -> Dict[str, Any]:
    """Append one task dict to a tasks.json queue file (atomic, idempotent).

    The single write path shared by the MCP tool, the CLI and any other
    no-MCP queue feeder (hooks, scripts): validates the required fields,
    normalises the test-less shape, refuses duplicate task_ids, and writes
    through a temp file so a concurrent reader never sees a torn queue.
    """
    task = dict(task)
    task.setdefault("max_attempts", 4)
    if not task.get("tests"):
        task["tests"] = []
    for field_name in ("task_id", "category", "prompt"):
        if not task.get(field_name):
            raise ValueError(f"'{field_name}' is required")
    if not task.get("tests") and not task.get("criteria"):
        raise ValueError("either 'tests' (non-empty list of {call, expected}) "
                         "or 'criteria' (success description for the "
                         "completion judge) is required")
    if task.get("tests") and (not isinstance(task["tests"], list)
                              or not task["tests"]):
        raise ValueError("'tests' must be a non-empty list of {call, expected}")
    path = Path(tasks_file)
    existing: List[Dict[str, Any]] = []
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    if any(t.get("task_id") == task["task_id"] for t in existing):
        return {"added": False,
                "reason": f"task_id '{task['task_id']}' already queued"}
    existing.append(task)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(existing, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    tmp.replace(path)
    return {"added": True, "task_id": task["task_id"],
            "tasks_total": len(existing), "tasks_file": str(path)}


#: Minimum new replay steps required before re-asking the LLM for a policy
#: (one full cycle's worth of evidence — keeps generation cost bounded).
POLICY_REGROW_STEPS = 4



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
        api_call_budget: int = 100,
        dream_iterations: int = 60,
        interval_seconds: float = 300.0,
        on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        max_tokens: int = 2048,
        enable_policy_code: bool = True,
        policy_revive: bool = False,
        enable_thoughts: bool = True,
        thought_steering: bool = True,
        explore_epsilon: float = 0.15,
        rng_seed: int = 0,
        enable_judge: bool = True,
        fair_queue: bool = True,
        sentinel_engine: Any = None,
        enable_evolution_stop: bool = False,
        alm: Optional["ArtifactLifecycleManager"] = None,
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
        # Baseline online exploration: with probability epsilon the greedy
        # fallback tries a never-expanded branch instead of the best-score
        # leaf. Without it, one high-score decoy (plausible code that never
        # passes hidden tests) monopolises the budget forever and the
        # recorded world never contains the evidence the replay gate needs.
        self.explore_epsilon = explore_epsilon
        # Automatic completion verdicts for test-less tasks (tests always
        # dominate; the judge never overrides a failing verifier).
        self.enable_judge = enable_judge
        self._judge = LLMJudge(client, max_tokens=256) if enable_judge else None
        self._rng = random.Random(rng_seed)
        # Fair queue: order the task list by processed-attempt count (least
        # processed first) so a budget-capped cycle can never starve the
        # tail of the backlog — starvation is fixed by PROCESSING, and the
        # prune in dream_once stays what it is: a backstop, not the fix.
        # Benchmarks pin this OFF so published arm numbers stay byte-stable.
        self.fair_queue = fair_queue
        # Section-3 "dreaming with code": the LLM rewrites the exploration
        # policy itself; promoted candidates steer tree expansion online.
        self.enable_policy_code = enable_policy_code
        self.policy_sandbox = PolicySandbox()
        # Revise mode (dream-rsi-sdk-inspired, opt-in): policy generation asks
        # the LLM to REPAIR the incumbent against replay feedback (structured
        # greedy-replay brief) instead of rewriting from scratch. Same gate,
        # same strict win — only the prompt changes. Bench arms pin this OFF
        # so published arm numbers stay byte-stable.
        self.policy_revive = policy_revive
        self._policy_gen = PolicyGenerator(client, sandbox=self.policy_sandbox,
                                           max_tokens=max_tokens)
        # Thought-conditioned branching: record the model's one-line plan per
        # node and feed the branch's tried-idea ledger back into proposals and
        # into the policy observation, so expansion steers semantically
        # ("that idea is dead", "this idea almost worked") not just by score.
        self.enable_thoughts = enable_thoughts
        # thought_steering=False keeps the ledger in prompts but disables the
        # semantic expansion pick — the ablation that separates "the model
        # sees the tried-idea ledger" from "the loop leaves dead idea
        # families" when measuring the arm.
        self.thought_steering = thought_steering
        # Optional SentinelEngine (open_dream_rsi.sentinel): when injected,
        # every verification result is also observed STRUCTURALLY and its
        # replay-safe facts land on the tree node (node.sentinel), and the
        # episode's stop cause is recorded (node.termination_reason).
        # Default None = trees byte-identical to pre-observation runs.
        self.sentinel = sentinel_engine
        # Anytime-valid evolution stopping (opt-in; arXiv:2610.04756): a
        # betting martingale per category over "did regeneration promote
        # this cycle". After an alarm the loop keeps the last promoted
        # policy and stops spending calls on regeneration attempts.
        self.enable_evolution_stop = enable_evolution_stop
        self._evolution = EvolutionStopper()
        self._cycle_policy_promoted: set = set()
        # Artifact Lifecycle Manager (issue #3): the loop OWNS learning —
        # what to create and whether a promotion is earned — the ALM OWNS
        # lifecycle bookkeeping (validation, activation events, staleness,
        # supersession, retirement) for every artifact the pipeline produces.
        # It is a mirror of decisions already made, never a second promotion
        # engine; the legacy JSON stores remain the runtime view (a later
        # issue cuts runtime reads over to the ALM's ACTIVE set).
        self.alm = alm if alm is not None else ArtifactLifecycleManager(memory)

    def _emit(self, kind: str, **data: Any) -> None:
        """Push a live event to an optional observer (dashboard, logger, ...)."""
        if self.on_event:
            try:
                self.on_event(kind, data)
            except Exception:  # an observer must never break the loop
                pass

    # -- ALM seams (issue #3) ---------------------------------------------------
    #
    # These helpers MIRROR decisions the learning pipeline has already made
    # into the lifecycle store. They never decide anything themselves and an
    # ALM error must never break a working loop — hence the guards.

    def _alm_sync(self, artifact_type: str, slot: str, payload: Dict[str, Any],
                  *, reason: str, actor: str) -> None:
        try:
            state = self.alm.sync_active(artifact_type, slot, payload,
                                         reason=reason, actor=actor)
            if state is not None:
                self._emit("artifact_active", artifact_id=state.artifact_id,
                           artifact_type=artifact_type, slot=slot,
                           version=state.version)
        except ArtifactLifecycleError as exc:
            self.memory.log_event("alm_error", slot=slot,
                                  artifact_type=artifact_type, error=str(exc)[:300])

    # -- one full cycle ----------------------------------------------------------

    def run_once(self) -> CycleReport:
        report = CycleReport()
        usage_before = dict(getattr(self.client, "usage_total", None) or {})
        tasks = self._tasks() if callable(self._tasks) else list(self._tasks)
        if self.fair_queue and len(tasks) > 1:
            # Least-processed first (stable: ties keep queue order) so a
            # budget-capped cycle rotates through the backlog instead of
            # always dying at the same head — starvation is fixed by
            # PROCESSING; the queue cap downstream is only a backstop.
            attempts = self.memory.get_all_task_attempts()
            tasks = sorted(tasks, key=lambda t: attempts.get(t.task_id, 0))
        self._emit("cycle_start", tasks=[t.task_id for t in tasks])
        attempted_categories: set = set()
        for task in tasks:
            if self.api_calls_used >= self.api_call_budget:
                report.stopped_reason = "api_budget_exhausted"
                break
            attempted_categories.add(task.category)
            solved, improved = self._work_on_task(task, report)
            self.memory.record_task_attempt(task.task_id)
            report.tasks_attempted += 1
            if solved:
                report.tasks_solved += 1
            if improved:
                report.improvements.append(improved)

        if self.enable_evolution_stop:
            for cat in sorted(attempted_categories):
                prev = self._evolution.is_frozen(cat)
                d = self._evolution.observe(
                    cat, promoted=cat in self._cycle_policy_promoted)
                if d["frozen"] and not prev:
                    self.memory.log_event(
                        "evolution_alarm", category=cat, rounds=d["rounds"],
                        wealth=round(d["wealth"], 4), p0=d["p0"])
                    self._emit("evolution_alarm", category=cat,
                               rounds=d["rounds"])
            self._cycle_policy_promoted.clear()
        event = report.to_dict()
        tokens = self._tokens_delta(usage_before)
        if tokens:
            event["tokens"] = tokens
        self.memory.log_event("cycle", **event)
        self._emit("cycle_done", **event)
        self._alm_maintenance()
        return report

    def _tokens_delta(self, before: Dict[str, int]) -> Dict[str, int]:
        """Token usage burned this cycle (OpenAI ``usage`` accounting).

        Empty dict when the client does not track usage (stub clients) or the
        endpoint reports no ``usage`` blocks — metrics then show "not
        instrumented" instead of a misleading zero.
        """
        total = dict(getattr(self.client, "usage_total", None) or {})
        if not total:
            return {}
        delta = {key: max(0, int(total.get(key, 0) or 0) - int(before.get(key, 0) or 0))
                 for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        return delta if any(delta.values()) else {}

    def _alm_maintenance(self) -> None:
        """Deterministic lifecycle housekeeping after each cycle: staleness
        detection and retention/GC for lifecycle artifacts. Pinned artifacts
        are excluded by the ALM itself; the event log is never pruned."""
        try:
            stale = self.alm.detect_stale()
            if stale:
                self.memory.log_event("artifacts_stale", artifact_ids=stale)
            gc = self.alm.gc()
            if gc.archived or gc.deleted:
                self.memory.log_event("artifacts_gc", **gc.to_dict())
        except ArtifactLifecycleError as exc:
            self.memory.log_event("alm_error", scope="maintenance",
                                  error=str(exc)[:300])

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
        calls_at_start = report.api_calls
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
        failures: List[Dict[str, Any]] = []
        last_node = None  # last verification node this episode created
        for _ in range(task.max_attempts):
            if self.api_calls_used >= self.api_call_budget:
                break
            state_id = self._next_expansion(tree, category)
            branch_code = self._branch_code(tree, state_id)
            ideas = self._ideas_ledger(tree) if self.enable_thoughts else ""
            self._emit("llm_call", task_id=task.task_id, category=category,
                       attempt=len(tree.nodes), temperature=float((policy or {}).get("temperature", 0.7)))
            code, thought = self._propose_candidate(task, policy, recipe, feedback, branch_code,
                                                   ideas)
            self.api_calls_used += 1
            report.api_calls += 1
            if code is None:
                continue
            if task.tests:
                result = self.verifier.run(code, task.tests)
            elif self.enable_judge:
                result = self._judge_attempt(task, code, report)
            else:
                result = ToolResult(False, 0.0,
                                    "task has no tests and the completion "
                                    "judge is disabled")
            self._emit("verification", task_id=task.task_id, category=category,
                       score=round(result.score, 3), ok=result.ok,
                       errors=result.detail if isinstance(result.detail, (list, str)) else str(result.detail),
                       thought=thought[:120],
                       code=code[:800])
            node = tree.add_node(
                f"{task.task_id}/try-{len(tree.nodes)}",
                action=f"write_code:{hashlib.sha1(code.strip().encode()).hexdigest()[:8]}",
                result={"code": code, "errors": result.detail},
                score=result.score,
                parent_id=state_id,
                thought=thought[:240] if self.enable_thoughts else "",
            )
            state_id = node.node_id
            if self.sentinel is not None:
                # Structured twin of the host annotation channel: the same
                # engine, but its OBSERVATION (not its prose) becomes part of
                # the world. to_world_dict() is replay-safe by construction;
                # the session id makes recurrence episode-local (per task),
                # and the verifier verdict is authoritative (status=).
                try:
                    obs = self.sentinel.observe_structured(
                        tool="verify", result=None, session_id=task.task_id,
                        status="ok" if result.ok else "error",
                        error_message=None if result.ok
                        else str(result.detail)[:2000])
                    node.sentinel = obs.to_world_dict()
                except Exception as exc:  # observation must never break the loop
                    self.memory.log_event("sentinel_error",
                                          task_id=task.task_id, error=str(exc)[:200])
            last_node = node
            if result.solved:
                solved = True
                if self.memory.save_recipe(category, code, score=result.score):
                    improved = f"{category}: new best solution (score {result.score})"
                    self._alm_sync("recipe", f"recipe:{category}",
                                   {"code": code, "score": result.score},
                                   reason=f"verified solution (score {result.score}) "
                                          "promoted by the loop",
                                   actor="dream_rsi_pipeline")
                break
            feedback = str(result.detail)[:600]
            failures.append({"action": node.action, "score": result.score,
                             "errors": [str(e)[:160] for e in (result.detail if isinstance(result.detail, list) else [result.detail])][:3]})

        # Honest stop cause for replay: "the task solved here" and "the
        # runtime ran out of attempts/API budget here" mean OPPOSITE things
        # to a counterfactual rollout (Dream-RSI prefix-only replay; the
        # TravelPlanner study: 91/91 undelivered cold episodes died at the
        # harness cap, i.e. the option was REMOVED, never declined).
        # Recorded only when a sentinel engine is wired — default trees are
        # byte-identical to pre-metadata runs.
        if self.sentinel is not None and last_node is not None:
            last_node.termination_reason = (
                TerminationReason.COMPLETED.value if solved
                else TerminationReason.BUDGET_GATE.value)

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
            self._alm_sync("policy_parameters", f"policy_parameters:{category}",
                           {"params": dict(best_policy)},
                           reason="offline dream improved the policy parameters",
                           actor="dream_rsi_pipeline")
        # -- section 3: the LLM rewrites the exploration policy itself ------------
        try:
            self._maybe_evolve_policy_code(task, tree, report, solved=solved)
        except Exception as exc:  # policy evolution must never break the loop
            self.memory.log_event("policy_error", task_id=task.task_id, error=str(exc)[:300])
        self.memory.record_task_outcome(category, solved=solved,
                                        calls=report.api_calls - calls_at_start)
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
        """Every visited node is expandable (re-expanding = branching, like MCTS).

        Delegates to ``frontier_entry`` so the online view and the replay
        harness show the policy exactly the same observation fields.
        """
        outcomes = outcome_map(tree)
        return [frontier_entry(n, outcomes) for n in tree.nodes.values()]

    def _branch_code(self, tree: DiscoveryTree, state_id: Optional[str]) -> Optional[str]:
        """Code of the node being expanded — the branch context for the proposal."""
        if not state_id:
            return None
        node = tree.nodes.get(state_id)
        if node is None or node.action == "warm_start":
            return None
        return (node.result or {}).get("code")

    def _ideas_ledger(self, tree: DiscoveryTree) -> str:
        """Render the tried-idea ledger of the WHOLE task history.

        Tree-wide on purpose: the point of the ledger is telling the model
        WHICH ideas are already dead anywhere in this task (score alone
        cannot distinguish 'promising idea unexplored' from 'decoy idea
        polishing'), so escaping a dead idea family does not depend on
        which single node happens to be expanded next.
        """
        # dedupe by idea text, keep the best outcome per idea (stable order)
        best: Dict[str, tuple] = {}
        for node in tree.nodes.values():
            thought = (getattr(node, "thought", "") or "").strip()
            if not thought:
                continue
            if thought not in best or node.score > best[thought][0]:
                best[thought] = (node.score, thought)
        if not best:
            return ""
        lines = [f'  - "{t}" -> score {s:.3f}' for s, t in best.values()][:8]
        return IDEAS_SECTION_TEMPLATE.format(lines="\n".join(lines))

    def _thought_aware_pick(self, tree: DiscoveryTree,
                            frontier: List[Dict[str, Any]]) -> Optional[str]:
        """Semantic escape from idea stagnation, or None to keep greedy flow.

        Score-greedy expansion locks onto the best-score leaf forever. When
        that leaf's expansions keep producing the SAME one-line idea (a
        child carries identical PLAN text — the branch is polishing a dead
        idea, not converging), expand instead the best-OUTCOME node outside
        that idea family (nodes labelled with the stagnant idea are excluded;
        the seed node, labelled by nothing, stays a candidate — its ladder
        still holds untried ideas). Deterministic: the escape is triggered by
        recorded evidence about the IDEA, not by luck. Returns None (=>
        epsilon/greedy decide) when nothing is stagnant or no candidate has
        an idea label to reason about.
        """
        by_id = tree.nodes

        def thought_of(nid: str) -> str:
            node = by_id.get(nid)
            return (node.thought or "").strip() if node else ""

        def branch_sig(nid: str) -> str:
            """Structural identity of the candidate a node produced (code
            hash from the node action) — works without thought labels."""
            node = by_id.get(nid)
            a = (node.action or "") if node else ""
            return a.split(":", 1)[1] if a.startswith("write_code:") else ""

        def stagnant(nid: str) -> bool:
            """Node's own idea is among its children's ideas -> dead polish.
            With thoughts off, a child reproducing the SAME candidate code
            is the same evidence: polishing, not converging."""
            node = by_id.get(nid)
            if node is None:
                return False
            t = thought_of(nid)
            if t and any(thought_of(c) == t for c in node.children):
                return True
            sig = branch_sig(nid)
            return bool(sig) and any(branch_sig(c) == sig for c in node.children)

        best = max(frontier, key=lambda n: n["score"])
        own = thought_of(best["node_id"])
        own_sig = branch_sig(best["node_id"])
        if not (own or own_sig) or not stagnant(best["node_id"]):
            return None  # not re-polishing a recorded idea — greedy is fine
        rank = lambda n: (n["outcome"], -n["children"])
        labeled_ok = bool(own)  # thoughts on: only idea-labelled escapes are trusted
        fresh = [n for n in frontier
                 if n["node_id"] != best["node_id"]
                 and not stagnant(n["node_id"])
                 and thought_of(n["node_id"]) != own
                 and branch_sig(n["node_id"]) != own_sig
                 and (thought_of(n["node_id"]) if labeled_ok else True)
                 ]
        if fresh:
            pick = max(fresh, key=rank)
            # Without thought labels trust raw score; with them keep the
            # proven outcome-rank (a low-score node may be the promising one)
            if own or pick["score"] >= best["score"] * 0.5:
                return pick["node_id"]
        # fallback: a node without an idea label (e.g. the seed) may still
        # hold untried ladder steps — expand it rather than the dead branch,
        # but only when no confidently-fresh candidate exists.
        unlabeled = [n for n in frontier
                     if n["node_id"] != best["node_id"]
                     and not thought_of(n["node_id"])
                     and not stagnant(n["node_id"])]
        if not unlabeled:
            return None
        return max(unlabeled, key=rank)["node_id"]

    def _judge_attempt(self, task: Task, code: str,
                       report: CycleReport) -> ToolResult:
        """Verdict for a test-less attempt: sandbox smoke-run as evidence,
        LLM judge as arbiter. The judge call is budget-charged like any other;
        a budget-starved cycle fails closed (no verdict -> not solved)."""
        if self.api_calls_used >= self.api_call_budget:
            return ToolResult(False, 0.0, "judge skipped: budget exhausted")
        smoke = self.verifier.smoke_run(code)
        if not smoke["ok"]:
            # Code does not even execute: no LLM call needed, hard fail,
            # traceback goes straight into the feedback channel.
            return ToolResult(False, 0.0, f"code does not execute: {smoke['error']}")
        if self._judge is None:
            return ToolResult(False, 0.0, "judge disabled")
        evidence = (f"imports cleanly; defines: {', '.join(smoke['defined']) or '(nothing)'}; "
                    f"stdout: {smoke['stdout'] or '(none)'}")
        evidence += f"\nSandbox stdout:\n{smoke['stdout']}" if smoke["stdout"] else ""
        self._emit("llm_call", task_id=task.task_id, category=task.category,
                   call_kind="judge")
        verdict = self._judge.judge(task.prompt, task.criteria, code, evidence)
        self.api_calls_used += 1
        report.api_calls += 1
        self._emit("judge_verdict", task_id=task.task_id, category=task.category,
                   solved=verdict["solved"], score=round(verdict["score"], 3),
                   reason=verdict["reason"][:200])
        return ToolResult(ok=verdict["solved"], score=verdict["score"],
                          detail=verdict["reason"], solved=verdict["solved"])

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
        step = len(tree.nodes)
        entry = self.memory.get_policy_code(category)
        if entry:
            run = self.policy_sandbox.choose(entry["code"], frontier, step)
            ids = {n["node_id"] for n in frontier}
            if run.ok and run.choice in ids:
                return run.choice
        # Thought-conditioned steering: when the score-greedy pick is a node
        # whose IDEA family already re-polished itself (children with the same
        # PLAN line), expand OUTSIDE that family — deterministic escape from
        # the decoy trap that needs neither scores improving nor luck.
        if self.enable_thoughts and self.thought_steering:
            pick = self._thought_aware_pick(tree, frontier)
            if pick:
                return pick
        # epsilon-exploration baseline: occasionally try a uniformly random
        # node instead of re-polishing the best-score leaf (escapes decoys
        # only by luck — this is the "epsilon-greedy" comparison arm)
        if self.explore_epsilon > 0 and self._rng.random() < self.explore_epsilon:
            if frontier:
                return self._rng.choice(frontier)["node_id"]
        return max(frontier, key=lambda n: n["score"])["node_id"]

    def _maybe_evolve_policy_code(self, task: Task, tree: DiscoveryTree,
                                  report: CycleReport, solved: bool = False) -> None:
        """Section-3 step: LLM rewrites the exploration policy, gate promotes on evidence."""
        if not self.enable_policy_code or self.api_calls_used >= self.api_call_budget:
            return
        if self.enable_evolution_stop and self._evolution.is_frozen(task.category):
            return
        entry = self.memory.get_policy_code(task.category)
        if solved and entry:
            # the category already solves itself — exploration calls are for
            # categories that actually struggle (cold-start generation still
            # runs once, so every category gets its first policy)
            return
        steps = sum(1 for n in tree.nodes.values() if n.parent_id)
        if steps < 2:
            return  # too little recorded history to score a policy fairly
        incumbent_source = entry["code"] if entry else None
        # Don't re-ask the LLM unless the replay world has grown by a full
        # cycle's worth of new evidence — dreaming stays free, calls do not.
        if entry and steps - int(entry.get("steps", 0)) < POLICY_REGROW_STEPS:
            return
        # Re-evaluate the incumbent on the CURRENT tree instead of trusting
        # its stored score: the replay world has grown since it was scored,
        # and an old threshold would let a candidate promote while being
        # genuinely worse than the incumbent on today's evidence
        # (old_incumbent < candidate < current_incumbent). Found by external
        # review, 2026-10.
        if incumbent_source:
            incumbent_score, _ = rollout_score(
                self.policy_sandbox, incumbent_source, tree)
            if incumbent_score == float("-inf"):
                # incumbent no longer scores on this tree (stale/broken) —
                # the greedy baseline is the honest floor
                incumbent_score = greedy_replay_score(tree)
        else:
            incumbent_score = greedy_replay_score(tree)
        self._emit("policy_gen", task_id=task.task_id, category=task.category,
                   incumbent_score=round(incumbent_score, 4))
        calls = [self.api_calls_used]
        result = self._policy_gen.generate_with_brief(
            task.category, incumbent_source, incumbent_score, tree,
            api_calls=calls) if self.policy_revive else \
            self._policy_gen.generate(
                task.category, incumbent_source, incumbent_score, tree,
                api_calls=calls)
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
        if self.memory.save_policy_code(task.category, result.source, result.score,
                                        steps=steps):
            self._cycle_policy_promoted.add(task.category)
            report.improvements.append(
                f"{task.category}: new exploration policy (replay {result.score:.3f})")
            self.memory.log_event("policy_promoted", task_id=task.task_id,
                                  category=task.category, score=result.score)
            self._alm_sync("policy_program", f"policy_program:{task.category}",
                           {"code": result.source, "score": result.score,
                            "steps": steps},
                           reason=f"replay promotion (score {result.score:.3f} "
                                  "beat the incumbent)",
                           actor="dream_rsi_promotion")
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
        branch_code: Optional[str] = None,
        ideas: str = "",
    ) -> tuple[Optional[str], str]:
        """Ask the model for the next attempt. Returns (code, thought).

        ``thought`` is the PLAN line the model states when thoughts are on
        ("" otherwise) — the semantic label recorded on the resulting node.
        """
        temperature = float((policy or {}).get("temperature", 0.7))
        # Test-call strings can embed huge fixtures (multi-KB HTML/CSV).
        # Tests may instead carry a 'setup' source defining fixture names;
        # render each unique setup ONCE so the model pays the tokens a
        # single time per proposal instead of once per call.
        setups: list = []
        seen_setup: set = set()
        for t in task.tests:
            s = t.get("setup")
            if s and s not in seen_setup:
                seen_setup.add(s)
                setups.append(s)
        tests_block = ""
        if setups:
            tests_block += "Fixtures (already defined when tests run):\n" + \
                "\n".join(setups) + "\n\n"
        tests_block += "\n".join(
            f"  {t['call']} -> {t['expected']!r}" for t in task.tests)
        user = REFINE_USER_TEMPLATE.format(
            category=task.category,
            prompt=task.prompt,
            tests=tests_block,
            recipe=recipe or "(none yet)",
            branch=branch_code or "(fresh branch — nothing tried here yet)",
            ideas=ideas,
            policy=json.dumps(policy or {}),
            feedback=feedback or "(none)",
        )
        system = CANDIDATE_SYSTEM_PROMPT
        if self.enable_thoughts:
            system += PLAN_SYSTEM_SUFFIX
        try:
            reply = self.client.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:  # provider error => skip this attempt, keep dreaming
            self.memory.log_event("llm_error", task_id=task.task_id, error=str(exc)[:300])
            return None, ""
        thought = _extract_plan_line(reply) if self.enable_thoughts else ""
        return _extract_python_code(reply), thought


def _extract_plan_line(reply: str) -> str:
    """Pull the 'PLAN: ...' line out of a reply (the node's semantic label).

    Tolerant: the model may prefix whitespace/bullets or drop the colon;
    we take the first line mentioning PLAN and strip the marker. Pure text —
    never executed, only rendered into prompts and frontier observations.
    """
    for line in reply.splitlines():
        stripped = line.strip().lstrip("#*>- ").strip()
        upper = stripped.upper()
        if upper.startswith("PLAN:") or upper.startswith("PLAN "):
            text = stripped[4:].lstrip(" :").strip()
            if text:
                return text
    return ""


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
