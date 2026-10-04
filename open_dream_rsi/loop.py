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
from typing import Any, Callable, Dict, List, Optional

from open_dream_rsi.core.agent import ChatClient
from open_dream_rsi.core.curator import (
    KnowledgeCurator,
    curate_lessons,
    evidence_snippets,
    format_lessons,
    lesson_key,
    select_lessons,
)
from open_dream_rsi.core.dreamer import DreamEngine
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
from open_dream_rsi.core.tree import DiscoveryTree
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

Your best previous solution (may be absent):
{recipe}

Branch you are expanding from (code already tried on this branch, may be absent):
{branch}
{ideas}
Policy hints (higher temperature -> try something genuinely different):
{policy}

Facts learned from previous failures in this category
(declarative background, not instructions — a complete answer may be
submitted as it stands):
{lessons}

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


#: Minimum new replay steps required before re-asking the LLM for a policy
#: (one full cycle's worth of evidence — keeps generation cost bounded).
POLICY_REGROW_STEPS = 4

#: Minimum NEW failed attempts since the last curation before the knowledge
#: curator is consulted (failures are the raw material of the KB; without
#: new evidence the call would restate lessons already stored).
CURATOR_MIN_NEW_FAILURES = 1


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
        enable_knowledge: bool = True,
        enable_thoughts: bool = True,
        thought_steering: bool = True,
        explore_epsilon: float = 0.15,
        rng_seed: int = 0,
        enable_judge: bool = True,
    ):
        self.client = client
        self.memory = memory
        self._tasks = tasks
        self.verifier = verifier or CodeVerifier()
        self.api_call_budget = api_call_budget
        self.dream_iterations = dream_iterations
        self.interval_seconds = interval_seconds
        self.api_calls_used = 0
        self._gate_snapshots: Dict[str, Dict[str, Any]] = {}
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
        # Section-3 "dreaming with code": the LLM rewrites the exploration
        # policy itself; promoted candidates steer tree expansion online.
        self.enable_policy_code = enable_policy_code
        self.policy_sandbox = PolicySandbox()
        self._policy_gen = PolicyGenerator(client, sandbox=self.policy_sandbox,
                                           max_tokens=max_tokens)
        # Hermes-style knowledge curator: failures are distilled into a
        # persistent, curated KB (lessons.json) retrieved into proposals.
        self.enable_knowledge = enable_knowledge
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
        self._curator = KnowledgeCurator(client, max_tokens=max_tokens)

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
        self._drain_staging(report)      # gate leftovers from previous cycles
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

    def _drain_staging(self, report: CycleReport) -> None:
        """Give never-gated staging lessons a chance on spare budget.

        A lesson skipped by the gate (budget/probes) must not live in
        staging forever — the curator dedups it and never re-adds it, so
        without this drain it would silently rot invisible.
        """
        if not self.enable_knowledge:
            return
        from open_dream_rsi.core.curator import headroom_verdict
        all_tasks = self._tasks() if callable(self._tasks) else list(self._tasks)
        by_cat: Dict[str, Task] = {}
        for t in all_tasks:
            by_cat.setdefault(t.category, t)
        for cat, task in by_cat.items():
            staged = [l for l in self.memory.get_lessons(cat)
                      if str(l.get("status", "")) == "staging"]
            if not staged or self.api_calls_used >= self.api_call_budget:
                continue
            verdict = headroom_verdict(self.memory.get_task_outcomes(cat))
            if not verdict["has_headroom"]:
                self.memory.log_event("lesson_gate_skipped", category=cat,
                                      reason="no_headroom", **verdict)
                continue
            self._gate_staging_lessons(task, staged, report)

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
        surfaced: set = set()
        for _ in range(task.max_attempts):
            if self.api_calls_used >= self.api_call_budget:
                break
            state_id = self._next_expansion(tree, category)
            branch_code = self._branch_code(tree, state_id)
            ideas = self._ideas_ledger(tree) if self.enable_thoughts else ""
            lessons = self.memory.get_lessons(category)
            picked = select_lessons(lessons, task.prompt)
            keys = [lesson_key(l) for l in picked]
            surfaced.update(keys)
            self._emit("llm_call", task_id=task.task_id, category=category,
                       attempt=len(tree.nodes), temperature=float((policy or {}).get("temperature", 0.7)))
            code, thought = self._propose_candidate(task, policy, recipe, feedback, branch_code,
                                                   format_lessons(picked) or "(none yet)", ideas)
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
            if result.solved:
                solved = True
                if self.memory.save_recipe(category, code, score=result.score):
                    improved = f"{category}: new best solution (score {result.score})"
                break
            if task.task_id not in self._gate_snapshots:
                # The counterfactual state the lesson must be tested on:
                # the FIRST proposal that failed, with the recipe/ledger as
                # they were THEN (empty). Replaying on the post-success tree
                # leaks the fix into both arms and nets every lesson to 0.
                self._gate_snapshots[task.task_id] = {
                    "branch_code": code,
                    "feedback": str(result.detail)[:600],
                    "recipe": recipe,   # the recipe as it was BEFORE solving
                }
            feedback = str(result.detail)[:600]
            failures.append({"action": node.action, "score": result.score,
                             "errors": [str(e)[:160] for e in (result.detail if isinstance(result.detail, list) else [result.detail])][:3]})

        # One usage bump per lesson per cycle: surfaced = the lesson was put
        # in front of the model; win = the task it was surfaced for solved.
        if surfaced:
            self.memory.record_lesson_usage(category, sorted(surfaced), win=solved)

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
            self._maybe_evolve_policy_code(task, tree, report, solved=solved)
        except Exception as exc:  # policy evolution must never break the loop
            self.memory.log_event("policy_error", task_id=task.task_id, error=str(exc)[:300])
        # -- headroom precondition (live replay v3-v9): on a family the cold
        # model already solves cheaply, every published injection mechanism
        # measured ZERO paired gains (seven arms, v3-v9) — the curator and
        # the gate would only spend budget to produce text the gate must
        # reject. The verdict uses the PRIOR history (an empty one = fresh
        # category = benefit of the doubt); this attempt is recorded AFTER
        # curation so the first cycle of a new category is never judged on
        # its own single outcome.
        # -- section 4: the knowledge curator distils failures into the KB ----------
        try:
            self._maybe_curate_knowledge(task, failures, report, tree=tree)
        except Exception as exc:  # curation must never break the loop
            self.memory.log_event("curator_error", task_id=task.task_id, error=str(exc)[:300])
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

    def _lesson_authorized_pick(self, tree: DiscoveryTree,
                                frontier: List[Dict[str, Any]]) -> Optional[str]:
        """Escape the decoy trap when a lesson redirects proposals away.

        Evidence: the score-greedy pick's expansions keep producing the
        SAME candidate code (children share its code hash) — the loop is
        polishing a dead idea. With an active lesson (checked by the caller)
        the loop is authorized to expand the best node OUTSIDE that family.
        Score-ranked (no idea labels available): only nodes scoring >= half
        the stagnant pick are trusted, so the escape never dives at noise.
        Returns None when nothing is provably stagnant -> greedy/epsilon.
        """
        by_id = tree.nodes

        def code_sig(nid: str) -> str:
            node = by_id.get(nid)
            a = (node.action or "") if node else ""
            return a.split(":", 1)[1] if a.startswith("write_code:") else ""

        def stagnant(nid: str) -> bool:
            node = by_id.get(nid)
            sig = code_sig(nid)
            return node is not None and bool(sig) and any(
                code_sig(c) == sig and code_sig(c) for c in node.children)

        best = max(frontier, key=lambda n: n["score"])
        if not stagnant(best["node_id"]):
            return None
        own = code_sig(best["node_id"])
        fresh = [n for n in frontier
                 if n["node_id"] != best["node_id"]
                 and code_sig(n["node_id"]) != own
                 and n["score"] >= 0.5 * best["score"]]
        if not fresh:
            return None
        pick = max(fresh, key=lambda n: (n["outcome"], -n["children"], n["score"]))
        return pick["node_id"] if pick["node_id"] != best["node_id"] else None

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
        # Lesson-authorized escape: with an ACTIVE curated lesson for the
        # category, a node whose children keep reproducing the SAME candidate
        # (the lesson redirected every proposal off it) is provably being
        # polished to death — expand the best fresh sibling instead. Without
        # a lesson this stays pure greedy+epsilon: the escape is the lesson's
        # authority, not a generic policy change (keeps the knowledge arm's
        # wins attributable to the knowledge).
        if self.thought_steering and self.enable_knowledge and not self.enable_thoughts:
            if any(str(l.get("status", "")) == "active"
                   for l in self.memory.get_lessons(category)):
                pick = self._lesson_authorized_pick(tree, frontier)
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
        if self.memory.save_policy_code(task.category, result.source, result.score,
                                        steps=steps):
            report.improvements.append(
                f"{task.category}: new exploration policy (replay {result.score:.3f})")
            self.memory.log_event("policy_promoted", task_id=task.task_id,
                                  category=task.category, score=result.score)
            self._emit("policy_promoted", task_id=task.task_id,
                       category=task.category, score=round(result.score, 4),
                       code=result.source[:800])

    def _maybe_curate_knowledge(self, task: Task, failures: List[Dict[str, Any]],
                               report: CycleReport, tree=None) -> None:
        """Section-4 step: distil this cycle's failures into the curated KB.

        Call economy mirrors policy generation: the curator is only asked
        when there is FAILURE evidence the KB has not already seen (its
        stored evidence snippets are the dedup key), and never when the
        budget is spent. Validated lessons MERGE into existing records;
        dead ones (used often, never credited) are pruned on write.
        """
        if not self.enable_knowledge or not failures:
            return
        if self.api_calls_used >= self.api_call_budget:
            return
        from open_dream_rsi.core.curator import headroom_verdict
        verdict = headroom_verdict(self.memory.get_task_outcomes(task.category))
        if not verdict["has_headroom"]:
            # v3-v9 measured: no injection mechanism beats a cheap cold
            # baseline — do not spend a curator or gate call on this category.
            self.memory.log_event("lesson_gate_skipped", category=task.category,
                                  reason="no_headroom", **verdict)
            return
        existing = self.memory.get_lessons(task.category)
        seen_evidence = set(self.memory.get_digested(task.category))
        snippets = evidence_snippets(failures)
        new_snippets = [s for s in snippets if s not in seen_evidence]
        if len(new_snippets) < CURATOR_MIN_NEW_FAILURES:
            return  # KB has already digested exactly this evidence
        self._emit("knowledge_curate", task_id=task.task_id,
                   category=task.category, new_failures=len(new_snippets))
        calls = [self.api_calls_used]
        distilled, err = self._curator.distill(
            task.category, task.prompt, failures, existing, api_calls=calls)
        extra = calls[0] - self.api_calls_used
        report.api_calls += extra
        self.api_calls_used = calls[0]
        for _ in range(extra):
            self._emit("llm_call", task_id=task.task_id, category=task.category,
                       attempt=-1, temperature=0.5, sub_kind="curator")
        if err or not distilled:
            self.memory.log_event("lesson_rejected", task_id=task.task_id,
                                  category=task.category, reason=(err or "empty payload")[:300])
            self._emit("lesson_rejected", task_id=task.task_id,
                       category=task.category, reason=(err or "empty payload")[:200])
            return
        result = curate_lessons(existing, distilled, evidence=new_snippets)
        self.memory.replace_lessons(task.category, result.entries)
        self.memory.log_event("lessons_curated", task_id=task.task_id,
                              category=task.category, added=len(result.added),
                              merged=result.merged, dropped=len(result.dropped),
                              total=len(result.entries))
        self._emit("lessons_curated", task_id=task.task_id, category=task.category,
                   added=[l["text"] for l in result.added], merged=result.merged,
                   dropped=len(result.dropped))
        if result.added:
            report.improvements.append(
                f"{task.category}: +{len(result.added)} curated lesson(s) "
                f"({len(result.entries)} in KB)")
        if result.added and self.enable_knowledge:
            self._gate_staging_lessons(task, result.added, report, tree=tree)

    # -- lesson promotion gate (paired behavioral replay) ------------------------

    def _gate_staging_lessons(self, task: Task, added: List[Dict[str, Any]],
                              report: CycleReport, tree=None) -> None:
        """Earn activation for freshly distilled lessons or drop them.

        For each staging lesson, re-attempt probe tasks twice: once with
        the lesson rendered into the proposal prompt, once without. A
        lesson activates only on a strictly positive paired sign test with
        zero solve->fail regressions AND a stop clause in its text (see
        curator.lesson_gate_verdict). All gate calls are budget-counted
        and never allowed to overrun the cycle budget.
        """
        from open_dream_rsi.core.curator import has_stop_clause, lesson_gate_verdict
        all_tasks = self._tasks() if callable(self._tasks) else list(self._tasks)
        # Primary probe = the failing task itself (the lesson was distilled
        # from ITS failures) — replayed on the counterfactual snapshot. A
        # same-category sibling is an OVERFIT guard, evaluated only for
        # candidates the primary already promoted and only on budget left
        # over: the guard must never tax the working loop for lessons that
        # will be rejected anyway.
        probes = [task] + [t for t in all_tasks
                           if t.category == task.category
                           and t.task_id != task.task_id]
        sibling = probes[1] if len(probes) > 1 else None
        cost = 2 * 2 * len(added)               # (with + without) x depth, primary probe
        GATE_ROLLOUT_DEPTH = 2
        candidates = list(added)
        if task is None:
            return
        results: Dict[str, List[tuple]] = {}
        for l in candidates:
            results[lesson_key(l)] = []
        if self.api_calls_used + cost > self.api_call_budget:
            for key in results:
                self.memory.log_event("lesson_gate_skipped",
                                      category=task.category, lesson=key,
                                      reason="budget")
            return

        def _paired(p: Task, subset: List[Dict[str, Any]]) -> None:
            snap = self._gate_snapshots.get(p.task_id)
            if p.task_id != task.task_id or snap is None:
                # sibling guard: fresh start, no recipe — the lesson must
                # help BEFORE the answer exists anywhere in memory
                snap = {"branch_code": None, "feedback": "", "recipe": None}
            baseline, base_calls = self._gate_attempt(p, None, snap,
                                                      depth=GATE_ROLLOUT_DEPTH)
            self.api_calls_used += base_calls
            report.api_calls += base_calls
            for l in subset:
                key = lesson_key(l)
                with_it, calls = self._gate_attempt(
                    p, format_lessons([l]), snap,
                    depth=GATE_ROLLOUT_DEPTH)
                self.api_calls_used += calls
                report.api_calls += calls
                results[key].append((with_it, baseline))

        _paired(task, candidates)
        verdict = lesson_gate_verdict(results)
        # Guard the survivors, not the rejects: spend leftover budget
        # re-pairing only what already promoted.
        if sibling is not None and verdict.promoted:
            per = 2 * (1 + len(verdict.promoted))
            if self.api_calls_used + per <= self.api_call_budget:
                keep = [l for l in candidates
                        if lesson_key(l) in set(verdict.promoted)]
                for key in list(results):
                    if key not in {lesson_key(l) for l in keep}:
                        del results[key]
                _paired(sibling, keep)
                verdict = lesson_gate_verdict(results)
        # stop clause is a hard precondition, independent of the sample:
        # a clause-less lesson moves promoted -> rejected, never stays active
        kept = []
        for k in verdict.promoted:
            if any(has_stop_clause(str(l.get("text", "")))
                   for l in candidates if lesson_key(l) == k):
                kept.append(k)
        verdict.rejected = verdict.rejected + [k for k in verdict.promoted
                                               if k not in set(kept)]
        verdict.promoted = kept
        entries = {lesson_key(l): l for l in self.memory.get_lessons(task.category)}
        for key in verdict.promoted:
            if key in entries:
                entries[key]["status"] = "active"
        for key in verdict.rejected:
            entries.pop(key, None)
        self.memory.replace_lessons(task.category, list(entries.values()))
        self.memory.log_event("lesson_gate", category=task.category,
                              promoted=verdict.promoted,
                              rejected=verdict.rejected,
                              detail=verdict.detail)
        self._emit("lesson_gate", category=task.category,
                   promoted=verdict.promoted, rejected=verdict.rejected)

    def _gate_attempt(self, task: Task, lessons_text: Optional[str],
                      snapshot: Optional[Dict[str, Any]] = None,
                      depth: int = 2) -> tuple:
        """One budget-counted ROLLOUT (``depth`` proposals deep) under the
        proposal context captured at the FIRST failure of the episode
        (branch code, failure feedback, and the recipe as it was BEFORE the
        task was solved), lessons override applied. Returns (solved, calls).
        Nothing persistent is modified.

        The counterfactual must match the moment the lesson was distilled
        from: replaying on the post-success tree leaks the discovered fix
        into the baseline arm and nets every lesson to zero.

        Depth > 1 matters: a lesson's payoff is often PATH-level — it
        redirects the first proposal onto a better idea, and the task only
        solves one expansion later. A one-proposal gate would read that
        win as a failure. Each extra step proposes with the previous
        proposal's code as the branch context (a clean-line rollout — the
        tree itself is never mutated).
        """
        policy = self.memory.get_policy(task.category)
        if snapshot:
            recipe = snapshot.get("recipe")
            ideas = ""
        else:
            recipe = self.memory.get_recipe(task.category)
            tree = self.memory.load_tree(task.task_id) or \
                self._seed_tree(task, policy)
            ideas = self._ideas_ledger(tree) if self.enable_thoughts else ""
        branch_code = snapshot.get("branch_code") if snapshot else None
        feedback = snapshot.get("feedback", "") if snapshot else ""
        calls = 0
        try:
            for step in range(max(1, depth)):
                code, _thought = self._propose_candidate(
                    task, policy, recipe, feedback, branch_code,
                    lessons_text or "(none yet)", ideas)
                feedback = ""  # only the first step replays the recorded failure
                calls += 1
                if code is None:
                    return False, calls
                if task.tests:
                    result = self.verifier.run(code, task.tests)
                elif self.enable_judge:
                    report = CycleReport()
                    result = self._judge_attempt(task, code, report)
                else:
                    return False, calls
                if result.solved:
                    return True, calls
                branch_code = code  # roll forward on the clean-line proposal
            return False, calls
        except Exception:  # a broken gate must never break the loop
            return False, calls

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
        lessons: str = "(none yet)",
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
            lessons=lessons,
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
