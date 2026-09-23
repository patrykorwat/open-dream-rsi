"""LLM-written exploration policies (paper section 3: "dreaming with code").

The discovery agent and evaluator stay frozen; the *exploration policy
itself* is Python code an LLM rewrites between cycles. This closes the gap
both peer implementations (TheAstrayDev/dream-rsi-sdk, robinber/dream-rsi-spark)
still carry on their roadmaps.

Pipeline per cycle:

    generate (LLM, costs 1 API call)
      -> static validation (AST, no sandbox needed)
      -> sandboxed execution (``python -I``, scrubbed env, timeout — never
         in-process: candidate code must not see the runtime's API keys)
      -> counterfactual rollout scoring on the recorded discovery history
      -> promotion gate: replace the incumbent only on evidence

Policy contract
---------------
A candidate module must define::

    def choose_action(frontier: list[dict], step: int) -> str | dict

where ``frontier`` is a list of node observations::

    {"node_id": str, "action": str, "score": float, "parent_id": str|None,
     "children": int, "outcome": float, "errors": list[str]}

and the return is the ``node_id`` to expand next (or ``{"node_id": ...}``).

Scoring is a **counterfactual rollout**, not a mean over recorded picks: the
candidate replays the recorded tree step by step, and each expansion yields
the node's *next recorded child* (falling back to a repeat of its last
child, or to its own score when the branch was never expanded). This rewards
policies that would have opened branches the logging policy neglected —
the decoy-trap case: a plausible high-score leaf that fails hidden tests
forever, hiding the fix one expansion away on a different branch. The
rollout is **prefix-only**: at each step the policy sees only descendants
its counterfactual path has revealed, and ``outcome`` is the best score
among *revealed* descendants — no future scores leak into present choices
(issue #1).
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from open_dream_rsi.core.tree import DiscoveryTree

#: Weight of branch diversity in the rollout score (the paper's beta_2:
#: replay objectives reward visiting *different* branches, not just high-
#: score ones — the counterfactual counterpart of UCB's visit-count term).
#: Applied identically to candidates and the greedy baseline.
DIVERSITY_BETA = 0.2
#: Fraction of rollout steps that must return a frontier-valid choice for a
#: candidate to be promotable at all.
MIN_VALIDITY = 0.8
#: Rollout horizon in steps; scaled up for bigger worlds.
MIN_HORIZON = 8

POLICY_CONTRACT = (
    "You are writing an exploration policy for a Dream-RSI loop. Reply with "
    "ONLY one fenced ```python``` block defining exactly:\n"
    "    def choose_action(frontier, step):\n"
    "        ...\n"
    "frontier is a list of dicts: {'node_id': str, 'action': str, "
    "'score': float, 'parent_id': str|None, 'children': int, "
    "'outcome': float, 'errors': list[str], 'thought': str}. Return the "
    "node_id (str) of the node to expand next. 'score' is what the verifier "
    "gave that node's own attempt; 'children' is how often it was expanded "
    "so far; 'outcome' is the best score found anywhere below it (equal to "
    "its score when unexplored); 'errors' lists which tests still fail on "
    "it; 'thought' is the model's one-line plan for that attempt (may be "
    "empty) — nodes sharing idea words belong to the same idea family. "
    "Beware DECOY "
    "TRAPS: a high-score leaf whose errors never change is plausible code "
    "that will fail hidden tests forever — re-expanding it burns the budget, "
    "while branches with different (or no) failures hide the real prize. "
    "Good policies balance exploiting promising branches against trying "
    "under-expanded ones. "
    "Rules: pure stdlib, no imports, no file/network/process access, no "
    "leading-underscore attributes, must terminate, must handle an empty "
    "frontier (return None)."
)

_BANNED_NODES = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.Try)
_BANNED_NAMES = frozenset({
    "exec", "eval", "compile", "open", "input", "__import__", "globals",
    "locals", "vars", "dir", "getattr", "setattr", "delattr", "breakpoint",
    "exit", "quit", "memoryview", "object", "type", "super",
})


class PolicyValidationError(ValueError):
    """A candidate policy program violated the sandbox contract."""


def extract_python_block(reply: str) -> Optional[str]:
    """Pull the first fenced ```python (or plain ```) block out of a reply."""
    marker = "```"
    if marker not in reply:
        return reply.strip() or None
    parts = reply.split(marker)
    body = parts[1] if len(parts) > 1 else ""
    if body.lower().startswith("python"):
        body = body[body.index("\n") + 1:] if "\n" in body else ""
    return body.strip() or None


def validate_policy_source(source: str) -> None:
    """Static gate: parse + reject anything that could escape the sandbox.

    Runs BEFORE the subprocess so obviously hostile code never executes at
    all; the subprocess isolation is defence-in-depth, not the only wall.
    Raises PolicyValidationError on the first violation.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise PolicyValidationError(f"syntax error: {exc}") from exc
    has_entry = any(
        isinstance(n, ast.FunctionDef) and n.name == "choose_action"
        for n in tree.body
    )
    if not has_entry:
        raise PolicyValidationError("missing def choose_action(frontier, step)")
    for node in ast.walk(tree):
        if isinstance(node, _BANNED_NODES):
            raise PolicyValidationError(f"forbidden statement: {type(node).__name__}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise PolicyValidationError(f"forbidden attribute: {node.attr!r}")
        if isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            raise PolicyValidationError(f"forbidden name: {node.id!r}")


# ---------------------------------------------------------------------------
# Sandboxed execution
# ---------------------------------------------------------------------------

#: Child script: exec the candidate, run choose_action on a frontier file.
_POLICY_HARNESS = """
import json, sys, traceback
src = open(sys.argv[1], encoding="utf-8").read()
frontier = json.load(open(sys.argv[2], encoding="utf-8"))
step = int(sys.argv[3])
report = {"choice": None}
try:
    ns = {}
    exec(compile(src, "policy.py", "exec"), ns)
    fn = ns.get("choose_action")
    if not callable(fn):
        raise RuntimeError("choose_action is not defined/callable")
    out = fn(frontier, step)
    if isinstance(out, str):
        report["choice"] = out
    elif isinstance(out, dict):
        report["choice"] = out.get("node_id")
except Exception:
    report["error"] = traceback.format_exc(limit=3)
print(json.dumps(report))
"""


@dataclass
class PolicyRun:
    choice: Optional[str]
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


#: Child script for the BATCHED counterfactual rollout: exec the candidate once,
#: then run the fixed reward harness loop in-process inside the sandbox
#: (one subprocess per candidate, not per step — sandbox spawns are ~40ms and a
#: rollout is 8-24 steps). The reward logic below must mirror _simulate().
_ROLLOUT_HARNESS = """
import json, sys, traceback
src = open(sys.argv[1], encoding="utf-8").read()
world = json.load(open(sys.argv[2], encoding="utf-8"))
nodes = world["nodes"]                 # [{node_id, action, score, parent_id, outcome, errors}]
child_lists = world["child_lists"]     # parent_id -> [child node_ids, recorded order]
horizon = world["horizon"]
report = {"rewards": [], "picks": [], "invalid": 0, "error": None}
try:
    ns = {}
    exec(compile(src, "policy.py", "exec"), ns)
    fn = ns.get("choose_action")
    if not callable(fn):
        raise RuntimeError("choose_action is not defined/callable")
    by_id = {n["node_id"]: n for n in nodes}
    child_index = {}
    for pid, kids in child_lists.items():
        for i, kid in enumerate(kids):
            child_index[kid] = i
    cursor = {}
    for step in range(horizon):
        revealed = set()
        for n in nodes:
            pid = n["parent_id"]
            if pid is None or cursor.get(pid, 0) > child_index.get(n["node_id"], 10**9):
                revealed.add(n["node_id"])
        asof = {n["node_id"]: n["score"] for n in nodes}
        for n in reversed(nodes):                    # children before parents
            for k in child_lists.get(n["node_id"], ()):
                if k in revealed and asof[k] > asof[n["node_id"]]:
                    asof[n["node_id"]] = asof[k]
        visible = []
        for n in nodes:
            if n["node_id"] in revealed:
                e = dict(n)
                e["children"] = cursor.get(n["node_id"], 0)
                e["outcome"] = asof[n["node_id"]]    # as-of-now: revealed only
                visible.append(e)
        visible_ids = {e["node_id"] for e in visible}
        try:
            out = fn(visible, step)
        except Exception:
            report["error"] = traceback.format_exc(limit=3)
            break
        pick = out if isinstance(out, str) else ((out or {}).get("node_id") if isinstance(out, dict) else None)
        if pick not in visible_ids:
            report["invalid"] += 1
            report["rewards"].append(0.0)
            report["picks"].append("!invalid")
            continue
        kids = child_lists.get(pick, [])
        c = cursor.get(pick, 0)
        if kids and c < len(kids):
            report["rewards"].append(by_id[kids[c]]["score"])       # next recorded child
        elif kids:
            # ladder exhausted: the branch rediscovers its LAST recorded child
            report["rewards"].append(by_id[kids[-1]]["score"])
        else:
            # never expanded in the real run: no invented continuations,
            # the policy pays the node's own score (counterfactual floor)
            report["rewards"].append(by_id[pick]["score"])
        report["picks"].append(pick)
        cursor[pick] = c + 1
except Exception:
    report["error"] = traceback.format_exc(limit=3)
print(json.dumps(report))
"""


class PolicySandbox:
    """Executes candidate policy code in an isolated subprocess.

    Same isolation boundary as :class:`~open_dream_rsi.tools.CodeVerifier`:
    ``python -I`` + scrubbed env + timeout. Invalid code is additionally
    gated by :func:`validate_policy_source` before we ever spawn a process.
    """

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout

    def choose(self, source: str, frontier: List[Dict[str, Any]], step: int) -> PolicyRun:
        try:
            validate_policy_source(source)
        except PolicyValidationError as exc:
            return PolicyRun(None, f"validation: {exc}")
        d = tempfile.mkdtemp(prefix="odr-policy-")
        src_path = Path(d, "policy.py")
        fr_path = Path(d, "frontier.json")
        harness_path = Path(d, "harness.py")
        src_path.write_text(source, encoding="utf-8")
        fr_path.write_text(json.dumps(frontier), encoding="utf-8")
        harness_path.write_text(_POLICY_HARNESS, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(harness_path), str(src_path), str(fr_path), str(step)],
                capture_output=True, text=True, timeout=self.timeout,
                env={"PATH": "/usr/bin:/bin"},  # no API keys inside the sandbox
                cwd=d,
            )
        except subprocess.TimeoutExpired:
            return PolicyRun(None, "policy execution timed out")
        finally:
            for p in (src_path, fr_path, harness_path):
                p.unlink(missing_ok=True)
            Path(d).rmdir()
        line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        try:
            report = json.loads(line)
        except (json.JSONDecodeError, IndexError):
            return PolicyRun(None, f"policy crash: {proc.stderr[:200]!r}")
        return PolicyRun(report.get("choice"), report.get("error"))


    def rollout(self, source: str, world: Dict[str, Any]) -> Tuple[Optional[List[float]], List[str], int, Optional[str]]:
        """Batched counterfactual rollout: candidate executes ONCE in the sandbox,
        the reward loop runs inside the child process (see _ROLLOUT_HARNESS).

        Returns ``(rewards, picks, invalid_count, error)``.
        """
        try:
            validate_policy_source(source)
        except PolicyValidationError as exc:
            return None, [], 0, f"validation: {exc}"
        d = tempfile.mkdtemp(prefix="odr-rollout-")
        src_path = Path(d, "policy.py")
        world_path = Path(d, "world.json")
        harness_path = Path(d, "harness.py")
        src_path.write_text(source, encoding="utf-8")
        world_path.write_text(json.dumps(world), encoding="utf-8")
        harness_path.write_text(_ROLLOUT_HARNESS, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(harness_path), str(src_path), str(world_path)],
                capture_output=True, text=True, timeout=self.timeout,
                env={"PATH": "/usr/bin:/bin"},  # no API keys inside the sandbox
                cwd=d,
            )
        except subprocess.TimeoutExpired:
            return None, [], 0, "policy rollout timed out"
        finally:
            for p in (src_path, world_path, harness_path):
                p.unlink(missing_ok=True)
            Path(d).rmdir()
        line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        try:
            report = json.loads(line)
        except (json.JSONDecodeError, IndexError):
            return None, [], 0, f"rollout crash: {proc.stderr[:200]!r}"
        if report.get("error"):
            return None, [], 0, str(report["error"])[:400]
        return (report.get("rewards") or [], report.get("picks") or [],
                int(report.get("invalid", 0)), None)


# ---------------------------------------------------------------------------
# Counterfactual rollout scoring on recorded discovery histories
# ---------------------------------------------------------------------------

def outcome_map(tree: DiscoveryTree) -> Dict[str, float]:
    """node_id -> best verifier score found anywhere below (or at) the node.

    Back-propagated like MCTS values: the seed that eventually led to the fix
    carries the fix's score, which is what lets the ONLINE policy credit a
    branch the recorded path neglected. Replay never uses this — the rollout
    recomputes outcomes step by step from revealed nodes only (prefix-only),
    so a policy can never see a score its counterfactual path has not
    reached (issue #1).
    """
    nodes = list(tree.nodes.values())
    values = {n.node_id: n.score for n in nodes}
    # propagate bottom-up (children always appear after parents in insertion order)
    for n in reversed(nodes):
        kids = [values.get(c, float("-inf")) for c in _child_ids(tree, n.node_id)]
        if kids:
            values[n.node_id] = max(values[n.node_id], max(kids))
    return values


def _child_ids(tree: DiscoveryTree, node_id: str) -> List[str]:
    node = tree.nodes.get(node_id)
    return list(node.children) if node else []


def frontier_entry(node: Any, outcomes: Dict[str, float]) -> Dict[str, Any]:
    """The observation dict a policy sees for one node.

    Field parity between online and replay; NOT value parity for
    ``outcome``: the online view passes the full-tree :func:`outcome_map`,
    the replay world passes an empty map and the harness overwrites
    ``outcome`` per step with the as-of-now value over revealed descendants
    only (prefix rule).
    """
    errors = []
    result = node.result
    if isinstance(result, dict):
        errors = result.get("errors") or []
    return {
        "node_id": node.node_id,
        "action": node.action,
        "score": node.score,
        "parent_id": node.parent_id,
        "children": len(node.children),
        "outcome": outcomes.get(node.node_id, node.score),
        "errors": [str(e)[:160] for e in errors[:3]],
        "thought": str(getattr(node, "thought", "") or "")[:160],
    }


def rollout_world_payload(tree: DiscoveryTree,
                          horizon: Optional[int] = None) -> Dict[str, Any]:
    """Serialisable counterfactual world: nodes + recorded child lists.

    ``child_lists[parent_id]`` are the node ids of that parent's children in
    recorded (insertion) order — the fixed "answers" the rollout receives:
    the k-th expansion of a node reveals its k-th recorded child and pays its
    verifier score. Visibility follows the same rule (a node becomes visible
    once its parent has been expanded past the child's index), so counter-
    factual rollouts only discover branches the recorded tree actually has.
    """
    nodes = list(tree.nodes.values())
    child_lists: Dict[str, List[str]] = {}
    for n in nodes:
        if n.parent_id:
            child_lists.setdefault(n.parent_id, []).append(n.node_id)
    steps = sum(1 for n in nodes if n.parent_id)
    horizon = horizon or max(MIN_HORIZON, steps)
    return {
        # outcome map deliberately EMPTY here: the harness recomputes each
        # node's outcome per step from revealed descendants only (prefix
        # rule) — the serialized world must never carry full-tree futures.
        "nodes": [frontier_entry(n, {}) for n in nodes],
        "child_lists": child_lists,
        "horizon": horizon,
    }


def _simulate_step_rewards(world: Dict[str, Any], pick_fn: Any) -> Tuple[List[float], List[str], int, Optional[str]]:
    """In-process mirror of the sandboxed rollout harness (keep in sync!).

    ``pick_fn(visible_entries, step) -> node_id|None``; returns
    ``(rewards, picks, invalid_count, error)``.
    """
    nodes, child_lists, horizon_n = world["nodes"], world["child_lists"], world["horizon"]
    by_id = {n["node_id"]: n for n in nodes}
    child_index: Dict[str, int] = {}
    for kids in child_lists.values():
        for i, kid in enumerate(kids):
            child_index[kid] = i
    cursor: Dict[str, int] = {}
    rewards: List[float] = []
    picks: List[str] = []
    invalid = 0
    for step in range(horizon_n):
        revealed: set = set()
        for n in nodes:
            pid = n["parent_id"]
            if pid is None or cursor.get(pid, 0) > child_index.get(n["node_id"], 10 ** 9):
                revealed.add(n["node_id"])
        asof = {n["node_id"]: n["score"] for n in nodes}
        for n in reversed(nodes):                    # children before parents
            for k in child_lists.get(n["node_id"], ()):
                if k in revealed and asof[k] > asof[n["node_id"]]:
                    asof[n["node_id"]] = asof[k]
        visible: List[Dict[str, Any]] = []
        for n in nodes:
            if n["node_id"] in revealed:
                e = dict(n)
                e["children"] = cursor.get(n["node_id"], 0)
                e["outcome"] = asof[n["node_id"]]    # as-of-now: revealed only
                visible.append(e)
        visible_ids = {e["node_id"] for e in visible}
        try:
            pick = pick_fn(visible, step)
        except Exception as exc:  # noqa: BLE001 — baseline errors are fatal too
            return rewards, picks, invalid, str(exc)
        if pick not in visible_ids:
            invalid += 1
            rewards.append(0.0)
            picks.append("!invalid")
            continue
        kids = child_lists.get(pick, [])
        c = cursor.get(pick, 0)
        if kids and c < len(kids):
            rewards.append(by_id[kids[c]]["score"])                  # next recorded child
        elif kids:
            rewards.append(by_id[kids[-1]]["score"])      # exhausted: last recorded child
        else:
            rewards.append(by_id[pick]["score"])     # never expanded: own score (no invention)
        picks.append(pick)
        cursor[pick] = c + 1
    return rewards, picks, invalid, None


def _rollout_objective(rewards: List[float], picks: List[str]) -> float:
    """Mean reward + branch-diversity bonus (beta_2), comparable across arms."""
    total = len(rewards) or 1
    distinct = len({p for p in picks if p != "!invalid"})
    return sum(rewards) / total + DIVERSITY_BETA * distinct / total


def rollout_score(
    sandbox: Optional[PolicySandbox],
    source: str,
    tree: DiscoveryTree,
    horizon: Optional[int] = None,
) -> Tuple[float, str]:
    """Counterfactual rollout of a candidate policy on the recorded tree.

    Step by step the rollout re-observes the visible frontier (with
    as-of-now ``children``/``outcome``/``errors``), asks the policy which node
    to expand, and receives that node's *next recorded child score* as the
    rollout reward. A node whose recorded children are exhausted repeats its
    last child; a node never expanded in the real run yields its own score —
    the honest counterfactual floor, no invented outcomes. Invalid choices
    score 0 and count against the validity floor.

    Returns ``(objective, error)`` where objective = mean reward + beta_2 *
    distinct-branch coverage; error is non-empty when the candidate crashed,
    was statically invalid, or dropped below the validity floor.
    """
    if not tree.nodes:
        return float("-inf"), "empty tree"
    world = rollout_world_payload(tree, horizon)
    rewards, picks, invalid, err = sandbox.rollout(source, world)  # type: ignore[union-attr]
    if err or rewards is None:
        return float("-inf"), err or "no rollout steps"
    total = len(rewards)
    if total == 0:
        return float("-inf"), "no rollout steps"
    if invalid / total > 1.0 - MIN_VALIDITY:
        return float("-inf"), f"invalid choices in {100 * invalid / total:.0f}% of steps"
    return _rollout_objective(rewards, picks), ""


def greedy_rollout_score(tree: DiscoveryTree,
                         horizon: Optional[int] = None) -> float:
    """Baseline: greedy expansion (best visible score) on the same rollout."""
    if not tree.nodes:
        return float("-inf")
    world = rollout_world_payload(tree, horizon)

    def pick(visible: List[Dict[str, Any]], step: int) -> Optional[str]:
        return max(visible, key=lambda e: e["score"])["node_id"] if visible else None

    rewards, picks, invalid, err = _simulate_step_rewards(world, pick)
    total = len(rewards)
    if err or total == 0:
        return float("-inf")
    if invalid / total > 1.0 - MIN_VALIDITY:
        return float("-inf")
    return _rollout_objective(rewards, picks)


def evaluate_policy(sandbox: PolicySandbox, source: str,
                    tree: DiscoveryTree) -> Tuple[float, str]:
    """Score a candidate policy by counterfactual rollout. (score, error)."""
    return rollout_score(sandbox, source, tree)


def greedy_replay_score(tree: DiscoveryTree) -> float:
    """Back-compat alias for :func:`greedy_rollout_score`."""
    return greedy_rollout_score(tree)


# ---------------------------------------------------------------------------
# Generation + promotion
# ---------------------------------------------------------------------------

GEN_USER_TEMPLATE = """Category: {category}

Current best policy code (may be absent — write the first one then):
```python
{incumbent}
```

Replay metrics of the current policy: {metrics}

Discovery history summary ({nodes} nodes, best score {best_score}):
{summary}

Previous candidate rejection reason: {feedback}

Return an improved complete policy as one python code block."""


@dataclass
class GenerationResult:
    source: Optional[str]
    score: float
    error: str


class PolicyGenerator:
    """Asks the LLM to rewrite the exploration policy and gates the result.

    One generation costs one API call (plus one repair call when the answer
    fails static validation). Promotion requires beating the greedy/incumbent
    rollout score in the same cycle — evidence over vibes.
    """

    def __init__(self, client: Any, sandbox: Optional[PolicySandbox] = None,
                 max_repair: int = 1, max_tokens: Optional[int] = None):
        self.client = client
        self.sandbox = sandbox or PolicySandbox()
        self.max_repair = max_repair
        self.max_tokens = max_tokens

    def generate(
        self,
        category: str,
        incumbent_source: Optional[str],
        incumbent_score: float,
        tree: DiscoveryTree,
        api_calls: Optional[List[int]] = None,
    ) -> GenerationResult:
        steps = sum(1 for n in tree.nodes.values() if n.parent_id)
        if steps < 2:
            return GenerationResult(None, float("-inf"), "no replay history yet")
        summary = self._tree_summary(tree)
        feedback = ""
        source: Optional[str] = None
        for attempt in range(1 + self.max_repair):
            user = GEN_USER_TEMPLATE.format(
                category=category,
                incumbent=incumbent_source or "(none — first policy)",
                metrics=f"{incumbent_score:.4f}",
                nodes=len(tree.nodes),
                best_score=f"{max((n.score for n in tree.nodes.values()), default=0.0):.3f}",
                summary=summary,
                feedback=feedback or "(none)",
            )
            try:
                kwargs: Dict[str, Any] = {"temperature": 0.9}  # policy writing benefits from variety
                if self.max_tokens:
                    kwargs["max_tokens"] = self.max_tokens
                reply = self.client.chat(
                    [{"role": "system", "content": POLICY_CONTRACT},
                     {"role": "user", "content": user}],
                    **kwargs,
                )
            except Exception as exc:
                return GenerationResult(None, float("-inf"), f"llm error: {exc}")
            if api_calls is not None:
                api_calls[0] += 1
            source = extract_python_block(reply or "")
            if not source:
                feedback = "reply contained no python code block"
                continue
            try:
                validate_policy_source(source)
            except PolicyValidationError as exc:
                feedback = f"static validation failed: {exc}"
                continue
            score, err = evaluate_policy(self.sandbox, source, tree)
            if err:
                feedback = f"replay rejected the policy: {err}"
                continue
            if source == incumbent_source:
                return GenerationResult(None, score, "identical to incumbent")
            if score < incumbent_score - 1e-9:
                return GenerationResult(
                    None, score,
                    f"replay score {score:.4f} < incumbent {incumbent_score:.4f}")
            return GenerationResult(source, score, "")
        return GenerationResult(None, float("-inf"), feedback or "generation failed")

    @staticmethod
    def _tree_summary(tree: DiscoveryTree, max_nodes: int = 12) -> str:
        lines = []
        for node in list(tree.nodes.values())[-max_nodes:]:
            result = node.result if isinstance(node.result, dict) else {}
            errs = result.get("errors") or []
            errs_txt = (" errors=" + "; ".join(str(e)[:40] for e in errs[:2])) if errs else ""
            thought = str(getattr(node, "thought", "") or "")[:60]
            thought_txt = f"  thought='{thought}'" if thought else ""
            lines.append(f"  {node.node_id}  action={node.action}  "
                         f"score={node.score:.3f}  children={len(node.children)}{errs_txt}{thought_txt}")
        return "\n".join(lines) or "(empty)"
