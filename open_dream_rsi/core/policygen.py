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
      -> off-policy replay scoring on the recorded discovery history
      -> promotion gate: replace the incumbent only on evidence

Policy contract
---------------
A candidate module must define::

    def choose_action(frontier: list[dict], step: int) -> str | dict

where ``frontier`` is a list of leaf observations
``{"node_id", "action", "score", "children"}`` and the return is the
``node_id`` to expand next (or ``{"node_id": ...}``).
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

#: Weight of action diversity in the replay score (beta_2 of the paper, library scale).
DIVERSITY_BETA = 0.2
#: Fraction of steps that must return a frontier-valid choice for a candidate
#: to be promotable at all.
MIN_VALIDITY = 0.8

POLICY_CONTRACT = (
    "You are writing an exploration policy for a Dream-RSI loop. Reply with "
    "ONLY one fenced ```python``` block defining exactly:\n"
    "    def choose_action(frontier, step):\n"
    "        ...\n"
    "frontier is a list of dicts: {'node_id': str, 'action': str, "
    "'score': float, 'children': int}. Return the node_id (str) of the leaf "
    "to expand next (higher score = better result so far). Rules: pure "
    "stdlib, no imports, no file/network/process access, no leading-underscore "
    "attributes, must terminate, must handle an empty frontier (return None)."
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


# ---------------------------------------------------------------------------
# Off-policy replay scoring on recorded discovery histories
# ---------------------------------------------------------------------------

def replay_world(tree: DiscoveryTree) -> List[Tuple[List[Dict[str, Any]], str]]:
    """Reconstruct recorded decision steps: (frontier_snapshot, chosen_node_id).

    nodes are replayed in insertion order; the parent of each appended node
    is the expansion the online loop actually chose at that step. The frontier
    at step *i* is every node seen so far (re-expanding a visited node is a
    legal branch), which is what gives replay its discriminating power:
    a policy must pick BETTER nodes than greedy, not just any leaf.
    """
    nodes = list(tree.nodes.values())
    world: List[Tuple[List[Dict[str, Any]], str]] = []
    for i in range(1, len(nodes)):
        child = nodes[i]
        parent_id = child.parent_id
        if parent_id:
            seen = nodes[:i]
            frontier = [
                {"node_id": n.node_id, "action": n.action, "score": n.score,
                 "children": sum(1 for m in seen if m.parent_id == n.node_id)}
                for n in seen
            ]
            if parent_id in {n["node_id"] for n in frontier}:
                world.append((frontier, parent_id))
    return world


def score_replay(policy_run_scores: List[Tuple[bool, float, str]]) -> float:
    """Combine per-step results into one replay score.

    Each item: (choice_in_frontier, score_of_chosen_node, action_of_chosen).
    score = mean(chosen scores) + DIVERSITY_BETA * distinct actions / steps.
    """
    steps = len(policy_run_scores)
    if not steps:
        return 0.0
    valid = [r for r in policy_run_scores if r[0]]
    validity = len(valid) / steps
    mean_score = sum(r[1] for r in valid) / steps  # invalid steps count as 0
    distinct = len({r[2] for r in valid})
    return mean_score + DIVERSITY_BETA * (distinct / steps)


def evaluate_policy(sandbox: PolicySandbox, source: str,
                    world: List[Tuple[List[Dict[str, Any]], str]]) -> Tuple[float, str]:
    """Score a candidate on one replay world. Returns (score, error).

    A candidate that crashes or is statically invalid scores -inf via error.
    """
    if not world:
        return float("-inf"), "empty replay world"
    results: List[Tuple[bool, float, str]] = []
    for step, (frontier, _recorded) in enumerate(world):
        run = sandbox.choose(source, frontier, step)
        if run.error is not None:
            return float("-inf"), run.error
        ids = {n["node_id"]: n for n in frontier}
        node = ids.get(run.choice or "")
        if node is None:
            results.append((False, 0.0, "invalid"))
        else:
            results.append((True, node["score"], node["action"]))
    validity = sum(1 for r in results if r[0]) / len(results)
    if validity < MIN_VALIDITY:
        return float("-inf"), f"invalid choices in {100 * (1 - validity):.0f}% of steps"
    return score_replay(results), ""


def greedy_replay_score(world: List[Tuple[List[Dict[str, Any]], str]]) -> float:
    """Baseline: the loop's current greedy expansion (best-score leaf)."""
    results = []
    for frontier, _ in world:
        best = max(frontier, key=lambda n: n["score"])
        results.append((True, best["score"], best["action"]))
    return score_replay(results)


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
    replay score in the same cycle — evidence over vibes.
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
        world = replay_world(tree)
        if not world:
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
            score, err = evaluate_policy(self.sandbox, source, world)
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
            lines.append(f"  {node.node_id}  action={node.action}  "
                         f"score={node.score:.3f}  children={len(node.children)}")
        return "\n".join(lines) or "(empty)"
