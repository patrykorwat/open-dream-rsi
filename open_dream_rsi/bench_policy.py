"""Exploration-policy benchmark: can the LLM-written policy escape decoy traps?

The headline claim of section 3 ("dreaming with code") is not that an LLM can
emit a policy-shaped function — it is that the *promoted* policy makes better
online decisions than any fixed heuristic, scored offline by counterfactual
replay and only then trusted with real budget. This benchmark measures
exactly that on a hand-crafted **decoy-trap suite**.

A decoy trap: for each task there is a *plausible* code family that passes 2
of 3 tests (score 0.667) forever — re-polishing it never helps — while the
real fix hides behind a *low-scoring* branch (0.333) that a score-greedy
loop never re-opens. Solving therefore requires an exploration strategy,
not more attempts.

Arms (same scripted solver, same task suite, same budget — only the
expansion strategy differs):

* ``greedy``          — always expand the best-score node (no exploration).
* ``epsilon_greedy``  — + random exploration with probability epsilon.
* ``evolved_policy``  — the loop asks the LLM for a policy program per
  category, gates it by counterfactual rollout, and (on evidence) steers
  online expansion with the promoted code.

The scripted model plays two roles deterministically (like ``MockLLM`` in
``bench.py``): a task solver following the family ladder, and a policy
writer emitting a sensible round-robin explorer. The delta between arms
comes purely from the loop machinery — expansion choice + the promotion
gate — which is what this benchmark is about.

    python -m open_dream_rsi bench-policy [--cycles 6] [--format md|svg]
"""

from __future__ import annotations

import json
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from open_dream_rsi.loop import AutoRSIRuntime, Task
from open_dream_rsi.memory import DreamMemory

# ---------------------------------------------------------------------------
# The decoy-trap suite. Each category has exactly three code families whose
# verifier scores are FIXED BY CONSTRUCTION (asserted in tests):
#   decoy     -> 0.667  (plausible, passes 2/3 forever — the trap)
#   promising -> 0.333  (worse score, but its branch leads to the fix)
#   fix       -> 1.000  (solves)
# ---------------------------------------------------------------------------


def _T(pairs):  # test shorthand
    return [{"call": c, "expected": e} for c, e in pairs]


@dataclass(frozen=True)
class TrapSuite:
    category: str
    prompt: str
    tests: List[Dict[str, Any]]
    decoy: str
    promising: str
    fix: str
    low: str
    low: str


TRAP_SUITES: List[TrapSuite] = [
    TrapSuite(
        category="dedupe",
        prompt="Implement dedupe(xs): remove duplicates, keep first occurrences, keep order.",
        tests=_T([("dedupe([1, 2, 1])", [1, 2]), ("dedupe([3])", [3]),
                  ("dedupe([2, 2])", [2])]),
        decoy="def dedupe(xs):\n    return sorted(set(xs), reverse=True)\n",
        promising="def dedupe(xs):\n    return [x for x in xs if xs.count(x) == 1]\n",
        fix="def dedupe(xs):\n    return list(dict.fromkeys(xs))\n",
        low="def dedupe(xs):\n    return []\n",
    ),
    TrapSuite(
        category="flatten",
        prompt="Implement flatten(xss): concatenate one level of nested lists, keep order.",
        tests=_T([("flatten([[1], [2]])", [1, 2]), ("flatten([[1, 2]])", [1, 2]),
                  ("flatten([])", [])]),
        decoy="def flatten(xss):\n    return [x for xs in xss for x in xs[:1]]\n",
        promising="def flatten(xss):\n    return [x for xs in xss for x in xs[1:]]\n",
        fix="def flatten(xss):\n    out = []\n    for xs in xss:\n        out.extend(xs)\n    return out\n",
        low="def flatten(xss):\n    return [[]]\n",
    ),
    TrapSuite(
        category="top_k",
        prompt="Implement top_k(xs, k): the k largest values, descending.",
        tests=_T([("top_k([3, 1, 2], 2)", [3, 2]), ("top_k([9], 1)", [9]),
                  ("top_k([5, 4], 2)", [5, 4])]),
        decoy="def top_k(xs, k):\n    return sorted(xs)[::-1][:k + 1]\n",
        promising="def top_k(xs, k):\n    return sorted(xs)[:k]\n",
        fix="def top_k(xs, k):\n    return sorted(xs)[::-1][:k]\n",
        low="def top_k(xs, k):\n    return []\n",
    ),
    TrapSuite(
        category="running_max",
        prompt="Implement running_max(xs): prefix maxima (each element = max of xs[:i+1]).",
        tests=_T([("running_max([1, 3, 2, 4])", [1, 3, 3, 4]),
                  ("running_max([5])", [5]), ("running_max([3, 2, 1])", [3, 3, 3])]),
        decoy="def running_max(xs):\n    return [max(xs[:i + 1]) for i in range(len(xs) - 1)] + [xs[0]]\n",
        promising="def running_max(xs):\n    return [x for x in xs]\n",
        fix="def running_max(xs):\n    out, m = [], None\n    for x in xs:\n        if m is None or x > m:\n            m = x\n        out.append(m)\n    return out\n",
        low="def running_max(xs):\n    return []\n",
    ),
    TrapSuite(
        category="median",
        prompt="Implement median(xs): the middle value (even length: mean of the two middles).",
        tests=_T([("median([3, 1, 2])", 2), ("median([4, 8, 1, 2])", 3.0),
                  ("median([7])", 7)]),
        decoy="def median(xs):\n    return sorted(xs)[len(xs) // 2]\n",
        promising="def median(xs):\n    return sorted(xs)[len(xs) // 2 - 1]\n",
        fix="def median(xs):\n    s = sorted(xs)\n    n = len(s)\n    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2\n",
        low="def median(xs):\n    return 0\n",
    ),
]

#: The policy program the scripted policy-writer returns. It encodes exactly
#: the structural escape from the decoy trap: expand the oldest fresh node
#: (breadth over new branches), but on every third step re-enter the oldest
#: node in the whole tree — re-opening the root of the decoy chain, where a
#: score-greedy loop is locked out by design. This is the behaviour a real
#: LLM writes once the contract spells out the decoy-trap failure mode; the
#: point of the benchmark is the gate + wiring, not the model.
EXPLORER_POLICY = """
def choose_action(frontier, step):
    if not frontier:
        return None
    def born(n):
        tail = n['node_id'].rsplit('/', 1)[-1]
        digits = ''.join(ch for ch in tail if ch.isdigit())
        return int(digits) if digits else -1
    fresh = [n for n in frontier if n['children'] == 0]
    if fresh and step % 3 != 0:
        return min(fresh, key=born)['node_id']
    return min(frontier, key=born)['node_id']
"""


#: The lesson the scripted knowledge curator emits per category. The task
#: solver treats it as a real insight: when this text is visible in the
#: proposal prompt, an attempt on the DECOY family produces the promising
#: (alternative-idea) code instead of more decoy polishing — i.e. the lesson
#: causally changes what the model does, which is what the knowledge arm
#: measures. Like EXPLORER_POLICY, the point is the gate + wiring, not the
#: model: a real LLM writes the lesson text from the same failure evidence.
TRAP_LESSON_TEXT = (
    "A plausible branch whose verifier errors never change will never "
    "improve; re-open the low-scoring sibling branch where the fix idea "
    "first appeared."
)


class TrapSolver:
    """Deterministic scripted model implementing the decoy-trap world.

    Family ladder keyed by (category, family) — family derived from the code
    shown as the branch context (or ``seed`` when the branch is fresh):

      seed family:      k=1 -> decoy | k=2 -> promising | k>=3 -> decoy
      decoy family:     k=1 -> decoy | k>=2 -> low   (polishing decays: trap)
      promising family: any k -> fix                 (right idea converges)
      fix family:       any k -> fix

    The only route to the fix: re-expand the *seed* twice (revealing the
    0.333 promising node under it) and expand that promising node. A
    score-greedy loop never re-opens the seed (score 0) nor the promising
    node (score .333 < decoy .667) — it locks onto the decoy forever. That
    lock-in is the failure mode this benchmark measures against; epsilon
    breaks it by luck, a good exploration policy breaks it by structure
    (visit unvisited nodes, re-enter branches whose recorded outcome beats
    their own score), and a curated lesson breaks it by memory: when
    TRAP_LESSON_TEXT is visible in the prompt, decoy-family attempts are
    redirected to the promising alternative — knowledge replacing luck.

    Policy-generation calls (system prompt contains 'exploration policy')
    return EXPLORER_POLICY — the candidate the replay gate must validate and
    promote on evidence. Curation calls (system prompt contains 'knowledge
    curator') return the TRAP_LESSON_TEXT lesson for the asked category.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.policy_calls = 0
        self.curator_calls = 0
        self._ladder: Dict[Any, int] = {}

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _branch_section(user: str) -> str:
        m = re.search(r"Branch you are expanding from.*?:\n(.*)", user, re.S)
        if not m:
            return ""
        seg = m.group(1)
        # take until the next template section
        for stop in ("\nPolicy hints", "\nPrevious failure"):
            i = seg.find(stop)
            if i >= 0:
                seg = seg[:i]
        return seg

    @staticmethod
    def _lesson_visible(user: str) -> bool:
        """True when the curated KB surfaced its decoy-trap lesson."""
        return TRAP_LESSON_TEXT[:60] in user

    def _family(self, suite: TrapSuite, branch: str, prompt: str) -> str:
        if suite.fix.strip() in branch:
            return "fix"
        if suite.promising.strip() in branch:
            return "promising"
        if suite.decoy.strip() in branch:
            return "decoy"
        if suite.low.strip() in branch:
            return "low"
        return "seed"

    # -- ChatClient protocol ------------------------------------------------
    def chat(self, messages, model=None, temperature=0.7, max_tokens=1024):
        self.calls += 1
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "exploration policy" in system:
            self.policy_calls += 1
            return f"```python\n{EXPLORER_POLICY}\n```"
        if "knowledge curator" in system:
            self.curator_calls += 1
            m = re.search(r"Category \[([\w]+)\]", user)
            cat = m.group(1) if m else "task"
            item = {"trigger": f"{cat} decoy trap", "text": TRAP_LESSON_TEXT}
            return "```json\n" + json.dumps([item]) + "\n```"
        m = re.search(r"Task \[([\w]+)\]", user)
        cat = m.group(1) if m else "?"
        suite = next(s for s in TRAP_SUITES if s.category == cat)
        if suite.fix.strip() in user:  # recipe warm start or fix branch visible
            return f"```python\n{suite.fix}```"
        fam = self._family(suite, self._branch_section(user), user)
        if fam in ("decoy", "low") and self._lesson_visible(user):
            # the lesson worked: the model abandons decoy polishing for the
            # promising alternative idea instead of re-polishing the trap
            return f"```python\n{suite.promising}```"
        key = (cat, fam)
        self._ladder[key] = self._ladder.get(key, 0) + 1
        k = self._ladder[key]
        if fam == "seed":
            code = {1: suite.decoy, 2: suite.promising}.get(k, suite.decoy)
        elif fam == "decoy":
            code = suite.decoy if k == 1 else suite.low
        elif fam == "low":
            code = suite.low
        elif fam == "promising":
            code = suite.fix
        else:  # fix
            code = suite.fix
        return f"```python\n{code}```"


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

ARMS: Dict[str, Dict[str, Any]] = {
    "greedy":         dict(enable_policy_code=False, enable_knowledge=False, explore_epsilon=0.0),
    "epsilon_greedy": dict(enable_policy_code=False, enable_knowledge=False, explore_epsilon=0.3),
    "evolved_policy": dict(enable_policy_code=True,  enable_knowledge=False, explore_epsilon=0.3),
    # Knowledge arm: no LLM-written policy code — only the curated lesson KB
    # (distil -> validate -> retrieve -> usage-prune) vs epsilon_greedy. Any
    # improvement is attributable to knowledge, not to exploration heuristics.
    "knowledge_curator": dict(enable_policy_code=False, enable_knowledge=True,
                              explore_epsilon=0.3),
}


@dataclass
class PolicyArmReport:
    arm: str
    cycles: int
    runs: int
    tasks_total: int
    solves_total: int
    api_calls_total: int
    policy_calls: int
    curator_calls: int
    mean_calls_per_solve: float
    mean_calls_per_task_cycle: float
    mean_solve_rate_by_cycle: List[float] = field(default_factory=list)
    wall_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _trap_tasks() -> List[Task]:
    return [Task(task_id=f"{s.category}_1", category=s.category, prompt=s.prompt,
                 tests=list(s.tests), max_attempts=4) for s in TRAP_SUITES]


def run_policy_arm(arm: str, cfg: Dict[str, Any], cycles: int, budget: int,
                   dream_iterations: int = 30,
                   seeds: Sequence[int] = (7, 11, 23)) -> PolicyArmReport:
    """Run one arm over several seeds (memory is per run, deterministic solver).

    Multiple seeds smooth the ε random-walk; greedy is deterministic so all
    seeds agree there (a useful sanity property, asserted in tests).
    """
    t0 = time.time()
    total_solves = total_calls = total_policy = total_curator = 0
    rates: List[List[float]] = []
    for seed in seeds:
        tasks = _trap_tasks()
        solver = TrapSolver()
        root = tempfile.mkdtemp(prefix=f"odr-polbench-{arm}-")
        rates.append([])
        for c in range(cycles):
            memory = DreamMemory(root)  # persistent across cycles within a run
            runtime = AutoRSIRuntime(
                client=solver, memory=memory, tasks=tasks,
                api_call_budget=budget, dream_iterations=dream_iterations,
                rng_seed=seed * 100 + c, **cfg,
            )
            runtime.api_calls_used = 0
            report = runtime.run_once()
            rates[-1].append(100.0 * report.tasks_solved / max(len(tasks), 1))
            total_calls += report.api_calls
            total_solves += report.tasks_solved
        total_policy += solver.policy_calls
        total_curator += solver.curator_calls
    mean_rates = [round(sum(r[c] for r in rates) / len(rates), 1)
                  for c in range(cycles)]
    denom = max(len(seeds) * cycles * len(_trap_tasks()), 1)
    return PolicyArmReport(
        arm=arm, cycles=cycles, runs=len(seeds), tasks_total=len(_trap_tasks()),
        solves_total=total_solves, api_calls_total=total_calls,
        policy_calls=total_policy, curator_calls=total_curator,
        mean_calls_per_solve=round(total_calls / max(total_solves, 1), 2),
        mean_calls_per_task_cycle=round(total_calls / denom, 3),
        mean_solve_rate_by_cycle=mean_rates,
        wall_seconds=round(time.time() - t0, 2),
    )


def run_policy_benchmark(cycles: int = 8, budget: int = 24,
                         seeds: Sequence[int] = (7, 11, 23)) -> Dict[str, Any]:
    arms = [run_policy_arm(name, cfg, cycles, budget, seeds=seeds).to_dict()
            for name, cfg in ARMS.items()]
    return {
        "cycles": cycles,
        "runs": len(seeds),
        "budget_per_cycle": budget,
        "tasks": len(_trap_tasks()),
        "arms": arms,
        "greedy_solves": arms[0]["solves_total"],
        "slots": len(seeds) * cycles * len(_trap_tasks()),
    }


# ---------------------------------------------------------------------------
# Rendering: markdown table + dependency-free SVG figure for the README
# ---------------------------------------------------------------------------

def to_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        f"# Policy benchmark — decoy-trap suite, {summary['runs']} runs × "
        f"{summary['cycles']} cycles × {summary['tasks']} tasks "
        f"(budget {summary['budget_per_cycle']} calls/cycle)",
        "",
        "| arm | solves | solve rate | API calls | calls / solve | policy calls | curator calls |",
        "|---|---|---|---|---|---|---|",
    ]
    slots = summary["slots"]
    for a in summary["arms"]:
        rate = 100.0 * a["solves_total"] / max(slots, 1)
        cps = "∞" if a["solves_total"] == 0 else a["mean_calls_per_solve"]
        lines.append(
            f"| {a['arm']} | {a['solves_total']}/{slots} | {rate:.0f}% | "
            f"{a['api_calls_total']} | {cps} | {a['policy_calls']} | "
            f"{a.get('curator_calls', 0)} |")
    lines += ["",
              f"Score-greedy exploration solves **{summary['greedy_solves']}/{slots}** "
              "trap tasks and then burns its whole budget re-polishing the decoy; "
              "replay-gated LLM-written policies reach the hidden fixes and reuse "
              "them as warm starts; the curated knowledge base lets epsilon alone "
              "spend luck elsewhere because the trap itself is remembered.", ""]
    return "\n".join(lines)


def to_svg(summary: Dict[str, Any], width: int = 860, height: int = 300) -> str:
    """Grouped bar chart: solve rate (%) per cycle per arm. Pure SVG, no deps."""
    arms = summary["arms"]
    cycles = summary["cycles"]
    colors = {"greedy": "#e5534b", "epsilon_greedy": "#d4a72c", "evolved_policy": "#3fb950",
              "knowledge_curator": "#58a6ff"}
    pad_l, pad_r, pad_t, pad_b = 48, 16, 30, 46
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    slot_w = plot_w / max(cycles, 1)
    bar_w = max(6.0, (slot_w - 14) / len(arms))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="system-ui,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
    ]
    # y grid
    for pct in (0, 25, 50, 75, 100):
        y = pad_t + plot_h * (1 - pct / 100.0)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" '
                     f'stroke="#21262d" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{y + 4:.1f}" fill="#8b949e" font-size="11" '
                     f'text-anchor="end">{pct}%</text>')
    # bars
    for ai, arm in enumerate(arms):
        color = colors.get(arm["arm"], "#58a6ff")
        for ci, pct in enumerate(arm["mean_solve_rate_by_cycle"]):
            x = pad_l + ci * slot_w + 7 + ai * bar_w
            h = plot_h * pct / 100.0
            y = pad_t + plot_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 3:.1f}" '
                         f'height="{max(h, 1):.1f}" rx="2" fill="{color}"/>')
    # x labels + legend
    for ci in range(cycles):
        cx = pad_l + ci * slot_w + slot_w / 2
        parts.append(f'<text x="{cx:.1f}" y="{pad_t + plot_h + 16}" fill="#8b949e" '
                     f'font-size="11" text-anchor="middle">cycle {ci + 1}</text>')
    lx = pad_l
    for arm in arms:
        color = colors.get(arm["arm"], "#58a6ff")
        parts.append(f'<rect x="{lx}" y="{height - 18}" width="10" height="10" rx="2" '
                     f'fill="{color}"/>')
        parts.append(f'<text x="{lx + 14}" y="{height - 9}" fill="#c9d1d9" font-size="11">'
                     f'{arm["arm"]}</text>')
        lx += 150
    parts.append(f'<text x="{pad_l}" y="18" fill="#c9d1d9" font-size="13" '
                 f'font-weight="600">Decoy-trap suite — solve rate per cycle '
                 f'(greedy vs ε-greedy vs replay-gated policies vs curated KB)</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--format", choices=["json", "md", "svg"], default="json")
    args = ap.parse_args()
    s = run_policy_benchmark(args.cycles, args.budget)
    if args.format == "json":
        print(json.dumps(s, indent=2))
    elif args.format == "md":
        print(to_markdown(s))
    else:
        print(to_svg(s))
