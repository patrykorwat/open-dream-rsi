"""API-efficiency benchmark: the dreaming loop vs a cold baseline.

Same task suite, same scripted model — the only difference is the library
machinery (persistent policies + recipes + offline dreaming vs cold start
every cycle). Mirrors the Dream-RSI paper's headline ablation (discovery
quality per online budget) at library scale.

    python -m open_dream_rsi bench --cycles 10 [--provider mock|openai|cursor|local]

Mock mode is fully deterministic and key-free: the LLM arm is a constant, so
the delta between arms comes purely from the runtime (warm starts, recipe
reuse, policy persistence). Plug a real client with --provider to measure
your own model.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from open_dream_rsi.loop import AutoRSIRuntime, Task
from open_dream_rsi.memory import DreamMemory


@dataclass
class ArmReport:
    arm: str
    cycles: int
    tasks_total: int
    solves_total: int
    api_calls_total: int
    mean_calls_per_task_cycle: float
    dream_iterations_total: int
    wall_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _make_client(provider: str, model: Optional[str]) -> Any:
    if provider == "mock":
        from open_dream_rsi.dashboard import MockLLM

        return MockLLM(latency=0.0)
    from open_dream_rsi.llm import LLMConfig, OpenAICompatibleClient

    cfg = LLMConfig.from_preset(provider)
    if model:
        cfg.model = model
    return OpenAICompatibleClient(cfg)


def _default_tasks() -> List[Task]:
    from open_dream_rsi.dashboard import DEMOS

    return [Task(max_attempts=3, **t) for t in DEMOS]


def run_arm(
    arm: str,
    tasks: List[Task],
    cycles: int,
    dream_iterations: int,
    client: Any,
    budget: int = 200,
) -> ArmReport:
    """One benchmark arm. dream_iterations=0 => cold baseline (fresh memory each cycle)."""
    root = tempfile.mkdtemp(prefix=f"odr-bench-{arm}-")
    t0 = time.time()
    api_calls = solves = dream_its = 0

    for _ in range(cycles):
        if dream_iterations == 0:
            shutil.rmtree(root, ignore_errors=True)  # cold start every cycle
        memory = DreamMemory(root)
        runtime = AutoRSIRuntime(
            client=client, memory=memory, tasks=tasks,
            api_call_budget=budget, dream_iterations=dream_iterations,
        )
        runtime.api_calls_used = 0
        report = runtime.run_once()
        api_calls += report.api_calls
        solves += report.tasks_solved
        dream_its += report.dream_iterations

    wall = time.time() - t0
    denom = max(cycles * len(tasks), 1)
    return ArmReport(
        arm=arm, cycles=cycles, tasks_total=len(tasks), solves_total=solves,
        api_calls_total=api_calls,
        mean_calls_per_task_cycle=round(api_calls / denom, 3),
        dream_iterations_total=dream_its, wall_seconds=round(wall, 2),
    )


def run_benchmark(cycles: int = 10, provider: str = "mock", model: Optional[str] = None,
                  dream_iterations: int = 60, tasks: Optional[List[Task]] = None) -> Dict[str, Any]:
    tasks = tasks or _default_tasks()
    arms: List[Dict[str, Any]] = []
    for arm_name, dream in (("cold_baseline", 0), ("dream_rsi_loop", dream_iterations)):
        client = _make_client(provider, model)  # fresh client per arm (scripted state resets)
        arms.append(run_arm(arm_name, tasks, cycles, dream, client).to_dict())

    base = arms[0]["api_calls_total"] or 1
    improved = arms[1]["api_calls_total"]
    summary = {
        "provider": provider,
        "cycles": cycles,
        "tasks": len(tasks),
        "arms": arms,
        "api_call_saving": round(1.0 - improved / base, 3),
    }
    return summary


def to_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        f"# Benchmark — {summary['provider']} client, {summary['cycles']} cycles × {summary['tasks']} tasks",
        "",
        "| arm | solves | API calls | calls / task·cycle | dream its | wall (s) |",
        "|---|---|---|---|---|---|",
    ]
    for a in summary["arms"]:
        lines.append(
            f"| {a['arm']} | {a['solves_total']} | {a['api_calls_total']} | "
            f"{a['mean_calls_per_task_cycle']} | {a['dream_iterations_total']} | {a['wall_seconds']} |"
        )
    lines += ["",
              f"API-call saving from dreaming + persistent memory: **{summary['api_call_saving']*100:.1f}%**",
              ""]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--provider", default="mock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--dream-iters", type=int, default=60)
    args = ap.parse_args()
    s = run_benchmark(args.cycles, args.provider, args.model, args.dream_iters)
    print(json.dumps(s, indent=2))
    print(to_markdown(s))
