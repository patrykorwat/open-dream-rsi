"""Command-line entry point: run the autonomous self-improvement loop.

    python -m open_dream_rsi loop --tasks tasks.json --once          # cron mode
    python -m open_dream_rsi loop --tasks tasks.json                 # daemon mode
    python -m open_dream_rsi status                                  # inspect memory
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from open_dream_rsi.loop import AutoRSIRuntime, Task
from open_dream_rsi.memory import DreamMemory


def _build_client(provider: str, model: str | None):
    from open_dream_rsi.llm import LLMConfig, OpenAICompatibleClient

    cfg = LLMConfig.from_preset(provider)  # honours OPENAI_BASE_URL / ODR_LLM_MODEL
    if model:
        cfg.model = model
    return OpenAICompatibleClient(cfg)


def cmd_loop(args: argparse.Namespace) -> int:
    tasks_raw = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    tasks = [Task(**t) for t in tasks_raw]
    memory = DreamMemory(args.memory)
    client = _build_client(args.provider, args.model)
    runtime = AutoRSIRuntime(
        client=client,
        memory=memory,
        tasks=tasks,
        api_call_budget=args.budget,
        dream_iterations=args.dream_iters,
        interval_seconds=args.interval,
    )
    if args.once:
        report = runtime.run_once()
        print(json.dumps(report.to_dict(), indent=2))
        return 0
    try:
        runtime.run_forever(max_cycles=args.cycles)
    except KeyboardInterrupt:
        print("\n[odr] supervisor stopped.", file=sys.stderr)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    memory = DreamMemory(args.memory)
    print(f"Memory root: {memory.root.resolve()}")
    for name, label in (("policies.json", "Dreamed policies"),
                        ("recipes.json", "Best-known solutions")):
        data = memory._load(memory.root / name, {})
        print(f"\n{label}: {len(data)}")
        for key, entry in data.items():
            print(f"  [{key}] {entry.get('updated_at') or entry.get('saved_at')} "
                  f"score={entry.get('score', '-')} "
                  f"{entry.get('params', '')}")
    events = memory.root / "events.jsonl"
    if events.exists():
        lines = events.read_text(encoding="utf-8").splitlines()
        print(f"\nEvent log: {len(lines)} events, last 5:")
        for line in lines[-5:]:
            print(" ", line[:200])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="odr", description="Open Dream-RSI autonomous supervisor")
    parser.add_argument("--memory", default=".dream_rsi", help="memory directory (one per instance)")
    sub = parser.add_subparsers(dest="command", required=True)

    loop = sub.add_parser("loop", help="run the self-improvement cycle")
    loop.add_argument("--tasks", required=True, help="JSON file: list of Task dicts")
    loop.add_argument("--provider", default="openai", choices=["openai", "cursor", "local"])
    loop.add_argument("--model", default=None)
    loop.add_argument("--budget", type=int, default=20, help="max API calls per cycle")
    loop.add_argument("--dream-iters", type=int, default=60)
    loop.add_argument("--interval", type=float, default=300.0, help="seconds between cycles")
    loop.add_argument("--cycles", type=int, default=None, help="stop after N cycles")
    loop.add_argument("--once", action="store_true", help="single cycle (for cron)")
    loop.set_defaults(func=cmd_loop)

    status = sub.add_parser("status", help="show learned policies, recipes, recent events")
    status.set_defaults(func=cmd_status)

    dash = sub.add_parser("dashboard", help="serve the live web dashboard")
    dash.add_argument("--host", default="0.0.0.0",
                      help="bind address (default 0.0.0.0 — reachable on the LAN)")
    dash.add_argument("--port", type=int, default=8765)
    dash.add_argument("--provider", default="mock", choices=["mock", "openai", "cursor", "local"])
    dash.add_argument("--model", default=None)
    dash.add_argument("--tasks", default=None, help="JSON task file (default: built-in demo set)")
    dash.add_argument("--interval", type=float, default=1.0, help="seconds between cycles")
    dash.add_argument("--budget", type=int, default=40)
    dash.set_defaults(func=cmd_dashboard)

    bench = sub.add_parser("bench", help="benchmark dreaming loop vs cold baseline")
    bench.add_argument("--cycles", type=int, default=10)
    bench.add_argument("--provider", default="mock", choices=["mock", "openai", "cursor", "local"])
    bench.add_argument("--model", default=None)
    bench.add_argument("--dream-iters", type=int, default=60)
    bench.add_argument("--tasks", default=None, help="JSON task file (default: built-in demo set)")
    bench.add_argument("--markdown", action="store_true", help="print markdown table")
    bench.set_defaults(func=cmd_bench)

    args = parser.parse_args(argv)
    return args.func(args)


def cmd_bench(args: argparse.Namespace) -> int:
    from open_dream_rsi.bench import run_benchmark, to_markdown

    tasks = None
    if args.tasks:
        raw = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
        tasks = [Task(**t) for t in raw]
    summary = run_benchmark(cycles=args.cycles, provider=args.provider,
                            model=args.model, dream_iterations=args.dream_iters,
                            tasks=tasks)
    print(json.dumps(summary, indent=2))
    if args.markdown:
        print(to_markdown(summary))
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    from open_dream_rsi.dashboard import run_dashboard

    run_dashboard(host=args.host, port=args.port, provider=args.provider,
                  model=args.model, tasks_file=args.tasks, interval=args.interval,
                  budget=args.budget, memory_root=args.memory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
