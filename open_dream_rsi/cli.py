"""Command-line entry point: run the autonomous self-improvement loop.

    python -m open_dream_rsi loop --tasks tasks.json --once          # cron mode
    python -m open_dream_rsi loop --tasks tasks.json                 # daemon mode
    python -m open_dream_rsi status                                  # inspect memory
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from open_dream_rsi.loop import AutoRSIRuntime, Task
from open_dream_rsi.memory import DreamMemory


def _build_client(provider: str, model: str | None):
    from open_dream_rsi.llm import LLMConfig, OpenAICompatibleClient

    cfg = LLMConfig.from_preset(provider)  # honours OPENAI_BASE_URL / ODR_LLM_MODEL
    if model:
        cfg.model = model
    _apply_no_thinking(cfg)
    return OpenAICompatibleClient(cfg)


def _apply_no_thinking(cfg) -> None:
    """ODR_LLM_NO_THINKING=1: ask vLLM chat templates to skip hidden reasoning.

    Reasoning models (Qwen3.x builds) can burn the whole completion budget on
    thinking and return empty or rambling content; disabling it made a real
    Spark run go from timeout to a correct answer in 19s. ON BY DEFAULT: the
    loop wants structured answers, not hidden reasoning. Endpoints that reject
    the flag self-heal (plain retry on HTTP 400); opt out with
    ODR_LLM_NO_THINKING=0 for models whose thinking is in-band.
    """
    import os

    if os.environ.get("ODR_LLM_NO_THINKING", "1").lower() not in ("0", "false", "no"):
        cfg.extra_payload.setdefault("chat_template_kwargs", {})["enable_thinking"] = False


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
        max_tokens=args.max_tokens,
        enable_policy_code=not args.no_policy_code,
        enable_knowledge=not args.no_knowledge,
        enable_thoughts=not args.no_thoughts,
        enable_judge=not args.no_judge,
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
                        ("recipes.json", "Best-known solutions"),
                        ("lessons.json", "Curated knowledge (lessons)")):
        data = memory._load(memory.root / name, {})
        print(f"\n{label}: {len(data)}")
        for key, entry in data.items():
            if name == "lessons.json":
                print(f"  [{key}] {len(entry)} lesson(s)")
                for l in entry:
                    print(f"    - [{l.get('trigger', '')}] {l.get('text', '')[:80]} "
                          f"(wins={l.get('wins', 0)} uses={l.get('uses', 0)})")
                continue
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


def cmd_sessions(args: argparse.Namespace) -> int:
    """Import host transcripts (Hermes state.db / Cursor state.vscdb) as
    curator evidence — the offline second door for lessons (see README)."""
    from open_dream_rsi.sessions import (
        episode_failures,
        load_episodes,
        read_cursor_sessions,
        read_hermes_sessions,
        write_episodes,
    )

    if args.sessions_cmd == "export":
        if args.source == "hermes":
            eps = read_hermes_sessions(args.db or args.memory,
                                       source=args.filter,
                                       limit=args.limit,
                                       redact=not args.no_redact)
        else:
            # --db accepts either the Cursor User dir or a state.vscdb copy
            db_arg = args.db
            db_file = db_arg if db_arg and Path(db_arg).suffix == ".vscdb" else None
            eps = read_cursor_sessions(user_dir=None if db_file else db_arg,
                                       db=db_file, limit=args.limit,
                                       redact=not args.no_redact)
        n = write_episodes(eps, args.out)
        msgs = sum(len(e.messages) for e in eps)
        print(f"[odr] exported {n} episode(s), {msgs} message(s) -> {args.out}")
        if not args.no_redact:
            print("[odr] secret redaction applied (best effort — see README)")
        return 0

    # distill: episodes -> staging lessons (gate-gated exactly like loop ones)
    from open_dream_rsi.core.curator import (
        KnowledgeCurator,
        curate_lessons,
        evidence_snippets,
    )

    eps = load_episodes(args.episodes)
    if not eps:
        print("[odr] no episodes to distil", file=sys.stderr)
        return 1
    memory = DreamMemory(args.memory)
    provider = getattr(args, "provider", "openai")
    if args.dry_run:
        for ep in eps[: args.limit or len(eps)]:
            fails = episode_failures(ep)
            print(f"[{ep.source}:{ep.session_id[:12]}] {ep.title[:60]} "
                  f"— {len(fails)} failure-shaped record(s)")
            for f in fails[:3]:
                print("   ", f["errors"][0][:100])
        return 0
    from open_dream_rsi.cli import _build_client
    client = _build_client(provider, args.model)
    curator = KnowledgeCurator(client)
    total_added = 0
    for ep in eps:
        failures = episode_failures(ep)
        if not failures:
            continue
        category = args.category or (
            Path(ep.cwd).name if ep.cwd else ep.source)
        distilled, err = curator.distill(category, ep.title or category,
                                         failures, memory.get_lessons(category))
        if err or not distilled:
            memory.log_event("lesson_rejected", source=ep.source,
                             session_id=ep.session_id, reason=(err or "empty")[:300])
            continue
        result = curate_lessons(memory.get_lessons(category), distilled,
                                evidence=evidence_snippets(failures))
        memory.replace_lessons(category, result.entries)
        memory.log_event("lessons_curated", source=ep.source,
                         session_id=ep.session_id, category=category,
                         added=len(result.added), merged=result.merged)
        total_added += len(result.added)
        print(f"[odr] {category}: +{len(result.added)} staging lesson(s) "
              f"from {ep.source}:{ep.session_id[:12]}")
    print(f"[odr] done — {total_added} staging lesson(s); they activate only "
          "through the paired-replay gate (odr loop) — never before")
    return 0


def cmd_dream(args: argparse.Namespace) -> int:
    """Automated cycle: evidence -> candidates -> promotion gate -> skills.

    The cadence is decided inside open_dream_rsi.dream (cheap maintenance
    per call; the full world dreamer only when evidence justifies the
    cost), so any trigger — a Hermes hook, the MCP tool, or this CLI —
    can fire as often as it likes without overrunning the budget."""
    from open_dream_rsi.dream import dream_once

    client = None
    if not args.offline:
        client = _build_client(args.provider, args.model)
    report = dream_once(
        args.memory, sessions_db=args.sessions, skills_out=args.skills_out,
        client=client, budget=args.budget, max_tokens=args.max_tokens,
        category=args.category)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def cmd_lessons(args: argparse.Namespace) -> int:
    """Git-backed lesson stores: publish active lessons, import others'
    knowledge as staging (see README 'Sharing lessons')."""
    from open_dream_rsi.share import (LessonShareError, export_lessons,
                                      import_lessons, store_status)

    memory = DreamMemory(args.memory)
    try:
        if args.lessons_cmd == "export":
            st = export_lessons(memory, args.store, categories=args.category,
                                 message=args.message, push=not args.no_push)
        elif args.lessons_cmd == "import":
            counts = import_lessons(args.store, memory,
                                    categories=args.category)
            if not counts:
                print("[odr] nothing imported (empty/filtered store)")
                return 0
            for cat, n in sorted(counts.items()):
                print(f"[odr] {cat}: +{n} staging lesson(s) — activate via "
                      f"the paired-replay gate (odr loop)")
            return 0
        else:  # status
            st = store_status(args.store)
    except LessonShareError as exc:
        print(f"[odr] store error: {exc}", file=sys.stderr)
        return 1
    if getattr(args, "json", False):
        print(json.dumps(st.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"[odr] store: {st.location} "
              f"({'git' if st.git else 'plain dir'}, ok={st.ok})")
        for cat, n in sorted(st.categories.items()):
            print(f"  {cat}: {n} lesson(s)")
        for e in st.errors:
            print(f"  ! {e}")
    return 0 if st.ok else 1


def cmd_skills(args: argparse.Namespace) -> int:
    """Render the curated lesson KB into Hermes-style skills (issue #3
    boundary: only lessons that EARNED activation become a prompt)."""
    from open_dream_rsi.sessions import lessons_to_skills

    memory = DreamMemory(args.memory)
    written = lessons_to_skills(
        memory, args.out,
        categories=args.category,
        only_active=not args.include_staging)
    if not written:
        print("[odr] no active lessons to export "
              "(staging lessons activate only through the paired-replay gate)")
        return 0
    for p in written:
        print(f"[odr] wrote {p}")
    print(f"[odr] {len(written)} skill(s) — point Hermes at the parent dir "
          "(skills auto-discovery) or copy into ~/.hermes/skills/")
    return 0


def cmd_artifacts(args: argparse.Namespace) -> int:
    """Inspect the Artifact Lifecycle Manager store (issue #3)."""
    from open_dream_rsi.lifecycle import ArtifactLifecycleManager

    memory = DreamMemory(args.memory)
    alm = ArtifactLifecycleManager(memory)
    states = alm.states(artifact_type=args.type,
                        lifecycle_state=args.state)
    if args.json:
        payload = {"materialized_matches_events": alm.verify_materialization(),
                   "artifacts": [s.to_dict() for s in states]}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    ok = alm.verify_materialization()
    print(f"Artifact store: {memory.root / 'artifacts'} "
          f"({len(alm.all_events())} events, {len(states)} artifacts shown)")
    print(f"Materialized view == rebuild(events): {'OK' if ok else 'MISMATCH'}")
    for s in states:
        print(f"  {s.artifact_id}  [{s.artifact_type}] slot={s.slot} "
              f"v{s.version} {s.lifecycle_state}"
              + (" PIN" if s.pinned else "")
              + (f" supersedes={','.join(s.supersedes)}" if s.supersedes else "")
              + (f" by={s.superseded_by}" if s.superseded_by else ""))
        if args.history:
            for e in alm.history(s.artifact_id):
                print(f"      #{e.sequence} {e.kind}: {e.from_state or '-'}"
                      f" -> {e.to_state}  ({e.actor}) {e.reason[:60]}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="odr", description="Open Dream-RSI autonomous supervisor")
    parser.add_argument("--memory", default=".dream_rsi", help="memory directory (one per instance)")
    sub = parser.add_subparsers(dest="command", required=True)

    loop = sub.add_parser("loop", help="run the self-improvement cycle")
    loop.add_argument("--tasks", required=True, help="JSON file: list of Task dicts")
    loop.add_argument("--provider", default="openai",
                      choices=["openai", "cursor", "local", "goose"],
                      help="LLM endpoint; 'goose' borrows goose's own "
                           "provider/model/key from ~/.config/goose")
    loop.add_argument("--model", default=None)
    loop.add_argument("--budget", type=int, default=20, help="max API calls per cycle")
    loop.add_argument("--max-tokens", type=int, default=2048,
                      help="completion budget per LLM call (raise for reasoning models)")
    loop.add_argument("--dream-iters", type=int, default=60)
    loop.add_argument("--no-policy-code", action="store_true",
                      help="disable LLM-written exploration policies (section-3 step)")
    loop.add_argument("--no-knowledge", action="store_true",
                      help="disable the knowledge curator (lessons.json KB, section-4 step)")
    loop.add_argument("--no-thoughts", action="store_true",
                      help="disable thought-conditioned branching (PLAN lines, "
                           "tried-idea ledger, semantic stagnation steering; "
                           "on by default)")
    loop.add_argument("--no-judge", action="store_true",
                      help="disable the LLM completion judge for test-less "
                           "tasks (such tasks then always verdict unsolved; "
                           "tasks WITH tests are never judged)")
    loop.add_argument("--interval", type=float, default=300.0, help="seconds between cycles")
    loop.add_argument("--cycles", type=int, default=None, help="stop after N cycles")
    loop.add_argument("--once", action="store_true", help="single cycle (for cron)")
    loop.set_defaults(func=cmd_loop)

    dr = sub.add_parser(
        "dream",
        help="automated cycle — evidence import, promotion gate, ALM "
             "maintenance, skills publish; safe to fire from any trigger "
             "(cadence is decided internally, not by the caller)")
    dr.add_argument("--sessions", default=None,
                    help="host state.db to import evidence from "
                         "(e.g. ~/.hermes/state.db)")
    dr.add_argument("--skills-out", default=None,
                    help="render ACTIVE lessons here as SKILL.md files")
    dr.add_argument("--provider", default="openai",
                    choices=["openai", "cursor", "local", "goose"])
    dr.add_argument("--model", default=None)
    dr.add_argument("--category", default=None,
                    help="force one KB category (default: per-episode cwd)")
    dr.add_argument("--budget", type=int, default=20,
                    help="max API calls for a FULL dream (default 20)")
    dr.add_argument("--max-tokens", type=int, default=4096)
    dr.add_argument("--offline", action="store_true",
                    help="no LLM: lifecycle maintenance only")
    dr.set_defaults(func=cmd_dream)

    status = sub.add_parser("status", help="show learned policies, recipes, recent events")
    status.set_defaults(func=cmd_status)

    art = sub.add_parser(
        "artifacts",
        help="inspect the Artifact Lifecycle Manager store (lifecycle states, "
             "lineage, transition history, materialization check)")
    art.add_argument("--type", default=None,
                     help="filter by artifact type (policy_parameters, "
                          "policy_program, recipe, lesson)")
    art.add_argument("--state", default=None,
                     help="filter by lifecycle state (CANDIDATE, VALIDATED, "
                          "ACTIVE, STALE, SUPERSEDED, QUARANTINED, ARCHIVED, REJECTED)")
    art.add_argument("--history", action="store_true",
                     help="print each artifact's transition history")
    art.add_argument("--json", action="store_true", help="machine-readable output")
    art.set_defaults(func=cmd_artifacts)

    sess = sub.add_parser(
        "sessions",
        help="import host transcripts (Hermes state.db / Cursor state.vscdb) "
             "as curator evidence — export to JSONL, then distill into "
             "staging lessons (promotion gate still decides activation)")
    sess_sub = sess.add_subparsers(dest="sessions_cmd", required=True)

    exp = sess_sub.add_parser("export", help="read a host store, write episodes.jsonl")
    exp.add_argument("--source", required=True, choices=["hermes", "cursor"])
    exp.add_argument("--db", default=None,
                     help="path to state.db (hermes) or Cursor User dir "
                          "(default: platform Cursor location)")
    exp.add_argument("--filter", default=None,
                     help="hermes only: session source filter (e.g. cli, desktop)")
    exp.add_argument("--limit", type=int, default=None)
    exp.add_argument("--out", required=True, help="output episodes JSONL path")
    exp.add_argument("--no-redact", action="store_true",
                     help="DISABLE secret redaction (not recommended — chat "
                          "history contains pasted credentials)")
    exp.set_defaults(func=cmd_sessions)

    dis = sess_sub.add_parser("distill",
                              help="distil exported episodes into staging lessons")
    dis.add_argument("episodes", help="episodes JSONL from 'sessions export'")
    dis.add_argument("--category", default=None,
                     help="KB category (default: cwd name of each episode)")
    dis.add_argument("--provider", default="openai",
                     choices=["openai", "cursor", "local", "goose"])
    dis.add_argument("--model", default=None)
    dis.add_argument("--limit", type=int, default=None)
    dis.add_argument("--dry-run", action="store_true",
                     help="show failure-shaped records, no LLM call")
    dis.set_defaults(func=cmd_sessions)

    sk = sub.add_parser(
        "skills",
        help="bridge the curated lesson KB and Hermes skills")
    sk_sub = sk.add_subparsers(dest="skills_cmd", required=True)
    skx = sk_sub.add_parser(
        "export",
        help="render active lessons into Hermes-style skills "
             "(<out>/<category>/SKILL.md)")
    skx.add_argument("--out", required=True,
                     help="target skills dir (e.g. ~/.hermes/skills/odr-curated)")
    skx.add_argument("--category", action="append", default=None,
                     help="limit to these categories (repeatable)")
    skx.add_argument("--include-staging", action="store_true",
                     help="export staging lessons too (not recommended: "
                          "staging has not earned a prompt)")
    skx.set_defaults(func=cmd_skills)

    ls = sub.add_parser(
        "lessons",
        help="git-backed lesson stores — publish active lessons to a repo, "
             "import others' knowledge as staging (schema odr-lessons/v1)")
    ls_sub = ls.add_subparsers(dest="lessons_cmd", required=True)

    lexp = ls_sub.add_parser("export", help="write active lessons into a store")
    lexp.add_argument("--store", required=True,
                      help="local dir or git URL (cloned/pulled under "
                           "<memory>/lesson_stores/)")
    lexp.add_argument("--category", action="append", default=None,
                      help="limit to these categories (repeatable)")
    lexp.add_argument("--message", default=None, help="git commit message")
    lexp.add_argument("--no-push", action="store_true",
                      help="commit locally but do not push")
    lexp.set_defaults(func=cmd_lessons)

    limp = ls_sub.add_parser("import",
                             help="merge a store's lessons into the KB as staging")
    limp.add_argument("--store", required=True,
                      help="local dir or git URL")
    limp.add_argument("--category", action="append", default=None)
    limp.set_defaults(func=cmd_lessons)

    lst = ls_sub.add_parser("status", help="verify manifest + checksums")
    lst.add_argument("--store", required=True)
    lst.add_argument("--json", action="store_true")
    lst.set_defaults(func=cmd_lessons)

    dash = sub.add_parser("dashboard", help="serve the live web dashboard")
    dash.add_argument("--host", default="0.0.0.0",
                      help="bind address (default 0.0.0.0 — reachable on the LAN)")
    dash.add_argument("--port", type=int, default=8765)
    dash.add_argument("--provider", default="mock", choices=["mock", "openai", "cursor", "local"])
    dash.add_argument("--model", default=None)
    dash.add_argument("--tasks", default=None, help="JSON task file (default: built-in demo set)")
    dash.add_argument("--interval", type=float, default=1.0, help="seconds between cycles")
    dash.add_argument("--budget", type=int, default=40)
    dash.add_argument("--max-tokens", type=int, default=2048,
                      help="completion budget per LLM call (raise for reasoning models)")
    dash.set_defaults(func=cmd_dashboard)

    bench = sub.add_parser("bench", help="benchmark dreaming loop vs cold baseline")
    bench.add_argument("--cycles", type=int, default=10)
    bench.add_argument("--provider", default="mock", choices=["mock", "openai", "cursor", "local"])
    bench.add_argument("--model", default=None)
    bench.add_argument("--dream-iters", type=int, default=60)
    bench.add_argument("--tasks", default=None, help="JSON task file (default: built-in demo set)")
    bench.add_argument("--markdown", action="store_true", help="print markdown table")
    bench.set_defaults(func=cmd_bench)

    polbench = sub.add_parser(
        "bench-policy",
        help="benchmark LLM-written exploration policies on the decoy-trap suite")
    polbench.add_argument("--cycles", type=int, default=8)
    polbench.add_argument("--budget", type=int, default=24)
    polbench.add_argument("--format", choices=["json", "md", "svg"], default="json")
    polbench.add_argument("--theme", choices=["dark", "light"], default="dark",
                          help="SVG palette: dark (README) or light (print/paper)")
    polbench.add_argument("--out", default=None, help="write output to a file")
    polbench.set_defaults(func=cmd_bench_policy)

    gate = sub.add_parser(
        "gate-replay",
        help="validate the lesson promotion gate offline on recorded "
             "baseline/warm arm JSON pairs (no model, no key)")
    gate.add_argument("--baseline", required=True,
                      metavar="NAME=PATH[:KEY][?f=v,...]",
                      help="no-lessons arm: rows under top-level KEY "
                           "(auto-detected when unambiguous), optional "
                           "field=value row filters")
    gate.add_argument("--compare", action="append", required=True,
                      metavar="NAME=PATH[:KEY][?f=v,...]",
                      help="lessons arm to judge against the baseline "
                           "(repeatable)")
    gate.add_argument("--key", default="task_id",
                      help="join field on rows (e.g. krs, task_id)")
    gate.add_argument("--solved-field", default="solved")
    gate.add_argument("--lessons", default=None, metavar="PATH",
                      help="optional lessons.json: report stop-clause "
                           "precondition per lesson text")
    gate.add_argument("--format", choices=["json", "md"], default="json")
    gate.set_defaults(func=cmd_gate_replay)

    mcp = sub.add_parser(
        "mcp",
        help="serve the loop as an MCP server (stdio for OpenCode/Goose/Claude "
             "Code/Codex/Hermes; --http for claude.ai/Cowork connectors)")
    mcp.add_argument("--tasks", default=None, help="task file (default $ODR_TASKS or tasks.json)")
    mcp.add_argument("--memory", default=None,
                     help="memory dir (default $ODR_MEMORY or .dream_rsi)")
    mcp.add_argument("--http", action="store_true",
                     help="serve stateless Streamable-HTTP MCP on POST /mcp")
    mcp.add_argument("--host", default="127.0.0.1")
    mcp.add_argument("--port", type=int, default=8800)
    mcp.set_defaults(func=cmd_mcp)

    proxy = sub.add_parser(
        "proxy",
        help="OpenAI-compatible proxy forwarding to goose's own model "
             "(borrow the host LLM; point OPENAI_BASE_URL at it)")
    proxy.add_argument("--host", default="127.0.0.1")
    proxy.add_argument("--port", type=int, default=8799)
    proxy.add_argument("--provider", default=None, help="override GOOSE_PROVIDER")
    proxy.add_argument("--model", default=None, help="override GOOSE_MODEL")
    proxy.add_argument("--base-url", default=None, help="override upstream base URL")
    proxy.add_argument("--api-key", default=None, help="override upstream key")
    proxy.add_argument("--passthrough-models", action="store_true",
                       help="keep caller's model instead of pinning GOOSE_MODEL")
    proxy.add_argument("--print-config", action="store_true",
                       help="show resolved upstream (key masked) and exit")
    proxy.set_defaults(func=cmd_proxy)

    sent = sub.add_parser(
        "sentinel",
        help="host-agnostic error-class sentinel: annotate failing tool "
             "results with recurrence facts (command-hook compatible)")
    sent_sub = sent.add_subparsers(dest="sentinel_command", required=True)
    chk = sent_sub.add_parser(
        "check", help="read one tool-result event JSON on stdin, print a "
                      "note if the error class is recurring (plain) or a "
                      "host response (--format claude)")
    chk.add_argument("--tool", default=None,
                     help="tool name (else event.tool_name / event.tool)")
    chk.add_argument("--session", default=None,
                     help="session id (else event.session_id / $CLAUDE_SESSION_ID)")
    chk.add_argument("--status", default=None, choices=["error", "ok"],
                     help="host-reported status (else inferred from payload)")
    chk.add_argument("--format", default="plain", choices=["plain", "claude"],
                     help="plain: note or nothing on stdout; claude: "
                          "PostToolUse hook JSON")
    chk.add_argument("--state", default=None,
                     help="state file (default $ODR_SENTINEL_STATE or "
                          "~/.local/state/odr-sentinel/state.json)")
    chk.set_defaults(func=cmd_sentinel_check)
    led = sent_sub.add_parser("ledger", help="show the tracked error-class ledger")
    led.add_argument("--state", default=None)
    led.set_defaults(func=cmd_sentinel_ledger)
    rst = sent_sub.add_parser("reset", help="clear the ledger")
    rst.add_argument("--state", default=None)
    rst.set_defaults(func=cmd_sentinel_reset)

    args = parser.parse_args(argv)
    return args.func(args)


def cmd_mcp(args: argparse.Namespace) -> int:
    from open_dream_rsi.mcp import main as mcp_main

    argv = []
    if args.tasks:
        argv += ["--tasks", args.tasks]
    if args.memory:  # --memory is a top-level flag; forward it explicitly
        argv += ["--memory", args.memory]
    if args.http:
        argv += ["--http", "--host", args.host, "--port", str(args.port)]
    return mcp_main(argv)


def cmd_proxy(args: argparse.Namespace) -> int:
    from open_dream_rsi.proxy import main as proxy_main

    argv = ["--host", args.host, "--port", str(args.port)]
    if args.provider:
        argv += ["--provider", args.provider]
    if args.model:
        argv += ["--model", args.model]
    if args.base_url:
        argv += ["--base-url", args.base_url]
    if args.api_key:
        argv += ["--api-key", args.api_key]
    if args.passthrough_models:
        argv += ["--passthrough-models"]
    if args.print_config:
        argv += ["--print-config"]
    return proxy_main(argv)


def cmd_sentinel_check(args: argparse.Namespace) -> int:
    """Command-hook adapter: one tool-result event JSON on stdin -> note.

    Event schema (all optional): {"tool_name"|"tool": str,
    "result"|"tool_response": any, "session_id": str, "status": "error"|"ok",
    "error_message": str}. Always exits 0 and prints at most the note (plain)
    or a valid hook response (claude) — never blocks the host."""
    import sys as _sys

    from open_dream_rsi.sentinel import SentinelEngine

    try:
        event = json.loads(_sys.stdin.read() or "{}")
        if not isinstance(event, dict):
            event = {}
    except Exception:
        event = {}
    tool = args.tool or str(event.get("tool_name") or event.get("tool") or "?")
    result = event.get("result", event.get("tool_response"))
    if not isinstance(result, str):
        result = json.dumps(result, default=str)
    session = (args.session or event.get("session_id")
               or os.environ.get("CLAUDE_SESSION_ID") or "default")
    status = args.status or event.get("status")
    error_message = event.get("error_message")
    if event.get("hook_event_name") == "PostToolUseFailure" and status is None:
        # the host already tells us this call failed — no need to guess
        status = "error"
        error_message = error_message or (result if isinstance(result, str) else None)
    engine = SentinelEngine(state_path=args.state,
                            intra_session_repeat=_cfg_sentinel(args, "intra_session_repeat", 2),
                            cross_session_count=_cfg_sentinel(args, "cross_session_count", 2))
    note = engine.observe(tool, result, session,
                          status=status, error_message=error_message)
    if args.format == "claude":
        payload = {}
        if note:
            payload = {"hookSpecificOutput": {
                "hookEventName": event.get("hook_event_name") or "PostToolUse",
                "additionalContext": note.strip()}}
        print(json.dumps(payload))
    elif note:
        print(note.strip())
    return 0


def _cfg_sentinel(args: argparse.Namespace, key: str, default: int) -> int:
    return int(os.environ.get("ODR_SENTINEL_" + key.upper(), default))


def cmd_sentinel_ledger(args: argparse.Namespace) -> int:
    from open_dream_rsi.sentinel import SentinelEngine

    print(SentinelEngine(state_path=args.state).ledger_text())
    return 0


def cmd_sentinel_reset(args: argparse.Namespace) -> int:
    from open_dream_rsi.sentinel import SentinelEngine

    SentinelEngine(state_path=args.state).reset()
    print("Sentinel: signature ledger cleared.")
    return 0


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


def cmd_gate_replay(args: argparse.Namespace) -> int:
    from open_dream_rsi.gate_replay import (
        filter_rows,
        gate_replay,
        load_arm_rows,
        to_markdown,
    )

    def parse_spec(spec: str) -> tuple[str, str, str | None, list[str]]:
        # name=path[:records_key][?field=value,field2=value2]
        rest, fsep, filt = spec.partition("?")      # filters split FIRST
        head, sep, tail = rest.partition("=")
        if not sep:                      # no NAME= prefix — whole spec is path
            head, tail = "", rest
        path, ksep, rec_key = tail.partition(":")
        filters = [f for f in filt.split(",") if f]
        return head or path, path, (rec_key if ksep else None), filters

    def load(spec: str) -> list[dict]:
        name, path, rec_key, filters = parse_spec(spec)
        return filter_rows(load_arm_rows(path, rec_key), filters)

    b_name, b_path, _, _ = parse_spec(args.baseline)
    baseline_rows = load(args.baseline)
    arms: dict[str, list[dict]] = {}
    for spec in args.compare:
        name = parse_spec(spec)[0]
        arms[name] = load(spec)
    lesson_texts = None
    if args.lessons:
        lessons = load(args.lessons)
        lesson_texts = [l.get("text", "") for l in lessons]
    report = gate_replay(baseline_rows, arms, key=args.key,
                         solved_field=args.solved_field,
                         lesson_texts=lesson_texts)
    if args.format == "md":
        print(to_markdown(report))
    else:
        print(json.dumps(report, indent=2))
    harmful = [n for n, d in report["arms"].items() if not d["promoted"]]
    print(f"# baseline arm: {b_name}; gate rejects: "
          f"{', '.join(harmful) if harmful else 'none'}", file=sys.stderr)
    return 0


def cmd_bench_policy(args: argparse.Namespace) -> int:
    from open_dream_rsi.bench_policy import (
        run_policy_benchmark,
        to_markdown as pol_markdown,
        to_svg,
    )

    summary = run_policy_benchmark(cycles=args.cycles, budget=args.budget)
    if args.format == "json":
        text = json.dumps(summary, indent=2)
    elif args.format == "md":
        text = pol_markdown(summary)
    else:
        text = to_svg(summary, theme=getattr(args, "theme", "dark"))
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    from open_dream_rsi.dashboard import run_dashboard

    run_dashboard(host=args.host, port=args.port, provider=args.provider,
                  model=args.model, tasks_file=args.tasks, interval=args.interval,
                  budget=args.budget, memory_root=args.memory, max_tokens=args.max_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
