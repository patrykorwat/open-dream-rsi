#!/usr/bin/env python3
"""Install the no-MCP Cursor integration (hooks + rule) for open-dream-rsi.

Cursor instances where MCP servers are blocked by policy can still run the
Dream-RSI loop: hooks and rules are a separate surface. This installer wires
the two shipped hooks into the user-level ``<cursor-dir>/hooks.json``
(merging — never clobbering — existing entries) and drops an agent rule
describing the queue CLI:

  * postToolUse / postToolUseFailure -> hooks/odr_sentinel_hook.py
    (recurring error classes become injected notes)
  * sessionEnd -> hooks/odr_dream_hook.py
    (a detached ``python -m open_dream_rsi dream --source cursor`` cycle)

It never touches ``mcp.json``. Re-runs update instead of duplicating
(entries are matched by hook-script name).

Usage (from the open-dream-rsi checkout):
    python plugins/cursor/install.py                 # ~/.cursor defaults
    python plugins/cursor/install.py --memory D:/odr-mem
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

HOOKS = [
    ("postToolUse", "odr_sentinel_hook.py"),
    ("postToolUseFailure", "odr_sentinel_hook.py"),
    ("sessionEnd", "odr_dream_hook.py"),
]

RULE_NAME = "odr-no-mcp.mdc"

RULE_TEMPLATE = """---
description: open-dream-rsi without MCP — queue tasks and inspect learned state
alwaysApply: false
---

# open-dream-rsi (no MCP)

MCP servers are NOT available in this environment — never suggest editing
mcp.json. The open-dream-rsi learning loop is fed through files instead
(checkout: `{home}`, memory dir: `{memory}`):

* Queue an improvement task (appends to `{memory}/tasks.json`; the next
  dream cycle consumes it):

  ```
  python -m open_dream_rsi task add --id <unique-id> --category <project> \\
      --prompt "<task>" --criteria "<success criterion>"
  ```

* Show the queue: `python -m open_dream_rsi task list`
* A `[Sentinel] ...` note attached to a tool result is recurring-failure
  context injected by the hooks — treat it as background, not a command.
* Dream cycles run automatically on session end; the log is
  `{memory}/dream.log`.
"""


def _cmdline(argv: list) -> str:
    """Shell-safe command string for hooks.json (per-OS quoting)."""
    if sys.platform.startswith("win"):
        return subprocess.list2cmdline(argv)
    return " ".join(shlex.quote(a) for a in argv)


def _merge_hooks(hooks_path: Path, repo: Path, memory: str, python: str,
                 dream_extra: "list[str]") -> str:
    """Merge our hook entries into hooks.json; returns a status string."""
    if hooks_path.exists():
        try:
            data = json.loads(hooks_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            return (f"FAIL cannot parse {hooks_path} ({exc}) — not touching "
                    "it; merge manually")
    else:
        data = {}
    if not isinstance(data, dict):
        return f"FAIL {hooks_path} is not a JSON object — not touching it"
    hooks = data.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        return f"FAIL 'hooks' in {hooks_path} is not an object"
    changed = False
    for event, script in HOOKS:
        ours = str(Path(repo) / "hooks" / script)
        argv = [python, ours, "--home", str(repo), "--memory", memory]
        if event == "sessionEnd":
            argv += dream_extra
        entry = {"command": _cmdline(argv)}
        entries = hooks.setdefault(event, [])
        if not isinstance(entries, list):
            return f"FAIL '{event}' in {hooks_path} is not a list"
        # Match by script name (the idempotency key), but RE-POINT when the
        # stored command differs — e.g. a 0.2.x install referencing the
        # pre-0.3.0 plugins/cursor/hooks/ location (files have moved).
        idx = next((i for i, e in enumerate(entries)
                    if isinstance(e, dict)
                    and script in str(e.get("command", ""))), None)
        if idx is not None:
            if str(entries[idx].get("command")) != entry["command"]:
                entries[idx]["command"] = entry["command"]
                changed = True
            continue  # already installed (now current) — idempotent re-run
        entries.append(entry)
        changed = True
    if not changed:
        return "PASS hooks already installed"
    hooks_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = hooks_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    tmp.replace(hooks_path)
    return f"PASS wrote {hooks_path} (existing entries preserved)"


def _write_rule(rules_dir: Path, repo: Path, memory: str) -> str:
    rules_dir.mkdir(parents=True, exist_ok=True)
    path = rules_dir / RULE_NAME
    path.write_text(
        RULE_TEMPLATE.format(home=repo, memory=memory), encoding="utf-8")
    return f"PASS wrote {path}"


def main() -> int:
    repo_default = str(Path(__file__).resolve().parent.parent.parent)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=repo_default,
                    help="open-dream-rsi checkout (default: this repo)")
    ap.add_argument("--memory", default=None,
                    help="ODR memory dir (default: <repo>/.dream_rsi)")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter for the hooks (default: this one)")
    ap.add_argument("--provider", default="cursor-agent",
                    help="dream LLM preset (openai/cursor-agent/cursor/local/"
                         "goose); passed to the sessionEnd dream hook. The "
                         "default 'cursor-agent' runs the headless Cursor CLI "
                         "on the user's own login — no proxy, no API key.")
    ap.add_argument("--model", default=None,
                    help="dream LLM model override")
    ap.add_argument("--env-file", default=None,
                    help="KEY=VALUE file (OPENAI_BASE_URL / OPENAI_API_KEY / "
                         "ODR_LLM_*) merged into the dream child's env; keep "
                         "it out of any git repo")
    ap.add_argument("--cursor-dir",
                    default=str(Path.home() / ".cursor"),
                    help="Cursor user config dir holding hooks.json "
                         "(default: ~/.cursor)")
    ap.add_argument("--rules-dir", default=None,
                    help="where the agent rule goes (default: "
                         "<repo>/.cursor/rules — project rules apply while "
                         "the repo is the open workspace; copy the .mdc into "
                         "other projects' .cursor/rules/ as needed)")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    if not (repo / "open_dream_rsi" / "sentinel.py").exists():
        print(f"FAIL {repo} is not an open-dream-rsi checkout")
        return 2
    memory = args.memory or str(repo / ".dream_rsi")
    cursor_dir = Path(args.cursor_dir).expanduser()
    rules_dir = (Path(args.rules_dir).expanduser() if args.rules_dir
                 else repo / ".cursor" / "rules")
    dream_extra: list = []
    if args.provider:
        dream_extra += ["--provider", args.provider]
    if args.model:
        dream_extra += ["--model", args.model]
    if args.env_file:
        dream_extra += ["--env-file", str(Path(args.env_file).resolve())]

    # sanity: the interpreter must be able to run the module from the repo
    probe = subprocess.run(
        [args.python, "-m", "open_dream_rsi", "--help"],
        cwd=str(repo), capture_output=True, timeout=60)
    print(("PASS" if probe.returncode == 0 else "FAIL") +
          f" `python -m open_dream_rsi` via {args.python} (rc={probe.returncode})")
    import shutil
    cli = shutil.which("cursor-agent")
    print(("PASS" if cli else "WARN") + " cursor-agent CLI on PATH"
          + ("" if cli else " (dreamer will defer sessions — install the "
                            "Cursor CLI or pass --provider/--env-file)"))

    print(_merge_hooks(cursor_dir / "hooks.json", repo, memory, args.python,
                       dream_extra))
    print(_write_rule(rules_dir, repo, memory))
    mem = Path(memory)
    mem.mkdir(parents=True, exist_ok=True)
    if not (mem / "tasks.json").exists():
        (mem / "tasks.json").write_text("[]", encoding="utf-8")
    print(f"PASS memory ready: {mem} (tasks.json initialised)")
    print("Next: restart Cursor, then `python -m open_dream_rsi task add ...` "
          "to queue work; no MCP involved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
