#!/usr/bin/env python3
"""Configure the open-dream-rsi extension inside goose's config.yaml.

Idempotent editor: locates the ``extensions:`` mapping in
``~/.config/goose/config.yaml``, removes any existing
``open-dream-rsi`` entry (including stale inline comments) and injects the
canonical block pointing the dreamer at the model-borrowing proxy. The file
is backed up next to itself before any change, rewritten only when the
result differs, and verified by reading it back.

This is deliberately a *targeted block editor*, not a YAML round-tripper:
goose rewrites this file itself, so we preserve every foreign byte and only
touch our own extension entry.

Usage (normally invoked by scripts/odr_goose_setup.sh)::

    python3 -m open_dream_rsi.utils.goose_config            # write proxy mode
    python3 -m open_dream_rsi.utils.goose_config --check     # dry run
    python3 -m open_dream_rsi.utils.goose_config --no-proxy  # direct provider
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

EXT_NAME = "open-dream-rsi"
PROXY_TOKEN = "proxy-internal"  # dummy: proxy authenticates upstream itself

DESCRIPTION = (
    "Dream-RSI self-improvement loop. Queue Python tasks with odr_add_task, "
    "run an improvement cycle with odr_run_once, reuse verified solutions via "
    "odr_recipes/odr_lessons, inspect learned state with odr_status."
)


def _extension_block(py: str, tasks: str, memory: str, proxy_url: Optional[str]) -> List[str]:
    """Canonical YAML lines (2-space extension name, 4-space keys)."""
    lines = [
        f"  {EXT_NAME}:",
        "    enabled: true",
        "    type: stdio",
        f"    name: {EXT_NAME}",
        f"    description: \"{DESCRIPTION}\"",
        f"    cmd: {py}",
        "    args: [\"-m\", \"open_dream_rsi\", \"mcp\",",
        f"           \"--tasks\", \"{tasks}\",",
        f"           \"--memory\", \"{memory}\"]",
    ]
    if proxy_url:
        lines += [
            "    envs:",
            f"      OPENAI_BASE_URL: \"{proxy_url}\"",
            f"      OPENAI_API_KEY: \"{PROXY_TOKEN}\"",
        ]
    lines.append("    timeout: 300")
    return lines


def _find_extensions_range(lines: List[str]) -> Optional[tuple]:
    """(start_idx, end_idx_exclusive) of the extensions: mapping block."""
    start = None
    for i, line in enumerate(lines):
        if line.rstrip() in ("extensions:", "extensions: {}"):
            start = i
            break
    if start is None or lines[start].rstrip() == "extensions: {}":
        return None if start is None else (start, start + 1)
    end = len(lines)
    for i in range(start + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not (lines[i][0] in " \t"):
            end = i
            break
    return (start, end)


def _find_entry_range(lines: List[str], ext_start: int, ext_end: int) -> Optional[tuple]:
    """Range of our extension's entry inside the extensions block."""
    target = f"  {EXT_NAME}:"
    start = None
    for i in range(ext_start + 1, ext_end):
        if lines[i].rstrip() == target:
            start = i
            break
    if start is None:
        return None
    end = ext_end
    for i in range(start + 1, ext_end):
        if lines[i][:1] not in (" ", "\t") and lines[i].strip():
            end = i
            break
        if lines[i].startswith("  ") and not lines[i].startswith("   "):
            # next sibling extension at the same indent
            end = i
            break
    # trim trailing blank lines back to the previous entry
    while end - 1 > start and not lines[end - 1].strip():
        end -= 1
    return (start, end)


def configure(config_path: Path, py: str, repo_dir: Path,
              proxy_url: Optional[str], check: bool = False) -> int:
    """Write/refresh our extension entry. Returns process exit code."""
    report: Dict[str, object] = {"config": str(config_path)}
    if not config_path.exists():
        print(json.dumps({"error": f"not found: {config_path} — configure a "
                                   "provider in goose first"}))
        return 1

    original = config_path.read_text(encoding="utf-8")
    lines = original.splitlines()

    block = _extension_block(py, str(repo_dir / "tasks.json"),
                             str(repo_dir / ".dream_rsi"), proxy_url)

    ext_range = _find_extensions_range(lines)
    if ext_range is None:
        lines = ["extensions:"] + lines  # prepend the mapping
        ext_range = (0, 1)

    entry_range = _find_entry_range(lines, *ext_range)
    if entry_range:
        new_lines = lines[:entry_range[0]] + block + lines[entry_range[1]:]
        action = "replaced"
    else:
        insert_at = ext_range[1]
        # keep a blank line before the following top-level key, as goose does
        tail = lines[insert_at:]
        new_lines = lines[:insert_at] + block
        if tail and tail[0].strip():
            new_lines.append("")
        new_lines += tail
        action = "added"

    new_text = "\n".join(new_lines) + ("\n" if original.endswith("\n") else "")
    report.update({"action": action, "changed": new_text != original})

    # read-back verification data
    report["has_proxy_env"] = ('OPENAI_BASE_URL: "http://127.0.0.1:8799/v1"' in new_text
                               if proxy_url else True)

    if check:
        report["action"] = "check-only"
        print(json.dumps(report, indent=2))
        return 0

    if new_text == original:
        report["action"] = "unchanged"
        print(json.dumps(report, indent=2))
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = config_path.with_name(config_path.name + f".bak-{stamp}")
    shutil.copy2(config_path, backup)
    report["backup"] = str(backup)
    config_path.write_text(new_text, encoding="utf-8")

    # verify on disk (never trust the write call alone)
    on_disk = config_path.read_text(encoding="utf-8")
    ok = (f"  {EXT_NAME}:" in on_disk) and (
        (f'OPENAI_BASE_URL: "{proxy_url}"' in on_disk) if proxy_url else True)
    report["verified"] = ok
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="odr-goose-config")
    parser.add_argument("--config", default=str(Path.home() / ".config/goose/config.yaml"))
    parser.add_argument("--repo-dir", default=str(Path.cwd()))
    parser.add_argument("--py", default=sys.executable)
    parser.add_argument("--proxy-url", default="http://127.0.0.1:8799/v1")
    parser.add_argument("--no-proxy", action="store_true",
                        help="omit envs (use odr_run_once provider='goose' instead)")
    parser.add_argument("--check", action="store_true", help="dry run")
    args = parser.parse_args(argv)
    return configure(Path(args.config).expanduser(), args.py, Path(args.repo_dir).expanduser(),
                     None if args.no_proxy else args.proxy_url, check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
