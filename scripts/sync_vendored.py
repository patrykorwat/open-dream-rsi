#!/usr/bin/env python3
"""Mirror open_dream_rsi/ into the vendored npm bundle copy — the invariant
tests/test_vendored_drift.py (CI job "Lint + vendored-copy drift") enforces.

    python scripts/sync_vendored.py           # copy changed, prune removed, verify
    python scripts/sync_vendored.py --check   # verify only, writes nothing

The vendored copy CANNOT be gitignored instead: plugins/deepseek_harness
ships it (npm installs the plugin without a pip step), so every clone must
carry it byte-identical. After syncing run the full suite
(`python -m unittest discover -s tests`) and `scripts/check_versions.py`
before pushing — the drift test alone does not cover the rest.
"""

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "open_dream_rsi"
DST = REPO / "plugins" / "deepseek_harness" / "open_dream_rsi"
EXCLUDED_DIRS = {"__pycache__"}


def tree(root: Path) -> dict:
    """Relative path -> absolute path for every non-excluded file."""
    if not root.exists():
        return {}
    return {
        str(p.relative_to(root)): p
        for p in root.rglob("*")
        if p.is_file() and not (EXCLUDED_DIRS & set(p.parts))
    }


def drift(src: dict, dst: dict) -> tuple:
    """(missing/stale upstream files, extras only in the vendored copy)."""
    copied = sorted(rel for rel, p in src.items()
                    if rel not in dst or dst[rel].read_bytes() != p.read_bytes())
    pruned = sorted(rel for rel in dst if rel not in src)
    return copied, pruned


def _prune_empty_dirs(start: Path) -> None:
    parent = start.parent
    while parent != DST and parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()
        parent = parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sync open_dream_rsi/ into plugins/deepseek_harness/open_dream_rsi/.")
    ap.add_argument("--check", action="store_true",
                    help="verify only: report drift, write nothing")
    args = ap.parse_args()

    src, dst = tree(SRC), tree(DST)
    to_copy, to_prune = drift(src, dst)
    if args.check:
        for rel in to_copy:
            print(f"DRIFT missing/stale: {rel}")
        for rel in to_prune:
            print(f"DRIFT extra in vendored copy: {rel}")
        if to_copy or to_prune:
            print(f"--check FAILED: {len(to_copy)} stale, {len(to_prune)} extra")
            return 1
        print(f"--check OK: {len(src)} files byte-identical")
        return 0

    for rel in to_copy:
        target = DST / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src[rel], target)
        print(f"copied  {rel}")
    for rel in to_prune:
        target = dst[rel]
        target.unlink()
        _prune_empty_dirs(target)
        print(f"pruned  {rel}")
    if not to_copy and not to_prune:
        print(f"nothing to do ({len(src)} files already byte-identical)")
        return 0

    # Re-verify after writing: the exit code is the proof, not the intent.
    src, dst = tree(SRC), tree(DST)
    leftover_copy, leftover_prune = drift(src, dst)
    if leftover_copy or leftover_prune:
        print("FAIL: residual drift after sync")
        return 1
    print(f"mirror OK: {len(src)} files byte-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
