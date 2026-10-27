"""Fail when the release version drifts across the files that carry it.

The version lives in six places (pyproject.toml, package __init__, the
Cursor plugin + marketplace manifests, the npm bundle manifest, the Claude
Code plugin manifest); a version
left behind ships stale vendored code under an old number. Run before any
release (the publish workflow enforces it) and after any version bump.
"""

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _first(pattern: str, text: str, label: str) -> str:
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        raise SystemExit(f"FAIL: no version found in {label}")
    return m.group(1)


def collect() -> dict:
    versions = {
        "pyproject.toml": _first(
            r'^version\s*=\s*"([^"]+)"',
            (REPO / "pyproject.toml").read_text(encoding="utf-8"),
            "pyproject.toml"),
        "open_dream_rsi/__init__.py": _first(
            r'^__version__\s*=\s*"([^"]+)"',
            (REPO / "open_dream_rsi" / "__init__.py").read_text(encoding="utf-8"),
            "open_dream_rsi/__init__.py"),
    }
    plugin = json.loads((REPO / ".cursor-plugin" / "plugin.json").read_text())
    versions[".cursor-plugin/plugin.json"] = plugin["version"]
    market = json.loads((REPO / ".cursor-plugin" / "marketplace.json").read_text())
    entries = [("metadata", market["metadata"]["version"])]
    entries += [(f"plugins[{i}]", p["version"])
                for i, p in enumerate(market.get("plugins", []))]
    for label, ver in entries:
        versions[f".cursor-plugin/marketplace.json:{label}"] = ver
    versions["plugins/deepseek_harness/package.json"] = json.loads(
        (REPO / "plugins" / "deepseek_harness" / "package.json").read_text()
    )["version"]
    claude = REPO / ".claude-plugin" / "plugin.json"
    if claude.is_file():
        versions[".claude-plugin/plugin.json"] = json.loads(
            claude.read_text(encoding="utf-8"))["version"]
    return versions


def main() -> int:
    versions = collect()
    expected = versions["pyproject.toml"]
    drift = {k: v for k, v in versions.items() if v != expected}
    print(f"expected version: {expected}")
    for name, ver in versions.items():
        mark = "OK " if ver == expected else "DRIFT"
        print(f"  {mark} {name}: {ver}")
    if drift:
        print(f"FAIL: {len(drift)} file(s) drifted from {expected}: "
              + ", ".join(sorted(drift)))
        return 1
    print("version sync OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
