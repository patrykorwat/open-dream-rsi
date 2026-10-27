"""Fail when the release version drifts across the files that carry it.

A root VERSION file is the single source of truth (the basal-skills model):
every artifact that carries a version — pyproject.toml, the package
__init__, the Cursor plugin + marketplace manifests, the npm bundle
manifest, the Claude Code plugin manifest — must equal it, and authored
content changed since the last release tag requires a VERSION bump. Run
before any release (the publish workflow enforces it), after any version
bump, and in CI.
"""

import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Docs and infra don't require a VERSION bump; everything shipped does.
DOCS_ONLY = {
    "VERSION", "CHANGELOG.md", "README.md", "CONTRIBUTING.md",
    "LICENSE", "SECURITY.md", ".gitignore",
}
EXEMPT_PREFIXES = (".github/", "docs/", "paper/")


def _first(pattern: str, text: str, label: str) -> str:
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        raise SystemExit(f"FAIL: no version found in {label}")
    return m.group(1)


def _last_tag() -> str | None:
    """Newest tag reachable from HEAD, or None when tags are unavailable
    (shallow CI checkouts)."""
    run = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"],
        capture_output=True, text=True, cwd=REPO)
    return run.stdout.strip() if run.returncode == 0 else None


def collect() -> dict:
    versions = {"VERSION": (REPO / "VERSION").read_text().strip()}
    versions["pyproject.toml"] = _first(
        r'^version\s*=\s*"([^"]+)"',
        (REPO / "pyproject.toml").read_text(encoding="utf-8"),
        "pyproject.toml")
    versions["open_dream_rsi/__init__.py"] = _first(
        r'^__version__\s*=\s*"([^"]+)"',
        (REPO / "open_dream_rsi" / "__init__.py").read_text(encoding="utf-8"),
        "open_dream_rsi/__init__.py")
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


def bump_missing(versions: dict) -> str | None:
    """Why the bump rule fails, or None when it is satisfied or not
    checkable (no tags, empty diff, tag predating the VERSION file)."""
    tag = _last_tag()
    if tag is None:
        return None
    diff = subprocess.run(
        ["git", "diff", "--name-only", f"{tag}..HEAD"],
        capture_output=True, text=True, cwd=REPO)
    if diff.returncode != 0 or not diff.stdout.strip():
        return None
    changed = [f for f in diff.stdout.splitlines() if f.strip()]
    authored = [f for f in changed
                if f not in DOCS_ONLY and not f.startswith(EXEMPT_PREFIXES)]
    if not authored:
        return None
    old = subprocess.run(
        ["git", "show", f"{tag}:VERSION"],
        capture_output=True, text=True, cwd=REPO)
    if old.returncode != 0:
        return None  # tag predates the VERSION file
    if old.stdout.strip() == versions["VERSION"]:
        return (f"authored content changed since {tag} but VERSION is still "
                f"{versions['VERSION']} — bump it; changed: {authored[:5]}")
    return None


def main() -> int:
    versions = collect()
    expected = versions["VERSION"]
    drift = {k: v for k, v in versions.items()
             if k != "VERSION" and v != expected}
    print(f"expected version (VERSION file): {expected}")
    for name, ver in versions.items():
        mark = ("src" if name == "VERSION"
                else "OK " if ver == expected else "DRIFT")
        print(f"  {mark} {name}: {ver}")
    if drift:
        print(f"FAIL: {len(drift)} artifact(s) drifted from VERSION "
              f"{expected}: " + ", ".join(sorted(drift)))
        return 1
    problem = bump_missing(versions)
    if problem:
        print(f"FAIL: {problem}")
        return 1
    print("version sync OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
