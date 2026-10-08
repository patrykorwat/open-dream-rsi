"""Git-backed lesson stores: publish and consume curated knowledge (issue #3
follow-up: sharing).

A *store* is a directory of portable lesson files — optionally a git
checkout, so it can live on GitHub and be consumed by any other instance:

    .dream_rsi/lessons.json          (private, runtime view)
        |  export: active lessons only, secrets redacted, checksummed manifest
        v
    <store>/manifest.json            (schema odr-lessons/v1 + sha256 per file)
    <store>/lessons/<category>.json  (portable records + provenance)
        ^  import: structural gate, lands as STAGING — never ACTIVE
        |
    any other .dream_rsi instance

Trust model, identical to the session import: sharing knowledge, not
authority. Export ships only lessons that EARNED activation (paired-replay
gate); import re-validates structurally, registers CANDIDATE -> VALIDATED
in the ALM, and leaves activation to the *local* promotion gate. A checksum
mismatch against the manifest aborts the import — a tampered store is
detected, not trusted.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from open_dream_rsi.sessions import redact_secrets

SCHEMA = "odr-lessons/v1"
MAX_EVIDENCE_PER_SHARED_LESSON = 5

_URL_RE = re.compile(r"^(https?://|ssh://|git://|git@)")


class LessonShareError(RuntimeError):
    """Store resolution, git or integrity failure."""


@dataclass
class StoreStatus:
    ok: bool
    location: str
    git: bool
    categories: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "location": self.location, "git": self.git,
                "categories": self.categories, "errors": self.errors}


def _slug(category: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(category).lower()).strip("-") \
        or "uncategorized"


def _default_branch(cwd: Path) -> str:
    """Resolve the remote's default branch name (robust to dangling HEADs)."""
    try:
        _git(cwd, "remote", "set-head", "origin", "-a")
        ref = _git(cwd, "rev-parse", "--abbrev-ref", "origin/HEAD")
        return ref.split("/", 1)[1] if "/" in ref else "main"
    except LessonShareError:
        return "main"


def _pull(cwd: Path) -> None:
    """Best-effort fast-forward sync. Never fatal: an empty/new remote, a
    misconfigured tracking branch or offline use must not abort an export —
    the local commit (and later push) still records the truth."""
    try:
        _git(cwd, "pull", "--ff-only")
    except LessonShareError:
        try:
            _git(cwd, "fetch", "origin")
            _git(cwd, "merge", "--ff-only", "FETCH_HEAD")
        except LessonShareError:
            pass


def _git(cwd: Path, *args: str, check: bool = True) -> str:
    cmd = ["git", "-C", str(cwd), *args]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except FileNotFoundError as exc:
        raise LessonShareError("git executable not found — stores need git") from exc
    if check and r.returncode != 0:
        raise LessonShareError(
            f"git {' '.join(args)} failed: {r.stderr.strip()[:300]}")
    return r.stdout.strip()


def _memory_root(memory_or_root: Any) -> Path:
    if isinstance(memory_or_root, (str, Path)):
        return Path(memory_or_root)
    return Path(memory_or_root.root)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_git_store(path: Path) -> bool:
    try:
        _git(path, "rev-parse", "--git-dir", check=False)
        return (path / ".git").exists() or _git(path, "rev-parse", "--git-dir") != ""
    except LessonShareError:
        return False


def resolve_store(store: "str | Path", *, cache_root: Optional[Path] = None,
                  pull: bool = True, create: bool = False) -> Path:
    """Resolve a store reference to a local directory.

    A path is used in place (git-managed or not); a URL is cloned into
    ``<cache_root>/lesson_stores/<name>`` and pulled on use. With
    ``create=True`` a missing local path is created (export semantics).
    """
    s = str(store)
    if _URL_RE.match(s) or s.endswith(".git"):
        base = Path(cache_root) if cache_root else Path.cwd()
        leaf = s.rstrip("/").split("/")[-1]
        name = re.sub(r"[^A-Za-z0-9._-]+", "_",
                      leaf[:-4] if leaf.endswith(".git") else leaf) or "store"
        dest = base / "lesson_stores" / name
        if dest.exists():
            if pull:
                _pull(dest)
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        _git(dest.parent, "clone", s, dest.name)
        return dest
    p = Path(store)
    if not p.exists():
        if create:
            p.mkdir(parents=True, exist_ok=True)
            return p
        raise LessonShareError(f"store not found: {store}")
    if pull and is_git_store(p):
        _pull(p)
    return p


# -- export ---------------------------------------------------------------------

def _portable(entry: Dict[str, Any]) -> Dict[str, Any]:
    """One shared lesson record: contract fields + provenance, redacted.
    Internal bookkeeping (keys, digests, status) never leaves the instance —
    consumers re-derive their own."""
    return {
        "trigger": str(entry.get("trigger", "")),
        "text": redact_secrets(str(entry.get("text", ""))),
        "evidence": [redact_secrets(str(e)) for e in
                     (entry.get("evidence") or [])][:MAX_EVIDENCE_PER_SHARED_LESSON],
        "wins": int(entry.get("wins", 0)),
        "uses": int(entry.get("uses", 0)),
        "provenance": {"source": "dream_rsi_curator",
                       "promoted_by": "paired_replay_gate",
                       "promoted_at": float(entry.get("updated_at", 0.0))},
    }


def export_lessons(memory_or_root: Any, store: "str | Path", *,
                   categories: Optional[List[str]] = None,
                   message: Optional[str] = None,
                   push: bool = True) -> StoreStatus:
    """Write ACTIVE lessons into ``<store>/lessons/<category>.json`` plus a
    checksummed manifest. Git stores: pull first, commit (and push unless
    ``push=False``). Secrets redacted. Staging/rejected entries are never
    shipped — unearned knowledge stays home."""
    root = _memory_root(memory_or_root)
    kb = _load_json(root / "lessons.json", {})
    store_dir = resolve_store(store, cache_root=root, create=True)
    git = is_git_store(store_dir)
    (store_dir / "lessons").mkdir(parents=True, exist_ok=True)
    manifest_path = store_dir / "manifest.json"
    manifest = _load_json(manifest_path, {"schema": SCHEMA, "categories": {}})
    if not isinstance(manifest, dict):
        manifest = {"schema": SCHEMA, "categories": {}}
    manifest.setdefault("categories", {})
    written: Dict[str, int] = {}
    for category, entries in sorted(kb.items()):
        if categories and category not in categories:
            continue
        keep = [l for l in entries
                if str(l.get("status", "active")) == "active"]
        path = store_dir / "lessons" / f"{_slug(category)}.json"
        payload = {"schema": SCHEMA, "category": category,
                   "lessons": [_portable(l) for l in keep]}
        _atomic(path, json.dumps(payload, indent=2, sort_keys=True,
                                 ensure_ascii=False))
        manifest["categories"][category] = {
            "file": f"lessons/{_slug(category)}.json",
            "sha256": _sha(path), "lessons": len(keep)}
        written[category] = len(keep)
    manifest["schema"] = SCHEMA
    manifest["updated_at"] = time.time()
    manifest["generator"] = "open-dream-rsi"
    _atomic(manifest_path, json.dumps(manifest, indent=2, sort_keys=True,
                                      ensure_ascii=False))
    if git:
        _git(store_dir, "add", "-A")
        if _git(store_dir, "status", "--porcelain"):
            name = os.environ.get("ODR_GIT_AUTHOR_NAME", "open-dream-rsi")
            email = os.environ.get("ODR_GIT_AUTHOR_EMAIL", "odr@users.noreply.github.com")
            _git(store_dir, "-c", f"user.name={name}", "-c", f"user.email={email}",
                 "commit", "-m", message or
                 f"odr lessons export: {sum(written.values())} lesson(s) "
                 f"in {len(written)} category(ies)")
            if push:
                _git(store_dir, "push")
    return StoreStatus(ok=True, location=str(store_dir), git=git,
                       categories=written)


# -- import ---------------------------------------------------------------------

def import_lessons(store: "str | Path", memory_or_root: Any, *,
                   categories: Optional[List[str]] = None) -> Dict[str, int]:
    """Merge lessons from a store into the local KB as STAGING.

    Nothing is trusted: manifest checksums are verified (mismatch aborts),
    every lesson passes the structural gate, and the ALM records
    CANDIDATE -> VALIDATED — activation still requires the local
    paired-replay gate. Returns per-category import counts."""
    from open_dream_rsi.core.curator import (curate_lessons,
                                             validate_lesson_items)
    from open_dream_rsi.lifecycle import (ArtifactLifecycleError,
                                          ArtifactLifecycleManager)

    root = _memory_root(memory_or_root)
    store_dir = resolve_store(store, cache_root=root)
    manifest = _load_json(store_dir / "manifest.json", None)
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise LessonShareError(
            f"missing or unsupported manifest.json (want schema {SCHEMA})")
    alm = ArtifactLifecycleManager(root)
    imported: Dict[str, int] = {}
    for category, meta in sorted(manifest.get("categories", {}).items()):
        if categories and category not in categories:
            continue
        rel = str(meta.get("file", ""))
        path = store_dir / rel
        if not path.exists():
            continue  # removed upstream: keep the local copy, delete nothing
        if _sha(path) != meta.get("sha256"):
            raise LessonShareError(
                f"checksum mismatch for {rel} — the store may be tampered; "
                "import aborted")
        payload = _load_json(path, {})
        items, evidence = [], []
        for l in payload.get("lessons", []):
            if not isinstance(l, dict):
                continue
            try:
                items.extend(validate_lesson_items(
                    [{"trigger": l.get("trigger", ""),
                      "text": l.get("text", "")}], 1))
                evidence.extend(str(e) for e in (l.get("evidence") or []))
            except Exception:
                continue  # structural failure: skip this lesson, keep the rest
        if not items:
            continue
        result = curate_lessons(_local_lessons(root, category), items,
                                evidence=evidence[:12], staging=True)
        _write_category(root, category, result.entries)
        for entry in result.added:
            key = str(entry.get("trigger", ""))
            try:
                from open_dream_rsi.core.curator import lesson_key
                key = lesson_key(entry)
                cand = alm.register_candidate(
                    "lesson",
                    {k: entry.get(k) for k in ("trigger", "text", "evidence")
                     if entry.get(k)},
                    slot=f"lesson:{category}:{key}",
                    created_by=f"lesson_store:{store_dir.name}")
                alm.validate(cand.artifact_id, actor="alm")
            except ArtifactLifecycleError:
                pass  # already live under this identity — merge was enough
        imported[category] = len(items)
    return imported


def _local_lessons(root: Path, category: str) -> List[Dict[str, Any]]:
    data = _load_json(root / "lessons.json", {})
    return [dict(l) for l in data.get(category, [])]


def _write_category(root: Path, category: str,
                    entries: List[Dict[str, Any]]) -> None:
    data = _load_json(root / "lessons.json", {})
    if entries:
        data[category] = entries
    else:
        data.pop(category, None)
    _atomic(root / "lessons.json", json.dumps(data, indent=2,
                                              ensure_ascii=False))


# -- status -----------------------------------------------------------------------

def store_status(store: "str | Path", *, cache_root: Optional[Path] = None) -> StoreStatus:
    """What a store currently holds (no writes)."""
    try:
        store_dir = resolve_store(store, cache_root=cache_root)
    except LessonShareError as exc:
        return StoreStatus(ok=False, location=str(store), git=False,
                           errors=[str(exc)])
    manifest = _load_json(store_dir / "manifest.json", None)
    errors: List[str] = []
    cats: Dict[str, int] = {}
    if isinstance(manifest, dict) and manifest.get("schema") == SCHEMA:
        for category, meta in sorted(manifest.get("categories", {}).items()):
            path = store_dir / str(meta.get("file", ""))
            if not path.exists():
                errors.append(f"{category}: file missing ({meta.get('file')})")
                continue
            if _sha(path) != meta.get("sha256"):
                errors.append(f"{category}: checksum MISMATCH")
            cats[category] = int(meta.get("lessons", 0))
    else:
        errors.append("no valid manifest.json")
    return StoreStatus(ok=not errors, location=str(store_dir),
                       git=is_git_store(store_dir), categories=cats,
                       errors=errors)
