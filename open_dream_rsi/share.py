"""Git-backed stores: share recorded worlds and policy programs.

A *store* is a directory of portable artifact files — optionally a git
checkout, so it can live on GitHub and be consumed by any other instance:

    <store>/manifest.json            (schema manifest + sha256 per file)
    <store>/trees/<task>.json        (recorded discovery trees)
    <store>/policies/<category>.json (policy programs)

Trust model: sharing evidence and candidates, not authority. Imported
worlds land under ``trees/imported/`` (evidence, never live tasks) and
imported policy programs register as CANDIDATE -> VALIDATED in the ALM —
activation stays with the *local* replay gate. A checksum mismatch
against the manifest aborts the import — a tampered store is detected,
not trusted.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

from open_dream_rsi.memory import DreamMemory


SCHEMA = "odr-lessons/v1"  # store manifest format (name is historical)
_URL_RE = re.compile(r"^(https?://|ssh://|git://|git@)")


_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),                     # OpenAI-style
    re.compile(r"ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}"),  # GitHub tokens
    re.compile(r"AKIA[0-9A-Z]{16}"),                           # AWS
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),              # Slack
    re.compile(r"(?i)\b(password|passwd|secret|token|api[_-]?key|access[_-]?key)"
               r"\b(\s*[=:]\s*)\S+"),                           # key: value pairs
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"),         # bearer headers
    re.compile(r"[A-Za-z0-9+/_\-]{40,}={0,2}"),                # long opaque blobs
]


def redact_secrets(text: str) -> str:
    """Best-effort scrub of credential-shaped substrings. Deliberately
    over-eager: a false positive costs one garbled word in an evidence
    snippet, a false negative writes a live token to disk."""
    out = text or ""
    for pat in _SECRET_PATTERNS:
        out = pat.sub("[REDACTED]", out)
    return out



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



MAX_SHARED_STRING_CHARS = 4000
TREES_DIR = "trees"
POLICIES_DIR = "policies"
IMPORTED_TREES_SUBDIR = "imported"


def _redact_deep(value: Any, depth: int = 0) -> Any:
    """Recursively secret-redact and length-cap a JSON-ish structure."""
    if depth > 6:
        return str(value)[:200]
    if isinstance(value, str):
        return redact_secrets(value)[:MAX_SHARED_STRING_CHARS]
    if isinstance(value, list):
        return [_redact_deep(v, depth + 1) for v in value[:50]]
    if isinstance(value, dict):
        return {str(k): _redact_deep(v, depth + 1)
                for k, v in list(value.items())[:50]}
    return value


def _world_file_name(task_id: str, taken: Dict[str, str]) -> str:
    """Slug + short-hash disambiguation: two task ids may slug identically."""
    slug = _slug(task_id)
    if slug not in taken:
        taken[slug] = task_id
        return f"{slug}.json"
    if taken[slug] == task_id:
        return f"{slug}.json"
    short = hashlib.sha1(task_id.encode("utf-8")).hexdigest()[:8]
    name = f"{slug}-{short}"
    taken[name] = task_id
    return f"{name}.json"


def _manifest(store_dir: Path) -> Dict[str, Any]:
    manifest = _load_json(store_dir / "manifest.json",
                          {"schema": SCHEMA, "categories": {}})
    if not isinstance(manifest, dict):
        manifest = {"schema": SCHEMA, "categories": {}}
    manifest.setdefault("categories", {})
    return manifest


def _write_manifest(store_dir: Path, manifest: Dict[str, Any]) -> None:
    manifest["schema"] = SCHEMA
    manifest["updated_at"] = time.time()
    manifest["generator"] = "open-dream-rsi"
    _atomic(store_dir / "manifest.json", json.dumps(
        manifest, indent=2, sort_keys=True, ensure_ascii=False))


def _commit_store(store_dir: Path, git: bool, message: str, push: bool) -> None:
    if not git:
        return
    _git(store_dir, "add", "-A")
    if _git(store_dir, "status", "--porcelain"):
        name = os.environ.get("ODR_GIT_AUTHOR_NAME", "open-dream-rsi")
        email = os.environ.get("ODR_GIT_AUTHOR_EMAIL", "odr@users.noreply.github.com")
        _git(store_dir, "-c", f"user.name={name}", "-c", f"user.email={email}",
             "commit", "-m", message)
        if push:
            _git(store_dir, "push")


def export_worlds(memory_or_root: Any, store: "str | Path", *,
                  task_ids: Optional[List[str]] = None,
                  message: Optional[str] = None,
                  push: bool = True) -> StoreStatus:
    """Ship archived discovery trees (``trees/*.json``) as bundle worlds.

    The tree JSON is secret-redacted and length-capped; the manifest gains a
    ``worlds`` section (sha256 per file). Recorded worlds are EVIDENCE — the
    recipient imports them into ``trees/imported/``, never over a local tree.
    """
    root = _memory_root(memory_or_root)
    store_dir = resolve_store(store, cache_root=root, create=True)
    git = is_git_store(store_dir)
    (store_dir / TREES_DIR).mkdir(parents=True, exist_ok=True)
    manifest = _manifest(store_dir)
    manifest.setdefault("worlds", {})
    written: Dict[str, int] = {}
    taken: Dict[str, str] = {}
    trees_dir = root / TREES_DIR
    paths = sorted(trees_dir.glob("*.json")) if trees_dir.exists() else []
    for path in paths:
        task_id = path.stem
        if task_ids and task_id not in task_ids:
            continue
        data = _load_json(path, None)
        if not isinstance(data, dict) or not isinstance(data.get("nodes"), list):
            continue
        payload = {"task_id": task_id, "tree": _redact_deep(data)}
        out = store_dir / TREES_DIR / _world_file_name(task_id, taken)
        _atomic(out, json.dumps(payload, indent=2, sort_keys=True,
                                ensure_ascii=False))
        manifest["worlds"][task_id] = {
            "file": f"{TREES_DIR}/{out.name}", "sha256": _sha(out),
            "nodes": len(data["nodes"])}
        written[task_id] = len(data["nodes"])
    _write_manifest(store_dir, manifest)
    _commit_store(store_dir, git,
                  message or f"odr worlds export: {len(written)} tree(s)",
                  push)
    return StoreStatus(ok=True, location=str(store_dir), git=git,
                       categories=written)


def import_worlds(store: "str | Path", memory_or_root: Any, *,
                  task_ids: Optional[List[str]] = None) -> Dict[str, int]:
    """Import bundle worlds as recorded evidence under ``trees/imported/``.

    Checksums are verified (mismatch aborts the whole import). Every world
    must round-trip through :class:`~open_dream_rsi.core.tree.DiscoveryTree`
    (unknown node fields are dropped, not trusted); imported trees are
    never wired into the runtime's ``load_tree`` path and never executed —
    they are historical evidence a replay harness can point at explicitly.
    """
    from open_dream_rsi.core.tree import TreeNode

    root = _memory_root(memory_or_root)
    store_dir = resolve_store(store, cache_root=root)
    manifest = _load_json(store_dir / "manifest.json", None)
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise LessonShareError(
            f"missing or unsupported manifest.json (want schema {SCHEMA})")
    imported: Dict[str, int] = {}
    allowed_fields = {f.name for f in fields(TreeNode)}
    for task_id, meta in sorted(manifest.get("worlds", {}).items()):
        if task_ids and task_id not in task_ids:
            continue
        rel = str(meta.get("file", ""))
        path = store_dir / rel
        if not path.exists():
            continue
        if _sha(path) != meta.get("sha256"):
            raise LessonShareError(
                f"checksum mismatch for {rel} — the store may be tampered; "
                "import aborted")
        payload = _load_json(path, {})
        tree_data = payload.get("tree") if isinstance(payload, dict) else None
        if not isinstance(tree_data, dict):
            continue
        nodes = []
        ok = True
        for n in tree_data.get("nodes", []):
            if not isinstance(n, dict) or "node_id" not in n:
                ok = False
                break
            nodes.append({k: n[k] for k in allowed_fields if k in n})
        if not ok or not nodes:
            continue
        out_dir = root / TREES_DIR / IMPORTED_TREES_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        _atomic(out_dir / f"{_slug(task_id)}.json", json.dumps(
            {"root_id": tree_data.get("root_id"), "nodes": nodes},
            indent=2, ensure_ascii=False))
        imported[task_id] = len(nodes)
    if imported:
        DreamMemory(root).log_event(
            "world_imported", count=sum(imported.values()),
            store=Path(store_dir).name, tasks=sorted(imported)[:20])
    return imported


def export_policies(memory_or_root: Any, store: "str | Path", *,
                    categories: Optional[List[str]] = None,
                    message: Optional[str] = None,
                    push: bool = True) -> StoreStatus:
    """Ship promoted exploration-policy programs (``policy_codes.json``).

    Only programs that still pass the structural gate are exported, after
    secret redaction (a redaction that breaks the program makes it fail the
    gate, so broken code is never shipped).
    """
    from open_dream_rsi.core.policygen import (PolicyValidationError,
                                               validate_policy_source)

    root = _memory_root(memory_or_root)
    codes = _load_json(root / "policy_codes.json", {})
    store_dir = resolve_store(store, cache_root=root, create=True)
    git = is_git_store(store_dir)
    (store_dir / POLICIES_DIR).mkdir(parents=True, exist_ok=True)
    manifest = _manifest(store_dir)
    manifest.setdefault("policies", {})
    written: Dict[str, int] = {}
    for category, entry in sorted(codes.items()):
        if categories and category not in categories:
            continue
        if not isinstance(entry, dict):
            continue
        code = redact_secrets(str(entry.get("code", "")))
        try:
            validate_policy_source(code)
        except PolicyValidationError:
            continue  # never ship a program the structural gate rejects
        payload = {"schema": SCHEMA, "category": category, "code": code,
                   "score": float(entry.get("score", 0.0)),
                   "steps": int(entry.get("steps", 0)),
                   "provenance": {"source": "dream_rsi_policygen",
                                  "promoted_by": "counterfactual_replay_gate",
                                  "promoted_at": float(entry.get("updated_at", 0.0))}}
        out = store_dir / POLICIES_DIR / f"{_slug(category)}.json"
        _atomic(out, json.dumps(payload, indent=2, sort_keys=True,
                                ensure_ascii=False))
        manifest["policies"][category] = {"file": f"{POLICIES_DIR}/{out.name}",
                                          "sha256": _sha(out)}
        written[category] = 1
    _write_manifest(store_dir, manifest)
    _commit_store(store_dir, git,
                  message or f"odr policy export: {len(written)} program(s)",
                  push)
    return StoreStatus(ok=True, location=str(store_dir), git=git,
                       categories=written)


def import_policies(store: "str | Path", memory_or_root: Any, *,
                    categories: Optional[List[str]] = None) -> Dict[str, int]:
    """Register bundle policy programs as ALM CANDIDATE -> VALIDATED.

    Nothing imported becomes the runtime incumbent: adoption is the local
    promotion machinery's job. A re-import of an identical program is a
    no-op (payload sha dedupe — no version churn).
    """
    from open_dream_rsi.core.policygen import (PolicyValidationError,
                                               validate_policy_source)
    from open_dream_rsi.lifecycle import (_payload_sha, ArtifactLifecycleError,
                                          ArtifactLifecycleManager)

    root = _memory_root(memory_or_root)
    store_dir = resolve_store(store, cache_root=root)
    manifest = _load_json(store_dir / "manifest.json", None)
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise LessonShareError(
            f"missing or unsupported manifest.json (want schema {SCHEMA})")
    alm = ArtifactLifecycleManager(DreamMemory(root))
    imported: Dict[str, int] = {}
    for category, meta in sorted(manifest.get("policies", {}).items()):
        if categories and category not in categories:
            continue
        rel = str(meta.get("file", ""))
        path = store_dir / rel
        if not path.exists():
            continue
        if _sha(path) != meta.get("sha256"):
            raise LessonShareError(
                f"checksum mismatch for {rel} — the store may be tampered; "
                "import aborted")
        payload = _load_json(path, {})
        code = str(payload.get("code", ""))
        try:
            validate_policy_source(code)
        except PolicyValidationError:
            continue
        record = {"code": code, "score": float(payload.get("score", 0.0)),
                  "steps": int(payload.get("steps", 0)), "category": category}
        slot = f"policy_program:{category}:imported"
        sha = _payload_sha(record)
        if any(s.payload_sha256 == sha for s in alm.states(slot=slot)):
            continue  # identical program already registered — no churn
        try:
            cand = alm.register_candidate("policy_program", record, slot=slot,
                                          created_by=f"store:{Path(store_dir).name}")
            alm.validate(cand.artifact_id, actor="alm")
            imported[category] = 1
        except ArtifactLifecycleError:
            pass  # identity collision — the merge was already recorded
    return imported


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
        for section in ("worlds", "policies"):
            for name, meta in sorted((manifest.get(section) or {}).items()):
                path = store_dir / str(meta.get("file", ""))
                if not path.exists():
                    errors.append(f"{name}: file missing ({meta.get('file')})")
                    continue
                if _sha(path) != meta.get("sha256"):
                    errors.append(f"{name}: checksum MISMATCH")
                if section == "worlds":
                    cats[f"world:{name}"] = int(meta.get("nodes", 0))
                else:
                    cats[f"policy:{name}"] = 1
    else:
        errors.append("no valid manifest.json")
    return StoreStatus(ok=not errors, location=str(store_dir),
                       git=is_git_store(store_dir), categories=cats,
                       errors=errors)
