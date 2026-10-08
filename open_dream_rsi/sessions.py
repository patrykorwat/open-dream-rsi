"""Session ingestion: turn real agent transcripts into curator raw material.

The Dream-RSI loop normally distils lessons from the failures it produces
itself. This module opens the second legitimate door: *host* transcripts —
Hermes sessions (``state.db``) and Cursor chats (``state.vscdb``) — are a
record of real tool failures the user already paid for. They are imported
as **evidence episodes**, never trusted as truth: distillation runs them
through the same structural gate, staging default and promotion semantics
as loop-collected failures.

Transcripts are hostile input. Everything read here is:

* opened **read-only** (URI mode; Cursor may be running and holds WAL locks),
* **secret-redacted** by default (chat history contains whatever was pasted
  into it — tokens, passwords, keys; see README before disabling),
* parsed defensively (undocumented, version-drifting schemas — a missing
  field yields an empty episode, never a crash).

Only the *shape* of the two stores is assumed (documented in README); all
semantics come from the curator's validation gate.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# -- secret redaction ---------------------------------------------------------
# Transcripts leak whatever was pasted into a chat. Redact before anything
# leaves the machine (tasks.json, lessons.json, SKILL.md, LLM prompts).

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


# -- normalized episode model ---------------------------------------------------

@dataclass
class SessionMessage:
    role: str            # "user" | "assistant" | "tool"
    text: str
    timestamp: float = 0.0
    name: str = ""       # tool name for tool rows


@dataclass
class Episode:
    """One imported conversation, normalized across hosts."""
    source: str                       # "hermes" | "cursor"
    session_id: str
    title: str = ""
    cwd: str = ""
    started_at: float = 0.0
    messages: List[SessionMessage] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "session_id": self.session_id,
                "title": self.title, "cwd": self.cwd,
                "started_at": self.started_at,
                "messages": [vars(m) for m in self.messages]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Episode":
        msgs = [SessionMessage(**m) for m in d.get("messages", [])]
        return cls(source=d.get("source", ""), session_id=d.get("session_id", ""),
                   title=d.get("title", ""), cwd=d.get("cwd", ""),
                   started_at=float(d.get("started_at", 0.0)), messages=msgs)


# -- Hermes sessions --------------------------------------------------------------

def read_hermes_sessions(db_path: "str | Path", *, source: Optional[str] = None,
                         limit: Optional[int] = None,
                         redact: bool = True) -> List[Episode]:
    """Read episodes from a Hermes ``state.db`` (read-only).

    Assumes the documented Hermes schema (``sessions`` + ``messages``).
    Archived/hidden sessions are skipped; a missing table yields nothing.
    """
    uri = f"file:{Path(db_path).as_posix()}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    try:
        cols = {r[1] for r in con.execute("pragma table_info(sessions)")}
        if not cols:
            return []
        where, params = [], []
        if source and "source" in cols:
            where.append("source = ?")
            params.append(source)
        for flag in ("archived", "hidden"):
            if flag in cols:
                where.append(f"coalesce({flag}, 0) = 0")
        sql = ("select id, title, display_name, cwd, started_at from sessions")
        if where:
            sql += " where " + " and ".join(where)
        sql += " order by started_at desc"
        if limit is not None:
            sql += f" limit {max(0, int(limit))}"
        rows = con.execute(sql, params).fetchall()
        out: List[Episode] = []
        for sid, title, display, cwd, started in rows:
            msgs = con.execute(
                "select role, content, tool_name, timestamp from messages "
                "where session_id = ? and coalesce(active, 1) = 1 "
                "order by id", (sid,)).fetchall()
            ep = Episode(source="hermes", session_id=str(sid),
                         title=str(title or display or ""), cwd=str(cwd or ""),
                         started_at=float(started or 0.0))
            for role, content, tool_name, ts in msgs:
                role = str(role or "")
                if role not in ("user", "assistant", "tool"):
                    continue
                text = str(content or "").strip()
                if not text:
                    continue
                ep.messages.append(SessionMessage(
                    role=role, text=redact_secrets(text) if redact else text,
                    timestamp=float(ts or 0.0), name=str(tool_name or "")))
            if ep.messages:
                out.append(ep)
        return out
    finally:
        con.close()


# -- Cursor chats -------------------------------------------------------------------

def default_cursor_user_dir() -> Path:
    """Platform default Cursor user-data dir (documented in README)."""
    import os
    import sys
    home = Path(os.path.expanduser("~"))
    if sys.platform == "darwin":
        return home / "Library/Application Support/Cursor/User"
    if sys.platform.startswith("win"):
        appdata = os.environ.get("APPDATA", str(home / "AppData/Roaming"))
        return Path(appdata) / "Cursor/User"
    return home / ".config/Cursor/User"


def _cursor_workspace_cwds(user_dir: Path) -> Dict[str, str]:
    """Map workspaceStorage hash dir -> project folder path (best effort)."""
    out: Dict[str, str] = {}
    ws = user_dir / "workspaceStorage"
    if not ws.is_dir():
        return out
    for d in ws.iterdir():
        wj = d / "workspace.json"
        try:
            folder = json.loads(wj.read_text(encoding="utf-8")).get("folder", "")
            out[d.name] = re.sub(r"^file://", "", str(folder))
        except Exception:
            continue
    return out


def _cursor_composer_workspaces(user_dir: Path) -> Dict[str, str]:
    """composerId -> workspace hash, from per-workspace composer registries."""
    owner: Dict[str, str] = {}
    ws = user_dir / "workspaceStorage"
    if not ws.is_dir():
        return owner
    for d in ws.iterdir():
        db = d / "state.vscdb"
        if not db.exists():
            continue
        try:
            con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro&immutable=1",
                                  uri=True)
            row = con.execute(
                "select value from ItemTable where key='composer.composerData'"
            ).fetchone()
            con.close()
            if not row or row[0] is None:
                continue
            for c in json.loads(row[0]).get("allComposers", []):
                cid = c.get("composerId")
                if cid:
                    owner[cid] = d.name
        except Exception:
            continue
    return owner


def read_cursor_sessions(user_dir: "str | Path | None" = None,
                         *, db: "str | Path | None" = None,
                         limit: Optional[int] = None,
                         redact: bool = True) -> List[Episode]:
    """Read episodes from Cursor's ``globalStorage/state.vscdb``.

    The schema is UNDOCUMENTED (observed 2026): ``cursorDiskKV`` holds
    ``composerData:{id}`` headers whose ``fullConversationHeadersOnly`` (or
    ``conversation``) lists bubble ids in order, and ``bubbleId:{cid}:{bid}``
    rows hold one message each (type 1 = user, 2 = assistant; empty text =
    tool-only row). Re-verify after Cursor upgrades; a row that does not
    parse is skipped, never fatal.

    The live database is WAL-locked while Cursor runs: with ``immutable=1``
    you read the committed state — for a consistent snapshot of the newest
    messages copy ``state.vscdb`` + ``-wal`` + ``-shm`` and pass the copy
    via ``db=`` (see README).
    """
    if db is not None:
        global_db = Path(db)
        user_dir_path = global_db.parent.parent  # .../User/globalStorage/x
    else:
        user_dir_path = Path(user_dir or default_cursor_user_dir())
        global_db = user_dir_path / "globalStorage" / "state.vscdb"
    if not global_db.exists():
        return []
    cwds = _cursor_workspace_cwds(user_dir_path) if user_dir_path else {}
    composer_ws = _cursor_composer_workspaces(user_dir_path) if user_dir_path else {}
    con = sqlite3.connect(f"file:{global_db.as_posix()}?mode=ro&immutable=1",
                          uri=True)
    out: List[Episode] = []
    try:
        headers = con.execute(
            "select key, value from cursorDiskKV where key like 'composerData:%'"
        ).fetchall()
        headers.sort(key=lambda r: r[0])
        if limit:
            headers = headers[-int(limit):]
        for key, value in headers:
            if value is None:
                continue
            try:
                data = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(data, dict):
                continue
            cid = str(data.get("composerId") or key.split(":", 1)[-1])
            order = [b.get("bubbleId") for b in
                     (data.get("fullConversationHeadersOnly")
                      or data.get("conversation") or [])
                     if isinstance(b, dict) and b.get("bubbleId")]
            if order:
                rows = []
                for bid in order:  # order matters: out-of-order = wrong pairing
                    r = con.execute(
                        "select value from cursorDiskKV where key = ?",
                        (f"bubbleId:{cid}:{bid}",)).fetchone()
                    if r:
                        rows.append(r)
            else:
                rows = con.execute(
                    "select value from cursorDiskKV where key like ?",
                    (f"bubbleId:{cid}:%",)).fetchall()
            ep = Episode(
                source="cursor", session_id=cid,
                title=str(data.get("name") or ""),
                started_at=float(data.get("createdAt") or 0.0) / 1000.0,
                cwd=cwds.get(composer_ws.get(cid, ""), ""))
            for (raw,) in rows:
                try:
                    b = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(b, dict):
                    continue
                text = str(b.get("text") or "").strip()
                if not text:  # tool-only bubble: no prose to distil
                    continue
                role = "user" if b.get("type") == 1 else "assistant"
                ep.messages.append(SessionMessage(
                    role=role, text=redact_secrets(text) if redact else text,
                    timestamp=float(b.get("createdAt") or 0.0) / 1000.0))
            if ep.messages:
                out.append(ep)
        return out
    finally:
        con.close()


# -- episode -> curator evidence -----------------------------------------------------

_ERROR_MARKERS = ("error", "traceback", "exception", "failed", "failure",
                  '"ok": false', '"ok":false')

#: Clean tool-envelope shapes (Hermes JSON results embed "error": null /
#: exit_code 0 in EVERY result); stripping them before the marker scan
#: keeps success envelopes out of the failure evidence.
_CLEAN_ENVELOPE = re.compile(
    r'"error"\s*:\s*(null|""|0)\b|"exit_code"\s*:\s*0\b|"status"\s*:\s*"ok"'
    r'|"ok"\s*:\s*true', re.IGNORECASE)


def episode_failures(ep: Episode, max_items: int = 6) -> List[Dict[str, Any]]:
    """Records shaped like the loop's failure evidence (``action`` /
    ``score`` / ``errors``) so :class:`KnowledgeCurator` can distil from a
    host transcript exactly as from its own verifier failures. Heuristic
    marker scan only — the curator's structural gate and the promotion gate
    remain the only authorities on what the lesson ends up being."""
    out: List[Dict[str, Any]] = []
    for m in ep.messages:
        probe = _CLEAN_ENVELOPE.sub(" ", m.text.lower())
        if m.role != "tool" and not any(k in probe for k in _ERROR_MARKERS):
            continue
        if m.role == "tool" and not any(k in probe for k in _ERROR_MARKERS):
            continue  # clean tool envelope: nothing to learn from it
        out.append({"action": m.name or f"{ep.source}:{m.role}",
                    "score": 0.0,
                    "errors": [m.text[:160]]})
    return out[-max_items:]


def write_episodes(episodes: List[Episode], path: "str | Path") -> int:
    """JSONL sink (one episode per line). Returns the line count."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8") as fh:
        for ep in episodes:
            fh.write(json.dumps(ep.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def load_episodes(path: "str | Path") -> List[Episode]:
    out: List[Episode] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(Episode.from_dict(json.loads(line)))
        except (json.JSONDecodeError, TypeError):
            continue  # truncated line: skip, never abort the batch
    return out


# -- curator lessons -> Hermes skills -------------------------------------------------

#: Facts the skills frontmatter carries; only ACTIVE lessons export —
#: staging/rejected knowledge has not earned a prompt anywhere, and a
#: Hermes skill IS a prompt.
_SKILL_TEMPLATE = """---
name: {name}
description: "Dream-RSI curated facts for {category} ({n} lesson(s), replay-gated). Use when working on {category} tasks."
metadata:
  source: open-dream-rsi curator
  lessons: {n}
---

# {title}

Facts learned from recorded failures in the `{category}` task family and
promoted by paired-replay evidence (net gain, zero solve->fail
regressions, explicit stop clause). Declarative background, not
instructions — a complete answer may be submitted as it stands.

{body}

Generated by `odr skills export` from `lessons.json` (memory:
`{memory}`). Regenerate after promotion passes; edit here and the edits
are lost on the next export — edit the knowledge base instead.
"""


def lessons_to_skills(memory_or_root: Any, out_dir: "str | Path",
                      *, categories: Optional[List[str]] = None,
                      only_active: bool = True) -> List[Path]:
    """Render the curated lesson KB into Hermes-style skills.

    One directory per category (``<out>/<slug>/SKILL.md``): Hermes picks
    each up as a skill with its own trigger description. Only ``active``
    lessons are exported by default — staging and rejected entries have
    not earned a prompt, and a skill IS a prompt (the curator's whole
    measured contract: activation must be earned). Returns the written
    SKILL.md paths."""
    if isinstance(memory_or_root, (str, Path)):
        root = Path(memory_or_root)
    else:
        root = Path(memory_or_root.root)
    lessons_path = root / "lessons.json"
    if not lessons_path.exists():
        return []
    try:
        data = json.loads(lessons_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    out_root = Path(out_dir)
    written: List[Path] = []
    for category, entries in sorted(data.items()):
        if categories and category not in categories:
            continue
        keep = [l for l in entries
                if (not only_active
                    or str(l.get("status", "active")) == "active")]
        if not keep:
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", str(category).lower()).strip("-") \
            or "lessons"
        body = "\n".join(
            f"- **[{l.get('trigger', '')}]** {redact_secrets(str(l.get('text', '')))}"
            for l in keep)
        text = _SKILL_TEMPLATE.format(
            name=f"odr-{slug}", category=category, n=len(keep),
            title=f"Curated facts: {category}", body=body,
            memory=str(root))
        skill_dir = out_root / slug
        skill_dir.mkdir(parents=True, exist_ok=True)
        path = skill_dir / "SKILL.md"
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written
