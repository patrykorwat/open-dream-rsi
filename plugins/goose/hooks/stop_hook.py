#!/usr/bin/env python3
"""Goose Stop-hook adapter for the ODR sentinel (reactive delivery arm).

The MCP server observes every sandbox tool failure server-side and appends
recurrence notes to $SENTINEL_NOTE_FILE (JSONL). Goose has no post-tool
injection channel for stdio extensions, but Stop CAN block: this hook
consumes one pending note and blocks the turn exactly once, with the note
as the reason — the model sees the recurrence facts before it can retry.

Zero notes -> clean allow (exit 0, empty stdout).
"""
import json
import os
import sys


def main() -> None:
    note_file = os.environ.get(
        "SENTINEL_NOTE_FILE", "/tmp/odr_sentinel_notes.jsonl")
    try:
        with open(note_file, encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
    except FileNotFoundError:
        return
    if not lines:
        return
    first = lines[0]
    rest = lines[1:]
    tmp = note_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for ln in rest:
            f.write(ln + "\n")
    os.replace(tmp, note_file)
    try:
        note = json.loads(first).get("note", "")
    except json.JSONDecodeError:
        note = first
    if not note:
        return
    print(json.dumps({"decision": "block",
                      "reason": "ODR sentinel (recurrence detected in this "
                                "session): " + note +
                                "\nDo not retry a call that differs only by "
                                "parameter; change strategy or answer."}))
    sys.exit(0)


if __name__ == "__main__":
    main()
