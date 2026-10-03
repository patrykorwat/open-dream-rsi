# Sentinel — Claude Code adapter

Sentinel watches every tool result and, when the **same error class**
(different arguments, same failure — URLs/paths/numbers/hexes normalized
away) repeats, appends a short declarative note to the tool result:
recurrence facts plus an explicit stop condition. Clean calls pay zero
tax; the note never blocks anything.

Claude Code's `PostToolUse` / `PostToolUseFailure` command hooks are the
generic integration channel for any agent that speaks the same stdin/stdout
JSON contract (shell hooks). This adapter needs no code beyond the shipped
CLI: `open_dream_rsi sentinel check --format claude`.

## Install

1. Make the CLI importable — either `pip install open-dream-rsi` or run
   from a checkout of this repo.

2. Merge the snippet below into `~/.claude/settings.json` (or the project's
   `.claude/settings.json`). Adjust the command if you run from an
   environment where `python -m open_dream_rsi` needs a venv prefix.

   See [`settings.snippet.json`](settings.snippet.json).

3. Verify:

   ```bash
   echo '{"tool_name":"Bash","tool_response":{"exit_code":7},"session_id":"t"}' \
     | python -m open_dream_rsi sentinel check --format claude   # {} (1st hit)
   echo '{"tool_name":"Bash","tool_response":{"exit_code":7},"session_id":"t"}' \
     | python -m open_dream_rsi sentinel check --format claude   # hookSpecificOutput
   ```

## Configuration

| env var                              | default | meaning                          |
|--------------------------------------|---------|----------------------------------|
| `ODR_SENTINEL_STATE`                 | `~/.local/state/odr-sentinel/state.json` | ledger file |
| `ODR_SENTINEL_INTRA_SESSION_REPEAT`  | 2       | hits in one session before note  |
| `ODR_SENTINEL_CROSS_SESSION_COUNT`   | 2       | distinct sessions before note    |

Ledger: `python -m open_dream_rsi sentinel ledger` — clear: `... sentinel reset`.

## Notes

- The note rides `hookSpecificOutput.additionalContext` (appended to the
  tool result Claude sees). `PostToolUseFailure` events are covered by the
  same command; the hook event name is echoed from the event, so both work.
- Sessions are keyed by the event's `session_id` field.
- The adapter always exits 0 and prints valid JSON or nothing — a sentinel
  failure can never break your session (hooks' failure policy is annotate,
  never gate).
