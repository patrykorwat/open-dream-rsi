# Sentinel — Hermes adapter

Thin adapter over the host-agnostic engine (`open_dream_rsi/sentinel.py`).
What the sentinel is and why (error-class repeats re-appear within a
session after a median of ~1.5 minutes — faster than any curator schedule):
see the benchmark section of the repo README.

## Install

```bash
cp -r plugins/hermes_sentinel "$HERMES_HOME/plugins/sentinel"
hermes plugins enable sentinel
```

The plugin prefers the installed `open_dream_rsi` package for its engine
and falls back to the vendored `_sentinel_engine.py` copy, so the plain
directory copy installs standalone with no Python package required.

Config (optional), under `plugins.entries.sentinel.settings`:

| key                    | default | meaning                         |
|------------------------|---------|---------------------------------|
| `intra_session_repeat` | 2       | hits in one session before note |
| `cross_session_count`  | 2       | distinct sessions before note   |
| `max_tracked`          | 500     | LRU cap on tracked classes      |

Slash command: `/sentinel` (ledger), `/sentinel reset` (clear).

## Files

- `__init__.py` — Hermes hook adapter (transform_tool_result) + slash command
- `_sentinel_engine.py` — vendored copy of the engine (kept byte-identical
  to `open_dream_rsi/sentinel.py` by the test suite)
- `plugin.yaml` — manifest
- `test_sentinel.py` — adapter contract tests (faked PluginContext)

Other hosts: Claude Code adapter in `plugins/claude_code/`; any host with
command hooks via `python -m open_dream_rsi sentinel check`.
