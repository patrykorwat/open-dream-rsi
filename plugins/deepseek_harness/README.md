# ODR — DeepSeek Harness adapter

[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (`dsh`)
is an all-plugin agent harness. ODR speaks MCP, and DSH ships a first-party
MCP client bridge (`@deepseek-ai/dsh-mcp-client`) — so this adapter is pure
configuration: a Cordis overlay that inserts the ODR MCP server as an
external stdio (or Streamable HTTP) server. DSH starts the child with its
plugin lifecycle, discovers the six `odr_*` tools, and exposes them as
`mcp__open_dream_rsi__<tool>`.

No code ships here — only two overlay files and this guide.

## Files

| file                        | transport       | use when                                  |
|-----------------------------|-----------------|-------------------------------------------|
| `odr.cordis.yml`            | stdio           | DSH and the ODR checkout on one machine    |
| `odr-http.cordis.yml`       | streamable-http | ODR served remotely (tunnel / other host)  |

## Install (stdio)

1. Make the server importable: `pip install open-dream-rsi`, or export
   `PYTHONPATH=<ODR checkout>` in the overlay's `env:` block.

2. Edit both `/ABSOLUTE/PATH/...` placeholders in
   [`odr.cordis.yml`](odr.cordis.yml) to your project's `tasks.json` and
   memory dir (DSH does not expand `~`; use absolute paths).

3. Apply it once:

   ```bash
   dsh --patch /ABS/PATH/TO/odr.cordis.yml web
   ```

   To persist across runs, merge the overlay's `- insert:` entry into
   `$DSH_HOME/profiles/web/cordis.patch.yml` (do **not** overwrite that file
   — it may already hold unrelated patches). Verify the layer mounted:

   ```bash
   dsh --patch /ABS/PATH/TO/odr.cordis.yml --dump-config | grep -A2 open_dream_rsi
   ```

## Install (remote)

Serve the endpoint first, then apply `odr-http.cordis.yml` the same way.
The URL is a credential — see the note in the overlay.

## Environment scrubbing

DSH scrubs the spawned child's environment (names matching
`KEY|PASSWORD|SECRET|TOKEN` and everything `DSH_*`). The stdio overlay
therefore re-passes `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `ODR_LLM_*`
explicitly from the harness process's environment. Without them the dreamer
falls back to `http://127.0.0.1:8000/v1` (vLLM/Ollama default) and
auto-detects the served model. No key is needed for a local endpoint.

## Timeouts

`toolCallTimeoutMs: 1800000` (30 min) replaces the 60 s default because
`odr_run_once` / `odr_dream` run bounded improvement cycles that can take
tens of minutes. `odr_status`, `odr_recipes`, `odr_lessons` and
`odr_add_task` return in milliseconds.

## Verify

In a DSH session:

1. `mcp__open_dream_rsi__odr_status` — returns the loop's learned-state
   summary (policies, recipes, events). Expected on a fresh memory dir:
   zeros plus a hint, not an error.
2. `mcp__open_dream_rsi__odr_add_task` with a small task (prompt + tests),
   then `mcp__open_dream_rsi__odr_run_once` — the cycle needs a resolvable
   LLM endpoint (env, goose config, or localhost vLLM); with none reachable
   it reports the resolution failure instead of hanging.

Model-facing behavior notes (recipes as warm start, add-task-then-dream
workflow) live in the main README's "Plug in your favourite agent" section.
