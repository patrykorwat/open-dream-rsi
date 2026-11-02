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
|-----------------------------|-----------------|--------------------------------------------|
| `odr.cordis.yml`            | stdio           | DSH and the ODR checkout on one machine    |
| `odr-http.cordis.yml`       | streamable-http | ODR served remotely (tunnel / other host)  |
| `cordis.patch.yml` + `package.json` | bundle (npm) | installable `dsh plugin add` package    |

The npm bundle ships the ODR Python server inside the package
(`open_dream_rsi/`, stdlib-only) plus a tiny bridge (`bridge.js`) that does
two things: it publishes where the vendored entry point lives
(`run_mcp.py`), and it **is the dreamer's brain** — an in-process
OpenAI-compatible proxy that forwards every ODR LLM call to the model the
harness itself is configured with (`ctx.llm.stream`), so the Python child
needs no API keys, no presets and no extra endpoints. The layer inserts the
in-box `@deepseek-ai/dsh-mcp-client` row pointing the system `python3` at
the vendored entry point. No pip, no PYTHONPATH, no build step and no build
allowance even for git installs. Bundle defaults resolve `tasks.json` /
`.dream_rsi` beside the directory `dsh` launches from; override the
`mcp-open-dream-rsi` row from your profile's `cordis.patch.yml` (same id —
a config override replaces the whole config).

## Install as a bundle (`dsh plugin add`)

The bundle ships the Python server inside the package — no pip, no
PYTHONPATH. Only a system `python3` (3.10+) on PATH is required. From a
directory holding your `tasks.json`:

```bash
# from the npm registry, a tarball, a checkout, or straight from GitHub — pick one:
dsh plugin --profile <name> add dsh-open-dream-rsi            # or pin: dsh-open-dream-rsi@0.3.1
dsh plugin --profile <name> add /path/to/dsh-open-dream-rsi-0.3.1.tgz
dsh plugin --profile <name> add /path/to/open-dream-rsi/plugins/deepseek_harness
dsh plugin --profile <name> add github:patrykorwat/open-dream-rsi#path:plugins/deepseek_harness

dsh --profile <name> --dump-config    # shows the "# == dsh-open-dream-rsi" layer
```

Git installs fetch sources; this bundle needs no build step and no build
allowance. Pin a commit so a later push cannot silently change what runs:

```bash
dsh plugin --profile <name> add "github:patrykorwat/open-dream-rsi#<sha>&path:plugins/deepseek_harness"
```

(The pin must point at a commit that already contains this directory —
pnpm fetches sources, not releases.)

## Updating

The bundle is self-contained (the Python server is vendored, no pip, no
build step), so updating is a re-install of the same package id:

```bash
dsh plugin --profile <name> add dsh-open-dream-rsi
```

Installed from git instead? Pin a commit that already contains the bundle
files — an older sha registers a synthetic `0.0.0` package and dsh dies in
`resolveBundleDir`:

```bash
dsh plugin --profile <name> add github:patrykorwat/open-dream-rsi#<sha>&path:plugins/deepseek_harness
```

Verify the installed version afterwards with your profile's plugin list
(see `dsh plugin --help`). Re-installs never touch your memory:
`tasks.json` / `.dream_rsi` live beside the directory dsh launches from.

## Install (stdio, `--patch` overlay)

1. Make the server importable (not on PyPI — install from the repo):
   `python3 -m pip install git+https://github.com/patrykorwat/open-dream-rsi`,
   or export `PYTHONPATH=<ODR checkout>` in the overlay's `env:` block.

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

## Where the dreamer's model comes from

The bundle needs **no LLM configuration at all**. The bridge runs an
OpenAI-compatible proxy inside the harness process; the Python child is
pointed at it via `OPENAI_BASE_URL=http://127.0.0.1:<port>/v1` and every
`/v1/chat/completions` call is served by the harness's own configured model
through the dsh-llm service (`ctx.llm.stream`). The route is resolved once,
in order of intent: `ODR_LLM_MODEL` pins the model; otherwise the harness's
own agent default route (`ctx.agentDefaultModel`); otherwise the first
registered provider plus its first catalog model (`ctx.llm.listModels`).
`GET /v1/models` reports that
model, so ODR's auto-detection picks it up.

Escape hatch: if the harness process already has a real `OPENAI_BASE_URL`
set, the child is pointed there directly and the proxy stays idle (direct
mode). Set `ODR_LLM_MODEL` to pin a specific model in either mode.

Environment scrubbing still applies: DSH drops child-env names matching
`KEY|PASSWORD|SECRET|TOKEN` and everything `DSH_*`, and the bridge
(`childEnv()`) re-passes only LLM vars that are **actually set** — an empty
`ODR_LLM_PRESET=""` never reaches the child (older ODR releases fed it into
preset selection and died with `Unknown preset ''`).

The hand-rolled `--patch` overlay (`odr.cordis.yml`) has no bridge: it
re-passes the vars it needs (`process.env.VAR || undefined`) and ODR then
resolves an endpoint itself — `OPENAI_*` env, a preset
(`ODR_LLM_PRESET=local|openai|cursor|goose`), or its local default
(`http://127.0.0.1:8000/v1` with model auto-detection).

## Timeouts

`toolCallTimeoutMs: 1800000` (30 min) replaces the 60 s default because
`odr_run_once` / `odr_dream` run bounded improvement cycles that can take
tens of minutes. `odr_status`, `odr_recipes` and
`odr_add_task` return in milliseconds.

## Verify

In a DSH session:

1. `mcp__open_dream_rsi__odr_status` — returns the loop's learned-state
   summary (policies, recipes, events). Expected on a fresh memory dir:
   zeros plus a hint, not an error.
2. `mcp__open_dream_rsi__odr_add_task` with a small task (prompt + tests),
   then `mcp__open_dream_rsi__odr_run_once` — with the bundle, the cycle
   calls the model your harness session already uses; with the overlay, it
   needs a resolvable LLM endpoint (env, preset, or localhost vLLM) and
   reports the resolution failure instead of hanging.

Model-facing behavior notes (recipes as warm start, add-task-then-dream
workflow) live in the main README's "Plug in your favourite agent" section.
