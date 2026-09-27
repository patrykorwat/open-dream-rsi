# Plug Open Dream-RSI into your agent harness (OpenCode, Goose, …)

Open Dream-RSI speaks **MCP** (Model Context Protocol) over stdio, so any
MCP-speaking harness — OpenCode, Goose, Claude Code, Cursor, Zed, … — can use
the self-improvement loop as a tool provider: your agent queues tasks, the
dreamer solves them offline, and the agent pulls back verified recipes and
lessons.

No extra Python packages are required — the server is stdlib-only.

## 0. One-time: install the library

```bash
git clone https://github.com/patrykorwat/open-dream-rsi.git
cd open-dream-rsi
pip install -e .
```

Sanity check the server (it waits on stdin — Ctrl-C to exit)::

    python3 -m open_dream_rsi mcp

If it starts without a traceback, you are ready to plug in.

### Point it at your model (optional but recommended)

The loop uses the same OpenAI-compatible env vars as everywhere else:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1   # vLLM / Ollama / any gateway
export OPENAI_API_KEY=***                  # any token for local servers
export ODR_LLM_MODEL=your-model-name               # exact served model name
```

---

## OpenCode

Create (or merge into) `opencode.json` **in the project root where you work** —
the same directory you will run `opencode` from:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "open-dream-rsi": {
      "type": "local",
      "command": ["python3", "-m", "open_dream_rsi", "mcp",
                  "--tasks", "./tasks.json", "--memory", "./.dream_rsi"],
      "enabled": true
    }
  }
}
```

Start OpenCode and type `/mcps` — you should see `open-dream-rsi: connected`.

Then just talk to the agent, for example:

> "Queue a task for the Dream-RSI loop: category `strutil`, implement
> `snake_case(s)` that splits on camel case and punctuation, tests
> `snake_case('HelloWorld') == 'hello_world'` and
> `snake_case('https://Example.com') == 'https_example_com'`."

or, when it is stuck on something the loop may already know:

> "Check odr_lessons and odr_recipes for category `strutil` before you try
> again."

The five tools the agent sees:

| Tool | What it does |
|---|---|
| `odr_status` | what the loop has learned (policies, recipes, events) |
| `odr_recipes` | best **verified** solution for a category — use as warm start |
| `odr_lessons` | curated failure lessons for a category / search query |
| `odr_add_task` | queue a task (prompt + tests) for the dreamer |
| `odr_run_once` | run one improvement cycle now (bounded API budget) |

`odr_run_once` runs with **thought-conditioned branching enabled by
default** (like the library itself): attempts record their one-line `PLAN:`,
proposals see the tried-idea ledger, and expansion leaves dead idea
families — the strongest configuration from the decoy-trap benchmark, no
setup needed. Pass `thoughts: false` to the tool to ablate it.

Tip — let OpenCode consult the loop proactively: add to your project
`AGENTS.md`:

```markdown
## Self-improvement loop (MCP: open-dream-rsi)
- Before implementing a self-contained Python utility, call `odr_recipes`
  and `odr_lessons` for its category; reuse a verified recipe verbatim.
- When you finish a hard, testable function, queue it with `odr_add_task`
  (task_id = function name, tests = your test cases) so the loop can
  dream over it offline.
```

## Goose

### Recommended: let the script wire everything

```bash
cd open-dream-rsi
./scripts/odr_goose_setup.sh            # diagnose + configure goose + start proxy
./scripts/odr_goose_setup.sh --check    # diagnose only, change nothing
./scripts/odr_goose_setup.sh --no-proxy # skip the proxy (use provider='goose')
```

The script verifies the install, resolves your goose provider (CLI
`GOOSE_*` **or** desktop `active_provider:` + nested `providers:` block),
starts the model-borrowing proxy, sends one real completion through it, and
writes the extension block into `~/.config/goose/config.yaml` itself
(`open_dream_rsi.utils.goose_config` — targeted block editor: backs the file
up with a timestamp, replaces a stale entry including inline comments, stays
idempotent, never touches foreign entries, verifies the result by reading
the file back). Afterwards: **fully quit goose (Cmd+Q — the desktop app
caches config at startup), relaunch, activate `open-dream-rsi` in the
session's extensions picker.**

### Manual block

Add the extension to `~/.config/goose/config.yaml`
(or run `goose configure` → *Add Extension* → *Command-line Extension* and
enter the same command/args):

```yaml
extensions:
  open-dream-rsi:
    enabled: true
    type: stdio
    name: open-dream-rsi
    description: "Dream-RSI self-improvement loop. Use odr_add_task to queue a
      Python task with tests, odr_run_once to run an improvement cycle,
      odr_recipes/odr_lessons to reuse verified solutions and failure lessons,
      odr_status to inspect what the loop has learned."
    cmd: python3
    args: ["-m", "open_dream_rsi", "mcp",
           "--tasks", "/ABSOLUTE/PATH/projects/myproject/tasks.json",
           "--memory", "/ABSOLUTE/PATH/projects/myproject/.dream_rsi"]
    timeout: 300
```

Goose spawns extensions **without a shell and with a scrubbed environment**:
use absolute paths (no `~`, no relative `./`) and pass any env the server
needs via `envs:` — it will not inherit your export'ed `OPENAI_*`.

One-shot alternative (no config edit), from the project directory::

    goose session --with-extension "python3 -m open_dream_rsi mcp --tasks ./tasks.json --memory ./.dream_rsi"

Inside the session, ask e.g.:

> "Ask the odr_status tool what the self-improvement loop has learned, then
> run one cycle with odr_run_once if there are queued tasks."

Approve the extension tools when Goose prompts for permission. No LLM model
will call these tools on its own initiative — the trigger must come from you,
from recipe `instructions:`, or from a hook that surfaces them at the right
moment.

### Borrow goose's model (no second API key)

Goose does not implement MCP sampling (`sampling/createMessage`), so an
extension cannot ask the host for completions through the protocol. Two
first-class ways to run the dreamer on **exactly the model goose uses**:

1. **`provider: "goose"` (zero daemon).** Pass it to `odr_run_once` (or
   `--provider goose` on `loop`): Open Dream-RSI reads
   `~/.config/goose/config.yaml` — CLI style (`GOOSE_PROVIDER` /
   `GOOSE_MODEL`) or desktop style (`active_provider:` + nested
   `providers:` block) — plus `custom_providers/*.json`, `secrets.yaml` and
   the macOS keychain, and calls that same endpoint itself. The credential
   stays in goose's storage; nothing is duplicated in the extension config.
   Caveat: an MCP server spawned by goose gets a scrubbed environment, so a
   key that lives only in your shell export is invisible here — the proxy
   (option 2) is the robust path for goose-spawned servers; `provider:
   "goose"` shines for the standalone `loop` daemon and cron.

2. **`proxy` (plain OpenAI-compatible endpoint).** If you want anything
   OpenAI-compatible pointed at goose's brain (not only ODR), start::

       python3 -m open_dream_rsi proxy --port 8799

   and set, in this extension's `envs:` or anywhere else::

       OPENAI_BASE_URL: "http://127.0.0.1:8799/v1"
       OPENAI_API_KEY: "proxy-internal"   # dummy; proxy authenticates upstream

   The proxy resolves the upstream **per request** (switching models in
   goose takes effect live), pins outgoing calls to `GOOSE_MODEL`
   (`--passthrough-models` opts out), speaks OpenAI and Anthropic upstreams,
   binds to loopback only, and exposes `GET /health` (resolved upstream with
   the key masked) and `GET /v1/models`. Check the resolution without
   serving: `python3 -m open_dream_rsi proxy --print-config`.

   Keep the proxy alive across reboots with a launchd agent — save as
   `~/Library/LaunchAgents/dev.open-dream-rsi.proxy.plist` (adjust paths),
   then `launchctl load ~/Library/LaunchAgents/dev.open-dream-rsi.proxy.plist`:

   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
   <plist version="1.0"><dict>
     <key>Label</key><string>dev.open-dream-rsi.proxy</string>
     <key>ProgramArguments</key><array>
       <string>/opt/homebrew/bin/python3</string>
       <string>-m</string><string>open_dream_rsi</string>
       <string>proxy</string><string>--port</string><string>8799</string>
     </array>
     <key>EnvironmentVariables</key><dict>
       <!-- fallback credential for the upstream, if not in goose storage -->
       <key>OPENAI_API_KEY</key><string>REPLACEME</string>
     </dict>
     <key>RunAtLoad</key><true/>
     <key>KeepAlive</key><true/>
     <key>StandardOutPath</key><string>/tmp/odr-proxy.log</string>
     <key>StandardErrorPath</key><string>/tmp/odr-proxy.log</string>
   </dict></plist>
   ```

## Any other MCP client (Hermes, Codex, Claude Code, Cowork, Zed, …)

### Zero configuration (all of them)

Since 0.2 the server needs **no LLM setup at all**. `odr_run_once` resolves
the brain automatically, in this order:

1. `OPENAI_BASE_URL` / `OPENAI_API_KEY` in the server's environment
   (e.g. pointed at the `proxy`),
2. **your local goose config** (`~/.config/goose`: `active_provider` +
   `providers:` block + `custom_providers/*.json` + `secrets.yaml` + macOS
   keychain) — read from disk, so it survives the env scrubbing that
   Hermes/Codex/Claude Code apply to spawned servers,
3. `http://127.0.0.1:8000/v1` (vLLM/Ollama default).

Self-hosted endpoints additionally run with hidden reasoning disabled
(`chat_template_kwargs.enable_thinking=false`) by default — endpoints that
reject the flag get a clean automatic retry without it. An empty task queue
is a benign no-op with a hint, never an error. So: register the server,
restart the client, and the first message that says "run one Dream-RSI
cycle" just works — no prompts to babysit, no keys to copy.

The five tools are registered prefixed per client (e.g. Hermes
`mcp_open_dream_rsi_odr_run_once`).

### Hermes Agent

Add under `mcp_servers` in `~/.hermes/config.yaml` (or via the dashboard's
MCP catalog):

```yaml
mcp_servers:
  open-dream-rsi:
    command: "python3"
    args: ["-m", "open_dream_rsi", "mcp",
           "--tasks", "/ABSOLUTE/PATH/open-dream-rsi/tasks.json",
           "--memory", "/ABSOLUTE/PATH/open-dream-rsi/.dream_rsi"]
    timeout: 300
```

Restart Hermes. Tools appear as `mcp_open_dream_rsi_*` in every platform
toolset. Hermes also supports MCP **sampling** — if you want the dreamer to
use Hermes' own model through the protocol instead of config-file
resolution, that is the one host where it works; today ODR does not request
sampling (it resolves the endpoint itself).

### Codex CLI

```bash
codex mcp add open-dream-rsi -- python3 -m open_dream_rsi mcp \
  --tasks /ABSOLUTE/PATH/open-dream-rsi/tasks.json \
  --memory /ABSOLUTE/PATH/open-dream-rsi/.dream_rsi
```

(equivalently `[mcp_servers.open-dream-rsi]` with `command`/`args` in
`~/.codex/config.toml`). Verify with `codex mcp list`.

### Claude Code

```bash
claude mcp add open-dream-rsi -- python3 -m open_dream_rsi mcp \
  --tasks /ABSOLUTE/PATH/open-dream-rsi/tasks.json \
  --memory /ABSOLUTE/PATH/open-dream-rsi/.dream_rsi
```

Check with `/mcp` inside a session.

### Claude Cowork / claude.ai connectors (remote only)

Cowork connects to **remote** MCP URLs (the connection is brokered from
Anthropic's cloud; local stdio is not available there). Run the server in
its HTTP mode and expose it through your own tunnel:

```bash
python3 -m open_dream_rsi mcp --http --host 0.0.0.0 --port 8800 \
  --tasks /ABS/PATH/tasks.json --memory /ABS/PATH/.dream_rsi
# then tunnel 8800 (tailscale funnel / cloudflared) and add the public
# https URL as a custom connector (Settings -> Connectors -> Add custom)
```

The endpoint implements the stateless Streamable-HTTP profile
(`POST /mcp`, one JSON-RPC message per request; `GET /health`).
**Treat the URL as a credential**: whoever reaches it can queue tasks and
spend your LLM budget — keep the tunnel private and share it only when
you mean to.

Optional flags/env (any client): `--tasks <file>` (or `ODR_TASKS`),
`--memory <dir>` (or `ODR_MEMORY`). One memory dir per improving instance —
share it between the MCP server and a long-running
`python -m open_dream_rsi loop` so the harness sees exactly what the daemon
has learned (that is the recommended setup: the daemon dreams, the harness
queries).

## Troubleshooting

- **Server shows "failed/unknown" in the harness** — run the command manually
  in the same directory; any traceback you see there is the real error
  (usually `open_dream_rsi` not importable → re-run `pip install -e .`).
- **`odr_run_once` is slow / 0 solved** — it is making real LLM calls; check
  `OPENAI_BASE_URL`/`ODR_LLM_MODEL` and try `provider: "mock"` for a key-free
  smoke test of the plumbing.
- **Tool call denied** — Goose prompts per tool the first time; approve it.
- **Tools exist but the model never calls them** — by design no model
  self-initiates unfamiliar tools: trigger via recipe `instructions:`, a
  hook, or an explicit ask ("call the odr_status tool"). In the **desktop
  app**, also check the per-session extensions picker (enabled in
  config.yaml ≠ active in this chat) and do a full Cmd+Q restart after any
  config edit — re-run `./scripts/odr_goose_setup.sh --check` to confirm
  the block is still in place.
