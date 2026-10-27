# Plug Open Dream-RSI into your agent harness (OpenCode, Goose, …)

Open Dream-RSI speaks **MCP** (Model Context Protocol) over stdio, so any
MCP-speaking harness — OpenCode, Goose, Claude Code, Cursor, Zed, … — can use
the self-improvement loop as a tool provider: your agent queues tasks, the
dreamer solves them offline, and the agent pulls back verified recipes and
lessons.

No extra Python packages are required — the server is stdlib-only.

## 0. One-time: install the library

```bash
pip install open-dream-rsi        # zero runtime deps, Python 3.10+
```

The package installs the `odr` console script (0.3.9+). From a checkout:
`pip install -e .`.

Sanity check the server (it waits on stdin — Ctrl-C to exit):

```bash
odr mcp        # or: python3 -m open_dream_rsi mcp
```

If it starts without a traceback, you are ready to plug in.

### Point it at your model (optional but recommended)

The loop uses the same OpenAI-compatible env vars as everywhere else:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1   # vLLM / Ollama / any gateway
export OPENAI_API_KEY=***                  # any token for local servers
export ODR_LLM_MODEL=your-model-name       # optional — the first model the
# endpoint serves (GET /v1/models) is auto-detected when unset; pin only to
# select among several served models
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

The six tools the agent sees:

| Tool | What it does |
|---|---|
| `odr_status` | what the loop has learned (policies, recipes, events) |
| `odr_recipes` | best **verified** solution for a category — use as warm start |
| `odr_lessons` | curated failure lessons for a category / search query |
| `odr_add_task` | queue a task (prompt + tests, or prompt + criteria) for the dreamer — test-less tasks are completion-judged |
| `odr_run_once` | run one improvement cycle now (bounded API budget) |
| `odr_dream` | the automated cycle: import host session evidence, distill curated lessons, run the promotion gate, publish skills |

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

## How it fits together (and where the data lives)

```
  ┌────────────────────────────┐      ┌───────────────────────────────────────┐
  │ AGENT HOSTS                │      │ MEMORY — one dir per dreaming instance│
  │  Mac:  goose, claude, codex│      │  recipes.json      verified code+score│
  │  your server:  Hermes (Docker)     │      │  lessons.json      failure lessons    │
  │  web:  Cowork via tunnel   │      │  policies*.json    exploration params │
  └───────────┬────────────────┘      │  trees/            dream trees (replay)│
              │ stdio (local hosts)   │  events.jsonl      audit log: per-task│
              │ or HTTP (remote)      │                    api_calls, solved   │
              ▼                       └───────────────────▲───────────────────┘
  ┌────────────────────────────┐                          │ read/write
  │ odr-mcp server             │   resolves the brain:    │
  │ (same package, stdlib only)│   OPENAI_* env → goose   │  only VERIFIED code
  │ odr_add_task / odr_run_once│   config → localhost     │  (sandbox: python -I,
  │ odr_status / recipes /...  │   vLLM; self-hosted →    │  scrubbed env) enters
  └───────────┬────────────────┘   enable_thinking=false  │  recipes → next task
              │ chat/completions                          │  of the category is
              ▼                                           │  warm-started from it
      the SAME model the host uses                        │  (this is the compression)
      (one endpoint, no key duplication)
```

Data placement (example deployment):

| data | host path | notes |
|---|---|---|
| production memory | `<hermes-compose-dir>/data/dream-rsi/.dream_rsi/` | bind-mounted at `/opt/data` — survives rebuilds |
| task queue | `<hermes-compose-dir>/data/dream-rsi/tasks.json` | created lazily on first `odr_add_task` |
| benchmark runs | `open-dream-rsi/.bench_real/` | deliberately outside the production memory |
| package itself | `<hermes-compose-dir>/open-dream-rsi` → mounted `:ro`, `PYTHONPATH` | no pip layer in the image |

Sharing across agents: memory is plain JSON — mount the same directory to
several hosts (SSHFS/tailnet) and one dreamer feeds many consumers.
Never hand-edit `recipes.json`/`lessons.json`; go through the tools so the
verification gate holds.

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

The six tools are registered prefixed per client (e.g. Hermes
`mcp_open_dream_rsi_odr_run_once`).

### Hermes Agent

Easiest: install the bundled plugin. Hermes discovers plugin manifests in a
repo root or one directory deep — this repo ships `odr-trigger/plugin.yaml`
(the event-driven dream cycle: every session end fires `dream_once()` —
evidence ingest, lesson distillation, promotion gate, ALM maintenance —
event-driven, no cron, no schedule; overlapping triggers collapse on a lock).

Desktop app: **Settings → Plugins → Install from Git**, paste
`https://github.com/patrykorwat/open-dream-rsi`, review, install, enable
`odr-trigger`.

![Hermes desktop — Install from Git](images/hermes-install-dialog.png)

CLI (same result, works headless):

```bash
pip install open-dream-rsi        # the plugin imports the library in-process
hermes plugins install "https://github.com/patrykorwat/open-dream-rsi#odr-trigger"
hermes plugins enable odr-trigger
```

The `#odr-trigger` fragment points discovery at the plugin directory (the
repo also carries a Cursor manifest, so a whole-repo scan can be ambiguous).
Point the dreamer's brain at your model with env — the zero-config chain
applies (`OPENAI_BASE_URL` + `OPENAI_API_KEY`; the served model is
auto-detected via `/v1/models`). For a server install, see
[Hermes in Docker](#hermes-in-docker-servercontainer-setup).

#### MCP tools beside the trigger plugin

The plugin automates the dream cycle. To also have the loop's MCP tools
(`odr_recipes`, `odr_add_task`, …) inside Hermes sessions, add the server
under `mcp_servers` in `~/.hermes/config.yaml` (or via the dashboard's MCP
catalog):

```yaml
mcp_servers:
  open-dream-rsi:
    command: "python3"
    args: ["-m", "open_dream_rsi", "mcp",
           "--tasks", "/ABSOLUTE/PATH/tasks.json",
           "--memory", "/ABSOLUTE/PATH/.dream_rsi"]
    timeout: 300
```

(0.3.9+ pip installs put an `odr` console script on PATH — `command: "odr"`,
`args: ["mcp", …]` is equivalent when the gateway inherits your PATH.)
Restart Hermes; tools appear as `mcp_open_dream_rsi_*` in every platform
toolset. Hermes also supports MCP **sampling** — the one host where the
dreamer could use the host's own model through the protocol; today ODR
resolves its endpoint itself (zero-config chain).

### Hermes in Docker (server/container setup)

The stdio MCP server is spawned **inside** the Hermes container, so the
host mount + one ENV make it work without any pip layer in the image:

1. Put the repo on the Docker host, e.g. `git clone git@github.com:patrykorwat/open-dream-rsi.git ~/git/open-dream-rsi`.
2. Mount it read-only into the container and export `PYTHONPATH` (in the
   Dockerfile once, so every service inherits it):

   ```dockerfile
   ENV PYTHONPATH=/opt/open-dream-rsi
   ```

   ```yaml
   # compose.yaml -> services.hermes
   volumes:
     - /home/USER/git/open-dream-rsi:/opt/open-dream-rsi:ro
     - ./data:/opt/data
   ```

   Read-only mount + `PYTHONPATH` is the whole integration — the package is
   stdlib-only, no install step. Updating = `git pull` on the host + restart
   the Hermes container (no image rebuild). If you prefer a self-contained
   image instead of a mount, `RUN git clone … /opt/open-dream-rsi` in the
   Dockerfile — at the cost of a rebuild per upstream change.
3. Add the server to `~/.hermes/config.yaml` (on the host that is
   `./data/config.yaml` of the compose project — it is bind-mounted at
   `/opt/data`):

   ```yaml
   mcp_servers:
     open-dream-rsi:
       command: "python3"
       args: ["-m", "open_dream_rsi", "mcp",
              "--tasks", "/opt/data/dream-rsi/tasks.json",
              "--memory", "/opt/data/dream-rsi/.dream_rsi"]
       timeout: 300
       env:
         # the container has no goose config, so point the dreamer straight
         # at your OpenAI-compatible endpoint (vLLM/Ollama/proxy):
         OPENAI_BASE_URL: "http://YOUR-VLLM-HOST:8000/v1"
         OPENAI_API_KEY: "***"
   ```

   Paths under `/opt/data` persist with the Hermes data volume — the dreamer
   remembers everything across rebuilds. `OPENAI_API_KEY` may be any token
   for vLLM/Ollama; the zero-config chain would otherwise fall back to
   localhost:8000, which does not exist inside a bridge network.
4. Restart the container. Verify without a chat session (inside the
   container or via a throwaway `docker compose run --rm hermes bash`):
   `python3 -c "import open_dream_rsi; print('ok')"` and the JSON-RPC
   handshake (`printf … | python3 -m open_dream_rsi mcp`, see Troubleshooting
   elsewhere in this doc).

### Codex CLI

```bash
codex mcp add open-dream-rsi -- python3 -m open_dream_rsi mcp \
  --tasks /ABSOLUTE/PATH/open-dream-rsi/tasks.json \
  --memory /ABSOLUTE/PATH/open-dream-rsi/.dream_rsi
```

(equivalently `[mcp_servers.open-dream-rsi]` with `command`/`args` in
`~/.codex/config.toml`). Verify with `codex mcp list`.

### Claude Code

Install as a plugin from this repository — Claude Code prompts for the two
paths when the plugin is enabled, and no `pip install` is needed (the
manifest runs the bundled stdlib-only checkout through `PYTHONPATH`;
a system `python3` (3.10+) on PATH is the only requirement):

```bash
claude plugin marketplace add patrykorwat/open-dream-rsi
claude plugin install open-dream-rsi@open-dream-rsi
```

Inside a session the same works as `/plugin marketplace add …` and
`/plugin install …`. The manifest's version is synced with the release
version (`scripts/check_versions.py` watches `.claude-plugin/plugin.json`).
The server resolves its brain the zero-configuration way (env → local goose
config → `http://127.0.0.1:8000/v1`).

Manual route — one MCP server from a pip-installed package:

```bash
claude mcp add open-dream-rsi -- odr mcp \
  --tasks /ABSOLUTE/PATH/tasks.json \
  --memory /ABSOLUTE/PATH/.dream_rsi
```

(`odr` is the console script installed with the package; on a bare checkout
use `python3 -m open_dream_rsi` instead.)

Check with `/mcp` inside a session.

**Session import:** Claude Code transcripts under `~/.claude/projects` are
first-class curator evidence — `odr --memory .dream_rsi dream --source
claude` runs the whole cycle on your existing history (redacted, staged,
gate-gated like every other source).

### Cursor (without MCP — hooks and files)

Cursor installs where **MCP servers are blocked by policy** can still run
the whole Dream-RSI loop: hooks and rules are a separate surface, and
everything the loop needs travels through files (`tasks.json`, the memory
dir), never through MCP.

One command wires it (run inside the open-dream-rsi checkout):

```bash
python plugins/cursor/install.py
# options: --memory D:/odr-mem  --provider cursor-agent (default) | openai | local | goose
#          --env-file D:/odr/env.list   (openai/local only: OPENAI_BASE_URL / OPENAI_API_KEY ...)
```

The installer merges three entries into `~/.cursor/hooks.json` (existing
entries preserved; it never touches `mcp.json`), writes an agent rule to
`<repo>/.cursor/rules/odr-no-mcp.mdc`, and creates the memory dir with an
empty `tasks.json`. Restart Cursor afterwards.

What runs where:

| Moment | What happens |
|---|---|
| A tool call fails the same way twice | `odr_sentinel_hook.py` (postToolUse / postToolUseFailure) injects a `[Sentinel]` recurrence note as additional context |
| A session ends | `odr_dream_hook.py` detaches `python -m open_dream_rsi --memory <mem> dream --source cursor` |
| You queue work | `python -m open_dream_rsi task add --id <id> --category <project> --prompt "<task>" --criteria "<done when...>"` (writes `<memory>/tasks.json`; `task list` shows the queue) |
| The dream cycle runs | it imports Cursor chats as evidence (state.vscdb is read directly — no MCP), consumes the queue, runs the lesson gate, publishes earned knowledge; log: `<memory>/dream.log` |

The dreamer's LLM is any OpenAI-compatible endpoint: pass
`--provider/--model` (plus `--env-file` for `OPENAI_BASE_URL` /
`OPENAI_API_KEY` / `ODR_LLM_MODEL` — keys stay out of the command line), or
export the variables if Cursor sessions inherit your shell profile.
Cadence and budget are decided inside the `dream` module, so firing on
every session end is safe.

When MCP **is** allowed and you want the dreamer on Cursor's own models:
Cursor has no public OpenAI-compatible chat endpoint (a `crsr_...` key is
the Cloud Agents API; `api2.cursor.sh` is the IDE's internal gRPC backend).
Since 0.3.0 the zero-proxy route is the `cursor-agent` provider: the
dreamer runs the headless Cursor CLI (`cursor-agent -p --force
--output-format text`) on the user's own login — no API key, no proxy; the
model is whatever the user's Cursor uses (`--model` / `ODR_LLM_MODEL` to
pin). A machine without the CLI is safe: the dream defers those sessions
(never burns evidence). Install the CLI:
`irm 'https://cursor.com/install?win32=true' | iex` (Windows) or
`curl https://cursor.com/install -fsS | bash` (macOS/Linux/WSL); verify
with `cursor-agent --version` in a fresh terminal. The older local-proxy
route remains:

```bash
npm i -g cursor-openai-api
cursor-openai-api login && cursor-openai-api serve   # /v1 on :3000
export ODR_LLM_PRESET=cursor                          # preset default: :3000
```

Feeding Cursor chats in by hand instead of the hook: `sessions export
--source cursor` (see the README's "Importing real sessions" section).
The live database is write-ahead-logged — quit Cursor or copy
`state.vscdb`+`-wal`+`-shm` together and pass the copy via `--db` for a
complete snapshot.

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

### DeepSeek Harness

[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (`dsh`)
runs external MCP servers through its first-party bridge
(`@deepseek-ai/dsh-mcp-client`). The bundle is published on npm:

```bash
dsh plugin --profile <name> add dsh-open-dream-rsi
```

The tools appear as `mcp__open_dream_rsi__<tool>` (`odr_status`,
`odr_recipes`, `odr_lessons`, `odr_add_task`, `odr_run_once`, `odr_dream`).
The bundle ships the Python server inside the package — only `python3`
(3.10+) on PATH is required, no pip install. Bundle defaults point
`--tasks`/`--memory` at `tasks.json`/`.dream_rsi`
beside the directory `dsh` launches from; override the
`mcp-open-dream-rsi` row from your profile's `cordis.patch.yml` (same id).
DSH scrubs the spawned child's environment (names matching
`KEY|PASSWORD|SECRET|TOKEN` and all `DSH_*`). The bundle's bridge
(`childEnv()`) re-passes only LLM vars that are **actually set** — an empty
`ODR_LLM_PRESET=""` never reaches the child, because older ODR releases fed
it straight into provider selection and died with `Unknown preset ''` —
and, when the harness process has no `OPENAI_BASE_URL`, points the child at
its own in-process OpenAI-compatible proxy: every dreamer LLM call is then
served by the model the harness is already configured with
(`ctx.llm.stream`), no API keys or presets needed. A real `OPENAI_BASE_URL`
in the harness env bypasses the proxy (direct mode). The hand-rolled
overlay re-passes `OPENAI_*`/`ODR_LLM_*` via `process.env.VAR || undefined`
and lets ODR resolve an endpoint itself (env, a preset like
`ODR_LLM_PRESET=local`, or the `http://127.0.0.1:8000/v1` auto-detect).
`toolCallTimeoutMs: 1800000` replaces the 60 s
default because `odr_run_once`/`odr_dream` cycles run for many minutes.

Prefer a hand-rolled overlay?
`dsh --patch /ABS/PATH/TO/open-dream-rsi/plugins/deepseek_harness/odr.cordis.yml web`
inserts the same row with explicit absolute paths (a remote Streamable-HTTP
variant, `odr-http.cordis.yml`, points at the `mcp --http` endpoint — the
URL is a credential, same caveat as the Cowork section above). Full
walkthrough:
[plugins/deepseek_harness/README.md](../plugins/deepseek_harness/README.md).

## Troubleshooting

- **Server shows "failed/unknown" in the harness** — run the command manually
  in the same directory; any traceback you see there is the real error
  (usually `open_dream_rsi` not importable → re-run `pip install -e .`).
- **`odr_run_once` is slow / 0 solved** — it is making real LLM calls; check
  `OPENAI_BASE_URL` (the served model is auto-detected via `/v1/models`
  unless `ODR_LLM_MODEL` is pinned) and try `provider: "mock"` for a key-free
  smoke test of the plumbing.
- **Tool call denied** — Goose prompts per tool the first time; approve it.
- **Tools exist but the model never calls them** — by design no model
  self-initiates unfamiliar tools: trigger via recipe `instructions:`, a
  hook, or an explicit ask ("call the odr_status tool"). In the **desktop
  app**, also check the per-session extensions picker (enabled in
  config.yaml ≠ active in this chat) and do a full Cmd+Q restart after any
  config edit — re-run `./scripts/odr_goose_setup.sh --check` to confirm
  the block is still in place.
