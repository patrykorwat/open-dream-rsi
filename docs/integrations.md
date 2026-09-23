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
   `~/.config/goose/config.yaml` (`GOOSE_PROVIDER` / `GOOSE_MODEL`),
   `custom_providers/*.json` and `secrets.yaml` — including the macOS
   keychain — and calls that same endpoint itself. The credential stays in
   goose's storage; nothing is duplicated in the extension config.

2. **`proxy` (plain OpenAI-compatible endpoint).** If you want anything
   OpenAI-compatible pointed at goose's brain (not only ODR), start::

       python3 -m open_dream_rsi proxy --port 8799

   and set, in this extension's `envs:` or anywhere else::

       OPENAI_BASE_URL: "http://127.0.0.1:8799/v1"
       OPENAI_API_KEY: "pr..."        # proxy authenticates upstream itself

   The proxy resolves the upstream **per request** (switching models in
   goose takes effect live), pins outgoing calls to `GOOSE_MODEL`
   (`--passthrough-models` opts out), speaks OpenAI and Anthropic upstreams,
   binds to loopback only, and exposes `GET /health` (resolved upstream with
   the key masked) and `GET /v1/models`. Check the resolution without
   serving: `python3 -m open_dream_rsi proxy --print-config`.

## Any other MCP client (Claude Code, Cursor, Zed, …)

Register a **stdio MCP server** with command:

    python3 -m open_dream_rsi mcp

Optional flags/env: `--tasks <file>` (or `ODR_TASKS`), `--memory <dir>` (or
`ODR_MEMORY`). One memory dir per improving instance — share it between the
MCP server and a long-running `python -m open_dream_rsi loop` so the harness
sees exactly what the daemon has learned (that is the recommended setup:
the daemon dreams, the harness queries).

## Troubleshooting

- **Server shows "failed/unknown" in the harness** — run the command manually
  in the same directory; any traceback you see there is the real error
  (usually `open_dream_rsi` not importable → re-run `pip install -e .`).
- **`odr_run_once` is slow / 0 solved** — it is making real LLM calls; check
  `OPENAI_BASE_URL`/`ODR_LLM_MODEL` and try `provider: "mock"` for a key-free
  smoke test of the plumbing.
- **Tool call denied** — Goose prompts per tool the first time; approve it.
