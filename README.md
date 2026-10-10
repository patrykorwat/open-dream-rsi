# Open Dream-RSI

> **📄 Preprint:** *Open Dream-RSI: An Open-Source Library for Recursive
> Self-Improvement Around a Frozen LLM, with Replay-Gated Learned Policies and
> a Curated Knowledge Base* — P. Orwat, 2026.
> [**PDF**](paper/main.pdf) · [**LaTeX source**](paper/main.tex)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-321%20passing-brightgreen)](tests/)

## What is this?

Every AI agent pays the same tax over and over: it fails a task, works out
what went wrong, and then throws that understanding away when the session
ends. Open Dream-RSI keeps what was learned — **without touching the model**.

The LLM stays **frozen**. What improves is everything *around* it:

- the **exploration policy** (even re-written as Python code by the LLM
  itself, gated by replay before it is trusted),
- a library of **verified solutions** (recipes) reused as warm starts,
- **curated lessons** distilled from failures — every one of them promoted
  only after it *measurably* helps on replay,
- and the **world model** the policy is scored against: a Discovery Tree of
  every attempt, enriched with replay-safe error-class facts by the Sentinel
  engine.

Between runs the system "dreams": it replays thousands of strategies over
the recorded history in-process, at **zero API cost**, and keeps only what
beats the incumbent. An open implementation of *Recursive Self-Improvement
through Evolving Worlds* (Dream-RSI, Zheng et al.,
[arXiv:2609.14858](https://arxiv.org/abs/2609.14858)).

**Zero runtime dependencies** — Python 3.10+ stdlib only, ~10.2k lines of
library code, ~5.1k lines of tests, no build step.

**Evaluation policy:** this project makes performance claims on exactly
**one benchmark — goose × TravelPlanner**, public and scored by the
benchmark's own evaluators ([below](#benchmark-goose--travelplanner-public-externally-scored)).
The scripted suites (`bench`, `bench-policy`) are **deterministic
self-checks** — behavioural regression contracts for the machinery — and
carry no model-quality claim.

---

## How it works

```
                        ┌───────────────┐
                        │  Frozen LLM   │
                        └──────┬────────┘
                               │ proposals
              online loop      ▼      offline dream
        ┌──────────── DiscoveryTree ──────────── ReplaySimulator ──────────┐
        │                     │                          │                 │
        │              node.sentinel              counterfactual          │
        │              (Sentinel facts,           rollout scoring         │
        │               replay-safe)                    │                 │
        │                     │                         ▼                 │
        │                     └────→ Policy improvement ────→ next cycle  │
        │                                                                   │
   SENTINEL MODE (tool-execution layer)                                     │
        │          │             │                                          │
   error-class   nudge       permute-gate ──→ block tool calls             │
   ledger        (once/session)              past 80% of budget ───────────┘
```

1. **Online execution** — the agent attempts each task with the real LLM and
   a sandboxed verifier, appending every attempt to a per-task
   *Discovery Tree* (code, test feedback, score, one-line plan, and — when a
   Sentinel engine is wired — structured error-class facts and an explicit
   termination reason).
2. **Offline dreaming** — instead of paying for real-world rollouts, the
   agent "dreams" over the recorded history: thousands of strategy variants
   are scored in an in-process replay simulator at zero external-call cost.
   The best parameters, programs and recipes are persisted and steer the
   next cycle.

Beyond parameter-level dreaming, the loop closes section 3 of the paper
("dreaming with code"): each cycle the LLM may **rewrite the exploration
policy itself** as a small Python program, `choose_action(frontier, step)`.
Candidates are statically validated (AST gate), executed only in a hardened
subprocess sandbox, and scored by **counterfactual replay rollout** on the
recorded tree. A candidate replaces the incumbent only on evidence — and
the incumbent is re-scored on the *current* tree, not trusted at its stored
score. A crashing or cheating policy can never break the loop: expansion
falls back to the greedy baseline.

---

## Quick start

```bash
git clone https://github.com/patrykorwat/open-dream-rsi.git
cd open-dream-rsi
pip install -e .
```

**Point it at a model — or don't.** Any OpenAI-compatible endpoint works;
keys are read only from the environment. The *model name is
auto-detected* from the endpoint (`GET /v1/models`), so pointing at a local
vLLM/Ollama needs only the base URL:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1   # vLLM / Ollama / any gateway
export OPENAI_API_KEY=***                         # any token for local servers
# no ODR_LLM_MODEL needed: the first model the endpoint serves is used.
# Pin ODR_LLM_MODEL only to select among several served models.
```

Queue a task in `tasks.json`:

```json
[{
  "task_id": "add1",
  "category": "math",
  "prompt": "Implement add(a, b) returning the sum.",
  "tests": [{"call": "add(2, 3)", "expected": 5}],
  "max_attempts": 3
}]
```

Run the supervisor — once, as a daemon, or as the recommended automated
`dream` cycle (one command that imports session evidence, runs the promotion
gate when the evidence justifies it, and publishes earned lessons as skills):

```bash
python -m open_dream_rsi --memory .dream_rsi dream \
    --sessions ~/.hermes/state.db --skills-out ~/.hermes/skills/odr-curated
python -m open_dream_rsi loop --tasks tasks.json --interval 300   # daemon mode
python -m open_dream_rsi loop --tasks tasks.json --once           # single cycle
python -m open_dream_rsi status                                   # what it learned
```

No key? Everything deterministic still runs: `bench`, `bench-policy`,
`gate-replay` and the dashboard all use a scripted mock client. A full
in-process walkthrough with a mock OpenAI server: `examples/live_loop_demo.py`.

Don't want the CLI at all? **The same loop runs as an MCP server inside your
existing coding agent** — that is the fastest way in:

```bash
python3 -m open_dream_rsi mcp --tasks ./tasks.json --memory ./.dream_rsi
```

…see the next section for the exact block for your agent.

---

## Plug in your favourite agent

The loop speaks **MCP** (Model Context Protocol). One config block and your
everyday agent can queue tasks for the dreamer, pull back verified recipes
and consult curated lessons — **zero extra LLM setup**: the dreamer resolves
its own brain (env → local goose config → localhost vLLM), and asks the
endpoint itself which model to use.

| Tool | What it does |
|---|---|
| `odr_status` | what the loop has learned (policies, recipes, events) |
| `odr_recipes` | best **verified** solution for a category — use as warm start |
| `odr_lessons` | curated failure lessons for a category / search query |
| `odr_add_task` | queue a task (prompt + tests, or prompt + criteria) — test-less tasks are completion-judged |
| `odr_run_once` | run one improvement cycle now (bounded API budget) |
| `odr_dream` | the automated cycle: import host session evidence, distill curated lessons, run the promotion gate, publish skills |

Tools are registered prefixed per client (e.g. Hermes
`mcp_open_dream_rsi_odr_run_once`). **No model calls tools on its own
initiative** — trigger them via a recipe's `instructions:`, a hook, or an
explicit ask ("check odr_recipes before you try again"). A tip that makes
the loop proactive: add to your project `AGENTS.md`:

```markdown
## Self-improvement loop (MCP: open-dream-rsi)
- Before implementing a self-contained Python utility, call `odr_recipes`
  and `odr_lessons` for its category; reuse a verified recipe verbatim.
- When you finish a hard, testable function, queue it with `odr_add_task`
  (task_id = function name, tests = your test cases) so the loop can
  dream over it offline.
```

### OpenCode

Create (or merge into) `opencode.json` **in the project root where you
work**:

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

### Goose

Recommended — the script wires everything (diagnoses the install, resolves
your goose provider, optionally starts the model-borrowing proxy, and
writes the extension block into `~/.config/goose/config.yaml` itself):

```bash
./scripts/odr_goose_setup.sh            # diagnose + configure goose + start proxy
./scripts/odr_goose_setup.sh --check    # diagnose only, change nothing
./scripts/odr_goose_setup.sh --no-proxy # skip the proxy (use provider='goose')
```

Or add the extension manually to `~/.config/goose/config.yaml`:

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
needs via `envs:` — it will not inherit your export'ed `OPENAI_*`. After
any config edit: fully quit goose (the desktop app caches config at
startup), relaunch, and activate `open-dream-rsi` in the session's
extensions picker. One-shot alternative (no config edit):
`goose session --with-extension "python3 -m open_dream_rsi mcp --tasks ./tasks.json --memory ./.dream_rsi"`.

Two first-class ways to run the dreamer on **exactly the model goose uses**
(no second API key): `provider: "goose"` reads `~/.config/goose/config.yaml`
itself (CLI and desktop dialects, `custom_providers/*.json`, `secrets.yaml`,
macOS keychain), and `python3 -m open_dream_rsi proxy --port 8799` exposes
goose's brain as a plain OpenAI-compatible endpoint — details in
[docs/integrations.md](docs/integrations.md).

### Hermes Agent

Add under `mcp_servers` in `~/.hermes/config.yaml` (or via the dashboard's
MCP catalog):

```yaml
mcp_servers:
  open-dream-rsi:
    command: "python3"
    args: ["-m", "open_dream_rsi", "mcp",
           "--tasks", "/ABS/PATH/open-dream-rsi/tasks.json",
           "--memory", "/ABS/PATH/open-dream-rsi/.dream_rsi"]
    timeout: 300
    env:
      OPENAI_BASE_URL: "http://YOUR-VLLM-HOST:8000/v1"
      OPENAI_API_KEY: "***"        # any token for local servers
```

Restart Hermes; tools appear as `mcp_open_dream_rsi_*` in every platform
toolset. Hermes is also the one host with MCP **sampling** support, if you
ever want the dreamer to borrow the host's model through the protocol
(today ODR resolves the endpoint itself instead).

**Hermes in Docker:** mount the repo read-only and export `PYTHONPATH`
(no pip layer in the image), then add the server block with an explicit
`env:` (the container has no goose config):

```dockerfile
ENV PYTHONPATH=/opt/open-dream-rsi
```

```yaml
# compose.yaml -> services.hermes
volumes:
  - /home/USER/git/open-dream-rsi:/opt/open-dream-rsi:ro
  - ./data:/opt/data
```

Point `--tasks`/`--memory` at paths under the mounted data volume (e.g.
`/opt/data/dream-rsi/...`) so memory survives rebuilds. Updating = `git
pull` on the host + restart the container — no image rebuild.

### Codex CLI

```bash
codex mcp add open-dream-rsi -- python3 -m open_dream_rsi mcp \
  --tasks /ABS/PATH/tasks.json --memory /ABS/PATH/.dream_rsi
```

Verify with `codex mcp list` (or `[mcp_servers.open-dream-rsi]` with
`command`/`args` in `~/.codex/config.toml`).

### Claude Code

```bash
claude mcp add open-dream-rsi -- python3 -m open_dream_rsi mcp \
  --tasks /ABS/PATH/tasks.json --memory /ABS/PATH/.dream_rsi
```

Check with `/mcp` inside a session.

### Claude Cowork / claude.ai (remote only)

Cowork connects to **remote** MCP URLs brokered from Anthropic's cloud.
Serve HTTP, tunnel, add the URL as a custom connector:

```bash
python3 -m open_dream_rsi mcp --http --host 0.0.0.0 --port 8800 \
  --tasks /ABS/PATH/tasks.json --memory /ABS/PATH/.dream_rsi
# tunnel 8800 (tailscale funnel / cloudflared) -> Settings -> Connectors -> Add custom
```

The endpoint implements the stateless Streamable-HTTP profile (`POST /mcp`,
one JSON-RPC message per request; `GET /health`). **Treat the URL as a
credential**: whoever reaches it can queue tasks and spend your LLM budget.

### DeepSeek Harness

[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (`dsh`)
runs external MCP servers through its first-party bridge
(`@deepseek-ai/dsh-mcp-client`) — the adapter is a Cordis overlay, no
extra code:

```bash
dsh --patch /ABS/PATH/TO/open-dream-rsi/plugins/deepseek_harness/odr.cordis.yml web
```

The overlay starts `python3 -m open_dream_rsi mcp --tasks … --memory …` as
a stdio child and the tools appear as `mcp__open_dream_rsi__<tool>`. DSH
spawns the child with a scrubbed environment (names matching
`KEY|PASSWORD|SECRET|TOKEN` and all `DSH_*`), so the overlay re-passes
`OPENAI_BASE_URL` / `OPENAI_API_KEY` / `ODR_LLM_*` from the harness
process's env explicitly — without them the dreamer falls back to
`http://127.0.0.1:8000/v1` with model auto-detection (fine for a local
vLLM/Ollama, needs no key). `toolCallTimeoutMs: 1800000` replaces the 60 s
default because `odr_run_once`/`odr_dream` cycles run for many minutes.
Edit the two `/ABSOLUTE/PATH/...` placeholders to your project first. To
persist the layer, merge the entry into
`$DSH_HOME/profiles/web/cordis.patch.yml` (do not overwrite that file — it
may already hold unrelated patches); verify with
`dsh --patch … --dump-config`. A remote variant
(`odr-http.cordis.yml`, Streamable HTTP) points at the
`mcp --http` endpoint — the URL is a credential, same caveat as Cowork
below. Full walkthrough: [plugins/deepseek_harness/README.md](plugins/deepseek_harness/README.md).

### Zero configuration (all of them)

`odr_run_once` resolves the brain automatically, in this order:

1. `OPENAI_BASE_URL` / `OPENAI_API_KEY` in the server's environment,
2. **your local goose config** (`~/.config/goose`: `active_provider` +
   `providers:` block + `custom_providers/*.json` + `secrets.yaml` + macOS
   keychain) — read from disk, so it survives the env scrubbing that
   Hermes/Codex/Claude Code apply to spawned servers,
3. `http://127.0.0.1:8000/v1` (vLLM/Ollama default).

The served model is auto-detected (`GET /v1/models`) unless pinned.
Self-hosted endpoints run with hidden reasoning disabled by default
(`chat_template_kwargs.enable_thinking=false` — a thinking-on Qwen otherwise
burns the whole completion budget); endpoints that reject the flag get a
clean automatic retry without it. An empty task queue is a benign no-op
with a hint, never an error.

Full per-host instructions (including the launchd keep-alive for the goose
proxy and troubleshooting): **[docs/integrations.md](docs/integrations.md)**.

---

## Providers

The agent talks to any **OpenAI-compatible** `POST {base_url}/chat/completions`
server. API keys are read **only** from environment variables.

| Provider | Preset | Base URL | Key env var |
|---|---|---|---|
| OpenAI | `openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| Cursor Models API | `cursor` | `https://api2.cursor.sh` | `CURSOR_API_KEY` |
| Local (vLLM / Ollama / LM Studio) | `local` | `http://127.0.0.1:8000/v1` | `OPENAI_API_KEY` |
| Borrowed from goose config | `goose` | resolved from `~/.config/goose` | — |

Precedence is always: explicit overrides → environment → **endpoint model
discovery** (first id from `GET /v1/models`) → preset defaults. A redeployed
local server is picked up automatically — and a pinned model the endpoint no
longer serves self-heals: the client re-discovers once and retries instead
of failing the whole run. Self-hosted reasoning endpoints get
`enable_thinking: false` automatically; an HTTP 400 rejection of that flag
triggers a self-heal retry.

Library-level use (no CLI):

```python
from open_dream_rsi import (DiscoveryTree, DreamEngine, ReplaySimulator,
                            DreamAgent, LLMConfig, OpenAICompatibleClient)

client = OpenAICompatibleClient(LLMConfig.from_preset("local"))
tree = DiscoveryTree()
dreamer = DreamEngine(simulator=ReplaySimulator(tree))
dreamer.run_offline_optimization(iterations=100)
agent = DreamAgent(dreamer=dreamer, client=client)
```

---

## What persists between runs (`--memory` dir)

| Artifact | File | Effect |
|---|---|---|
| Dreamed policy parameters | `policies.json` | next cycle starts from the last one's optimum |
| Promoted policy programs | `policy_codes.json` | LLM-written `choose_action` steers expansion |
| Recipe library | `recipes.json` | verified solutions replayed as warm starts |
| Knowledge base | `lessons.json` | curated, replay-gated lessons (declarative text) |
| Discovery trees | `trees/` | offline dreaming always has history; nodes optionally carry replay-safe Sentinel facts |
| Audit log | `events.jsonl` | append-only record of every decision |

Switches worth knowing: `--no-policy-code`, `--no-knowledge`, `--no-thoughts`,
`--no-judge` ablate the four optional machinery layers; `--budget N` caps API
calls per cycle (dreaming stays free).

---

## Automated dreaming: when does the loop actually run?

The seam is `dream_once()` (CLI `odr dream`, MCP `odr_dream`, or a host
hook). **You do not set a schedule — the module decides the tier per call**,
so firing it after every task is safe and costs nothing extra when there is
nothing new to learn:

| Tier | Triggered when | What runs |
|---|---|---|
| **maintenance** | quiet call (no new evidence, recent last dream) | ingest scan (cheap), ALM stale/supersede/archive/GC, skills republish — **zero LLM** |
| **full** | `min_new_evidence` (default 3) new failure-shaped sessions, OR `max_age_seconds` (default 6 h) since the last full dream | + online attempts, dreaming, paired-replay promotion gate (`budget`, default 20 API calls) |

A busy day dreams a few times, a quiet day once, an idle day never;
concurrent calls collapse on a single-instance lock. **Probes are queued
automatically**: every failure-shaped session appends its *own request* to
`tasks.json` as a `probe:<session>` task (`max_attempts=1`, appended after
manual tasks), so the promotion gate never depends on hand-written task
files — the evidence you already produced is the counterparty.

For a mass install the whole thing is two artifacts, no host glue:

```bash
pip install open-dream-rsi                       # library + CLI + MCP server
cp -r plugins/hermes_odr_trigger $HERMES_HOME/plugins/odr-trigger
hermes plugins enable odr-trigger                # restart to load the hook
# optional overrides:
hermes config set plugins.entries.odr-trigger.settings.provider local
hermes config set plugins.entries.odr-trigger.settings.base_url http://.../v1
```

---

## Benchmark: goose × TravelPlanner (public, externally scored)

This is the **only benchmark** in this project — everything else below the
heading line is machinery self-check. A real agent (goose v1.53) plans real
itineraries against the official **TravelPlanner** offline database
(`osunlp/TravelPlanner`) over a stdlib MCP sandbox, scored by the benchmark's
*own* commonsense and hard-constraint evaluators (nothing in the scoring path
is ours), on the model `Qwen3.8-Flash-Next` served by vLLM. Same endpoint and
model across every study in this repo and the paper.

Frozen protocol: iterations only on `train.csv` (45 tasks); one preregistered
eval pass on official validation rows 102–181 (79 tasks, 60 hard),
configuration unchanged from the dev split; official test split never
touched. Arms interleaved task-by-task per pass (paired, McNemar-ready).
The gate arm is *choice removal*, not persuasion: past 36 of 45 tool calls
the sandbox refuses to serve (all text channels off).

| split | arm | delivered plans | commonsense pass | hard pass | final pass |
|---|---|---|---|---|---|
| train (45) | cold | 15/45 (33%) | 4 | 3 | 6.7% |
| train (45) | gate | 41/45 (91%) | 18 | 9 | 20.0% |
| **eval (79)** | cold | 18/79 (23%) | 12 | 6 | 7.6% |
| **eval (79)** | gate | **74/79 (94%)** | 40 | 17 | **21.5%** |

McNemar on paired delivery: eval 56↔0 discordant, p=1.4e-17 (train 28↔2,
p=4.3e-7).

![goose × TravelPlanner: delivery + failure anatomy](docs/figures/fig_tp.png)

Why it works: **every single undelivered cold episode (91/91 across both
splits) died pinned at the harness call cap with the sandbox data already
in hand** — the binding failure under budget pressure is *termination*, not
error recovery, and three text-in-context channels measured earlier
(pull tool, Stop-hook note, budget nudge) converted zero of those episodes.
Removing the option to keep exploring is the only lever that moved delivery.
Honest caveats: the *choice* of gate over persuasion was made knowing
results on the earlier 101-task validation pool (selection era, archived —
the eval pass is a preregistered confirmation, not a first exposure); the
gate's 5 residual eval failures are episodes that called through the
refusals to the cap anyway. Everything behind these numbers is in this
directory: `benchmarks/travelplanner/` — the sandbox MCP server, the
episode runner, the official-evaluator wrapper, the exact split CSVs and
the raw per-episode records. `python3 score.py results/submissions/…`
reproduces the table above to the task from the committed records; the
directory README documents prerequisites and the one-command rerun. The
figure re-renders from `fixtures/tp_v3_summary.json` via
`paper/make_figures.py`. Paper §Evaluation.

---

## Machinery self-checks (deterministic, key-free)

Not benchmarks — behavioural regression contracts: a scripted solver makes
the model arm a constant, so every delta is attributable to the library's
gates, sandboxes, promotion and retrieval logic. They exist so a broken
mechanism fails CI instead of silently mutating a published number, and they
re-run from a fresh clone in seconds.

### `bench` — the dreaming loop against a cold baseline

| arm | solves | API calls | calls / task·cycle | dream its |
|---|---|---|---|---|
| cold_baseline (fresh memory each cycle) | 50 | 150 | 3.0 | 0 |
| dream_rsi_loop (policies + recipes + dreaming) | 50 | 69 | 1.38 | 3000 |

**54% fewer API calls at equal solve quality** (re-measured on this commit:
identical to the published table).

```bash
python -m open_dream_rsi bench --cycles 10 --markdown
```

### `bench-policy` — decoy traps (exploration contracts)

Solve-rate self-checks saturate on easy suites and cannot tell good
exploration from luck, so `bench-policy` hides each fix behind a
*low-scoring* branch (passes 1/3 tests) past a *plausible decoy* (passes 2/3
forever). A score-greedy loop locks onto the decoy and starves; escaping
requires structural exploration. Five arms, one scripted solver, one budget:

```bash
python -m open_dream_rsi bench-policy --cycles 8 --format md
```

| arm | solves | solve rate | API calls | calls / solve | policy calls | curator calls |
|---|---|---|---|---|---|---|
| greedy | 0/120 | 0% | 480 | ∞ | 0 | 0 |
| epsilon_greedy | 29/120 | 24% | 405 | 13.97 | 0 | 0 |
| **evolved_policy** | **92/120** | **77%** | **252** | **2.74** | 27 | 0 |
| **knowledge_curator** | **88/120** | **73%** | 332 | 3.77 | 0 | 34 |
| **thought_guided** | **120/120** | **100%** | **164** | **1.37** | 0 | 0 |

- **evolved_policy** — the loop asks the LLM for a policy program per
  category, validates and replay-scores it, and promotes only on evidence.
  100% solve rate by cycle 5 while spending *fewer total calls* than ε
  wasted on luck.
- **knowledge_curator** — ε-greedy with zero policy calls: every escape
  above the ε baseline came from remembered knowledge (73%, at parity with
  the gated policies at ~80% of ε's calls).
- **thought_guided** (library default) — each attempt records its one-line
  `PLAN:`; expansion leaves a branch as soon as its idea repeats itself.
  With ε forced to 0 and no policy/curator calls it escapes every trap
  **from cycle 1**; the ledger-only ablation (same text in prompts, pick
  disabled) collapses back to the baselines — the win is the expansion
  rule, not prompt length.

![Decoy-trap suite: per-cycle solve rate, five arms](docs/screenshots/odr_policy_bench.png)

### `gate-replay` — would a lesson set have been promoted?

Applies the shipped promotion rule (`lesson_gate_verdict`) to per-task
outcome pairs from any recorded replay harness — no model, no GPU:

```bash
python -m open_dream_rsi gate-replay \
  --baseline cold=/tmp/arms.json?label=cold \
  --compare lessons=/tmp/arms.json?label=warm \
  --key task_id --format md
```

---

## What the loop learned about memory (honest results)

The knowledge curator distils verifier failures into short, validated,
deduplicated **lessons** — pure text, never executed. Live-model replay,
however, proved that fresh lessons are *hazardous by default*:

- Structurally perfect lessons (passed every schema/caps gate) dropped
  solve-rate **18/20 → 6/20** on a greedy decoder: "keep checking"-style
  advice turns greedy decoding into an over-exploration loop.
- The same facts rewritten as **declarative background** (no imperative
  verbs, explicit stop clause): 18/20 — indistinguishable from cold
  (McNemar p=1.0). The hazard is imperative mood, not content.
- Seven further published injection strategies (abstract workflows,
  exemplars, evidence certificates, reactive-on-error notes,
  end-of-prompt position) scored **zero** paired gains against a
  contemporaneous cold baseline; moving safe text to the end of the prompt
  was actively harmful (11/20, p=0.008 — recency makes the model *act* on
  facts that were inert mid-prompt).

The shipped design encodes all of that: lessons are `staging` candidates
that must pass a paired-replay gate (net ≥ 1, **zero** solve→fail
regressions, explicit stop clause) before any prompt sees them; proposal
prompts frame the KB as declarative background; and a per-category
**headroom verdict** (skip curation when recent solve-rate ≥ 0.7 *and*
calls/solve ≤ 5) spends zero curator/gate calls where the cold baseline
already wins. Guidance pays a perturbation tax — the loop only pays it
where there is measurable headroom to spend it against.

---

## Learning vs lifecycle vs history

The repository keeps four planes strictly apart:

| Plane | Owner | Question it answers |
|---|---|---|
| **Learning** | Dream-RSI pipeline (dreamer, replay gate, lesson gate) | Does this artifact *deserve* to be active? |
| **Current artifact state** | `ArtifactLifecycleManager` (`lifecycle.py`) | What is true about this artifact *now*? |
| **Transition history** | append-only `artifacts/events.jsonl` | How did it get here? |
| **Immutable evidence** | Discovery Trees + the event logs | Why do these artifacts exist, and what world produced them? |

The ALM manages the four artifact types (`policy_parameters`,
`policy_program`, `recipe`, `lesson`) through an explicit state machine
(`CANDIDATE → VALIDATED → ACTIVE → STALE/SUPERSEDED/QUARANTINED → ARCHIVED`,
terminal `REJECTED`), with hard rules: activation is *only* an explicit
promotion call carrying the evidence; every state change appends exactly
one immutable event **before** exposing the new state; the materialized
view must always equal `rebuild_artifact_state(events)` (a hand-edited view
is rebuilt from the log, never trusted); merges create NEW artifacts with
lineage and never mutate sources; deletion never prunes the event log.
Discovery Trees and audit events are **evidence, not artifacts** — the ALM
refuses to register them. Inspect the store:
`odr artifacts [--type lesson --state ACTIVE --history]`.

---

## Importing real sessions: Hermes and Cursor (curator's second door)

The loop distils lessons from failures it produces itself. `sessions.py`
opens a second legitimate door: **transcripts from host agents are evidence
you already paid for**. Imports feed the *same* pipeline as loop-collected
failures — structural gate, staging-by-default, paired replay promotion.
Nothing imported is trusted; activation is still earned. Transcripts are
treated as **hostile input**: opened read-only, credential-scrubbed, parsed
defensively — and lessons that never passed the promotion gate **never reach
a prompt**.

### From Hermes sessions

Hermes persists every session to `<HERMES_HOME>/state.db` (SQLite:
`sessions` + `messages`). Export, then distil:

```bash
python -m open_dream_rsi --memory .dream_rsi sessions export \
    --source hermes --db ~/.hermes/state.db --limit 50 \
    --out episodes.jsonl

# inspect what WOULD be distilled — no LLM calls, no writes:
python -m open_dream_rsi --memory .dream_rsi sessions distill \
    episodes.jsonl --dry-run --limit 5

# real run: staging lessons into the KB (per-category = episode cwd name)
python -m open_dream_rsi --memory .dream_rsi sessions distill episodes.jsonl
```

The next `odr loop` cycle gates the staging entries exactly like
loop-collected ones; `odr artifacts --type lesson --history` shows the
lifecycle events behind every activation.

### From Cursor chats

Cursor keeps conversations in `globalStorage/state.vscdb` (Linux:
`~/.config/Cursor/User`, macOS: `~/Library/Application Support/Cursor/User`,
Windows: `%APPDATA%\Cursor\User`), with `workspaceStorage` mapping chats to
project folders for the category. The reader handles the undocumented
layout and the bubble-ordering rules; re-verify after major Cursor
upgrades.

```bash
python -m open_dream_rsi --memory .dream_rsi sessions export \
    --source cursor --out episodes.jsonl --limit 100
python -m open_dream_rsi --memory .dream_rsi sessions distill episodes.jsonl
```

Notes: the live DB is write-ahead-logged — `immutable=1` may miss the
newest chats, so quit Cursor or copy `state.vscdb`+`-wal`+`-shm` together
and point at the copy for a complete snapshot. Chat history contains
everything ever pasted into it: export redacts credential-shaped substrings
by default (`--no-redact` exists for local processing only) — review
`episodes.jsonl` before sharing it.

### Curator lessons as Hermes skills

```bash
python -m open_dream_rsi --memory .dream_rsi skills export \
    --out ~/.hermes/skills/odr-curated
```

Renders only **active** (promotion-gated) lessons into Hermes-style skills —
one directory per category, declarative bullets, never imperative commands
(the measured hazard). The export is one-way and regenerated wholesale:
edit `lessons.json` (or distil again), not the skills.

### Sharing lessons: git-backed stores

`share.py` turns curated knowledge into a portable, verifiable store that
other instances (or other people) can consume — a directory of JSON files,
optionally a git checkout, so GitHub is just storage:

```bash
# one-time: point the store at a repo (clone/pull handled for you)
python -m open_dream_rsi --memory .dream_rsi lessons export \
    --store git@github.com:you/dream-rsi-lessons.git

# on the consumer side — someone else's store becomes your staging:
python -m open_dream_rsi --memory .dream_rsi lessons import \
    --store https://github.com/you/dream-rsi-lessons.git
```

Trust model: **export ships only what earned activation** (staging and
rejected lessons excluded by construction, secrets scrubbed, sha256
manifest); **import lands as staging, never active** (checksum mismatch
aborts the whole import — tampering detected, not trusted; each lesson
still needs *your* paired-replay gate to reach a prompt). A local directory
works identically (`--store ./store`) — git is just the transport.

### Full Hermes wiring (what exists today)

| Path | Direction | What it does |
|---|---|---|
| `odr mcp` (stdio) / `--http` | Hermes → ODR | Hermes calls `odr_run_once` / `odr_dream` / `odr_status` as tools |
| `plugins/hermes_sentinel/` | Hermes → ODR engine | Sentinel observes tool results, annotates, gates |
| `plugins/hermes_odr_trigger/` | Hermes → ODR engine | session-end hook spawns the detached `odr dream` cycle (no shell/cron) |
| `plugins/deepseek_harness/` | DeepSeek Harness (`dsh`) → ODR | Cordis overlay: the ODR MCP server behind `@deepseek-ai/dsh-mcp-client` |
| `odr sessions export/distill` | Hermes → curator | session history becomes staging lessons |
| `odr skills export` | curator → Hermes | active lessons become discovered SKILL.md files |
| `odr dream` | any trigger | one automated cycle: evidence → gate → ALM → skills |

---

## Security model

Candidate code — both task solutions and policy programs — runs in a
hardened subprocess (`open_dream_rsi/sandbox.py`): `python -I`, scrubbed
environment (only `PATH` — API keys never cross the wall), a timeout that
kills the **entire process group**, and POSIX address-space / CPU /
thread-count limits. Policy code additionally passes an AST gate *before*
any spawn: required entry point, no imports, no dunder access, no
`__builtins__` reference (name plus constructed-key indexing is itself an
escape route).

**This is defence in depth, not a container.** The subprocess does not
restrict filesystem or network access, and an AST gate cannot be provably
complete against code that is not statically gated. For hostile task
sources set `ODR_SANDBOX_CMD` to a
[bubblewrap](https://github.com/containers/bubblewrap)/nsjail wrapper
(e.g. `bwrap --unshare-all --die-with-parent --ro-bind / /`) or run the
loop inside a container.

---

## Live dashboard

One command, no dependencies, stdlib http server:

```bash
python -m open_dream_rsi dashboard                   # mock LLM, port 8765
python -m open_dream_rsi dashboard --provider cursor # real model
python -m open_dream_rsi dashboard --host 0.0.0.0    # LAN access
```

Dark single-page UI: KPIs (cycles / solved / API budget / dream iterations),
live event feed, per-task attempt boards with score bars and code diff-downs,
dreamed-policy gauges per category and the learned recipe library.

---

## Positioning

- **The paper this implements:** Dream-RSI
  ([arXiv:2609.14858](https://arxiv.org/abs/2609.14858)); official repo
  [zhengkid/Dream-RSI](https://github.com/zhengkid/Dream-RSI), explainer
  [dream-rsi.com](https://dream-rsi.com/).
- **Peer implementations** (this repo is an independent open one, not the
  first): [TheAstrayDev/dream-rsi-sdk](https://github.com/TheAstrayDev/dream-rsi-sdk)
  (adapter SDK, strict replay; LLM-written policies on their roadmap),
  [robinber/dream-rsi-spark](https://github.com/robinber/dream-rsi-spark)
  (local Qwen + CUDA kernel exploration), plus skill variants
  ([hermes-dream-rsi](https://github.com/lesterppo/hermes-dream-rsi),
  [dream-rsi-skill](https://github.com/Harkit2004/dream-rsi-skill),
  [Pi-RSI](https://github.com/mailbobg/Pi-RSI)).
  What differentiates this repo: an **always-on autonomous supervisor**
  with budget guards, **LLM-written exploration policies** gated by
  counterfactual replay (section 3 closed), a **public-benchmark
  measurement** (goose × TravelPlanner) rather than self-scored claims, a
  deterministic decoy-trap self-check suite, persistent cross-run memory,
  and a **negative-results section on memory** that the loop itself
  enforces.
- **OpenRSI / Frontis-MA1** ([FrontisAI/OpenRSI](https://github.com/FrontisAI/OpenRSI))
  is a different layer: they post-train *weights*; this library optimises
  the *exploration policy around a frozen model* — no training, no GPU, no
  weight access. Complementary: their trained improver could be the frozen
  LLM behind our client.

---

## Module architecture

| Module | Role |
|---|---|
| `core.tree` | DiscoveryTree — hypotheses, results, actions, thoughts, Sentinel facts, termination reason |
| `core.simulator` | replay simulation without touching the environment |
| `core.dreamer` | offline optimisation over recorded history |
| `core.agent` | LLM agent driven by the rewarded policy |
| `core.policygen` | policy generation, AST gate, rollout scoring, promotion |
| `core.curator` | lesson distillation, headroom verdict, gate verdict |
| `core.judge` | completion judge for test-less (`criteria`) tasks |
| `loop` | `AutoRSIRuntime` — the autonomous supervisor (opt-in `sentinel_engine`) |
| `dream` | `dream_once` — automated cadence seam: evidence ingest, tier decision, gate + ALM, skills publish; lock-guarded |
| `lifecycle` | `ArtifactLifecycleManager` — state machine, event-sourced history, lineage, GC |
| `sessions` | Hermes `state.db` / Cursor `state.vscdb` readers, redaction, lessons→skills bridge |
| `memory` | `DreamMemory` — policies, recipes, lessons, trees, events |
| `sandbox` | `run_isolated` — the single untrusted-code execution boundary |
| `tools` | `CodeVerifier` — sandboxed verification of candidate solutions |
| `llm` | OpenAI-compatible client (stdlib `urllib` transport; model discovery + 404 self-heal) |
| `cli` | `loop / status / dashboard / bench / bench-policy / gate-replay / mcp / proxy / sentinel` |
| `mcp`, `mcp_server` | MCP stdio server for loop tools and sentinel checks |
| `proxy` | loopback OpenAI-compatible proxy borrowing goose's upstream |
| `dashboard` | zero-dependency live web UI |
| `gate_replay` | offline validation of the lesson promotion rule |
| `sentinel` | host-agnostic error-class engine + `SentinelObservation` world contract |
| `utils.goose` | goose config resolver (CLI + desktop dialects, keychain) |
| `plugins/*` | Hermes / Claude Code / goose / DeepSeek Harness adapters |
| `benchmarks/travelplanner` | **the benchmark**: sandbox MCP server, episode runner, official-evaluator wrapper, split CSVs, raw published episode records |

## Tests

```bash
python -m unittest discover -s tests          # or: pytest tests/ -q
```

321 tests (+10 subtests), stdlib-only: benchmark behavioural contracts
(greedy must solve nothing; the gated arm must dominate ε; the rollout must
be prefix-only), sandbox escape regressions, gate semantics, MCP protocol,
goose config dialects, sentinel channel semantics (nudge fires once per
budget, merges with recurrence notes, clean calls count toward the budget;
the permute-gate serves until ceil(0.8·budget) then refuses forever, per
session; the 30-day production replay must reproduce its recorded note count
exactly), model discovery and stale-pin self-heal contracts, and the
structured seam (cross-session counts never reach `to_world_dict()`; the
frontier-feature attribution pattern — beats greedy with metadata, ties
without; the vendored Hermes engine pinned byte-for-byte to upstream by
`test_vendored_engine_matches_upstream`).

## License

MIT — see [LICENSE](LICENSE).
