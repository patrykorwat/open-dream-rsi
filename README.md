# Open Dream-RSI

> **📄 Preprint:** *Open Dream-RSI: An Open-Source Library for Recursive
> Self-Improvement Around a Frozen LLM, with Replay-Gated Learned Policies and
> a Curated Knowledge Base* — P. Orwat, 2026.
> [**PDF**](paper/main.pdf) · [**LaTeX source**](paper/main.tex)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-244%20passing-brightgreen)](tests/)

An open, dependency-free implementation of **Dream-RSI** (*Recursive
Self-Improvement through Evolving Worlds*, Zheng et al., Google / Google
DeepMind / UMD, [arXiv:2609.14858](https://arxiv.org/abs/2609.14858)).

The model stays **frozen**. What improves is everything *around* it: the
exploration policy, the verified-solution library, the curated lessons
learned from failures, and — since v0.2 — the **world model itself**: the
error-class Sentinel feeds structured, replay-safe observations onto
discovery-tree nodes, so counterfactual replay can tell *"the policy chose
to stop"* from *"the runtime removed the option"*. The loop optimises its
own problem-solving strategy between runs — persistent memory, replay-gated
promotion, budget guards — and never touches model weights.

**Zero runtime dependencies** (stdlib only), ~6.4k lines of library code,
~3.1k lines of tests, MIT license, installable with no build step.

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

The loop alternates two phases:

1. **Online execution** — the agent attempts each task with the real LLM and
   a sandboxed verifier, appending every attempt to a per-task
   *Discovery Tree* (code, test feedback, score, one-line plan, and — when a
   Sentinel engine is wired — structured error-class facts and an explicit
   termination reason).
2. **Offline dreaming** — instead of paying for real-world rollouts, the agent
   "dreams" over the recorded history: thousands of strategy variants are
   scored in an in-process replay simulator at zero external-call cost. The
   best parameters, programs and recipes are persisted and steer the next cycle.

Beyond parameter-level dreaming, the loop closes section 3 of the paper
("dreaming with code"): each cycle the LLM may **rewrite the exploration
policy itself** as a small Python program, `choose_action(frontier, step)`.
Candidates are statically validated (AST gate), executed only in a hardened
subprocess sandbox, and scored by **counterfactual replay rollout** on the
recorded tree — where each frontier node can now carry Sentinel features
(`sentinel_signature`, `sentinel_repeat`, `sentinel_blocked`), so a policy
can learn to abandon a branch whose failure class already blocks it. A
candidate replaces the incumbent only on evidence — and the incumbent is
re-scored on the *current* tree, not trusted at its stored score. A crashing
or cheating policy can never break the loop: expansion falls back to the
greedy baseline.

---

## Quick start

```bash
git clone https://github.com/patrykorwat/open-dream-rsi.git
cd open-dream-rsi
pip install -e .
```

Queue tasks in `tasks.json`:

```json
[{
  "task_id": "add1",
  "category": "math",
  "prompt": "Implement add(a, b) returning the sum.",
  "tests": [{"call": "add(2, 3)", "expected": 5}],
  "max_attempts": 3
}]
```

Run the supervisor — as a daemon, or as a single cycle from cron:

```bash
export OPENAI_API_KEY=***            # any OpenAI-compatible endpoint
python -m open_dream_rsi loop --tasks tasks.json --interval 300
python -m open_dream_rsi loop --tasks tasks.json --once    # cron-friendly
python -m open_dream_rsi status                             # what it has learned
```

No key? Everything deterministic still runs: `bench`, `bench-policy`,
`gate-replay` and the dashboard all use a scripted mock client.
A full in-process walkthrough with a mock OpenAI server:
`examples/live_loop_demo.py`.

### What persists between runs (`--memory` dir)

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

## Providers

The agent talks to any **OpenAI-compatible** `POST {base_url}/chat/completions`
server. API keys are read **only** from environment variables.

| Provider | Preset | Base URL | Key env var |
|---|---|---|---|
| OpenAI | `openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| Cursor Models API | `cursor` | `https://api2.cursor.sh` | `CURSOR_API_KEY` |
| Local (vLLM / Ollama / LM Studio) | `local` | `http://127.0.0.1:8000/v1` | `OPENAI_API_KEY` |
| Borrowed from goose config | `goose` | resolved from `~/.config/goose` | — |

Precedence is always: explicit overrides → environment → preset defaults.
Self-hosted reasoning endpoints get `enable_thinking: false` automatically
(a thinking-on Qwen otherwise burns the whole completion budget); an HTTP 400
rejection of that flag triggers a self-heal retry.

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

## Plugging into your coding agent (MCP)

The loop speaks **MCP** — stdio for local agents (Goose, Hermes, Codex CLI,
Claude Code, OpenCode, Zed) and Streamable-HTTP (`mcp --http --port 8800`)
for remote connectors (Claude Cowork / claude.ai). One config block and your
everyday agent can queue tasks for the dreamer, pull verified recipes and
consult curated lessons — **zero extra LLM setup**: the dreamer resolves its
own brain (env → local goose config → localhost vLLM).

```bash
python3 -m open_dream_rsi mcp --tasks ./tasks.json --memory ./.dream_rsi
```

Already inside goose and want the dreamer to use **goose's own model** (no
second API key)? `./scripts/odr_goose_setup.sh` diagnoses the install, starts
the loopback model-borrowing proxy (`python -m open_dream_rsi proxy`, port
8799) and writes the extension block into `~/.config/goose/config.yaml` for
you.

Copy-paste instructions for every supported host:
**[docs/integrations.md](docs/integrations.md)**.

---

## Benchmark: goose × TravelPlanner (public, externally scored)

This is the **only benchmark** in this project — everything else below the
heading line is machinery self-check. A real agent (goose v1.53) plans real
itineraries against the official **TravelPlanner** offline database
(`osunlp/TravelPlanner`) over a stdlib MCP sandbox, scored by the benchmark's
*own* commonsense and hard-constraint evaluators (nothing in the scoring path
is ours), on the model `Qwen3.8-Flash-Next` served by vLLM — a
state-of-the-art-class local model. Same endpoint and model across every
study in this repo and the paper.

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

## Sentinel: from prompt channel to world layer

Curation is *epistemic* — it reads outcomes after episodes end and wakes on
a schedule. Measured on a real install's session store (~60k tool messages /
30 days; anonymized fixture in `fixtures/sentinel_audit.json`): a recurring
error class re-appears **within one session** at a median gap of 4.3 minutes,
and half of failing calls are same-class repeats. No schedule wins that race,
so the mechanism lives in the tool-execution layer of the host runtime.

The engine (`open_dream_rsi/sentinel.py`) is host-independent: error-class
fingerprints (URLs/paths/numbers normalized — different arguments, same
failure), a durable ledger, and one **declarative** note per class per
session at a repeat threshold (recurrence facts plus a stop condition,
never a command — the replay arms showed imperative framing measurably
extends loops). Clean calls pay zero prompt tax: the note rides only failing
results. Adapters: Hermes plugin (`plugins/hermes_sentinel/`), Claude Code
hooks (`plugins/claude_code/`), goose Stop-hook (`plugins/goose/`), and plain
stdin-JSON CLI for any command-hook host
(`python -m open_dream_rsi sentinel check`).

**Reactive channels that were measured, in order:** the recurrence note and
the **finalize nudge** (counts *all* tool calls, fires at most once per
session at ~80% of the episode's call budget, rides an ordinary tool result —
never a blocking hook). The TravelPlanner study showed the nudge converts
**0 of 25** cap-dying episodes it triggered: this model family does not stop
because text asks it to. What moved the numbers (table above) is **the
permute-gate — choice removal**: past 80% of the call budget the sandbox
itself refuses to serve (`SENTINEL_GATE_BUDGET` / `SENTINEL_GATE_AT` on the
MCP sandbox server, `open_dream_rsi/mcp_server.py`), every further call
returns `isError` and all persuasion channels stay silent, so answering with
data in hand becomes the only option.

**The structured seam (new).** Inside the library, Sentinel is no longer
only a prompt-annotation channel — it is part of the world model, in a
replay-safe way:

- `SentinelEngine.observe_structured()` returns a frozen `SentinelObservation`
  (signature, error class, in-session and cross-session counts, recurring,
  nudge fired, budget, blocked) instead of prose. `observe()` remains a
  *rendering* of the same facts on the same ledger — the host-facing text
  contract is byte-pinned and unchanged.
- `SentinelObservation.to_world_dict()` is the **only** view allowed into
  `TreeNode.sentinel`: signature, episode-local `repeat_world`, `recurring`,
  `blocked`, budget fraction. Cross-session counters are stripped by
  construction — a fact from another (possibly future) session leaking into
  the historical world would falsify prefix-only counterfactual replay.
- `TreeNode.termination_reason` records a closed `TerminationReason`
  (`completed / policy_stop / tool_failure / budget_gate / sentinel_block /
  unknown`) on the node where an episode stopped: replay can now distinguish
  *the decision-maker declined to continue* from *the runtime removed the
  option* — the exact anatomy the TravelPlanner cold arm died by (91/91 at
  the cap).
- The LLM-written policy sees these as primitive frontier features
  (`sentinel_signature`, `sentinel_repeat`, `sentinel_blocked`) with
  clean-and-unblocked defaults for nodes without metadata — an ON/OFF
  ablation changes **values, not interface**, so cold arms and legacy trees
  cost nothing.
- Wiring is opt-in: `AutoRSIRuntime(sentinel_engine=...)`; the default
  (`None`) leaves archived trees byte-identical, and both self-check suites
  reproduce their published numbers unchanged with the seam merged.

Attribution is pinned by tests (`tests/test_policygen_sentinel.py`,
`tests/test_sentinel_world.py`): the same policy code that reads
`sentinel_blocked` beats greedy on a tree with a blocked trap and **ties**
greedy on the identical tree without metadata — the delta is the feature,
not the code; and a malformed node carrying `cross_session_count` still
cannot leak into replay. A live-model arm consuming the new features is
future work; the seam itself ships tested, the *uplift* claim is not made.

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

The knowledge curator (section 4) distils verifier failures into short,
validated, deduplicated **lessons** — pure text, never executed. Live-model
replay, however, proved that fresh lessons are *hazardous by default*:

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

## Learning vs lifecycle vs history (issue #3: the Artifact Lifecycle Manager)

The repository keeps four planes strictly apart:

| Plane | Owner | Question it answers |
|---|---|---|
| **Learning** | Dream-RSI pipeline (dreamer, replay gate, lesson gate) | Does this artifact *deserve* to be active? |
| **Current artifact state** | `ArtifactLifecycleManager` (`lifecycle.py`) | What is true about this artifact *now* (`ArtifactState`)? |
| **Transition history** | append-only `artifacts/events.jsonl` | How did it get here (`ArtifactTransitionEvent`)? |
| **Immutable evidence** | Discovery Trees + the event logs | Why do these artifacts exist, and what world produced them? |

**Dream-RSI owns learning. ALM owns artifact lifecycle. The event log owns
transition history. Discovery Trees own historical world evidence.**

The ALM manages four artifact types — `policy_parameters`, `policy_program`,
`recipe`, `lesson` — through an explicit state machine
(`CANDIDATE → VALIDATED → ACTIVE → STALE/SUPERSEDED/QUARANTINED → ARCHIVED`,
plus terminal `REJECTED`). Its hard rules:

- activation is *only* an explicit `activate()`/promotion call — never
  "newer, similar or plausible"; the lesson gate and replay promotion remain
  the sole behavioral authorities, the ALM just records their decisions;
- every state-changing method appends exactly one immutable event **before**
  exposing the new state; an invalid transition is never committed;
- the materialized `states.json` view must always equal
  `rebuild_artifact_state(events)` — a hand-edited view is rebuilt from the
  log, never trusted (`verify_materialization()`);
- merge creates a NEW artifact with `supersedes` lineage to every source and
  never mutates the sources; rollback derives a new ACTIVE version instead
  of erasing the promoted version's history;
- pinned artifacts are skipped by automatic staleness and GC; quarantined
  artifacts have no automatic path back to ACTIVE (restore re-enters through
  VALIDATED + successful validation);
- physical deletion is separate from lifecycle state and NEVER prunes the
  event log — deletion itself is an `artifact.deleted` audit event.

Discovery Trees and audit events are **evidence, not artifacts**: the ALM
refuses to register, activate, supersede or merge them; replay keeps
operating on historical trees without mutation. Inspect the store with
`odr artifacts [--type lesson --state ACTIVE --history]`.

---

## Importing real sessions: Hermes and Cursor (curator's second door)

The loop distils lessons from failures it produces itself. `sessions.py`
opens the second legitimate door: **transcripts from host agents are
evidence you already paid for**. Imports feed the *same* pipeline as
loop-collected failures — structural gate, staging-by-default, paired
replay promotion. Nothing imported is trusted; activation is still earned.

```
Hermes state.db  ─┐                                    ┌─> lessons.json
                  ├─> episodes.jsonl ─> odr sessions    │    (staging)
Cursor state.vscdb┘        (redacted)     distill ─────┘         │
                                                                 ▼
                              odr loop (paired-replay gate) → ACTIVE
                                                                 │
                              odr skills export ─> Hermes SKILL.md
```

Two rules make this safe: transcripts are **hostile input** (opened
read-only, credential-scrubbed, schema parsed defensively), and lessons
that never passed the promotion gate **never reach a prompt** — neither a
proposal prompt nor a generated SKILL.md (a skill IS a prompt, and the
v5–v9 measurements say unearned prompts cost solve-rate).

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

The next `odr loop` cycle gates the staging entries against recorded probe
episodes exactly like loop-collected ones; `odr artifacts --type lesson
--history` shows the lifecycle events behind every activation.

### From Cursor chats

Cursor keeps every conversation in one SQLite file (undocumented — the
readers here encode the layout observed in 2026, re-verify after major
upgrades):

| Platform | Cursor user dir |
|---|---|
| macOS | `~/Library/Application Support/Cursor/User` |
| Linux | `~/.config/Cursor/User` |
| Windows | `%APPDATA%\Cursor\User` |

Inside it: `globalStorage/state.vscdb` holds the message bodies in
`cursorDiskKV` (rows `composerData:<id>` = headers whose
`fullConversationHeadersOnly` gives bubble ORDER; `bubbleId:<cid>:<bid>` =
one message each, `type 1` user / `2` assistant, empty `text` = tool-only),
while `workspaceStorage/<hash>/state.vscdb` + `workspace.json` map chats to
project folders (the reader joins them for the `cwd`).

```bash
# read-only export (works while Cursor runs; see the WAL note below)
python -m open_dream_rsi --memory .dream_rsi sessions export \
    --source cursor --out episodes.jsonl --limit 100

python -m open_dream_rsi --memory .dream_rsi sessions distill episodes.jsonl
```

Notes:

* **WAL:** the live DB is write-ahead-logged; `immutable=1` reads the
  committed state, which may miss the newest chats. For a complete
  snapshot, quit Cursor or copy the three files together and point at the
  copy: `cp state.vscdb state.vscdb-wal state.vscdb-shm /tmp/c/ &&
  ... sessions export --source cursor --db /tmp/c/state.vscdb`.
* **Secrets:** chat history contains everything ever pasted into it.
  Export redacts credential-shaped substrings by default; `--no-redact`
  exists only if you are processing locally and want the raw text. Review
  `episodes.jsonl` before sharing it — redaction is best-effort, not a
  guarantee.
* **Ordering:** bubbles are fetched by header order, not key order — out
  of order, a tool call attaches to the wrong prompt and the distilled
  "lesson" describes a world that never happened.

### Curator lessons as Hermes skills

`odr skills export` renders the **active** (promotion-gated) lessons into
Hermes-style skills — one directory per category:

```bash
python -m open_dream_rsi --memory .dream_rsi skills export \
    --out ~/.hermes/skills/odr-curated
```

Each `SKILL.md` carries frontmatter Hermes discovers automatically and
declarative bullets (`[trigger] text`) — never imperative commands (the
measured hazard). Re-run after promotion passes; the export is one-way and
regenerated wholesale, so edit `lessons.json` (or better: distil again),
not the skills. Staging and rejected lessons are excluded by construction;
`--include-staging` exists for audits, not for prompts.

The reverse direction (Hermes → ODR) is the session import above: Hermes
sessions become evidence, never directly-active knowledge.

### Sharing lessons: git-backed stores

`share.py` turns curated knowledge into a **portable, verifiable store**
that other instances (or other people) can consume — the publishing
mechanism the curator layer deliberately never had. A store is a directory
of JSON files, optionally a git checkout, so GitHub is just storage:

```
.dream_rsi/lessons.json  (private runtime view)
      │  export: ACTIVE lessons only · secrets redacted · sha256 manifest
      ▼
<store>/manifest.json                 schema "odr-lessons/v1"
<store>/lessons/<category>.json       portable records + provenance
      ▲  import: checksum verified → structural gate → STAGING
      │
any other .dream_rsi instance / repo / fork
```

Publish your curated knowledge to a repo, then anyone pulls it:

```bash
# one-time: point the store at a repo (clone/pull handled for you,
# cached under <memory>/lesson_stores/)
python -m open_dream_rsi --memory .dream_rsi lessons export \
    --store git@github.com:you/dream-rsi-lessons.git

# later exports pull, commit and push incrementally
python -m open_dream_rsi --memory .dream_rsi lessons export \
    --store git@github.com:you/dream-rsi-lessons.git \
    --message "promotions from oct-week-2"

# on the consumer side — someone else's store becomes your staging:
python -m open_dream_rsi --memory .dream_rsi lessons import \
    --store https://github.com/you/dream-rsi-lessons.git

python -m open_dream_rsi --memory .dream_rsi lessons status \
    --store <dir-or-url>          # manifest + checksum verification
```

A local directory works identically (`--store ./store`), so the mechanism
is not GitHub-specific — git is just the transport. `--no-push` commits
locally for review before publishing; `--category` filters both directions.

Trust model — identical to the session import, because a shared lesson is
the same kind of object:

* **Export ships only what earned activation.** Staging and rejected
  lessons are excluded by construction; internal bookkeeping (keys,
  digests, statuses) never leaves the instance. Every record carries
  provenance (`promoted_by: paired_replay_gate`).
* **Import lands as staging, never active.** Each lesson passes the
  structural gate, registers `CANDIDATE → VALIDATED` in the ALM, and
  still needs *your* paired-replay gate to reach a prompt. Importing
  someone's store shares their knowledge, not their authority.
* **Tampering is detected, not trusted.** `manifest.json` records the
  sha256 of every file; a mismatch aborts the whole import. `lessons
  status` verifies the same checksums read-only.
* **Secrets are scrubbed before anything is published** — the same
  redactor the session export uses (review before pushing anyway).

Git identity for the auto-commit comes from `ODR_GIT_AUTHOR_NAME` /
`ODR_GIT_AUTHOR_EMAIL` (defaults `open-dream-rsi` /
`odr@users.noreply.github.com`); push auth is whatever your git already
uses — the code never handles credentials.

### Full Hermes wiring (what exists today)

| Path | Direction | What it does |
|---|---|---|
| `odr mcp` (stdio) / `--http` | Hermes → ODR | Hermes calls `odr_run_once` / `odr_dream` / `odr_status` as tools |
| `plugins/hermes_sentinel/` | Hermes → ODR engine | Sentinel observes tool results, annotates, gates |
| `plugins/hermes_odr_trigger/` | Hermes → ODR engine | session-end hook fires `dream_once` in-process (no shell/cron) |
| `odr sessions export/distill` | Hermes → curator | session history becomes staging lessons |
| `odr skills export` | curator → Hermes | active lessons become discovered SKILL.md files |
| `odr dream` | any trigger | one automated cycle: evidence → gate → ALM → skills |

### Cadence: when does the world dreamer actually run?

The seam is `dream_once()` (CLI `odr dream`, MCP `odr_dream`, or the Hermes
session-end hook). **You do not set a schedule — the module decides the tier
per call**, so firing it after every task is safe and costs nothing extra
when there is nothing new to learn:

| Tier | Triggered when | What runs |
|---|---|---|
| **maintenance** | quiet call (no new evidence, recent last dream) | ingest scan (cheap), ALM stale/supersede/archive/GC, skills republish — **zero LLM** |
| **full** | `min_new_evidence` (default 3) new failure-shaped sessions, OR `max_age_seconds` (default 6 h) since the last full dream | + online attempts, dreaming, paired-replay promotion gate (`budget`, default 20 API calls) |

So a busy day dreams a few times, a quiet day once, an idle day never —
and concurrent calls collapse on a single-instance lock. The cadence knobs
(`min_new_evidence`, `max_age_seconds`) are arguments, not config sprawl.

**Why not per-session-full-dream?** the promotion gate costs ~4 LLM calls
per candidate lesson per probe; firing it every session would spend the
whole budget re-proving lessons already proven. **Why not cron?** a fixed
schedule is blind to evidence — it dreams on idle days and starves busy
ones. The evidence-justified tier gets both wrongness out.

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
| `lifecycle` | `ArtifactLifecycleManager` — state machine, event-sourced history, lineage, GC |
| `sessions` | Hermes `state.db` / Cursor `state.vscdb` readers, redaction, lessons→skills bridge |
| `memory` | `DreamMemory` — policies, recipes, lessons, trees, events |
| `sandbox` | `run_isolated` — the single untrusted-code execution boundary |
| `tools` | `CodeVerifier` — sandboxed verification of candidate solutions |
| `llm` | OpenAI-compatible client (stdlib `urllib` transport) |
| `cli` | `loop / status / dashboard / bench / bench-policy / gate-replay / mcp / proxy / sentinel` |
| `mcp`, `mcp_server` | MCP stdio server for loop tools and sentinel checks |
| `proxy` | loopback OpenAI-compatible proxy borrowing goose's upstream |
| `dashboard` | zero-dependency live web UI |
| `gate_replay` | offline validation of the lesson promotion rule |
| `sentinel` | host-agnostic error-class engine + `SentinelObservation` world contract |
| `utils.goose` | goose config resolver (CLI + desktop dialects, keychain) |
| `plugins/*` | Hermes / Claude Code / goose adapters |
| `benchmarks/travelplanner` | **the benchmark**: sandbox MCP server, episode runner, official-evaluator wrapper, split CSVs, raw published episode records |

## Tests

```bash
python -m unittest discover -s tests          # or: pytest tests/ -q
```

244 tests (+10 subtests), stdlib-only: benchmark behavioural contracts
(greedy must solve nothing; the gated arm must dominate ε; the rollout must
be prefix-only), sandbox escape regressions, gate semantics, MCP protocol,
goose config dialects, sentinel channel semantics (nudge fires once per
budget, merges with recurrence notes, clean calls count toward the budget;
the permute-gate serves until ceil(0.8·budget) then refuses forever, per
session; the 30-day production replay must reproduce its recorded note count
exactly), and the structured seam (cross-session counts never reach
`to_world_dict()`; the frontier-feature attribution pattern — beats greedy
with metadata, ties without; the vendored Hermes engine pinned byte-for-byte
to upstream by `test_vendored_engine_matches_upstream`).

## License

MIT — see [LICENSE](LICENSE).
