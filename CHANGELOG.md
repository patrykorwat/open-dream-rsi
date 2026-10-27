# Changelog

All notable changes to `open-dream-rsi` are documented here. Versions 0.3.0 and
earlier are summarized from the commit history; the full log is the source of
truth.

## [0.3.15] — 2026-10-27

### Added

- **Claude Code plugin**: `.claude-plugin/marketplace.json` (the repository
  doubles as its own marketplace) and `.claude-plugin/plugin.json` (manifest
  with the MCP server). `userConfig` prompts for the task-queue
  (`tasks_path`, type `file`) and memory (`memory_path`, type `directory`)
  paths at enable time and substitutes them into the server's arguments;
  `PYTHONPATH=${CLAUDE_PLUGIN_ROOT}` runs the bundled stdlib-only checkout,
  so no `pip install` is required. `scripts/check_versions.py` now watches
  the new manifest's version (guarded by file existence, so checkouts of
  older tags keep working).
- Docs: the README install table and `docs/integrations.md` lead the Claude
  Code section with the plugin route; `claude mcp add` remains the manual
  fallback. Hermes row corrected: plugins install via the desktop app's
  **Capabilities → Plugins**, not Settings.

## [0.3.14] — 2026-10-27

### Added
- **Lesson effectiveness in `odr metrics`**: per-lesson `uses -> wins`
  (credit rate) rendered from the `lessons.json` records that
  `record_lesson_usage` already maintains (surfaced into proposals /
  task solved); dead lessons (>=4 uses, 0 wins — the curator's prune
  rule) are flagged so the operator sees what never helped.

### Changed
- **Curator contract**: explicit trigger few-shots — GOOD: 1-4 lowercase
  keywords ('pip cache', 'ssh key perms', 'docker compose env'); BAD:
  sentence-like 'traceback code does not execute' (5 words, rejected by
  validation). Cuts the "trigger must be 1-4 words" rejection class.

## [0.3.13] — 2026-10-25

### Added
- **Retry with exponential backoff + full jitter** on transient LLM failures
  (connection refused/reset/remote closed, HTTP 429/5xx). Generation timeouts
  are NOT retried by default — each timed-out attempt already burned the full
  call timeout, and on a saturated shared endpoint blind timeout retries just
  multiply wall clock (`ODR_LLM_RETRY_TIMEOUTS=1` opts in). Tunables:
  `ODR_LLM_RETRIES` (default 3), `ODR_LLM_BACKOFF` (base seconds, default 5),
  `ODR_LLM_BACKOFF_MAX` (cap, default 120). Error messages keep the
  classification substrings ("timed out", "Could not connect") so
  `odr metrics` buckets stay stable.
- **Token usage accounting**: the client accumulates the OpenAI `usage`
  blocks (`usage_total`/`last_usage`); the loop logs a per-cycle
  `tokens` delta into `cycle` events; `odr metrics` reports
  `tokens: prompt / completion / total (N instrumented cycles)` and renders
  "not instrumented" for pre-0.3.13 history instead of a misleading zero.

## [0.3.12] — 2026-10-23

### Added
- **Fleet runner** (`scripts/odr_fleet_dream.sh`): env-configured loop that
  fires one `odr dream` cycle per memory dir matching `ODR_WS_GLOB`, with the
  LLM env merged from `ODR_ENV_FILE`. The timer half of a service-based fleet
  deployment (systemd oneshot unit + timer, or any cron line); per-workspace
  cadence still lives inside the dream module. Documented in the new README
  section "Fleet deployments", which also names the three ingredients that
  separate a learning fleet from a heartbeat-only one (`ODR_SOURCE`, a real
  `ODR_PROVIDER`, the workspace glob).

## [0.3.11] — 2026-10-23

### Added
- **Wake self-test** (`odr dream --wake`, MCP `odr_dream.wake`, env
  `ODR_WAKE`, default on): a FULL dream that finds the task queue empty
  seeds a one-shot `wake:selftest` probe — "write `add(a, b)`", verified
  by TESTS (never the judge), `max_attempts=1` — so an idle deployment
  still exercises its configured LLM endpoint once per full dream.
  `odr metrics` then shows real `api_calls` and a live solve rate
  (category `wake`) instead of a silent zero-call heartbeat. The probe
  never touches `tasks.json` and never displaces real work; opt out per
  deployment with `ODR_WAKE=0` or per call with `--no-wake`.

## [0.3.10] — 2026-10-22

### Added
- **Success metrics** (`odr metrics`, MCP `odr_metrics`): one command answers
  "is the loop actually working?" — cycle health (api spend, budget
  exhaustion rate), per-category solve rates with no-solve flags,
  llm-error and lesson-rejection classes, curation throughput (gate
  verdicts, lessons added/merged, user corrections), lesson states,
  backlog size and staleness; `--hours` windows the event-derived sections,
  all-time outcomes always included; empty memory degrades to zeros.

### Changed
- **Default full-cycle budget 20 → 100** (CLI `loop`/`dream`, `dream_once`,
  `AutoRSIRuntime`, MCP `odr_run_once`, the Hermes `odr-trigger` plugin):
  20 starved the promotion gate (`lesson_gate_skipped reason=budget`) and
  even skipped judge verdicts on queued tasks; 100 buys real full cycles
  (~25 min) while staying safe for unattended triggers.

## [0.3.9] — 2026-10-22

### Added
- **Hermes one-line plugin install**: the `odr-trigger` plugin now lives at
  the repo's top level (`odr-trigger/`), where Hermes plugin discovery finds
  it — `hermes plugins install
  "https://github.com/patrykorwat/open-dream-rsi#odr-trigger"` or the
  desktop **Install from Git** dialog; the old manual `cp -r` path is gone.
- **Evaluation and session-import docs split out of the README** —
  [docs/benchmarks.md](docs/benchmarks.md) (TravelPlanner protocol, tables, honest
  memory findings) and [docs/sessions.md](docs/sessions.md) (transcript import,
  corrections, stores); README down to ~470 lines with a per-agent install
  table as the Quick start; the key-free demo and Docker sections moved to
  [docs/try-it.md](docs/try-it.md).
- **`odr` console script** (`[project.scripts]`): `odr status`, `odr mcp …` —
  one-line MCP config blocks; `uvx --from open-dream-rsi odr …` works with no
  prior install.
- **Claude Code session importer** (`read_claude_code_sessions`,
  `dream --source claude`, `sessions export --source claude`): JSONL
  transcripts under `~/.claude/projects` become curator evidence — tool
  results joined to their calls, subagent sidechains skipped, resume
  re-appends deduplicated; same redaction and promotion gate as every other
  source.
- **Docker image** (`Dockerfile`, `.dockerignore`): one command to try the
  project in a pinned, non-root environment; CI builds the image on every
  push.
- **Clean mypy gate in CI** (`--ignore-missing-imports`, pinned version): the
  tree typechecks with zero errors — the `py.typed` promise is now enforced,
  not just declared.
- **Community files**: feature-request issue template, PR template,
  `CODE_OF_CONDUCT.md`, this changelog, `CITATION.cff` (the "Cite this
  repository" button), and a release job in `publish.yml` that creates a
  GitHub Release from every version tag with the matching changelog section.
- **[docs/comparison.md](docs/comparison.md)** — how the loop differs from
  memory plugins (Mem0, Letta), reflection (Reflexion), prompt optimizers
  (DSPy) and skill libraries: what is stored, how entries earn trust, what it
  costs.
- **[docs/holdout-reservations.md](docs/holdout-reservations.md)** — a
  walkthrough for `odr holdout` (single-use evaluation splits).
- `docs/README.md` — index of the documentation directory.

### Changed
- `docs/launch-kit.md` (internal launch notes) removed from the public
  repository.
- Git-backed lesson stores now carry **worlds** and **policies**: export ships
  redacted `trees/*.json` with sha256 in the manifest; imports verify
  checksums, drop unknown fields, and land policies as VALIDATED (an import
  never replaces an incumbent). New commands: `odr lessons
  worlds-export|worlds-import|policies-export|policies-import`.
- **Holdout reservations** (`open_dream_rsi/holdout.py`, `odr holdout
  claim|release|show|split`): single-use eval-split claims — same-purpose
  re-claim is idempotent (a crashed phase can resume), cross-purpose claims are
  refused. The loop never reads the registry.
- **Policy revise mode** (opt-in `policy_revive`): a rejected LLM-written
  policy gets a greedy-replay trace as revision feedback; same strict
  promotion gate.
- **Sandbox result cache** (opt-in): LRU+TTL keyed by file contents; only
  completed runs are cached.

### Fixed
- **LLM call timeout is env-tunable** (`ODR_LLM_TIMEOUT`, default 900 s; was a
  hard 120 s): proposal calls legitimately generate multi-kB reports and queue
  behind other consumers on a shared vLLM — at 120 s those aborts became the
  dominant failure and still burned API budget.
- **Category recipes are labelled as cross-task style references** in the
  proposal prompt: a same-category "best solution" from a different task used
  to steer whole probe trees (e.g. a Fibonacci snippet seeded into unrelated
  task families).
- **Curator salvage**: mixed lesson payloads no longer die all-or-nothing —
  valid entries survive one malformed sibling; comma/semicolon triggers fold
  into the 1-4 word budget; the contract demands single-space keywords.
- **odr-trigger (Hermes plugin)**: `settings.odr_root` pins the checkout the
  dream child imports from, `ODR_*` environment passes through to the child,
  and `OPENAI_*` stays settings-driven (no stray host override retargeting
  the endpoint).

## [0.3.8] — 2026-10-18

### Added
- **PyPI publishing** (tag-driven, trusted publishing; 0.3.8 live on PyPI) and
  **npm bundle publishing** in the same workflow.
- `py.typed` ships with the package; PEP 639 metadata
  (`License-Expression: MIT`).
- CI: vendored-copy drift test; version-sync check in the lint job;
  `set -o pipefail` on every piped step.
- Wilson 95% CIs on the TravelPlanner table (README + paper caption).

### Changed
- README split: per-agent setup moved to `docs/integrations.md`, self-check
  details to `docs/self-checks.md`.
- Evaluation scope note: the TravelPlanner gate arm measures the **fixed call
  cutoff** — no learned content in those episodes; the learned machinery is
  validated by the scripted suites.

## [0.3.7] — 2026-10-17
- Key-free CI (Python 3.10–3.13), `SECURITY.md`, `CONTRIBUTING.md`, working
  key-free demo.

## [0.3.6] — 2026-10-17
- Curator: lesson retrieval budget 3000 → 15000 chars.

## [0.3.4] — 2026-10-17
- Curator: lesson text cap 1200 → 80000 chars (detailed lessons).

## [0.3.3] — 2026-10-17
- Curator: detailed lessons — text cap 280 → 1200 chars, retrieval budget
  700 → 3000.

## [0.3.2] — 2026-10-17
- Fair task queue (least-processed-first) — fixes backlog starvation by
  processing, not pruning.

## [0.3.1] — 2026-10-13
- **User corrections as first-class evidence** (issue #6): correction /
  approach_redirect / preference / new_requirement classification, provenance
  recorded in the evidence itself, same trust gate as loop-collected failures.
- DSH bundle 0.3.1.

## [0.3.0] — 2026-10-12
- **Cursor plugin**: repository import works (`.cursor-plugin` manifest + root
  `hooks/` + `rules/`); hooks carry args, not code.
- **cursor-agent** headless CLI as the default dreamer brain (no proxy, no API
  key; Windows `.cmd` shims wrapped via `cmd.exe /c`).
- A dead endpoint **defers** the session (stays unseen) instead of burning the
  evidence backlog.

## Earlier versions
See the [commit history](https://github.com/patrykorwat/open-dream-rsi/commits/main)
for 0.2.x and the initial release.

[0.3.10]: https://github.com/patrykorwat/open-dream-rsi/compare/v0.3.9...HEAD
[0.3.9]: https://github.com/patrykorwat/open-dream-rsi/compare/v0.3.8...v0.3.9
[0.3.8]: https://github.com/patrykorwat/open-dream-rsi/releases/tag/v0.3.8
