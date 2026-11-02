# Machinery self-checks (deterministic, key-free)

Not benchmarks — behavioural regression contracts. A scripted solver makes
the model arm a constant, so every delta is attributable to the library's
gates, sandboxes, promotion and retrieval logic. They exist so a broken
mechanism fails CI instead of silently mutating a published number, and they
re-run from a fresh clone in seconds. The README carries the summary tables;
this file holds the design notes and the per-arm contracts.

## `bench` — the dreaming loop against a cold baseline

```bash
python -m open_dream_rsi bench --cycles 10 --markdown
```

| arm | solves | API calls | calls / task·cycle | dream its |
|---|---|---|---|---|
| cold_baseline (fresh memory each cycle) | 50 | 150 | 3.0 | 0 |
| dream_rsi_loop (policies + recipes + dreaming) | 50 | 71 | 1.42 | 3000 |

**53% fewer API calls at equal solve quality** (re-measured on this commit:
identical to the published table). The cold baseline re-creates its memory
every cycle, so the only difference between the arms is what the loop
persists and reuses — recipes, policies, the dream phase.

## `bench-policy` — decoy traps (exploration contracts)

Solve-rate self-checks saturate on easy suites and cannot tell good
exploration from luck, so `bench-policy` hides each fix behind a
*low-scoring* branch (passes 1/3 tests) past a *plausible decoy* (passes 2/3
forever). A score-greedy loop locks onto the decoy and starves; escaping
requires structural exploration. Five arms, one scripted solver, one budget:

```bash
python -m open_dream_rsi bench-policy --cycles 8 --format md
```

| arm | solves | solve rate | API calls | calls / solve | policy calls |
|---|---|---|---|---|---|
| greedy | 0/120 | 0% | 480 | ∞ | 0 |
| epsilon_greedy | 29/120 | 24% | 405 | 13.97 | 0 |
| **evolved_policy** | **93/120** | **78%** | **249** | **2.68** | 26 |
| **thought_guided** | **120/120** | **100%** | **164** | **1.37** | 0 |
| **sentinel_guided** | **102/120** | **85%** | 219 | 2.15 | 15 |

- **evolved_policy** — the loop asks the LLM for a policy program per
  category, validates and replay-scores it, and promotes only on evidence.
  100% solve rate by cycle 5 while spending *fewer total calls* than ε
  wasted on luck.
- **thought_guided** (library default) — each attempt records its one-line
  `PLAN:`; expansion leaves a branch as soon as its idea repeats itself.
  With ε forced to 0 and no policy calls it escapes every trap
  **from cycle 1**; the ledger-only ablation (same text in prompts, pick
  disabled) collapses back to the baselines — the win is the expansion
  rule, not prompt length.
- **sentinel_guided** — replay-gated policies on a tree that carries
  structured Sentinel facts: nodes whose error class keeps repeating are
  recorded **blocked** (choice removal, the TravelPlanner-measured
  mechanism), and the promoted policy reads `sentinel_blocked` and skips
  them. 85% at 2.15 calls/solve with ε = 0; the
  engine-off ablation (same policy code, no facts) falls to 75/120 — the
  win is the features, not the code.

Pinned invariants (asserted by `tests/test_bench_policy.py`): the greedy arm
solves nothing; every policy arm beats ε; `evolved_policy` reaches
100% in the final cycle with fewer calls than ε; the sentinel-guided win
survives the engine-off ablation. Seed sets are swept when a headline margin
looks thin, and the shipped numbers are seed-invariant.

## `gate-replay` — would an arm have been promoted?

Applies the shipped promotion rule (`gate_verdict`) to per-task
outcome pairs from any recorded replay harness — no model, no GPU:

```bash
python -m open_dream_rsi gate-replay \
  --baseline cold=/tmp/arms.json?label=cold \
  --compare arm=/tmp/arms.json?label=warm \
  --key task_id --format md
```

The tool joins baseline/compare records on a key you choose, reports
unmatched rows and duplicates explicitly, and prints the same accept/reject
verdict the live promotion gate would have issued — the decision rule is
validated by replaying it over recorded pairs, not by burning new runs.

## Where these fit

`bench`, `bench-policy` and `gate-replay` are **machinery self-checks**. The
only performance claim in this project is
[goose × TravelPlanner](benchmarks.md),
scored by the benchmark's own evaluators.
