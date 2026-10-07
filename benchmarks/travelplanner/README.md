# goose × TravelPlanner — the benchmark

This directory contains the **entire harness behind the only
performance-claiming benchmark in this repository** (see the top-level
README, "Benchmark: goose × TravelPlanner"): the sandbox MCP server, the
episode runner, the official evaluator wrapper, and every raw episode
record published in the paper.

```
sandbox_server.py   stdio MCP server: official TravelPlanner tools 1:1,
                    arms wired via env (sentinel off / nudge / permute-gate)
runner.py           goose episodes, task-interleaved arms, resumable,
                    McNemar-ready per-episode jsonl + submission assembly
score.py            official commonsense+hard evaluators (from the pinned
                    TravelPlanner checkout), offline, deterministic
build_df_cache.py   one-time pickle cache of the official DB frames
thinking_proxy.py   loopback proxy forcing enable_thinking:false (arm parity)
splits/             the exact task CSVs used (see Splits below)
results/            raw per-episode records published in the paper
```

## What is measured

Frozen local model, no weight access; the only moving part is what the
**sandbox does at the end of the episode**:

| arm | sandbox behaviour past 80% of the 45-call cap |
| --- | --- |
| `cold` | keeps serving until goose cuts the episode (score 0, no answer) |
| `osi_gate` | **refuses to serve** (`budget_exhausted`) — option removal |
| `osi_nudge` | keeps serving; a finalize nudge rides the next tool result |
| `osi_full` | nudge + pull tool + Stop-hook recurrence notes |

Published (frozen protocol, official evaluators):
delivery 18/79 → 74/79, final pass 7.6% → 21.5% on the preregistered
holdout (paired McNemar 56↔0, p=1.4e-17); details and per-constraint
breakdown in `results/` and `fixtures/tp_v3_summary.json` at repo root.

## Prerequisites

1. **Model endpoint** — any OpenAI-compatible server serving
   `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` (the paper's model) with
   `chat_template_kwargs` support. vLLM: `vllm serve <model>`.
2. **goose** — v1.53 (`goose --version`). Binary path via `ODR_GOOSE`
   (default: `goose` on `PATH`).
3. **Official TravelPlanner** — pinned checkout, used for the tool classes
   *and* the evaluators (this harness does not fork their logic):

   ```
   git clone https://github.com/OSU-NLP-Group/TravelPlanner /tmp/tp
   cd /tmp/tp && git checkout e52c87f4ac348a3410c46dc3553c519db5ec5e23
   ```

   Place the official offline database at `/tmp/tp/database` (the ~1 GB of
   CSVs distributed with the TravelPlanner repo; `flights/`,
   `accommodations/`, `restaurants/`, `attractions/`, `background/`).
   Override the checkout location with `TP_REPO`.
4. **pandas** in a venv for the sandbox process (the official tool classes
   are pandas-based); the rest of the harness is stdlib-only:

   ```
   python3 -m venv .venv-tp && .venv-tp/bin/pip install pandas
   ```

   Path via `TP_PY`.

## Reproduce

```
export ODR_UPSTREAM=http://127.0.0.1:8000       # your vLLM
TP_PY=.venv-tp/bin/python python3 build_df_cache.py        # ~2 min, once
python3 thinking_proxy.py &                     # forces enable_thinking:false

# eval split: both arms, one interleaved pass (79 tasks x 2 arms, ~6 h)
python3 runner.py --csv splits/validation_holdout.csv --arms cold,osi_gate
python3 runner.py --csv splits/validation_holdout.csv --assemble

# score with the official evaluators
python3 score.py results/submissions/validation_holdout_cold.jsonl \
    --csv splits/validation_holdout.csv
python3 score.py results/submissions/validation_holdout_osi_gate.jsonl \
    --csv splits/validation_holdout.csv
```

The runner is resumable (appends `results/tp_results_<split>.jsonl`,
skips finished (task, arm) pairs) and aborts on a dead endpoint rather
than aggregating broken arms. `--cases 1,2` pilots a subset first.

## Checks (what must come out)

`score.py` prints per-constraint counts + `final_pass_rate` per arm.
Scoring the assembled submissions from the committed raw records
(`results/tp_results_validation_holdout.jsonl`) must reproduce:

| split | arm | delivery | commonsense | final |
| --- | --- | --- | --- | --- |
| train (45) | cold | 15 | 4 | 3 |
| train (45) | osi_gate | 41 | 18 | 9 |
| holdout (79) | cold | 18 | 12 | 6 |
| holdout (79) | osi_gate | 74 | 40 | 17 |

Model sampling is not bit-reproducible; a fresh full run lands within a
few tasks of these counts (the effect size, not the last digit, is the
claim). The committed raw records let you re-score the *published*
episodes offline without re-running any episode:
`python3 runner.py --csv splits/X.csv --assemble` reads only `results/`.

## Protocol (frozen; what makes the numbers claimable)

* **Splits are committed verbatim** — `train.csv` (official train, 45
  tasks) as dev; `validation_holdout.csv` = official validation rows
  102–181 (79 tasks, 60 hard) as the single preregistered eval after
  freezing the arm; `validation_selection_era.csv` = the first 101
  validation tasks used during development (results archived as
  contamination, see below). The official test split was never touched.
* Arms run **task-interleaved in one pass** so cold-vs-gate is paired per
  task (McNemar-ready); one variable differs between arms (the sandbox
  closing behaviour), everything else — system prompt, cap, model,
  thinking-off proxy — is shared.
* **Contamination, stated honestly:** the *choice* of gate-over-persuasion
  was made knowing results on the first 101 validation tasks; the holdout
  pass is a preregistered confirmation, not a first exposure. The
  selection-era records are committed for exactly this disclosure.
* Scoring is the official `commonsense_constraint.evaluation` +
  `hard_constraint.evaluation` from the pinned checkout (`score.py`
  bypasses only `load_dataset`'s prompt-template step; constraint logic is
  the untouched upstream code).

## Attribution

Task data, tool implementations and evaluators are from
OSU-NLP-Group/TravelPlanner (MIT), pinned at the commit above. This
directory ships the task CSVs (small); the tool/DB frames are not
redistributed here — build them once from the official checkout.
