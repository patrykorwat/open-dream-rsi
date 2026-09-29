### Problem

#1 reported two real defects, both reproduced on `f573f67` and confirmed still present at HEAD:

1. **Replay leaked future scores.** `rollout_world_payload()` stamped every node with the full-tree `outcome_map()`, so a policy saw (and camping on a node was *paid* with) descendant scores its counterfactual path had not reached: camping on `root` in `root -> child(.1) -> future(1.0)` received `[0.1, 1.0, 1.0]`. Exhausted/never-expanded branches paid the back-propagated best descendant instead of any recorded continuation. The `rollout_score` docstring claimed "as-of-now" outcomes; it was not.
2. **The dreaming objective did not evaluate choices.** `_evaluate_policy_in_dream()` summed the same recorded node scores for every candidate and scaled them by a fixed function of temperature; `exploration_depth` was mutated but never read by the objective (only by the unused `utils/evaluator.py`), so hill-climbing on it was a random walk.

### Approach

**Prefix-only rollout (policygen).** `outcome` is now recomputed *at every rollout step* from descendants already revealed by the counterfactual path, in both the sandbox harness (`_ROLLOUT_HARNESS`) and its in-process mirror (`_simulate_step_rewards`) — kept byte-for-byte equivalent. The serialized world carries no full-tree futures (`frontier_entry(n, {})`). Exhausted ladders pay their **last recorded child**; never-expanded nodes pay **their own score** — no invented continuations. The online view keeps the full-tree `outcome_map` (MCTS-style values are legitimate there: the tree is the live state; `outcome` has been in the contract since `fc37bbc`). The module docstrings now state exactly this.

**Dreaming that simulates choices (dreamer).** The two parameters now define a parameterised exploration policy — softmax over as-of-now outcomes (temperature = sharpness) gated by a re-polish cap (exploration_depth = max re-expansions per node) — scored with the **same prefix-only counterfactual objective** that gates section-3 policy code, averaged over seeded episodes (deterministic hill climbing). A parameter earns score only when the choices it produces would have opened better recorded branches. `TEMP_OPTIMAL` shaping is gone; the incumbent seeds the climb, candidates replace it only on evidence.

### Tests

- 7 new regression tests (suite: 81 → **88 passed**): the issue #1 reproduction verbatim, exhausted-ladder credit, visibility-before-outcome ordering, causal sensitivity of **both** dream parameters, tree-dependence and determinism of the objective. All new tests fail on pre-fix code (verified by reverting sources and re-running: 5 fail).
- Full suite green; `examples/live_loop_demo.py` green.

### Re-measured published numbers

Closing the leak makes the promotion gate stricter *and* stronger: campers can no longer win on future credits, so `evolved_policy` improved to **92/120 (77%), 252 calls** (was 86/72%/274). All other arms are deterministic-identical. README + paper tables, seed-robustness paragraph (policy 84–92, knowledge 77–87, ε 15–29 over five seed sets), both bench figures (dark/light) and the compiled PDF were regenerated from fresh runs — per repo policy, no remembered figures.

### Exclusions

- `utils/evaluator.py` `PolicyEvaluator` remains a separate, unused-by-the-loop utility (its `exploration_depth` discount is its own heuristic). Left untouched — out of scope.
- The benchmark arms in `bench_policy.py` are unchanged; no arm-specific reactions were added, so no rigging risk.

Co-authored-by: Hermes <agent@nousresearch.com>
