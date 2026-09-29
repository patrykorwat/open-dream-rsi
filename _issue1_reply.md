Thanks for the careful review — both points reproduce on `f573f67` (and on current `main`), and both were real defects, not misreadings. Fixed on branch `fix/issue-1-prefix-replay-and-dream-sim` (commit `d8b7197`), closing.

### 1. Replay exposed future outcomes — confirmed, fixed

Your reproduction is exactly right: `rollout_world_payload()` stamped every node with the full-tree `outcome_map()` before the rollout started, so (a) the policy saw scores its counterfactual path had not reached and (b) camping on an exhausted node was paid the grandchild's score. The `rollout_score` docstring even claimed "as-of-now" outcomes that the code did not implement.

The fix makes the prefix rule actual rather than claimed:

- `outcome` is now recomputed **at every rollout step** from descendants already *revealed* by the counterfactual path (the same ladder/visibility cursor the rewards use), in both the sandboxed harness and its in-process mirror. The serialized replay world carries no full-tree futures at all (`frontier_entry(n, {})`).
- Exhausted ladders pay their **last recorded child's score**; a node never expanded in the real run pays **its own score** — no back-propagated best-descendant credit for branches the rollout never visited. Your `root → child(0.1) → future(1.0)` case now yields rewards `[0.1, 0.1, 0.1]` and `outcome(root) = 0.0` at step 0.
- The **online** frontier keeps the full-tree outcome map (MCTS-style values): there the tree is the live state and the agent legitimately knows what was found below. Replay is the counterfactual setting, and it is now prefix-only by construction. Your minimal repro is checked in as a regression test (`test_replay_never_leaks_future_scores`), alongside tests pinning the visibility→outcome ordering and the exhausted-ladder floor; all fail on pre-fix code.

Side effect worth noting: the promotion gate got both stricter and stronger. With campers no longer winning on leaked futures, the measured `evolved_policy` arm improved from 86/120 (72%) to **92/120 (77%) at 252 calls** (was 274); README/paper tables and figures were re-measured, not carried over.

### 2. The parameter-level objective did not evaluate choices — confirmed, fixed

Also accurate: the old `_evaluate_policy_in_dream()` summed the same recorded node scores for every candidate and scaled them by a fixed function of temperature; `exploration_depth` was mutated but never read by the objective (only by an unused utility), so hill-climbing on it was a random walk over the hard-coded `t · exp(−(t−0.8)²/0.5)` shaping. So it optimized the shaping function, not exploration quality — your framing was correct.

The objective was replaced with one that evaluates the choices themselves: `temperature` and `exploration_depth` now *define* a parameterised exploration policy — softmax over as-of-now outcomes (sharpness) gated by a re-polish cap (max re-expansions per node) — and it is scored by the **same prefix-only counterfactual rollout objective that gates section-3 policy code**, averaged over seeded episodes so hill-climbing compares candidates on identical trajectories. A parameter earns score only when the choices it produces would have opened better recorded branches; there is no parameter-shaped reward term left. Regression tests assert the objective responds causally to *both* parameters and depends on the recorded tree, not only on `(T, depth)` — they fail on the old objective.

Full write-up of both fixes, tests, and re-measured numbers is in the branch commit: https://github.com/patrykorwat/open-dream-rsi/commit/d8b7197
