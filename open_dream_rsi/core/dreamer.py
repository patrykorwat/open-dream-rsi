"""Offline dreaming: hill-climb exploration parameters by *simulating the
choices they would make* on the recorded discovery tree (issue #1).

The two parameters define a parameterised exploration policy:

* ``temperature``       — softmax sharpness over the as-of-now outcomes of
  the visible frontier (``T -> 0``: deterministic greedy, ``T`` large:
  near-uniform random exploration).
* ``exploration_depth`` — how often one node may be re-polished before the
  policy forces a new branch: nodes already expanded ``depth`` or more
  times drop out of the candidate pool.

The objective is the SAME counterfactual rollout objective that gates
LLM-written policy code (``core.policygen``: mean recorded-child reward +
``beta_2`` branch diversity), averaged over deterministic seeded episodes
and computed with the prefix-only reward harness — a parameter only earns
score when the choices it produces would have opened better branches in the
recorded history. No hard-coded reward shaping.
"""

import math
import random
from typing import Any, Dict, List

from open_dream_rsi.core.policygen import _rollout_objective, _simulate_step_rewards, rollout_world_payload
from open_dream_rsi.core.simulator import ReplaySimulator

#: Hard bounds for policy parameters — dreaming explores inside a safe region.
TEMP_MIN, TEMP_MAX = 0.1, 1.5
DEPTH_MIN, DEPTH_MAX = 1.0, 6.0
#: Seeded episodes per candidate evaluation (deterministic: hill climbing
#: must compare candidates on identical trajectories, issue #1 review).
EPISODES = 6
#: Base seed for the episode RNGs.
DREAM_SEED = 1234


class DreamEngine:
    """Offline exploration-strategy optimization engine (Offline Dreaming)."""

    def __init__(self, simulator: ReplaySimulator):
        self.simulator = simulator
        self.policy_parameters: Dict[str, float] = {"temperature": 0.7, "exploration_depth": 3.0}
        #: Dedicated RNG: mutation moves AND episode draws are seeded, so a
        #: dream run on the same tree is reproducible (benchmarks + tests).
        self._rng = random.Random(DREAM_SEED)

    def run_offline_optimization(self, iterations: int = 50) -> Dict[str, float]:
        """Optimize the policy parameters over the simulated history."""
        # Incumbent seeds the hill climb: a candidate only replaces it on a
        # strictly better simulated score (same promotion rule as section 3).
        best_policy = self.policy_parameters.copy()
        best_score = self._evaluate_policy_in_dream(best_policy)

        for _ in range(iterations):
            # 1. Mutate the strategy parameters (clamped to safe bounds)
            candidate_policy = {
                "temperature": min(TEMP_MAX, max(TEMP_MIN, self.policy_parameters["temperature"] + self._rng.uniform(-0.1, 0.1))),
                "exploration_depth": min(DEPTH_MAX, max(DEPTH_MIN, self.policy_parameters["exploration_depth"] + self._rng.uniform(-0.5, 0.5))),
            }

            # 2. Evaluate the candidate in the simulator
            simulated_score = self._evaluate_policy_in_dream(candidate_policy)

            # 3. Keep the best-so-far (hill climbing over dreams)
            if simulated_score > best_score:
                best_score = simulated_score
                best_policy = candidate_policy
                self.policy_parameters = candidate_policy

        self.policy_parameters = best_policy
        return best_policy

    # -- the simulated policy family ------------------------------------------------

    @staticmethod
    def _pick(visible: List[Dict[str, Any]], rng: random.Random,
              temperature: float, depth: float) -> str:
        """One choice of the parameterised policy on a visible frontier.

        Softmax over as-of-now outcomes with ``temperature`` sharpness, over
        the pool of nodes whose recorded re-expansion count is below
        ``exploration_depth`` (falls back to the full frontier when the cap
        would leave nothing to pick).
        """
        pool = [n for n in visible if n["children"] < depth] or visible
        vals = [n["outcome"] for n in pool]
        spread = max(vals) - min(vals)
        if spread <= 1e-9:
            return rng.choice(pool)["node_id"]
        top = max(vals)
        weights = [math.exp((v - top) / max(temperature, 1e-6)) for v in vals]
        total = sum(weights)
        draw = rng.random() * total
        acc = 0.0
        for node, w in zip(pool, weights):
            acc += w
            if draw <= acc:
                return node["node_id"]
        return pool[-1]["node_id"]

    def _evaluate_policy_in_dream(self, policy: Dict[str, float]) -> float:
        """Dreaming: roll the parameterised policy over the recorded history.

        ``exploration_depth`` gates re-polishing and ``temperature`` drives
        the softmax, so the score reflects the *choices* the candidate
        parameters make — not a fixed function of the parameters themselves.
        """
        if not self.simulator.tree.nodes:
            return float("-inf")
        t = min(TEMP_MAX, max(TEMP_MIN, float(policy["temperature"])))
        depth = min(DEPTH_MAX, max(DEPTH_MIN, float(policy["exploration_depth"])))
        world = rollout_world_payload(self.simulator.tree)
        total = 0.0
        for ep in range(EPISODES):
            rng = random.Random(f"{DREAM_SEED}:{ep}")
            rewards, picks, _invalid, err = _simulate_step_rewards(
                world, lambda visible, step: self._pick(visible, rng, t, depth))
            if err or not rewards:
                return float("-inf")
            total += _rollout_objective(rewards, picks)
        return total / EPISODES
