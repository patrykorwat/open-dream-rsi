import math
import random
from typing import Dict
from open_dream_rsi.core.simulator import ReplaySimulator

#: Hard bounds for policy parameters — dreaming explores inside a safe region.
TEMP_MIN, TEMP_MAX = 0.1, 1.5
DEPTH_MIN, DEPTH_MAX = 1.0, 6.0
#: Reward-optimal temperature: past this point extra randomness hurts more than it helps.
TEMP_OPTIMAL = 0.8


class DreamEngine:
    """Offline exploration-strategy optimization engine (Offline Dreaming)."""

    def __init__(self, simulator: ReplaySimulator):
        self.simulator = simulator
        self.policy_parameters: Dict[str, float] = {"temperature": 0.7, "exploration_depth": 3.0}

    def run_offline_optimization(self, iterations: int = 50) -> Dict[str, float]:
        """Optimize the policy parameters over the simulated history."""
        best_score = float("-inf")
        best_policy = self.policy_parameters.copy()

        for _ in range(iterations):
            # 1. Mutate the strategy parameters (clamped to safe bounds)
            candidate_policy = {
                "temperature": min(TEMP_MAX, max(TEMP_MIN, self.policy_parameters["temperature"] + random.uniform(-0.1, 0.1))),
                "exploration_depth": min(DEPTH_MAX, max(DEPTH_MIN, self.policy_parameters["exploration_depth"] + random.uniform(-0.5, 0.5))),
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

    def _evaluate_policy_in_dream(self, policy: Dict[str, float]) -> float:
        """Dreaming: virtual steps over the discovery tree with a concave reward.

        Temperature multiplies base reward but is penalised quadratically past
        TEMP_OPTIMAL (runaway randomness wastes budget), so the optimum is
        interior instead of the bound — without it, raw score * temperature
        drifts to the ceiling every run.
        """
        t = policy["temperature"]
        t_factor = t * math.exp(-((t - TEMP_OPTIMAL) ** 2) / 0.5)
        total_reward = 0.0
        for node_id in self.simulator.tree.nodes:
            _, score = self.simulator.step_offline(node_id, simulated_action="explore")
            total_reward += score * t_factor
        return total_reward
