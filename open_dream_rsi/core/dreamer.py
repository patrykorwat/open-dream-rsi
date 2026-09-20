import random
from typing import Dict
from open_dream_rsi.core.simulator import ReplaySimulator


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
            # 1. Mutate the strategy parameters
            candidate_policy = {
                "temperature": max(0.1, self.policy_parameters["temperature"] + random.uniform(-0.1, 0.1)),
                "exploration_depth": max(1.0, self.policy_parameters["exploration_depth"] + random.uniform(-0.5, 0.5)),
            }

            # 2. Evaluate the candidate in the simulator
            simulated_score = self._evaluate_policy_in_dream(candidate_policy)

            # 3. Accept the better policy
            if simulated_score > best_score:
                best_score = simulated_score
                best_policy = candidate_policy

        self.policy_parameters = best_policy
        return best_policy

    def _evaluate_policy_in_dream(self, policy: Dict[str, float]) -> float:
        """Dreaming: performs virtual steps over the discovery tree."""
        total_reward = 0.0
        for node_id in self.simulator.tree.nodes:
            _, score = self.simulator.step_offline(node_id, simulated_action="explore")
            total_reward += score * policy["temperature"]
        return total_reward
