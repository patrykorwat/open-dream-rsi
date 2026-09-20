from typing import Dict, List, Tuple

from open_dream_rsi.core.simulator import ReplaySimulator


class PolicyEvaluator:
    """Score and compare exploration policies over the recorded history."""

    def __init__(self, simulator: ReplaySimulator):
        self.simulator = simulator

    def score_policy(self, policy: Dict[str, float]) -> float:
        """Total reward of a policy across the whole tree history."""
        total = 0.0
        for node_id, node in self.simulator.tree.nodes.items():
            # exploration depth controls the discount applied to future rewards
            discount = 1.0 / (1.0 + len(node.children) / max(policy.get("exploration_depth", 3.0), 1e-6))
            _, score = self.simulator.step_offline(node_id, simulated_action=node.action)
            total += score * policy.get("temperature", 1.0) * discount
        return total

    def rank(self, policies: List[Dict[str, float]]) -> List[Tuple[Dict[str, float], float]]:
        """Sort policy candidates by score, descending."""
        return sorted(((p, self.score_policy(p)) for p in policies), key=lambda t: t[1], reverse=True)

    def compare(self, a: Dict[str, float], b: Dict[str, float]) -> int:
        """Return -1 if a is worse, 0 for a tie, 1 if a is better."""
        sa, sb = self.score_policy(a), self.score_policy(b)
        return (sa > sb) - (sa < sb)
