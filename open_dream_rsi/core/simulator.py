from typing import Any, Tuple
from open_dream_rsi.core.tree import DiscoveryTree


class ReplaySimulator:
    """Replay Simulator.

    Allows executing steps offline without invoking expensive external tools.
    """

    def __init__(self, tree: DiscoveryTree):
        self.tree = tree

    def step_offline(self, state_id: str, simulated_action: str) -> Tuple[Any, float]:
        """Return an approximate result and a dream reward (Offline Dream Step)."""
        node = self.tree.nodes.get(state_id)
        if not node:
            return None, 0.0

        # If the action matches the one recorded in the tree, return the exact offline result
        if node.action == simulated_action:
            return node.result, node.score

        # Dreaming heuristic for unseen actions, based on the parent/child state
        heuristic_score = node.score * 0.9
        return "simulated_trajectory_result", heuristic_score
