from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TreeNode:
    node_id: str
    action: str
    result: Any
    score: float
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    #: One-sentence plan the model recorded when producing this attempt
    #: ("why I try this"). Pure text — never executed — but it is what
    #: turns branch selection from purely numeric (score) into semantic:
    #: policies and future proposals can see WHICH IDEA a branch stands for.
    thought: str = ""


class DiscoveryTree:
    """Stores the agent's discovery history as a state tree."""

    def __init__(self):
        self.nodes: Dict[str, TreeNode] = {}
        self.root_id: Optional[str] = None

    def add_node(self, node_id: str, action: str, result: Any, score: float, parent_id: Optional[str] = None,
                 thought: str = "") -> TreeNode:
        node = TreeNode(node_id=node_id, action=action, result=result, score=score, parent_id=parent_id,
                        thought=thought or "")
        self.nodes[node_id] = node

        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].children.append(node_id)
        elif self.root_id is None:
            self.root_id = node_id

        return node

    def get_best_trajectory(self) -> List[TreeNode]:
        """Return the best path based on the score."""
        if not self.nodes:
            return []
        best_node = max(self.nodes.values(), key=lambda n: n.score)

        trajectory = []
        curr: Optional[TreeNode] = best_node
        while curr:
            trajectory.append(curr)
            curr = self.nodes.get(curr.parent_id) if curr.parent_id else None

        return list(reversed(trajectory))
