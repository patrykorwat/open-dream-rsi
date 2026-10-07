from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TerminationReason(str, Enum):
    """Why an episode's exploration stopped at its frontier node.

    Replay must be able to tell "the decision-maker chose to stop" from
    "the runtime removed the option" — they mean opposite things for a
    counterfactual rollout. Closed vocabulary; str values persist into
    archived trees, so they are part of the on-disk contract.
    """

    COMPLETED = "completed"          # the task solved at this node
    POLICY_STOP = "policy_stop"      # the policy/agent chose to stop exploring
    TOOL_FAILURE = "tool_failure"    # a tool error made further steps impossible
    BUDGET_GATE = "budget_gate"      # runtime cap (API budget / max attempts) removed the option
    SENTINEL_BLOCK = "sentinel_block"  # enforcement gate (e.g. PermuteGate) refused to serve
    UNKNOWN = "unknown"              # not recorded (older trees) or ambiguous


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
    #: Structured facts the Sentinel engine observed about this node's
    #: tool result, in its REPLAY-SAFE form (see
    #: ``SentinelObservation.to_world_dict``): signature, repeat_world,
    #: recurring, blocked, budget_fraction. Cross-session counters are
    #: stripped by construction — they belong to the live host / curator,
    #: never to the historical world (prefix-only replay invariant).
    sentinel: Optional[Dict[str, Any]] = None
    #: One of ``TerminationReason``'s str values, recorded on the node where
    #: the episode stopped. None on interior nodes (the episode continued).
    termination_reason: Optional[str] = None


class DiscoveryTree:
    """Stores the agent's discovery history as a state tree."""

    def __init__(self):
        self.nodes: Dict[str, TreeNode] = {}
        self.root_id: Optional[str] = None

    def add_node(self, node_id: str, action: str, result: Any, score: float,
                 parent_id: Optional[str] = None, thought: str = "",
                 sentinel: Optional[Dict[str, Any]] = None,
                 termination_reason: Optional[str] = None) -> TreeNode:
        node = TreeNode(node_id=node_id, action=action, result=result,
                        score=score, parent_id=parent_id,
                        thought=thought or "", sentinel=sentinel,
                        termination_reason=termination_reason)
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
