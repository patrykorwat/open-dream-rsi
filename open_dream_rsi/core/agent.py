from typing import Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from open_dream_rsi.core.tree import DiscoveryTree
from open_dream_rsi.core.dreamer import DreamEngine


@runtime_checkable
class ChatClient(Protocol):
    """Minimal LLM client interface (implemented by OpenAICompatibleClient and StubClient)."""

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str: ...


class DreamAgent:
    """LLM agent driven by the policy learned during offline dreaming.

    The agent does not modify model weights — it reuses the exploration policy
    from :class:`DreamEngine` (temperature, depth) to steer LLM calls made
    through an OpenAI-compatible endpoint (OpenAI, Cursor Models API, local servers).

    Args:
        dreamer:       Dream engine holding the dreamed policy.
        client:        LLM client (OpenAI-compatible) or StubClient for tests.
        system_prompt: System prompt defining the task domain.
    """

    def __init__(
        self,
        dreamer: Optional[DreamEngine] = None,
        client: Optional[ChatClient] = None,
        system_prompt: str = (
            "You are an agent optimizing a programming-task solution. "
            "Reply with a single action from the list: explore | refine | verify | commit."
        ),
    ):
        self.dreamer = dreamer
        if client is None:
            from open_dream_rsi.llm import StubClient  # lazy import — no import cycle

            client = StubClient()
        self.client: ChatClient = client  # type: ignore[assignment]
        self.system_prompt = system_prompt

    @property
    def policy(self) -> Dict[str, float]:
        return self.dreamer.policy_parameters if self.dreamer else {}

    def propose_action(self, context: str) -> str:
        """Propose the next action, applying the dreamed policy temperature."""
        temperature = float(self.policy.get("temperature", 0.7))
        reply = self.client.chat(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": context},
            ],
            temperature=temperature,
        )
        return reply.strip().lower().splitlines()[0] if reply else "explore"

    def execute_step(
        self,
        tree: DiscoveryTree,
        state_id: str,
        scorer: Callable[[str], Tuple[object, float]],
    ) -> str:
        """Execute one online step: the LLM proposes an action, the scorer grades it.

        ``scorer(action) -> (result, score)`` invokes the real tool
        (kernel compilation, tests). The new node is added to the Discovery Tree.
        Returns the node_id of the new node.
        """
        node = tree.nodes[state_id]
        context = f"State: {node.result!r}\nCurrent score: {node.score}\nChoose an action."
        action = self.propose_action(context)
        result, score = scorer(action)
        new_id = f"{state_id}/{action}-{len(tree.nodes)}"
        tree.add_node(new_id, action=action, result=result, score=score, parent_id=state_id)
        return new_id
