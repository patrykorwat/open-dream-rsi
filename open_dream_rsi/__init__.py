"""Open Dream-RSI — an open implementation of the Dream-RSI architecture.

Recursive Self-Improvement by "dreaming" offline over the agent's
discovery tree, without modifying model weights.
"""

from open_dream_rsi.core.tree import DiscoveryTree, TreeNode
from open_dream_rsi.core.simulator import ReplaySimulator
from open_dream_rsi.core.dreamer import DreamEngine
from open_dream_rsi.core.agent import DreamAgent
from open_dream_rsi.utils.evaluator import PolicyEvaluator
from open_dream_rsi.llm import LLMConfig, OpenAICompatibleClient, StubClient

__version__ = "0.1.0"

__all__ = [
    "DiscoveryTree",
    "TreeNode",
    "ReplaySimulator",
    "DreamEngine",
    "DreamAgent",
    "PolicyEvaluator",
    "LLMConfig",
    "OpenAICompatibleClient",
    "StubClient",
    "__version__",
]
