"""Open Dream-RSI — an open implementation of the Dream-RSI architecture.

Recursive Self-Improvement by "dreaming" offline over the agent's
discovery tree, without modifying model weights.
"""

from open_dream_rsi.core.tree import DiscoveryTree, TreeNode
from open_dream_rsi.core.simulator import ReplaySimulator
from open_dream_rsi.core.dreamer import DreamEngine
from open_dream_rsi.core.agent import DreamAgent
from open_dream_rsi.core.policygen import (
    PolicyGenerator,
    PolicySandbox,
    PolicyValidationError,
    extract_python_block,
    validate_policy_source,
)
from open_dream_rsi.core.curator import (
    KnowledgeCurator,
    LessonValidationError,
    curate_lessons,
    lesson_key,
    select_lessons,
    validate_lesson_items,
)
from open_dream_rsi.utils.evaluator import PolicyEvaluator
from open_dream_rsi.llm import LLMConfig, OpenAICompatibleClient, StubClient
from open_dream_rsi.loop import AutoRSIRuntime, Task, CycleReport
from open_dream_rsi.memory import DreamMemory
from open_dream_rsi.tools import CodeVerifier
from open_dream_rsi.bench_policy import (
    TRAP_SUITES,
    run_policy_arm,
    run_policy_benchmark,
)

__version__ = "0.2.0"

__all__ = [
    "DiscoveryTree",
    "TreeNode",
    "ReplaySimulator",
    "DreamEngine",
    "DreamAgent",
    "PolicyGenerator",
    "PolicySandbox",
    "PolicyValidationError",
    "extract_python_block",
    "validate_policy_source",
    "PolicyEvaluator",
    "KnowledgeCurator",
    "LessonValidationError",
    "curate_lessons",
    "lesson_key",
    "select_lessons",
    "validate_lesson_items",
    "LLMConfig",
    "OpenAICompatibleClient",
    "StubClient",
    "AutoRSIRuntime",
    "Task",
    "CycleReport",
    "DreamMemory",
    "CodeVerifier",
    "TRAP_SUITES",
    "run_policy_arm",
    "run_policy_benchmark",
    "__version__",
]
