"""Example: optimize a GPU kernel with the Dream-RSI loop.

Loop:
  1. Online  — the LLM agent (via an OpenAI-compatible endpoint, e.g. OpenAI
               or the Cursor Models API) proposes kernel variants;
               compilation/tests grade the runtime.
  2. Offline — DreamEngine "dreams" over the Discovery Tree and optimizes
               the exploration-policy parameters with zero API calls.
  3. Deploy  — the dreamed policy is handed back to the online agent.

API keys are read ONLY from environment variables:
    export OPENAI_API_KEY=...            # OpenAI / vLLM / Ollama
    export CURSOR_API_KEY=...            # Cursor Models API
    export ODR_LLM_PRESET=cursor         # switches the endpoint profile
"""

import random
from typing import Callable, Dict, Tuple

from open_dream_rsi import (
    DiscoveryTree,
    DreamAgent,
    DreamEngine,
    ReplaySimulator,
    StubClient,
)

ACTIONS = ["tile_32", "tile_64", "vectorize", "shared_mem", "unroll_4", "explore"]


def make_scorer(kernel_time: Dict[str, float]) -> Callable[[str], Tuple[object, float]]:
    """Simulated 'real' tester: measures each kernel variant's runtime (ms)."""

    def score(action: str) -> Tuple[object, float]:
        if action not in ACTIONS:
            return "compile_error", 0.0
        base = {"tile_32": 4.1, "tile_64": 3.2, "vectorize": 2.8,
                "shared_mem": 2.1, "unroll_4": 2.4, "explore": 3.5}[action]
        t = base + random.uniform(-0.2, 0.2)
        kernel_time[action] = min(kernel_time.get(action, 99.0), t)
        return {"kernel": action, "ms": round(t, 3)}, round(10.0 - t, 3)

    return score


def main() -> None:
    random.seed(42)
    kernel_time: Dict[str, float] = {}

    # --- 1. Online phase: a few agent steps in the real world -----------------
    tree = DiscoveryTree()
    tree.add_node("root", action="start", result="kernel_v0", score=5.0)

    # StubClient => works without an API key. To use a real LLM instead:
    #   from open_dream_rsi import OpenAICompatibleClient, LLMConfig
    #   client = OpenAICompatibleClient()                                # OPENAI_API_KEY
    #   client = OpenAICompatibleClient(LLMConfig.from_preset("cursor"))  # CURSOR_API_KEY
    agent = DreamAgent(client=StubClient(response="shared_mem"))

    state = "root"
    scorer = make_scorer(kernel_time)
    for _ in range(6):
        state = agent.execute_step(tree, state, scorer)

    # --- 2. Offline phase: dream over the history ------------------------------
    simulator = ReplaySimulator(tree)
    dreamer = DreamEngine(simulator=simulator)
    best_policy = dreamer.run_offline_optimization(iterations=100)
    print(f"Dreamed exploration policy: {best_policy}")

    # --- 3. Deploy: the agent adopts the dreamed policy ------------------------
    agent.dreamer = dreamer
    print(f"Online agent temperature: {agent.policy['temperature']:.3f}")

    best = tree.get_best_trajectory()
    print("Best kernel-optimization trajectory:")
    for node in best:
        print(f"  {node.action:>10} -> {node.result} (score {node.score})")


if __name__ == "__main__":
    main()
