# How Open Dream-RSI compares

"Another memory plugin?" is the first question visitors ask, so here is the
direct answer against the tools they already know. The categories blur at the
edges; the table captures the shipped center of each system.

| | What is stored | How entries earn trust | Runtime cost | Dependency model |
|---|---|---|---|---|
| **Open Dream-RSI** | Discovery Trees (full attempt history), replay-verified policies, recipes | Nothing activates on the model's word: candidates survive counterfactual replay against the recorded world and must strictly beat the incumbent | Dreaming is offline and in-process — zero API cost; runtime pays only for tasks | Zero runtime deps, Python 3.10+ stdlib |
| **Mem0** | Extracted memory items (facts, preferences) | LLM extraction + consolidation; no behavioral validation | Memory ops per turn; LLM calls for extraction | pip package + (typically) a vector store |
| **Letta (MemGPT)** | Paginated memory blocks in an agent OS | Context management heuristics; no replay validation | Always-on agent runtime | Managed service or self-hosted server |
| **Reflexion-style reflection** | Verbal self-critique appended to context | None — the reflection is trusted as written | Extra model calls per episode | Prompt pattern |
| **DSPy optimizers** | Optimized prompts / few-shot sets | Metric-guided search over prompts (requires labeled metrics) | Offline compilation; calls during search | pip package |
| **Skill libraries (e.g. Voyager)** | Reusable code skills | Skill is kept if it helped once; reuse is largely unguarded | Code exec per retrieval | Varies |

The structural difference that carries most of the weight:

1. **Validation runs against behavior, not judgment.** A policy or recipe
   becomes active only after a counterfactual replay shows it solving
   failures the incumbent could not — measured, paired, and strict (a tie is
   not evidence).

What ODR is **not**: a general memory layer for chat personalization, and not a
prompt optimizer. It is a loop around tool-using coding agents, for work that
repeats within a category.
