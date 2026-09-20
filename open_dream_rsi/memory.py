"""Persistent memory for the autonomous RSI loop (Hermes-style).

Everything the agent learns survives restarts:

* ``policies.json``  — dreamed policy parameters per task category. The next
  run of the same category starts where the last one finished.
* ``recipes.json``   — distilled winning solutions per category, used as warm
  starts (the loop literally improves itself between runs).
* ``trees/``         — archived Discovery Trees per task (offline-dream fodder).
* ``events.jsonl``   — append-only log of every runtime decision.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from open_dream_rsi.core.tree import DiscoveryTree, TreeNode


class DreamMemory:
    """JSON-backed store: cross-run policy library + recipe library + logs."""

    def __init__(self, root: "str | Path" = ".dream_rsi"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._policies = self._load(self.root / "policies.json", {})
        self._recipes = self._load(self.root / "recipes.json", {})

    # -- low-level -------------------------------------------------------------

    @staticmethod
    def _load(path: Path, default: Any) -> Any:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return default
        return default

    def _save(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)  # atomic

    # -- policies ----------------------------------------------------------------

    def get_policy(self, category: str) -> Optional[Dict[str, float]]:
        entry = self._policies.get(category)
        return dict(entry["params"]) if entry else None

    def save_policy(self, category: str, params: Dict[str, float]) -> None:
        self._policies[category] = {"params": dict(params), "updated_at": time.time()}
        self._save(self.root / "policies.json", self._policies)

    # -- recipes (learned solutions) ----------------------------------------------

    def get_recipe(self, category: str) -> Optional[str]:
        entry = self._recipes.get(category)
        return entry["code"] if entry else None

    def save_recipe(self, category: str, code: str, score: float) -> bool:
        old = self._recipes.get(category)
        if old and old.get("score", 0.0) >= score:
            return False  # keep the best-known solution
        self._recipes[category] = {"code": code, "score": score, "saved_at": time.time()}
        self._save(self.root / "recipes.json", self._recipes)
        return True

    # -- discovery trees -------------------------------------------------------------

    def archive_tree(self, task_id: str, tree: DiscoveryTree) -> None:
        payload = {
            "root_id": tree.root_id,
            "nodes": [vars(n) for n in tree.nodes.values()],
        }
        self._save(self.root / "trees" / f"{task_id}.json", payload)

    def load_tree(self, task_id: str) -> Optional[DiscoveryTree]:
        path = self.root / "trees" / f"{task_id}.json"
        if not path.exists():
            return None
        data = self._load(path, None)
        if not data:
            return None
        tree = DiscoveryTree()
        tree.root_id = data.get("root_id")
        for n in data.get("nodes", []):
            node = TreeNode(**n)
            tree.nodes[node.node_id] = node
        return tree

    # -- event log ---------------------------------------------------------------------

    def log_event(self, kind: str, **fields: Any) -> None:
        event = {"ts": time.time(), "kind": kind, **fields}
        with open(self.root / "events.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
