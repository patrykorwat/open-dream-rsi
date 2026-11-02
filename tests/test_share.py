"""Git-backed worlds/policies stores (share.py): export/import round-trip, checksum
integrity, staging-only import, git workflow.

Run:  python -m unittest discover -s tests
"""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from open_dream_rsi.share import (
    LessonShareError,
    export_policies,
    export_worlds,
    import_policies,
    import_worlds,
    resolve_store,
    store_status,
)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args],
                          capture_output=True, text=True,
                          check=True).stdout.strip()

class TestGitStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        base = Path(self.tmp)
        self.remote = base / "remote.git"
        subprocess.run(["git", "init", "--bare", str(self.remote)],
                       capture_output=True, check=True)
        self.seed = base / "seed"
        self.seed.mkdir()
        _git(self.seed, "init", "-q", "-b", "main")
        _git(self.seed, "config", "user.email", "t@t.t")
        _git(self.seed, "config", "user.name", "test")
        (self.seed / "README.md").write_text("odr store\n")
        _git(self.seed, "add", "-A")
        _git(self.seed, "commit", "-q", "-m", "seed")
        _git(self.seed, "push", "-q", str(self.remote), "main")

    def test_resolve_local_git_store_pulls(self):
        clone = Path(self.tmp) / "clone"
        subprocess.run(["git", "clone", "-q", str(self.remote), str(clone)],
                       capture_output=True, check=True)
        got = resolve_store(clone)
        self.assertEqual(got, clone)

    def test_missing_local_store_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(LessonShareError):
                resolve_store(Path(tmp) / "nope")
class TestWorldBundles(unittest.TestCase):
    """Worlds: archived discovery trees as checksummed, redacted replay data."""

    TREE = {
        "root_id": "t/seed",
        "nodes": [
            {"node_id": "t/seed", "action": "warm_start", "result": {"recipe": False},
             "score": 0.0, "parent_id": None, "children": ["t/a"],
             "thought": "", "sentinel": None, "termination_reason": None},
            {"node_id": "t/a", "action": "write_code:abc", "result": {"errors": []},
             "score": 1.0, "parent_id": "t/seed", "children": [],
             "thought": "PLAN: dict trick", "sentinel": None,
             "termination_reason": "completed"},
        ],
    }

    def _tree_root(self, tmp: str) -> Path:
        root = Path(tmp) / "mem"
        (root / "trees").mkdir(parents=True)
        (root / "trees" / "t1.json").write_text(json.dumps(self.TREE), encoding="utf-8")
        return root

    def test_export_import_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree_root(tmp)
            store = Path(tmp) / "store"
            st = export_worlds(root, store)
            self.assertTrue(st.ok)
            self.assertEqual(st.categories, {"t1": 2})
            self.assertTrue(store_status(store).ok)
            dst = Path(tmp) / "dst"
            counts = import_worlds(store, dst)
            self.assertEqual(counts, {"t1": 2})
            imported = json.loads((dst / "trees" / "imported" / "t1.json").read_text())
            self.assertEqual(imported["root_id"], "t/seed")
            self.assertEqual([n["node_id"] for n in imported["nodes"]],
                             ["t/seed", "t/a"])
            # thought/sentinel/termination round-trip; load_tree untouched
            self.assertEqual(imported["nodes"][1]["thought"], "PLAN: dict trick")
            from open_dream_rsi.memory import DreamMemory
            self.assertIsNone(DreamMemory(dst).load_tree("t1"))

    def test_import_is_idempotent_and_filters_task_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree_root(tmp)
            store = Path(tmp) / "store"
            export_worlds(root, store)
            dst = Path(tmp) / "dst"
            first = import_worlds(store, dst)
            second = import_worlds(store, dst)
            self.assertEqual(first, second)
            self.assertEqual(import_worlds(store, dst, task_ids=["nope"]), {})
            self.assertEqual(import_worlds(store, dst, task_ids=["t1"]),
                             {"t1": 2})

    def test_unknown_node_fields_dropped_not_trusted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree_root(tmp)
            # a foreign field smuggles future information into the world
            poisoned = json.loads(json.dumps(self.TREE))
            poisoned["nodes"][1]["future_hint"] = "the fix is X"
            (root / "trees" / "t1.json").write_text(json.dumps(poisoned), encoding="utf-8")
            store = Path(tmp) / "store"
            export_worlds(root, store)
            dst = Path(tmp) / "dst"
            import_worlds(store, dst)
            imported = json.loads((dst / "trees" / "imported" / "t1.json").read_text())
            self.assertNotIn("future_hint", imported["nodes"][1])

    def test_tampered_world_aborts_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree_root(tmp)
            store = Path(tmp) / "store"
            export_worlds(root, store)
            victim = store / "trees" / "t1.json"
            payload = json.loads(victim.read_text())
            payload["tree"]["nodes"][0]["score"] = 999.0
            victim.write_text(json.dumps(payload), encoding="utf-8")
            dst = Path(tmp) / "dst"
            with self.assertRaises(LessonShareError):
                import_worlds(store, dst)
            self.assertFalse((dst / "trees" / "imported").exists())

    def test_secrets_redacted_on_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree_root(tmp)
            poisoned = json.loads(json.dumps(self.TREE))
            poisoned["nodes"][1]["result"] = {"note": "password=hunter2secret"}
            (root / "trees" / "t1.json").write_text(json.dumps(poisoned), encoding="utf-8")
            store = Path(tmp) / "store"
            export_worlds(root, store)
            blob = (store / "trees" / "t1.json").read_text(encoding="utf-8")
            self.assertNotIn("hunter2secret", blob)
            self.assertIn("[REDACTED]", blob)


VALID_POLICY = (
    "def choose_action(frontier, step):\n"
    "    if not frontier:\n"
    "        return None\n"
    "    return max(frontier, key=lambda e: e.get('score', 0.0))['node_id']\n"
)

BAD_POLICY = "import os\ndef choose_action(f, s):\n    return None\n"

CODES = {"median": {"code": VALID_POLICY, "score": 0.9, "steps": 12,
                    "updated_at": 3.0},
         "broken": {"code": BAD_POLICY, "score": 0.5, "steps": 4,
                    "updated_at": 3.0}}


class TestPolicyBundles(unittest.TestCase):
    def _codes_root(self, tmp: str) -> Path:
        root = Path(tmp) / "mem"
        root.mkdir(parents=True, exist_ok=True)
        (root / "policy_codes.json").write_text(json.dumps(CODES), encoding="utf-8")
        return root

    def test_export_skips_gate_failures_and_imports_as_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._codes_root(tmp)
            store = Path(tmp) / "store"
            st = export_policies(root, store)
            self.assertEqual(st.categories, {"median": 1})  # bad code never ships
            dst = Path(tmp) / "dst"
            counts = import_policies(store, dst)
            self.assertEqual(counts, {"median": 1})
            from open_dream_rsi.lifecycle import ArtifactLifecycleManager
            alm = ArtifactLifecycleManager(dst)
            states = alm.states(slot="policy_program:median:imported")
            self.assertTrue(states)
            self.assertTrue(all(s.lifecycle_state == "VALIDATED" for s in states))
            self.assertTrue(alm.verify_materialization())
            # never becomes the runtime incumbent: the runtime view file is
            # not written by an import at all
            self.assertFalse((dst / "policy_codes.json").exists())

    def test_import_idempotent_no_version_churn(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._codes_root(tmp)
            store = Path(tmp) / "store"
            export_policies(root, store)
            dst = Path(tmp) / "dst"
            import_policies(store, dst)
            from open_dream_rsi.lifecycle import ArtifactLifecycleManager
            alm = ArtifactLifecycleManager(dst)
            n1 = len(alm.states(slot="policy_program:median:imported"))
            import_policies(store, dst)
            n2 = len(alm.states(slot="policy_program:median:imported"))
            self.assertEqual(n1, n2)

    def test_tampered_policy_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._codes_root(tmp)
            store = Path(tmp) / "store"
            export_policies(root, store)
            victim = store / "policies" / "median.json"
            payload = json.loads(victim.read_text())
            payload["code"] = BAD_POLICY
            victim.write_text(json.dumps(payload), encoding="utf-8")
            dst = Path(tmp) / "dst"
            with self.assertRaises(LessonShareError):
                import_policies(store, dst)


if __name__ == "__main__":
    unittest.main()
