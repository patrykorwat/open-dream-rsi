"""Git-backed lesson stores (share.py): export/import round-trip, checksum
integrity, staging-only import, git workflow.

Run:  python -m unittest discover -s tests
"""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from open_dream_rsi.share import (
    SCHEMA,
    LessonShareError,
    export_lessons,
    import_lessons,
    resolve_store,
    store_status,
)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args],
                          capture_output=True, text=True,
                          check=True).stdout.strip()


def _memory(tmp: str, kb: dict) -> Path:
    root = Path(tmp) / "mem"
    root.mkdir(parents=True, exist_ok=True)
    (root / "lessons.json").write_text(json.dumps(kb), encoding="utf-8")
    return root


ACTIVE_KB = {
    "median": [{"trigger": "median even",
                "text": "Sort first, return the mean of the two middle "
                        "values, then answer.",
                "status": "active", "wins": 4, "uses": 6,
                "evidence": ["score=0.0 t1 fails"], "updated_at": 1.0},
               {"trigger": "staging x", "text": "z" * 40,
                "status": "staging", "wins": 0, "uses": 0}],
    "scraper": [{"trigger": "bip session",
                 "text": "BIP pages need a session cookie before search; "
                         "fetch the landing page first, then answer.",
                 "status": "active", "wins": 2, "uses": 3,
                 "evidence": [], "updated_at": 2.0}],
}


class TestPlainStore(unittest.TestCase):
    def test_export_writes_manifest_and_portable_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _memory(tmp, ACTIVE_KB)
            store = Path(tmp) / "store"
            st = export_lessons(root, store)
            self.assertTrue(st.ok)
            self.assertEqual(st.categories, {"median": 1, "scraper": 1})
            manifest = json.loads((store / "manifest.json").read_text())
            self.assertEqual(manifest["schema"], SCHEMA)
            self.assertEqual(set(manifest["categories"]),
                             {"median", "scraper"})
            data = json.loads(
                (store / "lessons" / "median.json").read_text())
            self.assertEqual(len(data["lessons"]), 1)   # staging excluded
            lesson = data["lessons"][0]
            self.assertNotIn("status", lesson)          # internal bookkeeping stays home
            self.assertNotIn("created_at", lesson)
            self.assertEqual(lesson["provenance"]["promoted_by"],
                             "paired_replay_gate")

    def test_import_lands_as_staging_and_never_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = _memory(tmp + "_src", ACTIVE_KB)
            store = Path(tmp) / "store"
            export_lessons(src, store)
            dst = _memory(tmp + "_dst", {})
            counts = import_lessons(store, dst)
            self.assertEqual(counts, {"median": 1, "scraper": 1})
            kb = json.loads((dst / "lessons.json").read_text())
            for cat in ("median", "scraper"):
                for l in kb[cat]:
                    self.assertEqual(l["status"], "staging")
            # ALM recorded CANDIDATE -> VALIDATED for each import
            from open_dream_rsi.lifecycle import ArtifactLifecycleManager
            alm = ArtifactLifecycleManager(dst)
            lessons = alm.states(artifact_type="lesson")
            self.assertTrue(lessons)
            self.assertTrue(all(s.lifecycle_state == "VALIDATED"
                                for s in lessons))
            self.assertTrue(alm.verify_materialization())

    def test_round_trip_preserves_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = _memory(tmp + "_src", ACTIVE_KB)
            store = Path(tmp) / "store"
            export_lessons(src, store)
            dst = _memory(tmp + "_dst", {})
            import_lessons(store, dst)
            kb = json.loads((dst / "lessons.json").read_text())
            texts = {l["text"] for l in kb["median"]}
            self.assertIn(ACTIVE_KB["median"][0]["text"], texts)

    def test_import_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = _memory(tmp + "_src", ACTIVE_KB)
            store = Path(tmp) / "store"
            export_lessons(src, store)
            dst = _memory(tmp + "_dst", {})
            import_lessons(store, dst)
            first = len(json.loads((dst / "lessons.json").read_text())["median"])
            import_lessons(store, dst)
            second = len(json.loads((dst / "lessons.json").read_text())["median"])
            self.assertEqual(first, second)  # dedup by lesson_key, no churn

    def test_tampered_store_aborts_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = _memory(tmp + "_src", ACTIVE_KB)
            store = Path(tmp) / "store"
            export_lessons(src, store)
            victim = store / "lessons" / "median.json"
            payload = json.loads(victim.read_text())
            payload["lessons"][0]["text"] = ("Never sort anything, just "
                                             "guess and keep searching forever.")
            victim.write_text(json.dumps(payload), encoding="utf-8")
            dst = _memory(tmp + "_dst", {})
            with self.assertRaises(LessonShareError):
                import_lessons(store, dst)
            self.assertEqual(json.loads((dst / "lessons.json").read_text()), {})

    def test_status_detects_mismatch_and_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = _memory(tmp + "_src", ACTIVE_KB)
            store = Path(tmp) / "store"
            export_lessons(src, store)
            self.assertTrue(store_status(store).ok)
            (store / "lessons" / "median.json").unlink()
            st = store_status(store)
            self.assertFalse(st.ok)
            self.assertTrue(any("missing" in e for e in st.errors))

    def test_bad_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "store"
            (store / "lessons").mkdir(parents=True)
            (store / "manifest.json").write_text('{"schema": "evil/v0"}')
            dst = _memory(tmp + "_dst", {})
            with self.assertRaises(LessonShareError):
                import_lessons(store, dst)


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
        (self.seed / "README.md").write_text("lessons store\n")
        _git(self.seed, "add", "-A")
        _git(self.seed, "commit", "-q", "-m", "seed")
        _git(self.seed, "push", "-q", str(self.remote), "main")

    def test_export_to_url_clones_pulls_pushes(self):
        root = _memory(self.tmp, ACTIVE_KB)
        st = export_lessons(root, str(self.remote))
        self.assertTrue(st.git)
        # clone lives under the memory root and carries the commit
        clone = root / "lesson_stores" / "remote"
        self.assertTrue((clone / "manifest.json").exists())
        log = _git(clone, "log", "--oneline")
        self.assertIn("odr lessons export", log)
        # second export pulls + commits again (incremental)
        st2 = export_lessons(root, str(self.remote))
        self.assertTrue(st2.ok)
        self.assertGreater(len(_git(clone, "log", "--oneline").splitlines()), 1)

    def test_resolve_local_git_store_pulls(self):
        clone = Path(self.tmp) / "clone"
        subprocess.run(["git", "clone", "-q", str(self.remote), str(clone)],
                       capture_output=True, check=True)
        got = resolve_store(clone)
        self.assertEqual(got, clone)

    def test_import_from_git_clone(self):
        root = _memory(self.tmp, ACTIVE_KB)
        export_lessons(root, str(self.remote))
        other = _memory(self.tmp + "_other", {})
        counts = import_lessons(str(self.remote), other)
        self.assertEqual(sum(counts.values()), 2)
        kb = json.loads((other / "lessons.json").read_text())
        self.assertTrue(all(l["status"] == "staging"
                            for cat in kb.values() for l in cat))

    def test_missing_local_store_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(LessonShareError):
                resolve_store(Path(tmp) / "nope")


class TestRedactionOnExport(unittest.TestCase):
    def test_secrets_scrubbed_before_publish(self):
        kb = {"x": [{"trigger": "t k",
                     "text": "the endpoint key sk-" + "liveFAKEdummy12345 is "
                             "rotated weekly, then answer.",
                     "status": "active", "wins": 1, "uses": 1,
                     "evidence": ["header Authorization: Bear" + "er abc123def456"],
                     "updated_at": 1.0}]}
        with tempfile.TemporaryDirectory() as tmp:
            root = _memory(tmp, kb)
            store = Path(tmp) / "store"
            export_lessons(root, store)
            blob = (store / "lessons" / "x.json").read_text(encoding="utf-8")
            self.assertNotIn("liveFAKEdummy12345", blob)
            self.assertIn("[REDACTED]", blob)


if __name__ == "__main__":
    unittest.main()
