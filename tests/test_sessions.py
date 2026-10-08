"""Session ingestion: Hermes state.db + Cursor state.vscdb -> evidence
episodes -> staging lessons -> Hermes skills. Transcripts are hostile
input: read-only, redacted, parse-defensive; activation stays gate-gated.

Run:  python -m unittest discover -s tests
"""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from open_dream_rsi.sessions import (
    Episode,
    SessionMessage,
    episode_failures,
    lessons_to_skills,
    load_episodes,
    read_cursor_sessions,
    read_hermes_sessions,
    redact_secrets,
    write_episodes,
)


class TestRedaction(unittest.TestCase):
    def test_credential_shapes_scrubbed(self):
        # token-shaped literals are CONSTRUCTED at runtime: a committed
        # secret-shaped string (even a fake) trips GitHub push protection
        cases = [
            "key = sk-" + "aB3dEf6gHi9jKl2mNo",
            "ghp_" + "A" * 36,
            "AKIA" + "A" * 16,
            ("Authorization: Bear" + "er ") + "abcdef123456",
        ]
        for text in cases:
            out = redact_secrets(text)
            self.assertNotIn(text, out)
            self.assertIn("[REDACTED]", out)

    def test_prose_untouched(self):
        s = "the median of an even-length list is the mean of the middles"
        self.assertEqual(redact_secrets(s), s)


def _hermes_db(tmp: str) -> str:
    path = str(Path(tmp) / "state.db")
    con = sqlite3.connect(path)
    con.execute("create table sessions (id text, source text, title text, "
                "display_name text, cwd text, started_at real, archived int, "
                "hidden int)")
    con.execute("create table messages (id integer primary key, session_id text, "
                "role text, content text, tool_name text, timestamp real, "
                "active int)")
    con.execute("insert into sessions values "
                "('s1','cli','median task',null,'/repo/median',100,0,0)")
    con.execute("insert into sessions values "
                "('s2','cli','archived one',null,'/x',200,1,0)")
    con.execute("insert into messages values (1,'s1','user','sort the list',"
                "'',101,1)")
    con.execute("insert into messages values (2,'s1','tool','ValueError: x "
                "is not defined','bash',102,1)")
    con.execute("insert into messages values (3,'s1','assistant','fixed, "
                "api_key = sk-" + "liveTOKENdeadBEEF123','',103,1)")
    con.execute("insert into messages values (4,'s2','user','secret chat',"
                "'',201,1)")
    con.commit()
    con.close()
    return path


class TestHermesReader(unittest.TestCase):
    def test_reads_sessions_and_skips_archived(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _hermes_db(tmp)
            eps = read_hermes_sessions(path)
            self.assertEqual(len(eps), 1)
            ep = eps[0]
            self.assertEqual(ep.source, "hermes")
            self.assertEqual(ep.title, "median task")
            self.assertEqual([m.role for m in ep.messages],
                             ["user", "tool", "assistant"])
            # secrets scrubbed by default
            self.assertNotIn("liveTOKEN" + "deadBEEF123",
                             ep.messages[-1].text)
            # tool rows keep their name for failure shaping
            self.assertEqual(ep.messages[1].name, "bash")

    def test_source_filter_and_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _hermes_db(tmp)
            self.assertEqual(read_hermes_sessions(path, source="desktop"), [])
            self.assertEqual(len(read_hermes_sessions(path, source="cli")), 1)
            self.assertEqual(len(read_hermes_sessions(path, limit=0)), 0)

    def test_missing_tables_yield_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.db"
            sqlite3.connect(path).close()
            self.assertEqual(read_hermes_sessions(path), [])

    def test_database_is_never_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _hermes_db(tmp)
            before = Path(path).read_bytes()
            read_hermes_sessions(path)
            self.assertEqual(Path(path).read_bytes(), before)


def _cursor_user_dir(tmp: str, *, null_row: bool = False) -> str:
    """Synthetic Cursor store: global state.vscdb + one workspace registry."""
    user = Path(tmp) / "User"
    (user / "globalStorage").mkdir(parents=True)
    (user / "workspaceStorage" / "hash1").mkdir(parents=True)
    (user / "workspaceStorage" / "hash1" / "workspace.json").write_text(
        json.dumps({"folder": "file:///repo/scraper"}), encoding="utf-8")
    ws = sqlite3.connect(user / "workspaceStorage" / "hash1" / "state.vscdb")
    ws.execute("create table ItemTable (key TEXT UNIQUE, value BLOB)")
    ws.execute("insert into ItemTable values ('composer.composerData', ?)",
               (json.dumps({"allComposers": [
                   {"composerId": "c-1", "name": "scraper fix"}]}),))
    ws.commit()
    ws.close()

    g = sqlite3.connect(user / "globalStorage" / "state.vscdb")
    g.execute("create table ItemTable (key TEXT UNIQUE, value BLOB)")
    g.execute("create table cursorDiskKV (key TEXT UNIQUE, value BLOB)")
    header = {"composerId": "c-1", "name": "scraper fix",
              "createdAt": 1737136403732,
              "fullConversationHeadersOnly": [
                  {"bubbleId": "b2", "type": 2},   # ORDER differs from key sort
                  {"bubbleId": "b1", "type": 1}]}
    g.execute("insert into cursorDiskKV values ('composerData:c-1', ?)",
              (json.dumps(header),))
    if null_row:
        g.execute("insert into cursorDiskKV values ('composerData:c-null', NULL)")
    g.execute("insert into cursorDiskKV values ('composerData:c-junk', 'not json')")
    g.execute("insert into cursorDiskKV values ('bubbleId:c-1:b1', ?)",
              (json.dumps({"type": 1, "text": "the site blocks requests",
                           "createdAt": 1737136404000}),))
    g.execute("insert into cursorDiskKV values ('bubbleId:c-1:b2', ?)",
    (json.dumps({"type": 2, "text": "Error: 429 Too Many Requests "
               "token=sk-" + "SECRETabc123456789",
               "createdAt": 1737136405000}),))
    g.execute("insert into cursorDiskKV values ('bubbleId:c-1:b3', ?)",
              (json.dumps({"type": 2, "text": ""}),))  # tool-only: skipped
    g.commit()
    g.close()
    return str(user)


class TestCursorReader(unittest.TestCase):
    def test_order_follows_header_not_key_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            user = _cursor_user_dir(tmp)
            eps = read_cursor_sessions(user)
            self.assertEqual(len(eps), 1)
            ep = eps[0]
            self.assertEqual(ep.source, "cursor")
            self.assertEqual(ep.cwd, "/repo/scraper")
            # header order: b2 (assistant) first, then b1 (user)
            self.assertEqual([m.role for m in ep.messages],
                             ["assistant", "user"])
            self.assertEqual([m.text[:5] for m in ep.messages],
                             ["Error", "the s"])
            secret = "sk-" + "SECRETabc123456789"
            self.assertNotIn(secret, json.dumps(ep.to_dict()))
            self.assertIn("[REDACTED]", ep.messages[0].text)

    def test_null_and_junk_rows_never_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            user = _cursor_user_dir(tmp, null_row=True)
            eps = read_cursor_sessions(user)   # NULL value + non-JSON rows
            self.assertEqual(len(eps), 1)

    def test_empty_store_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(read_cursor_sessions(tmp), [])


class TestFailureShaping(unittest.TestCase):
    def test_error_markers_and_tool_rows_shape_evidence(self):
        ep = Episode(source="hermes", session_id="s", messages=[
            SessionMessage(role="assistant", text="looks good"),
            SessionMessage(role="tool", name="bash",
                           text="Traceback: KeyError 'krs'"),
        ])
        fails = episode_failures(ep)
        self.assertEqual(len(fails), 1)
        self.assertEqual(fails[0]["action"], "bash")
        self.assertEqual(fails[0]["score"], 0.0)
        self.assertTrue(fails[0]["errors"])


class TestEpisodeRoundTrip(unittest.TestCase):
    def test_write_then_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            eps = [Episode(source="cursor", session_id="c1", title="t",
                           cwd="/x", messages=[
                               SessionMessage(role="user", text="hi",
                                              timestamp=1.0)])]
            path = Path(tmp) / "ep.jsonl"
            self.assertEqual(write_episodes(eps, path), 1)
            path.open("a").write("garbage-not-json\n")  # resilience
            got = load_episodes(path)
            self.assertEqual(len(got), 1)
            self.assertEqual(got[0].messages[0].text, "hi")


GOOD = {"trigger": "median even",
        "text": "Sort first, return the mean of the two middle values, "
                "then answer immediately."}


def _memory_with_lessons(tmp: str) -> Path:
    root = Path(tmp) / "mem"
    root.mkdir()
    (root / "lessons.json").write_text(json.dumps({
        "median": [
            {**GOOD, "status": "active", "wins": 3, "uses": 4},
            {"trigger": "staging one", "text": "z" * 40, "status": "staging",
             "wins": 0, "uses": 0},
        ],
        "other": [{"trigger": "t", "text": "y" * 40, "status": "rejected"}],
    }), encoding="utf-8")
    return root


class TestSkillsExport(unittest.TestCase):
    def test_only_active_lessons_become_skills(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _memory_with_lessons(tmp)
            out = Path(tmp) / "skills"
            written = lessons_to_skills(root, out)
            self.assertEqual(len(written), 1)
            text = written[0].read_text(encoding="utf-8")
            self.assertIn("median", str(written[0]))
            self.assertTrue(text.startswith("---"))
            self.assertIn("name: odr-median", text)
            self.assertIn("Sort first, return the mean", text)
            self.assertNotIn("staging one", text)
            self.assertNotIn("rejected", written[0].read_text())

    def test_include_staging_option(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _memory_with_lessons(tmp)
            text = lessons_to_skills(root, Path(tmp) / "s2",
                                     only_active=False)[0].read_text()
            self.assertIn("staging one", text)

    def test_category_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _memory_with_lessons(tmp)
            self.assertEqual(lessons_to_skills(root, Path(tmp) / "s3",
                                               categories=["missing"]), [])


if __name__ == "__main__":
    unittest.main()
