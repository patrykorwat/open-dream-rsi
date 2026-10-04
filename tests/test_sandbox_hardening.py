"""Tests for the hardened sandbox boundary (external review, 2026-10).

Covers the three gaps the review found:
* the ``__builtins__`` name bypass of the AST gate;
* timeout killing only the direct child (grandchildren survived);
* no memory/CPU/process limits on sandboxed code.
Plus the promotion-gate fix: the incumbent is re-scored on the CURRENT tree.
"""

import json
import sys
import time
import unittest
from pathlib import Path

from open_dream_rsi.core.policygen import (
    PolicySandbox,
    PolicyValidationError,
    validate_policy_source,
)
from open_dream_rsi.sandbox import run_isolated


class BuiltinsGateTest(unittest.TestCase):
    def test_rejects_builtins_name(self):
        for name in ("__builtins__", "builtins"):
            src = (
                "def choose_action(frontier, step):\n"
                f"    b = {name}\n"
                "    return None\n"
            )
            with self.assertRaises(PolicyValidationError) as cm:
                validate_policy_source(src)
            self.assertIn(name, str(cm.exception))

    def test_rejects_constructed_import_via_builtins(self):
        # The exact bypass the review demonstrated: index __builtins__ with a
        # constructed key so no dunder appears as an attribute access.
        src = (
            "def choose_action(frontier, step):\n"
            "    imp = __builtins__['__imp' + 'ort__']\n"
            "    os = imp('os')\n"
            "    return os.getcwd()\n"
        )
        with self.assertRaises(PolicyValidationError):
            validate_policy_source(src)

    def test_plain_policy_still_passes(self):
        src = (
            "def choose_action(frontier, step):\n"
            "    if not frontier:\n"
            "        return None\n"
            "    return max(frontier, key=lambda n: n['score'])['node_id']\n"
        )
        validate_policy_source(src)  # must not raise


class ProcessGroupKillTest(unittest.TestCase):
    """A candidate that backgrounds a child must not outlive its timeout."""

    def test_timeout_kills_grandchildren(self):
        marker = Path("/tmp").resolve() / f"odr-pgkill-{time.time_ns()}"
        script = (
            "import subprocess, time\n"
            f"subprocess.Popen(['sh', '-c', 'sleep 30 && echo alive > {marker}'], "
            "start_new_session=False)\n"
            "time.sleep(60)\n"
        )
        with self.assertRaises(Exception):
            run_isolated(
                [sys.executable, "-I", "-c", script],
                timeout=2.0,
                env={"PATH": "/usr/bin:/bin", "PYTHONWARNINGS": "ignore"},
            )
        # give any surviving grandchild a chance to write the marker
        deadline = time.time() + 3.0
        wrote = False
        while time.time() < deadline:
            if marker.exists():
                wrote = True
                break
            time.sleep(0.25)
        self.assertFalse(wrote, "grandchild survived the group kill")
        marker.unlink(missing_ok=True)

    def test_normal_run_unaffected(self):
        proc = run_isolated([sys.executable, "-I", "-c", "print('ok')"], timeout=10)
        self.assertEqual(proc.stdout.strip(), "ok")
        self.assertEqual(proc.returncode, 0)


class ResourceLimitTest(unittest.TestCase):
    @unittest.skipUnless(sys.platform.startswith("linux"), "POSIX rlimits")
    def test_address_space_limit_enforced(self):
        script = "b = bytearray(3 * 1024 * 1024 * 1024); print('allocated')"
        proc = run_isolated([sys.executable, "-I", "-c", script], timeout=20)
        self.assertNotEqual(proc.returncode, 0)
        self.assertNotIn("allocated", proc.stdout)

    @unittest.skipUnless(sys.platform.startswith("linux"), "POSIX rlimits")
    def test_scrubbed_env_no_keys(self):
        script = "import os; print(repr(sorted(os.environ)))"
        proc = run_isolated([sys.executable, "-I", "-c", script], timeout=20)
        # CPython may inject LC_CTYPE in UTF-8 mode; nothing else may cross.
        keys = set(k for k in ["PATH", "LC_CTYPE"])
        self.assertTrue(eval(proc.stdout.strip()) and
                        set(eval(proc.stdout.strip())) <= keys,
                        f"env leaked: {proc.stdout!r}")


class IncumbentReevaluationTest(unittest.TestCase):
    """The promotion gate must compare candidate vs incumbent on the SAME tree."""

    def test_stale_incumbent_score_cannot_promote_a_worse_candidate(self):
        from open_dream_rsi.loop import AutoRSIRuntime, CycleReport, Task
        from open_dream_rsi.memory import DreamMemory
        from open_dream_rsi.core.tree import DiscoveryTree
        import tempfile

        # Decoy-trap tree: re-polishing the decoy chain never improves, the
        # fix lives under the low-score promising node one expansion away.
        tree = DiscoveryTree()
        seed = tree.add_node("seed", action="warm_start", result={}, score=0.0)
        decoy = tree.add_node("d0", action="write_code:decoy", result={
            "errors": ["t2 -> got 2 != 3"]}, score=0.667, parent_id=seed.node_id)
        tree.add_node("d1", action="write_code:decoy1", result={
            "errors": ["t2 -> got 2 != 3"]}, score=0.667, parent_id=decoy.node_id)
        prom = tree.add_node("p0", action="write_code:promising", result={
            "errors": ["t0 -> got [2] != [1,2]"]}, score=0.333, parent_id=seed.node_id)
        tree.add_node("fix", action="write_code:fixed", result={}, score=1.0,
                      parent_id=prom.node_id)

        # A strong incumbent policy (explores fresh branches first).
        strong = (
            "def choose_action(frontier, step):\n"
            "    if not frontier:\n"
            "        return None\n"
            "    fresh = [n for n in frontier if n['children'] == 0]\n"
            "    pool = fresh or frontier\n"
            "    return max(pool, key=lambda n: n['outcome'])['node_id']\n"
        )
        # A weak candidate: always picks the WORST outcome node.
        weak = (
            "def choose_action(frontier, step):\n"
            "    if not frontier:\n"
            "        return None\n"
            "    return min(frontier, key=lambda n: n['outcome'])['node_id']\n"
        )

        class Scripted:
            def __init__(self, source):
                self.source = source
                self.policy_msgs = 0

            def chat(self, messages, model=None, temperature=0.7, max_tokens=None):
                system = messages[0]["content"] if messages else ""
                if "exploration policy" in system:
                    self.policy_msgs += 1
                    return f"```python\n{self.source}\n```"
                return "```python\ndef add(a, b):\n    return a + b\n```"

        mem = DreamMemory(root=f"{tempfile.mkdtemp(prefix='odr-gate-')}/mem")
        # Stale stored score BELOW anything the weak candidate can reach...
        mem.save_policy_code("math", strong, score=-0.5, steps=0)
        llm = Scripted(weak)
        task = Task(task_id="t", category="math",
                    prompt="Implement add(a, b).",
                    tests=[{"call": "add(2, 3)", "expected": 5}],
                    max_attempts=1)
        rt = AutoRSIRuntime(client=llm, memory=mem, tasks=[task],
                            api_call_budget=30, dream_iterations=0, rng_seed=1)

        report = CycleReport()
        rt._maybe_evolve_policy_code(task, tree, report)

        # With the old stored score (-0.5) the weak candidate would pass the
        # gate; re-evaluated on THIS tree the strong incumbent beats it.
        entry = mem.get_policy_code("math")
        self.assertEqual(entry["code"], strong)
        self.assertEqual(report.improvements, [])


if __name__ == "__main__":
    unittest.main()
