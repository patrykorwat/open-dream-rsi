"""Fast test subset: everything except the slow benchmark/sandbox modules.

Full suite (the contract):  python -m unittest discover -s tests   (~2 min)
Fast subset (quick signal): python scripts/fast_tests.py            (~15 s)

CI runs both — the fast lane gives sub-minute feedback on every push, the
full matrix job stays the authority. A module belongs in SLOW_MODULES only
when its tests are multi-second (real subprocess spawns, benchmark
contracts, replay fixtures); keep the list short and re-time with
scripts/time_tests.py when in doubt.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SLOW_MODULES = {
    "test_bench_policy",           # ~87 s: deterministic bench-policy contracts
    "test_sandbox_hardening",      # ~5 s: real subprocess spawns
    "test_sentinel_audit_replay",  # ~4 s: 30-day replay fixture
    "test_bench",                  # ~2 s: bench contract
    "test_llm",                    # ~2 s: client retries / timeouts
}


def main() -> int:
    loader = unittest.TestLoader()
    suite = loader.discover("tests")

    fast = unittest.TestSuite()
    skipped = 0

    def walk(s: unittest.TestSuite) -> None:
        nonlocal skipped
        for t in s:
            if isinstance(t, unittest.TestSuite):
                walk(t)
            elif t.__class__.__module__ in SLOW_MODULES:
                skipped += 1
            else:
                fast.addTest(t)

    walk(suite)
    total = fast.countTestCases() + skipped
    print(f"fast subset: {fast.countTestCases()} tests "
          f"({skipped} in slow modules skipped of {total} total)")
    result = unittest.TextTestRunner(verbosity=1).run(fast)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
