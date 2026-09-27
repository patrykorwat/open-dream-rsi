"""Real-model compression benchmark: cold re-solve vs shared dreaming memory.

Answers the only question that matters for daily use: once the loop has
learned a category, how many API calls does the NEXT task of that category
cost? Same model, same prompts, same machine as a normal session — only the
memory differs.

    python3 scripts/bench_real.py --base-url http://YOUR-VLLM-HOST:8000/v1 \
        --model local-inference-lab/Qwen3.8-Flash-Next-NVFP4 [--reps 1]

Protocol (per category):
  COLD  — each eval task solved against a FRESH memory dir (what you pay
          today: no cross-task learning),  budget 6 calls per task.
  WARM  — a training task per category is solved first (the "dreaming"
          phase), then the SAME eval tasks run against the trained, shared
          memory — the warm_start recipe + policies + lessons kick in.

Headline numbers: mean API calls per solved eval task (COLD vs WARM) and
wall-time ratio. Everything is written to JSON so the table can be rebuilt
and audited from raw events (events.jsonl per run).
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from open_dream_rsi.loop import AutoRSIRuntime, Task  # noqa: E402
from open_dream_rsi.memory import DreamMemory  # noqa: E402

SUITE: Dict[str, Dict[str, Any]] = {
    "strutil": {
        "train": ("camel_to_snake: convert CamelCase (and punctuation) to "
                  "snake_case, lowercase, underscores.",
                  [{"call": "camel_to_snake('ParseHTMLFile')", "expected": "parse_html_file"},
                   {"call": "camel_to_snake('get.HTTPUrl')", "expected": "get_http_url"}]),
        "eval": [
            ("slugify_kebab: convert CamelCase + punctuation to kebab-case "
             "with hyphens, lowercase.",
             [{"call": "slugify_kebab('HelloWorld')", "expected": "hello-world"},
              {"call": "slugify_kebab('My.File-Name')", "expected": "my-file-name"}]),
            ("pascal_case: convert snake_case input to PascalCase.",
             [{"call": "pascal_case('hello_world')", "expected": "HelloWorld"},
              {"call": "pascal_case('parse_html_file')", "expected": "ParseHtmlFile"}]),
            ("words_to_camel: join a list of lowercase words into camelCase.",
             [{"call": "words_to_camel(['get','user','name'])", "expected": "getUserName"},
              {"call": "words_to_camel(['id'])", "expected": "id"}]),
        ],
    },
    "parse": {
        "train": ("extract_kv: parse 'key=value' lines (one per line) into a "
                  "dict; ignore blank lines and '#' comments.",
                  [{"call": "extract_kv('a=1\\nb=2')", "expected": {"a": "1", "b": "2"}},
                   {"call": "extract_kv('# x\\na=1')", "expected": {"a": "1"}}]),
        "eval": [
            ("extract_json_field: return value of a top-level JSON key from a "
             "JSON string, None if missing.",
             [{"call": "extract_json_field('{\"a\": 5}', 'a')", "expected": 5},
              {"call": "extract_json_field('{\"a\": 5}', 'b')", "expected": None}]),
            ("parse_range: parse '10-20' into (10, 20); a plain number N "
             "becomes (N, N).",
             [{"call": "parse_range('10-20')", "expected": (10, 20)},
              {"call": "parse_range('7')", "expected": (7, 7)}]),
            ("csv_to_dicts: split CSV text (first row = header) into list of "
             "dicts.",
             [{"call": "csv_to_dicts('a,b\\n1,2')", "expected": [{"a": "1", "b": "2"}]},
              {"call": "csv_to_dicts('x\\n5')", "expected": [{"x": "5"}]}]),
        ],
    },
    "mathy": {
        "train": ("clamp(x, lo, hi): limit x into [lo, hi].",
                  [{"call": "clamp(5, 0, 3)", "expected": 3},
                   {"call": "clamp(-1, 0, 3)", "expected": 0}]),
        "eval": [
            ("lerp(a, b, t): linear interpolation at t in [0,1].",
             [{"call": "lerp(0, 10, 0.5)", "expected": 5.0},
              {"call": "lerp(2, 4, 0)", "expected": 2}]),
            ("moving_avg(xs, n): trailing moving average of last n values "
             "as a list.",
             [{"call": "moving_avg([1,2,3,4], 2)", "expected": [1.5, 2.5, 3.5]},
              {"call": "moving_avg([5], 2)", "expected": [5.0]}]),
            ("round_sig(x, n): round x to n significant digits (float).",
             [{"call": "round_sig(1234.5678, 3)", "expected": 1230.0},
              {"call": "round_sig(0.00456, 2)", "expected": 0.0046}]),
        ],
    },
    "textproc": {
        "train": ("truncate_words(s, n): keep first n words, append '...' if "
                  "truncated.",
                  [{"call": "truncate_words('a b c d', 2)", "expected": "a b..."},
                   {"call": "truncate_words('a b', 5)", "expected": "a b"}]),
        "eval": [
            ("strip_accents: normalize unicode to ASCII dropping accents.",
             [{"call": "strip_accents('zażółć gęślą')", "expected": "zazolc gesla"},
              {"call": "strip_accents('naïve')", "expected": "naive"}]),
            ("word_freq(s): dict of lowercase word -> count.",
             [{"call": "word_freq('a b a')", "expected": {"a": 2, "b": 1}},
              {"call": "word_freq('')", "expected": {}}]),
            ("initials(full): 'Jan Kowalski' -> 'JK'.",
             [{"call": "initials('Jan Kowalski')", "expected": "JK"},
              {"call": "initials('anna maria nowak')", "expected": "AMN"}]),
        ],
    },
}


def make_task(cid: str, spec) -> Task:
    import hashlib
    prompt, tests = spec
    # stable id across runs (Python's str hash is salted per process)
    return Task(task_id=f"{cid}-{hashlib.sha1(prompt.encode()).hexdigest()[:8]}",
                category=cid, prompt=prompt, tests=[dict(t) for t in tests],
                max_attempts=4)


class EndpointDown(RuntimeError):
    """vLLM refused connections — the run is invalid, not a model failure."""


def _probe(base_url: str) -> None:
    import urllib.request
    try:
        u = base_url.rstrip("/")
        url = (u + "/models") if u.endswith("/v1") else (u + "/v1/models")
        with urllib.request.urlopen(url, timeout=10):
            return
    except Exception as exc:
        raise EndpointDown(str(exc)) from exc


def run_task(task: Task, memory_dir: str, args, label: str) -> Dict[str, Any]:
    from open_dream_rsi.cli import _build_client
    for attempt in range(6):  # ride out brief vLLM restarts
        try:
            _probe(args.base_url)
            break
        except EndpointDown:
            if attempt == 5:
                raise
            print("  endpoint down, retry %d/5 in 20s..." % (attempt + 1), flush=True)
            time.sleep(20)
    client = _build_client("local", args.model)
    if hasattr(client, "config"):
        client.config.timeout = 300.0
        client.config.extra_payload.setdefault(
            "chat_template_kwargs", {})["enable_thinking"] = not args.thinking
    t0 = time.time()
    rt = AutoRSIRuntime(client=client, memory=DreamMemory(memory_dir),
                        tasks=[task], api_call_budget=args.budget,
                        max_tokens=4096)
    rep = rt.run_once().to_dict()
    rep.update(task_id=task.task_id, category=task.category, label=label,
               wall=round(time.time() - t0, 1))
    # A solve reached through connection failures is NOT the same evidence
    # as a clean solve — count llm_error events per task.
    rep["llm_errors"] = 0
    events = Path(memory_dir) / "events.jsonl"
    if events.exists():
        for line in events.read_text().splitlines():
            try:
                if json.loads(line).get("kind") == "llm_error":
                    rep["llm_errors"] += 1
            except json.JSONDecodeError:
                pass
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=None,
                    help="OpenAI-compatible base; default: $OPENAI_BASE_URL "
                         "or http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--thinking", action="store_true",
                    help="keep hidden reasoning ON (default: off, as in normal use)")
    ap.add_argument("--reps", type=int, default=1, help="repeats per eval task")
    ap.add_argument("--out", default="bench_real_results.json")
    ap.add_argument("--resume", action="store_true",
                    help="keep existing --out results and skip completed runs")
    ap.add_argument("--categories", default=",".join(SUITE))
    args = ap.parse_args()
    import os
    args.base_url = (args.base_url or os.environ.get("OPENAI_BASE_URL")
                     or "http://127.0.0.1:8000/v1")
    os.environ.setdefault("OPENAI_BASE_URL", args.base_url)
    os.environ.setdefault("OPENAI_API_KEY", "bench")

    work = Path(".bench_real")
    if not args.resume:
        shutil.rmtree(work, ignore_errors=True)
    out_path = Path(args.out)
    results: List[Dict[str, Any]] = []
    done: set = set()
    if args.resume and out_path.exists():
        results = json.loads(out_path.read_text())
        done = {(r["label"], r["category"], r["task_id"]) for r in results}
        print(f"resuming: {len(done)} runs already on record", flush=True)

    def save() -> None:
        out_path.write_text(json.dumps(results, indent=2))
    def run_step(task, mem_dir, label, cid):
        key = (label, cid, task.task_id)
        if key in done:
            print(f"[skip {label} {cid}] (already done)", flush=True)
            return
        r = run_task(task, mem_dir, args, label)
        results.append(r); save()
        print(f"[{label} {cid}] solved={r['tasks_solved']} "
              f"calls={r['api_calls']} errs={r['llm_errors']} wall={r['wall']}s",
              flush=True)

    try:
        for cid in [c for c in args.categories.split(",") if c in SUITE]:
            suite = SUITE[cid]
            for rep_i in range(args.reps):
                for eval_i, spec in enumerate(suite["eval"]):
                    run_step(make_task(cid, spec),
                             str(work / f"cold-{cid}-{rep_i}-{eval_i}"),
                             "cold", f"{cid}#{eval_i}")
                run_step(make_task(cid, suite["train"]),
                         str(work / f"warm-{cid}-{rep_i}"), "train", cid)
                for eval_i, spec in enumerate(suite["eval"]):
                    run_step(make_task(cid, spec),
                             str(work / f"warm-{cid}-{rep_i}"),
                             "warm", f"{cid}#{eval_i}")
    except EndpointDown:
        save()
        print("\nENDPOINT DOWN — partial results saved to", out_path,
              "\nRe-run with --resume when the endpoint is back.", flush=True)
        return 3

    save()
    # summary
    def agg(label):
        evals = [r for r in results if r["label"] == label]
        solves = sum(r["tasks_solved"] for r in evals)
        calls = sum(r["api_calls"] for r in evals)
        wall = sum(r["wall"] for r in evals)
        return {"label": label, "eval_tasks": len(evals), "solved": solves,
                "api_calls": calls,
                "calls_per_solve": round(calls / solves, 2) if solves else None,
                "wall_s": round(wall, 1)}
    summary = [agg("cold"), agg("warm")]
    print("\n| arm | eval tasks | solved | api_calls | calls/solve | wall s | llm_errors |")
    print("|---|---|---|---|---|---|---|")
    for s in summary:
        rows = [r for r in results if r["label"] == s["label"]]
        s["llm_errors"] = sum(r.get("llm_errors", 0) for r in rows)
        print(f"| {s['label']} | {s['eval_tasks']} | {s['solved']} | "
              f"{s['api_calls']} | {s['calls_per_solve']} | {s['wall_s']} | "
              f"{s['llm_errors']} |")
    if any(s.get("llm_errors") for s in summary):
        print("WARNING: llm_errors > 0 - some runs hit endpoint failures; "
              "those rows are not clean evidence.")
    cold_c = summary[0]["calls_per_solve"]; warm_c = summary[1]["calls_per_solve"]
    if cold_c and warm_c:
        print(f"\ncompression: {cold_c} -> {warm_c} calls/solve "
              f"({round(100 * (1 - warm_c / cold_c))}% saving)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
