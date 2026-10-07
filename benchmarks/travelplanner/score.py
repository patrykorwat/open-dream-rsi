#!/usr/bin/env python3
"""Score a TravelPlanner submission.jsonl with the OFFICIAL evaluators.

Bypasses eval.py's load_dataset: reads query data from the committed local
CSV (the identical osunlp/TravelPlanner task rows), then runs
evaluation/commonsense_constraint.evaluation and
evaluation/hard_constraint.evaluation exactly as eval.py does (hard only if
commonsense is_not_absent and is_valid_information_in_sandbox both pass).
Constraint logic is the untouched upstream code from the pinned checkout.

Usage: python3 score.py results/submissions/validation_holdout_osi_gate.jsonl \
           --csv splits/validation_holdout.csv
Prints final pass rates (delivery, commonsense-final, hard-final) overall
and by level; per-constraint counts; details json beside the submission.
"""
import argparse
import contextlib
import csv as _csv
import io
import json
import os
import sys
import types
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
TP = os.environ.get("TP_REPO", "/tmp/tp")

ap = argparse.ArgumentParser()
ap.add_argument("submission")
ap.add_argument("--csv", default=str(HERE / "splits/validation_holdout.csv"))
args = ap.parse_args()
# resolve user paths BEFORE the evaluators' chdir below
args.csv = str(HERE / args.csv) if not Path(args.csv).is_absolute() and not Path(args.csv).exists() else str(Path(args.csv).resolve())
args.submission = str(Path(args.submission).resolve())

sys.path.insert(0, os.path.join(TP, "evaluation"))
sys.path.insert(0, TP)
if "gradio" not in sys.modules:  # utils.func only uses it for the demo UI
    sys.modules["gradio"] = types.ModuleType("gradio")
os.chdir(os.path.join(TP, "evaluation"))  # evaluators chdir themselves too
with contextlib.redirect_stdout(io.StringIO()):
    import commonsense_constraint as cc  # noqa: E402
    import hard_constraint as hc  # noqa: E402

query_data_list = []
with open(args.csv) as f:
    for row in _csv.DictReader(f):
        row["days"] = int(row["days"])
        row["visiting_city_number"] = int(row["visiting_city_number"])
        row["people_number"] = int(row["people_number"])
        row["budget"] = int(float(row["budget"]))
        if isinstance(row["local_constraint"], str):
            row["local_constraint"] = eval(row["local_constraint"])
        if isinstance(row["date"], str):
            row["date"] = eval(row["date"])
        query_data_list.append(row)

tested = []
with open(args.submission) as f:
    for ln in f:
        ln = ln.strip()
        if ln:
            d = json.loads(ln)
            plan = d["plan"]
            if isinstance(plan, str):
                plan = eval(plan)
            tested.append(plan)

assert len(tested) == len(query_data_list), (
    f"submission rows {len(tested)} != queries {len(query_data_list)}")

CS_KEYS = ["is_reasonable_visiting_city", "is_valid_restaurants",
           "is_valid_attractions", "is_valid_accommodation",
           "is_valid_transportation", "is_valid_information_in_current_city",
           "is_valid_information_in_sandbox", "is_not_absent"]
HARD_KEYS = ["valid_cost", "valid_room_rule", "valid_cuisine",
             "valid_room_type", "valid_transportation"]

from collections import defaultdict
cs_stat = defaultdict(Counter)
hard_stat = defaultdict(Counter)
by_level = {lvl: Counter() for lvl in ("easy", "medium", "hard")}
delivery = 0
cs_pass = 0
hard_pass = 0
details = []

def _first(v):
    # official evaluators mix (bool, msg) tuples and bare bools;
    # (None, None) means the constraint is not applicable -> pass
    if isinstance(v, (tuple, list)):
        if v[0] is None:
            return True
        return bool(v[0])
    return True if v is None else bool(v)


for q, plan in zip(query_data_list, tested):
    rec = {"idx": q.get("idx"), "level": q["level"], "days": q["days"]}
    if plan:
        delivery += 1
        by_level[q["level"]]["delivery"] += 1
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                cs = cc.evaluation(q, plan)
        except Exception as e:  # malformed plan -> fail closed
            cs = None
            rec["cs_error"] = str(e)[:120]
    else:
        cs = None
    cs_ok = bool(cs) and _first(cs["is_not_absent"]) and _first(cs["is_valid_information_in_sandbox"])
    if cs:
        for k in CS_KEYS:
            cs_stat[k][_first(cs[k])] += 1
    if cs_ok:
        cs_pass += 1
        by_level[q["level"]]["cs_pass"] += 1
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                h = hc.evaluation(q, plan)
        except Exception as e:
            h = None
            rec["hard_error"] = str(e)[:120]
        if h:
            for k in HARD_KEYS:
                if k in h:
                    hard_stat[k][_first(h[k])] += 1
            if all(_first(h[k]) for k in HARD_KEYS if k in h):
                hard_pass += 1
                by_level[q["level"]]["hard_pass"] += 1
                rec["solved"] = True
    details.append(rec)

n = len(query_data_list)
print(json.dumps({
    "n": n, "delivery": delivery,
    "commonsense_final": cs_pass, "hard_final": hard_pass,
    "commonsense_rate": round(cs_pass / n, 4),
    "final_pass_rate": round(hard_pass / n, 4),
    "per_constraint_commonsense": {k: dict(v) for k, v in cs_stat.items()},
    "per_constraint_hard": {k: dict(v) for k, v in hard_stat.items()},
    "by_level": {k: dict(v) for k, v in by_level.items()},
}, indent=1))
out = Path(args.submission).with_suffix(".score.json")
out.write_text(json.dumps({"summary": {"n": n, "delivery": delivery,
                                       "commonsense_final": cs_pass,
                                       "hard_final": hard_pass},
                           "details": details}, ensure_ascii=False, indent=1))
print("details ->", out, file=sys.stderr)
