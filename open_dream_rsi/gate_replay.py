"""Offline paired-arm comparison on recorded outcome rows.

The decision rule lives in :func:`open_dream_rsi.gate_replay.gate_verdict`;
this module applies THAT rule (never a re-implementation) to per-task
outcome pairs recorded by any replay harness: a baseline arm and one or
more treatment arms, joined on a shared task key. It answers the
counterfactual question "would the shipped gate have promoted this arm?"
without touching a model or an API key.

Input format (any replay harness that records per-task rows works):

* a JSON list of row objects, or
* a JSON object containing such a list, selected by ``records_key``
  (auto-detected when exactly one top-level list of objects exists).

Rows need a join key (``key``, e.g. ``task_id`` or ``krs``) and a solved flag
(``solved_field``, default ``"solved"``; any truthy value counts).

Typical use::

    python -m open_dream_rsi gate-replay \
        --baseline cold.json \
        --compare arm_v5=/tmp/warm_v5.json:warm \
        --compare neutral=/tmp/warm_v6.json:neutral \
        --key krs --format md
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


__all__ = ["filter_rows", "load_arm_rows", "arm_pairs", "gate_replay",
           "gate_verdict", "GateVerdict", "to_markdown"]


GATE_MIN_NET = 1


@dataclass
class GateVerdict:
    promoted: List[str]
    rejected: List[str]
    detail: Dict[str, Any]


def gate_verdict(results: Dict[str, List[Tuple[bool, bool]]]) -> GateVerdict:
    """Decide promotion from paired replay outcomes.

    ``results`` maps arm key -> list of (solved_treatment, solved_baseline)
    trial pairs. Promotion rule (sign test, harm-asymmetric):

    * net = #(fail->solve) - #(solve->fail) >= GATE_MIN_NET, and
    * zero solve->fail regressions.
    """
    promoted: List[str] = []
    rejected: List[str] = []
    detail: Dict[str, Any] = {}
    for key, pairs in results.items():
        gains = sum(1 for w, wo in pairs if w and not wo)
        regs = sum(1 for w, wo in pairs if wo and not w)
        net = gains - regs
        ok = net >= GATE_MIN_NET and regs == 0
        detail[key] = {"gains": gains, "regressions": regs, "net": net,
                       "promoted": ok}
        (promoted if ok else rejected).append(key)
    return GateVerdict(promoted, rejected, detail)



def filter_rows(rows: List[Dict[str, Any]],
                filters: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Keep rows matching every ``field=value`` filter (exact string match).

    Lets one multi-arm file (e.g. rows tagged ``label=cold``/``label=warm``)
    serve as multiple arms.
    """
    if not filters:
        return list(rows)
    out = list(rows)
    for spec in filters:
        field, sep, value = spec.partition("=")
        if not sep:
            raise ValueError(f"filter {spec!r} is not field=value")
        out = [r for r in out if str(r.get(field)) == value]
    return out


def load_arm_rows(path: str, records_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Read per-task rows from a replay-arm JSON file.

    Accepts a top-level list, or an object; with an object, ``records_key``
    selects the list, otherwise auto-detection requires exactly one
    top-level list of dict rows (otherwise the call is ambiguous and raises).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        if records_key is not None:
            if records_key not in data:
                raise ValueError(
                    f"{path}: records_key {records_key!r} not found "
                    f"(keys: {sorted(data)})")
            rows = data[records_key]
        else:
            candidates = [v for v in data.values()
                          if isinstance(v, list)
                          and (not v or isinstance(v[0], dict))]
            if len(candidates) != 1:
                raise ValueError(
                    f"{path}: {len(candidates)} candidate record lists — "
                    "pass an explicit records_key (name=path:key)")
            rows = candidates[0]
    else:
        raise ValueError(f"{path}: expected a JSON list or object, "
                         f"got {type(data).__name__}")
    if not all(isinstance(r, dict) for r in rows):
        raise ValueError(f"{path}: record list must contain objects")
    return rows


def arm_pairs(baseline_rows: List[Dict[str, Any]],
              arm_rows: List[Dict[str, Any]],
              key: str = "task_id",
              solved_field: str = "solved",
              ) -> Tuple[List[Tuple[bool, bool]], Dict[str, int]]:
    """Join an arm to the baseline on ``key`` and emit (with, without) pairs.

    Pair order matches ``gate_verdict``: ``(solved_with_treatment,
    solved_without_treatment)``. Rows whose key is missing or duplicated, or
    absent from the other arm, are skipped and counted in the stats dict.
    """
    def index(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, bool], int]:
        out: Dict[str, bool] = {}
        dup = 0
        for row in rows:
            k = row.get(key)
            if k is None:
                continue
            if k in out:
                dup += 1
                continue
            out[str(k)] = bool(row.get(solved_field))
        return out, dup

    base, dup_b = index(baseline_rows)
    arm, dup_a = index(arm_rows)
    shared = [k for k in base if k in arm]
    pairs = [(arm[k], base[k]) for k in shared]
    stats = {
        "baseline_rows": len(base),
        "arm_rows": len(arm),
        "joined": len(pairs),
        "skipped_unmatched": len(base) + len(arm) - 2 * len(shared),
        "duplicate_keys": dup_b + dup_a,
    }
    return pairs, stats


def gate_replay(baseline_rows: List[Dict[str, Any]],
                arms: Dict[str, List[Dict[str, Any]]],
                key: str = "task_id",
                solved_field: str = "solved",
                            ) -> Dict[str, Any]:
    """Replay the shipped gate's rule over recorded baseline/arm outcome pairs.

    ``baseline_rows``/arm rows should already be filtered to one arm
    (see :func:`filter_rows`). Returns a report dict: per-arm pair stats, the
    ``gate_verdict`` detail (the ACTUAL promotion rule, harm-asymmetric
    sign test).
    """
    result: Dict[str, Any] = {"key": key, "arms": {}}
    for name, rows in arms.items():
        pairs, stats = arm_pairs(baseline_rows, rows, key=key,
                                 solved_field=solved_field)
        verdict: GateVerdict = gate_verdict({name: pairs})
        detail = dict(verdict.detail[name])
        detail["promoted"] = name in verdict.promoted
        detail.update(stats)
        detail["pairs"] = pairs
        result["arms"][name] = detail
    return result


def to_markdown(report: Dict[str, Any]) -> str:
    """Render a gate-replay report as a markdown table."""
    lines = ["| arm | joined | gains | regressions | net | verdict |",
             "|---|---|---|---|---|---|"]
    for name, d in report["arms"].items():
        verdict = "PROMOTE" if d["promoted"] else "REJECT"
        lines.append(f"| {name} | {d['joined']} | {d['gains']} | "
                     f"{d['regressions']} | {d['net']:+d} | {verdict} |")
    return "\n".join(lines)
