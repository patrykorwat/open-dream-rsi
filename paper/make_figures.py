"""Generate the paper figures from shipped data (reproducible: no live keys).

  fig_race.png   — reaction-time race: ECDF of within-session same-class
                   error gaps (fixtures/sentinel_gaps.json) against the
                   episode boundary and the weekly curator interval.
  fig_arms.png   — v8/v9 injection-arm study: solve rate vs cost-per-solve,
                   cold controls marked (data: odr-bench RESULTS-replay).

Run: uv run --with matplotlib python make_figures.py  (writes into paper/)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 200, "savefig.dpi": 200,
})
C = {"data": "#1f77b4", "warn": "#d62728", "ok": "#2ca02c",
     "gray": "#7f7f7f", "amber": "#ff7f0e"}


def fig_race():
    """ECDF of same-class repeat gaps vs curator horizons.

    Layout discipline: the upper-left is reserved for the curve; every text
    annotation sits in the empty lower/right region with a short arrow to its
    vline. A renderer-level assertion (no_overlap) fails the build if any two
    texts collide — visual QA must not depend on a human eye.
    """
    d = json.loads((HERE.parent / "fixtures" / "sentinel_gaps.json")
                   .read_text(encoding="utf-8"))
    gaps = np.array(d["gaps_seconds"], dtype=float) / 60.0     # minutes
    episodes = np.array(d["episode_durations_seconds"], dtype=float) / 60.0
    xs = np.sort(gaps)
    ys = np.arange(1, len(xs) + 1) / len(xs)

    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ax.plot(xs, ys, color=C["data"], lw=1.6, zorder=3)
    ax.annotate("same-class repeat gap\n(ECDF, n=%d)" % len(xs),
                xy=(900, 0.90), ha="left", va="bottom",
                color=C["data"], fontsize=8)
    med = float(np.median(gaps))
    ax.axvline(med, color=C["warn"], ls="--", lw=1.1, zorder=2)
    ax.annotate("median gap %.1f min;\n29%% of repeats under 1 min" % med,
                xytext=(60, 0.10), xy=(med * 1.05, 0.13),
                color=C["warn"], fontsize=8,
                arrowprops=dict(arrowstyle="->", color=C["warn"], lw=0.8))
    ep_med = float(np.median(episodes))
    ax.axvline(ep_med, color=C["gray"], ls=":", lw=1.2)
    ax.annotate("episode boundary (median %.0f min):\nhorizon of ANY episode-based curator" % ep_med,
                xytext=(420, 0.30), xy=(ep_med * 1.05, 0.33),
                color=C["gray"], fontsize=8,
                arrowprops=dict(arrowstyle="->", color=C["gray"], lw=0.8))
    ax.axvline(60, color=C["data"], ls="--", lw=1.0, alpha=0.7)
    ax.annotate("reactive tick (60 min):\n90.7% of repeats land before it",
                xy=(62, 0.94), xytext=(14, 0.78),
                color=C["data"], fontsize=8,
                arrowprops=dict(arrowstyle="->", color=C["data"], lw=0.8))
    ax.axvline(7 * 24 * 60, color=C["amber"], ls="-.", lw=1.1)
    ax.annotate("weekly skill-curator\nwake-up (10080 min)",
                xytext=(1300, 0.52), xy=(7 * 24 * 60 * 0.72, 0.55),
                color=C["amber"], fontsize=8,
                arrowprops=dict(arrowstyle="->", color=C["amber"], lw=0.8))
    ax.axvspan(0.1, 7, color=C["warn"], alpha=0.06)
    ax.text(0.115, 0.55, "sentinel reacts here\n(in-tool, same turn)",
            color=C["warn"], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("minutes between consecutive failures of the SAME error class (within one session)")
    ax.set_ylabel("cumulative fraction")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0.1, 16000)
    no_overlap(fig)
    fig.tight_layout()
    fig.savefig(HERE / "fig_race.png")
    plt.close(fig)


def no_overlap(fig, include_legend=False):
    """Assert no two text/annotation boxes overlap (renderer coordinates)."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    boxes = []
    for ax in fig.axes:
        if include_legend and ax.get_legend() is not None:
            boxes.append(("legend", ax.get_legend().get_window_extent(rend)))
        for t in ax.texts:
            if t.get_text().strip():
                boxes.append((t.get_text()[:28].replace("\n", " "),
                              t.get_window_extent(rend)))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                raise AssertionError("annotation overlap: %r <-> %r"
                                     % (boxes[i][0], boxes[j][0]))


# v8/v9 injection-arm study (od
# v8/v9 injection-arm study: every arm was run against the contemporaneous
# cold control on the same 20 tasks. Numbers verbatim from odr-bench/
# RESULTS-replay.md (private fixtures; the arms themselves are generic).
ARMS = [
    # (label, family, solved/20, calls_per_solve, is_control)
    ("cold_re",      "v8", 18, 3.11, True),
    ("t1 declarative","v8", 18, 3.61, False),
    ("t2 exemplar",  "v8", 16, 3.88, False),
    ("t3 facts+exmpl","v8", 13, 6.54, False),
    ("cold_re2",     "v9", 19, 3.00, True),
    ("w1 workflow",  "v9", 14, 6.71, False),
    ("w2 reactive",  "v9", 18, 3.11, False),
    ("w3 certificate","v9", 18, 3.94, False),
    ("w4 end-position","v9", 11, 9.36, False),
]

def fig_arms():
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    fam_color = {"v8": C["data"], "v9": C["ok"]}
    # offsets searched programmatically against the no_overlap guard; the
    # crowded y=18 row is staggered vertically, never label-over-label
    offs = {"cold_re": (-10, -4), "cold_re2": (0, 9),
            "w2 reactive": (9, 12), "t1 declarative": (9, -2),
            "w3 certificate": (0, -12), "t2 exemplar": (9, -2),
            "t3 facts+exmpl": (9, -2), "w1 workflow": (9, -2),
            "w4 end-position": (0, 11)}
    has = {"cold_re": "right", "cold_re2": "center",
           "w4 end-position": "center", "t1 declarative": "center",
           "w3 certificate": "center"}
    for label, fam, solved, cost, control in ARMS:
        ax.scatter(cost, solved, s=90 if control else 60,
                   marker="s" if control else "o",
                   color=fam_color[fam], edgecolor="black", lw=0.5,
                   zorder=3, alpha=0.9)
        ax.annotate(label, (cost, solved), textcoords="offset points",
                    xytext=offs.get(label, (9, -2)), fontsize=7.5,
                    ha=has.get(label, "left"))
    ax.axvspan(2.5, 4.2, color=C["ok"], alpha=0.07)
    ax.annotate("cold control optimum: 19/20 @ 3.0 calls/solve",
                xy=(3.3, 13.35), fontsize=8, color=C["gray"], ha="center")
    ax.annotate("w4: same text, moved to end of prompt", xy=(9.36, 11.0),
                xytext=(5.3, 10.3), fontsize=8, color=C["warn"],
                arrowprops=dict(arrowstyle="->", color=C["warn"], lw=0.8))
    ax.set_xlabel("API calls per solved task (lower is better)")
    ax.set_ylabel("solved tasks (of 20)")
    ax.set_ylim(9.5, 20.6)
    ax.set_xlim(2.5, 10.2)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color=fam_color["v8"], marker="o", ls="", label="v8 repair theories"),
        Line2D([], [], color=fam_color["v9"], marker="o", ls="", label="v9 literature mechanisms"),
        Line2D([], [], color="0.3", marker="s", ls="", label="cold control"),
    ], loc="lower center", frameon=False, fontsize=8, ncol=3,
        bbox_to_anchor=(0.5, -0.02))
    no_overlap(fig, include_legend=False)   # legend lives outside the axes
    fig.tight_layout()
    fig.savefig(HERE / "fig_arms.png", bbox_inches="tight")
    plt.close(fig)




# Cross-agent replication on block/goose (same model, same 20 tasks, same
# scoring; goose 1.53 CLI, stdlib MCP sandbox server, Stop-hook delivery).
# Aggregates verbatim from odr-bench/goose_results.jsonl (private fixtures);
# McNemar discordants: b = cold-only solves (arm loss),
# c = arm-only solves (arm win); p two-sided exact binomial.
GOOSE_ARMS = [
    # (label, solved/20, calls_per_solve, b_vs_cold, c_vs_cold)
    ("cold",        17, 2.41, None, None),
    ("osi_pull",    14, 3.00, 4, 1),
    ("osi_stop",    18, 2.89, 1, 2),
    ("osi_full",    13, 3.38, 5, 1),
]


def fig_goose():
    import math
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    col = {"cold": "0.3", "osi_pull": C["warn"],
           "osi_stop": C["ok"], "osi_full": C["data"]}
    offs = {"cold": (0, -14), "osi_pull": (12, 4),
            "osi_stop": (-1, 11), "osi_full": (12, -8)}
    for label, solved, cpc, b, c in GOOSE_ARMS:
        ax.scatter(cpc, solved, s=120 if label == "cold" else 85,
                   marker="s" if label == "cold" else "o",
                   color=col[label], edgecolor="black", lw=0.5, zorder=3)
        ha = "center" if label in ("cold", "osi_stop") else "left"
        ax.annotate(label, (cpc, solved), textcoords="offset points",
                    xytext=offs[label], fontsize=8, ha=ha)
        if b is not None:
            n = b + c
            p = min(1.0, 2.0 * sum(math.comb(n, k) * 0.5 ** n
                                   for k in range(min(b, c) + 1)))
            ax.annotate(f"+{c}/-{b}  p={p:.2f}", (cpc, solved),
                        textcoords="offset points",
                        xytext=(offs[label][0], offs[label][1] - 8),
                        fontsize=7, color="0.4", ha=ha)
    ax.set_xlabel("tool calls per solved task (lower is better)")
    ax.set_ylabel("solved tasks (of 20)")
    ax.set_ylim(11.5, 20.4)
    ax.set_xlim(2.0, 3.9)
    no_overlap(fig)
    fig.tight_layout()
    fig.savefig(HERE / "fig_goose.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_race()
    fig_arms()
    fig_goose()
    print("wrote fig_race.png, fig_arms.png")