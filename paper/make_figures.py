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
    d = json.loads((HERE.parent / "fixtures" / "sentinel_gaps.json")
                   .read_text(encoding="utf-8"))
    gaps = np.array(d["gaps_seconds"], dtype=float) / 60.0     # minutes
    episodes = np.array(d["episode_durations_seconds"], dtype=float) / 60.0
    xs = np.sort(gaps)
    ys = np.arange(1, len(xs) + 1) / len(xs)

    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    ax.plot(xs, ys, color=C["data"], lw=1.6, zorder=3,
            label=f"same-class repeat gap (n={len(xs)})")
    med = np.median(gaps)
    ax.axvline(med, color=C["warn"], ls="--", lw=1.1, zorder=2)
    ax.annotate(f"median {med:.1f} min\n(29% under 1 min)",
                xy=(med, 0.5), xytext=(med * 3, 0.30),
                color=C["warn"], fontsize=8,
                arrowprops=dict(arrowstyle="->", color=C["warn"], lw=0.8))
    ep_med = np.median(episodes)
    ax.axvline(ep_med, color=C["gray"], ls=":", lw=1.2)
    ax.annotate(f"episode boundary (median {ep_med:.0f} min):\nthe horizon of ANY episode-based curator",
                xy=(ep_med, 0.92), xytext=(ep_med / 6, 0.80),
                color=C["gray"], fontsize=8,
                arrowprops=dict(arrowstyle="->", color=C["gray"], lw=0.8))
    ax.axvline(7 * 24 * 60, color=C["amber"], ls="-.", lw=1.1)
    ax.annotate("weekly skill-curator\nwake-up (10080 min)",
                xy=(7 * 24 * 60, 0.5), xytext=(140, 0.60),
                color=C["amber"], fontsize=8,
                arrowprops=dict(arrowstyle="->", color=C["amber"], lw=0.8))
    ax.axvspan(0.1, 7, color=C["warn"], alpha=0.06)
    ax.annotate("sentinel reacts here\n(in-tool, same turn)",
                xy=(7, 0.08), xytext=(0.13, 0.12), color=C["warn"], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("minutes between consecutive failures of the SAME error class (within one session)")
    ax.set_ylabel("cumulative fraction")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0.1, 16000)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "fig_race.png")
    plt.close(fig)


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
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    fam_color = {"v8": C["data"], "v9": C["ok"]}
    for label, fam, solved, cost, control in ARMS:
        ax.scatter(cost, solved, s=90 if control else 60,
                   marker="s" if control else "o",
                   color=fam_color[fam], edgecolor="black", lw=0.5,
                   zorder=3, alpha=0.9)
        ax.annotate(label, (cost, solved), textcoords="offset points",
                    xytext=(5, -3), fontsize=7.5)
    ax.axvspan(2.5, 4.2, color=C["ok"], alpha=0.07)
    ax.annotate("cold optimum: 19/20 @ 3.0 calls/solve\nno arm beats it (all gate-rejected)",
                xy=(3.2, 12.4), fontsize=8, color=C["gray"], ha="center")
    ax.annotate("w4: same text, moved to end of prompt\n11/20, 9.4 calls/solve, p=0.008",
                xy=(9.36, 11.2), xytext=(6.3, 10.6), fontsize=8, color=C["warn"],
                arrowprops=dict(arrowstyle="->", color=C["warn"], lw=0.8))
    ax.annotate("w2 reactive-on-error:\nzero prompt tax on clean episodes\n(net -1, still no gain here)",
                xy=(3.11, 18), xytext=(4.6, 19.2), fontsize=8, color=C["data"],
                arrowprops=dict(arrowstyle="->", color=C["data"], lw=0.8))
    ax.set_xlabel("API calls per solved task (lower is better)")
    ax.set_ylabel("solved tasks (of 20)")
    ax.set_ylim(9.5, 20.6)
    ax.set_xlim(2.5, 10.2)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color=fam_color["v8"], marker="o", ls="", label="v8 repair theories"),
        Line2D([], [], color=fam_color["v9"], marker="o", ls="", label="v9 literature mechanisms"),
        Line2D([], [], color="0.3", marker="s", ls="", label="contemporaneous cold control"),
    ], loc="lower left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "fig_arms.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_race()
    fig_arms()
    print("wrote fig_race.png, fig_arms.png")
