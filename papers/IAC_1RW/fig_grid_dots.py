"""Grouped dot plot of the baseline grid (optional item 3): architecture on y,
terminal pointing error on log x, median with the 25-75% interval, PD and planner
by marker shape; adjacent panel with the < 1 deg / < 5 deg / > 30 deg fractions.
Table 4 keeps the exact values; this makes '3+1 sits between' visible at a glance.
"""
import glob
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from papers.IAC_1RW._iac_sim import error_series  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fig_style  # noqa: E402
fig_style.apply(10.0)
from fig_style import ARCH, CTRL  # noqa: E402

OUT = os.path.join(HERE, "output_data")


def finals_from_pkls(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        with open(p, "rb") as f:
            out.append(float(error_series(pickle.load(f))[-1]))
    return np.asarray(out)


def finals_from_json(d, key):
    """Context cells: per-trial finals from the aggregate JSON, same statistics as
    the pkl-backed cells."""
    return summarize(np.asarray(d["cells"][key]["horizons"]["5554"]["finals_deg"], float))


def summarize(fin):
    return dict(med=float(np.median(fin)), q=(np.percentile(fin, 25), np.percentile(fin, 75)),
                f1=float(np.mean(fin <= 1)), f5=float(np.mean(fin <= 5)),
                f30=float(np.mean(fin > 30)), n=len(fin))


def main():
    d18 = json.load(open(os.path.join(OUT, "A_baseline_20260907_184702.json")))
    rows = []   # (task, arch, ctrl, stats)
    for task, key0, key3, pd_pat, pl_pat in (
            ("reduced", "0rw_reduced_pd", "3rw_reduced_pd",
             "wave/pd_reduced_kp1/*.pkl", "A_trials/1rw_reduced_planner_seed*.pkl"),
            ("full", "0rw_full_pd", "3rw_full_pd",
             "wave/pd_full_kp1/*.pkl", "tune_seed*_wave_planner_full.pkl")):
        rows.append((task, "3+0", "PD", finals_from_json(d18, key0)))
        rows.append((task, "3+1", "PD", summarize(finals_from_pkls(os.path.join(OUT, pd_pat)))))
        rows.append((task, "3+1", "planner", summarize(finals_from_pkls(os.path.join(OUT, pl_pat)))))
        rows.append((task, "3+3", "PD", finals_from_json(d18, key3)))

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.2),
                             gridspec_kw={"width_ratios": [2.4, 1.0]},
                             constrained_layout=True)
    order = [("3+3", "PD"), ("3+1", "planner"), ("3+1", "PD"), ("3+0", "PD")]
    ylab = ["3+3", "3+1 planner", "3+1 PD", "3+0"]
    ypos = {k: 3 - i for i, k in enumerate(order)}
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    for i, task in enumerate(("reduced", "full")):
        ax, axf = axes[i]
        for t, arch, ctrl, s in rows:
            if t != task:
                continue
            y = ypos[(arch, ctrl)]
            col = ARCH[arch]
            if s["q"] is not None:
                ax.plot(s["q"], [y, y], "-", color=col, lw=3.0, alpha=0.45,
                        solid_capstyle="butt")
            ax.plot(s["med"], y, CTRL[ctrl]["marker"], ms=9, color=col, mec=col,
                    mfc=col if ctrl == "PD" else "white", mew=1.6, zorder=3)
            f30 = s["f30"] if s["f30"] is not None else (1.0 - s["f5"])
            axf.barh(y + 0.22, s["f1"], height=0.2, color=col, alpha=0.95)
            axf.barh(y, s["f5"], height=0.2, color=col, alpha=0.5)
            axf.barh(y - 0.22, f30, height=0.2, color="white", edgecolor=col,
                     hatch="////", lw=0.6)
        ax.set_xscale("log"); ax.set_xlim(3e-2, 3e2)
        fig_style.log_decades_only(ax, "x")
        ax.set_yticks(list(ypos.values())); ax.set_yticklabels(ylab)
        ax.set_ylim(-0.6, 3.6)
        ax.grid(True, which="major", axis="x")
        ax.set_title(("(a) Boresight pointing" if task == "reduced"
                      else "(b) Full three-axis attitude"), loc="left")
        axf.set_xlim(0, 1.0); axf.set_xticks([0, 0.5, 1.0])
        axf.set_ylim(-0.6, 3.6); axf.set_yticks([])
        axf.spines["left"].set_visible(False)
        axf.grid(True, which="major", axis="x")
    axes[1][0].set_xlabel("terminal pointing error [deg]")
    axes[1][1].set_xlabel("fraction of trials")
    axes[0][0].legend(handles=[
        Line2D([], [], color="#444444", marker="o", ls="", ms=8, label="PD median"),
        Line2D([], [], color="#444444", marker="^", ls="", ms=8, mfc="white", mew=1.4,
               label="planner median"),
        Line2D([], [], color="#444444", lw=3.0, alpha=0.45, label="25–75%")],
        loc="upper right", handlelength=1.8, labelspacing=0.3)
    axes[0][1].legend(handles=[
        Patch(facecolor="#444444", alpha=0.95, label="< 1°"),
        Patch(facecolor="#444444", alpha=0.5, label="< 5°"),
        Patch(facecolor="white", edgecolor="#444444", hatch="////", label="> 30°")],
        loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, handlelength=1.3,
        columnspacing=0.9, handletextpad=0.4, borderaxespad=0.0)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_grid_dots.{ext}"))
    for r in rows:
        print(r[0], r[1], r[2], {k: (round(v, 2) if isinstance(v, float) else v)
                                 for k, v in r[3].items() if k != "q"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
