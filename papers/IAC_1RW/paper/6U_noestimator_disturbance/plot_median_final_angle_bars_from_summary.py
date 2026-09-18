"""Plot the median final-angle summary without loading Monte Carlo archives."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
PAPER_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
SUMMARY_PATH = OUTPUT_DIR / "6u_noestimator_disturbance_final_angle_error_bars_mc_summary.json"
OUTPUT_PATH = OUTPUT_DIR / "6u_noestimator_disturbance_median_final_angle_error_bars_mc.png"

ARCHITECTURES = ("3+0", "3+1", "3+3")
ALLOCATORS = ("LP", "QP")
GOAL_TYPES = ("quaternion", "vector")
COLORS = {"LP": "#0072B2", "QP": "#D55E00"}
HATCHES = {"quaternion": "", "vector": "//"}

import sys
sys.path.insert(0, str(PAPER_DIR))
from plot_style import configure_ieee_style


def main() -> None:
    configure_ieee_style()
    summary = json.loads(SUMMARY_PATH.read_text())
    rows = {
        (row["architecture"], row["allocator"], row["goal_type"]): row
        for row in summary["results"]
    }

    x = np.arange(len(ARCHITECTURES), dtype=float)
    width = 0.18
    fig, ax = plt.subplots(figsize=(3.45, 2.55))

    offsets = (-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width)
    for (goal_type, allocator), offset in zip(
        ((goal, alloc) for goal in GOAL_TYPES for alloc in ALLOCATORS), offsets
    ):
        values = []
        lower_errors = []
        upper_errors = []
        for architecture in ARCHITECTURES:
            row = rows[(architecture, allocator, goal_type)]
            median = row["final_angle_error_median_deg"]
            values.append(median)
            lower_errors.append(median - row["final_angle_error_p10_deg"])
            upper_errors.append(row["final_angle_error_p90_deg"] - median)

        ax.bar(
            x + offset,
            values,
            width,
            color=COLORS[allocator],
            hatch=HATCHES[goal_type],
            edgecolor="black",
            linewidth=0.35,
            zorder=2,
        )
        ax.errorbar(
            x + offset,
            values,
            yerr=(lower_errors, upper_errors),
            fmt="none",
            ecolor="black",
            elinewidth=0.65,
            capsize=2.0,
            zorder=3,
        )

    ax.set_xticks(x, ARCHITECTURES)
    ax.set_xlabel("Actuator architecture")
    ax.set_ylabel("Median final angle error [deg]")
    ax.set_title("6U final pointing error (100-run Monte Carlo)")
    ax.set_yscale("log")
    ax.grid(axis="y", zorder=0)

    allocator_handles = [
        Patch(facecolor=COLORS[allocator], edgecolor="black", label=allocator)
        for allocator in ALLOCATORS
    ]
    goal_handles = [
        Patch(facecolor="white", edgecolor="black", hatch=HATCHES[goal],
              label=goal.capitalize())
        for goal in GOAL_TYPES
    ]
    ax.legend(
        handles=allocator_handles + goal_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        ncol=2,
        frameon=True,
        framealpha=0.9,
        title="Error bars: 10th--90th percentile",
        title_fontsize=5.5,
        handlelength=1.5,
        columnspacing=1.0,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
