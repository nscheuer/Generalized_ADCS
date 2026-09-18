"""Load the Section IV Monte Carlo results and reproduce the summary figure."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


# Allow this script to be run from any working directory in the repository.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import ADCS  # noqa: E402
from ADCS.helpers.math_helpers import quat_inv, quat_mult, rot_mat  # noqa: E402


SECTION_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SECTION_DIR / "outputs"
FIGURE_PATH = SECTION_DIR / "outputs" / "planner_vs_pd_summary.png"
BODY_BORESIGHT = np.array([0.0, 1.0, 0.0])

BLUE = "#4C78A8"
RED = "#C65D5D"
GRAY = "#777777"


RESULT_FILES = {
    "planner_30_reduced": "mc100_altro_3+0_reduced_20260204_185357.sim",
    "planner_30_full": "mc100_altro_3+0_full_20260204_162440.sim",
    "planner_31_reduced": "mc100_altro_3+1_reduced_20260205_001823.sim",
    "planner_31_full": "mc100_altro_3+1_full_20260204_235053.sim",
    "pd_30_reduced": "mc100_lovera_3+0_reduced_20260205_003107.sim",
    "pd_30_full": "mc100_lovera_3+0_full_20260205_003347.sim",
    "pd_31_reduced": "mc100_lp_3+1_reduced_20260205_000536.sim",
    "pd_31_full": "mc100_lp_3+1_full_20260205_000958.sim",
}


def final_error_degrees(run, full: bool) -> float:
    """Return the final attitude or boresight error for one simulation run."""
    q = np.asarray(run.state_hist[-1].q)
    goal = np.asarray(run.target_hist[-1])

    if full:
        q_error = quat_mult(quat_inv(goal), q)
        return float(2.0 * np.arccos(np.clip(abs(q_error[0]), 0.0, 1.0)) * 180.0 / np.pi)

    goal_vector = goal[1:]
    boresight = rot_mat(q) @ BODY_BORESIGHT
    dot = np.clip(np.dot(boresight, goal_vector), -1.0, 1.0)
    return float(np.arccos(dot) * 180.0 / np.pi)


def load_errors(filename: str, full: bool) -> np.ndarray:
    results = ADCS.SimulationResults.load(OUTPUT_DIR / filename)
    return np.asarray([final_error_degrees(run, full) for run in results.runs])


def summarize(errors: np.ndarray) -> tuple[float, float]:
    return float(100.0 * np.mean(errors < 5.0)), float(np.mean(errors))


def make_figure(convergence: np.ndarray, mean_error: np.ndarray) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.titleweight": "normal",
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    labels = ["MTQ-only\nreduced goal", "3+1\nfull goal", "3+1\nreduced goal", "MTQ-only\nfull goal"]
    x = np.arange(len(labels), dtype=float)
    width = 0.34

    fig, (ax_rate, ax_error) = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.73, wspace=0.28)
    fig.suptitle(
        "Planner vs PD baseline — converges where PD does not (bold = controllability-appropriate task)",
        fontsize=15, fontweight="bold", y=0.965,
    )
    fig.legend(
        handles=[Patch(facecolor=BLUE, label="Planner (ALTRO)"), Patch(facecolor=RED, label="PD baseline (Lovera / LP)")],
        loc="upper center", bbox_to_anchor=(0.5, 0.895), ncol=2, frameon=False, fontsize=11,
    )

    def style_axis(ax, title: str, ylabel: str) -> None:
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, labels)
        ax.grid(axis="y", color="#B8B8B8", linewidth=0.7, alpha=0.45)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")
        ax.tick_params(axis="both", direction="in", length=4)
        for tick in ax.get_xticklabels():
            tick.set_fontweight("bold" if tick.get_text() == labels[0] else "normal")

    bars_rate = [ax_rate.bar(x - width / 2, convergence[:, 0], width, color=BLUE, edgecolor="none"),
                 ax_rate.bar(x + width / 2, convergence[:, 1], width, color=RED, edgecolor="none")]
    style_axis(ax_rate, "Convergence rate (100-trial MC)", "Converged [% <5° final]")
    ax_rate.set_ylim(0, 112)
    ax_rate.set_yticks(np.arange(0, 101, 20))

    bars_error = [ax_error.bar(x - width / 2, mean_error[:, 0], width, color=BLUE, edgecolor="none"),
                  ax_error.bar(x + width / 2, mean_error[:, 1], width, color=RED, edgecolor="none")]
    style_axis(ax_error, "Mean final error (linear scale)", "Mean final pointing error [deg]")
    ax_error.set_ylim(0, max(90.0, float(np.max(mean_error)) * 1.18))
    ax_error.set_yticks(np.arange(0, 101, 20))
    ax_error.axhline(5.0, color=GRAY, linestyle="--", linewidth=1.1)
    ax_error.text(x[-1] + 0.43, 5.0 + 1.5, "5° threshold", color=GRAY, fontsize=9, ha="right", va="bottom")

    for bars in bars_rate:
        for bar in bars:
            ax_rate.annotate(f"{bar.get_height():.0f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                             xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10)
    for bars in bars_error:
        for bar in bars:
            ax_error.annotate(f"{bar.get_height():.2f}°", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                              xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10)

    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved figure: {FIGURE_PATH}")


def main() -> None:
    loaded: dict[str, np.ndarray] = {}
    for key, filename in RESULT_FILES.items():
        loaded[key] = load_errors(filename, full="full" in key)

    # Table order: 3+0 reduced, 3+0 full, 3+1 reduced, 3+1 full.
    pairs = [
        ("3+0", "Reduced†", "planner_30_reduced", "pd_30_reduced"),
        ("3+0", "Full", "planner_30_full", "pd_30_full"),
        ("3+1", "Reduced", "planner_31_reduced", "pd_31_reduced"),
        ("3+1", "Full†", "planner_31_full", "pd_31_full"),
    ]
    convergence = []
    mean_error = []

    print("\nPlanner vs PD baseline")
    print(f"{'Config':<8} {'Task':<10} {'Planner conv.':>15} {'Planner mean err.':>19} {'PD conv.':>12} {'PD mean err.':>16}")
    print("-" * 86)
    for config, task, planner_key, pd_key in pairs:
        planner_summary = summarize(loaded[planner_key])
        pd_summary = summarize(loaded[pd_key])
        convergence.append([planner_summary[0], pd_summary[0]])
        mean_error.append([planner_summary[1], pd_summary[1]])
        print(f"{config:<8} {task:<10} {planner_summary[0]:>14.1f}% {planner_summary[1]:>18.2f}° {pd_summary[0]:>11.1f}% {pd_summary[1]:>15.2f}°")

    make_figure(np.asarray(convergence), np.asarray(mean_error))


if __name__ == "__main__":
    main()
