"""Plot final pointing-error bars for every disturbed no-estimator 6U allocator case."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
PAPER_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CAMPAIGN_PREFIX = "6u_noestimator_disturbance"
ARCHITECTURES = ((0, "3+0"), (1, "3+1"), (3, "3+3"))
ALLOCATORS = ("lp", "qp")
GOAL_TYPES = ("quaternion", "vector")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PAPER_DIR))

import ADCS
from ADCS.helpers.math_helpers import rot_mat
from plot_style import configure_ieee_style


def _latest_sim(number_rw: int, allocator: str, goal_type: str) -> Path:
    pattern = f"{CAMPAIGN_PREFIX}_{goal_type}_3mtq_{number_rw}rw_{allocator}_mc_*.sim"
    candidates = list(OUTPUT_DIR.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No saved MC result matching {OUTPUT_DIR / pattern}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _final_angle_error_deg(run) -> float:
    """Return the final full-attitude or boresight angle, according to goal type."""
    q_final = np.asarray(run.state_hist[-1].q, dtype=float)
    target_final = np.asarray(run.target_hist[-1], dtype=float)
    if np.isnan(target_final[0]):
        # Vector goals store [nan, x, y, z].  The spacecraft +z boresight is
        # the one configured in the 6U campaign satellite definition.
        goal_eci = target_final[1:]
        boresight_eci = rot_mat(q_final) @ np.array([0.0, 0.0, 1.0])
        cosine = np.clip(np.dot(boresight_eci, goal_eci), -1.0, 1.0)
        return float(np.rad2deg(np.arccos(cosine)))
    cosine = np.clip(abs(np.dot(q_final, target_final)), 0.0, 1.0)
    return float(np.rad2deg(2.0 * np.arccos(cosine)))


def _summary(number_rw: int, allocator: str, goal_type: str) -> tuple[float, float, Path]:
    path = _latest_sim(number_rw, allocator, goal_type)
    results = ADCS.SimulationResults.load(path)
    final_errors = np.asarray([_final_angle_error_deg(run) for run in results.runs])
    sigma = float(np.std(final_errors, ddof=1)) if len(final_errors) > 1 else 0.0
    return float(np.mean(final_errors)), sigma, path


def _plot(goal_type: str) -> Path:
    x = np.arange(len(ARCHITECTURES), dtype=float)
    width = 0.34
    fig, ax = plt.subplots(figsize=(4.3, 2.8))
    colors = {"lp": "#0072B2", "qp": "#D55E00"}

    for allocator, offset in zip(ALLOCATORS, (-width / 2.0, width / 2.0)):
        means, sigmas = [], []
        for number_rw, label in ARCHITECTURES:
            mean, sigma, path = _summary(number_rw, allocator, goal_type)
            means.append(mean)
            sigmas.append(sigma)
            print(f"{goal_type:10s} {label} {allocator.upper()}: {path.name}")
        ax.bar(x + offset, means, width, label=allocator.upper(), color=colors[allocator],
               edgecolor="black", linewidth=0.35, zorder=2)
        ax.errorbar(x + offset, means, yerr=sigmas, fmt="none", ecolor="black",
                    elinewidth=0.75, capsize=2.5, zorder=3)

    ax.set_xticks(x, [label for _, label in ARCHITECTURES])
    ax.set_xlabel("Actuator architecture")
    ax.set_ylabel("Final angle error [deg]")
    quantity = "quaternion pointing" if goal_type == "quaternion" else "vector pointing"
    ax.set_title(f"6U final {quantity} error (mean ± 1σ)")
    ax.set_yscale("log")
    ax.grid(axis="y", zorder=0)
    ax.legend(title="Allocator", frameon=True)
    fig.tight_layout()
    path = OUTPUT_DIR / f"{CAMPAIGN_PREFIX}_{goal_type}_final_angle_error_bars_mc.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    configure_ieee_style()
    for goal_type in GOAL_TYPES:
        print(f"Saved {_plot(goal_type)}")


if __name__ == "__main__":
    main()
