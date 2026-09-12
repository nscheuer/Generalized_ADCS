"""Plot final pointing-error bars for every disturbed no-estimator 6U allocator case."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


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


def _summary(number_rw: int, allocator: str, goal_type: str) -> tuple[float, float, Path, int]:
    path = _latest_sim(number_rw, allocator, goal_type)
    results = ADCS.SimulationResults.load(path)
    final_errors = np.asarray([_final_angle_error_deg(run) for run in results.runs])
    sigma = float(np.std(final_errors, ddof=1)) if len(final_errors) > 1 else 0.0
    return float(np.mean(final_errors)), sigma, path, len(results.runs)


def _plot() -> Path:
    x = np.arange(len(ARCHITECTURES), dtype=float)
    width = 0.18
    fig, ax = plt.subplots(figsize=(4.3, 2.8))
    colors = {"lp": "#0072B2", "qp": "#D55E00"}
    hatches = {"quaternion": "", "vector": "//"}

    offsets = (-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width)
    for (goal_type, allocator), offset in zip(
            ((goal, alloc) for goal in GOAL_TYPES for alloc in ALLOCATORS), offsets):
        means, sigmas = [], []
        for number_rw, label in ARCHITECTURES:
            mean, sigma, path, _ = _summary(number_rw, allocator, goal_type)
            means.append(mean)
            sigmas.append(sigma)
            print(f"{goal_type:10s} {label} {allocator.upper()}: {path.name}")
        ax.bar(x + offset, means, width, color=colors[allocator], hatch=hatches[goal_type],
               edgecolor="black", linewidth=0.35, zorder=2)
        ax.errorbar(x + offset, means, yerr=sigmas, fmt="none", ecolor="black",
                    elinewidth=0.75, capsize=2.5, zorder=3)

    ax.set_xticks(x, [label for _, label in ARCHITECTURES])
    ax.set_xlabel("Actuator architecture")
    ax.set_ylabel("Final angle error [deg]")
    ax.set_title("6U final pointing error (mean ± 1σ)")
    ax.set_yscale("log")
    ax.grid(axis="y", zorder=0)
    allocator_handles = [Patch(facecolor=colors[a], edgecolor="black", label=a.upper())
                         for a in ALLOCATORS]
    goal_handles = [Patch(facecolor="white", edgecolor="black", hatch=hatches[g],
                          label=g.capitalize()) for g in GOAL_TYPES]
    ax.legend(handles=allocator_handles + goal_handles,
              title="Color: allocator; hatch: goal", frameon=True, ncol=2)
    fig.tight_layout()
    path = OUTPUT_DIR / f"{CAMPAIGN_PREFIX}_final_angle_error_bars_mc.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_summary() -> Path:
    """Save the final-angle bar values and their one-sigma bounds as JSON."""
    rows = []
    for goal_type in GOAL_TYPES:
        for number_rw, architecture in ARCHITECTURES:
            for allocator in ALLOCATORS:
                mean, sigma, path, run_count = _summary(number_rw, allocator, goal_type)
                rows.append({
                    "goal_type": goal_type,
                    "architecture": architecture,
                    "number_reaction_wheels": number_rw,
                    "allocator": allocator.upper(),
                    "runs": run_count,
                    "final_angle_error_mean_deg": mean,
                    "final_angle_error_std_dev_deg": sigma,
                    "one_sigma_lower_deg": max(0.0, mean - sigma),
                    "one_sigma_upper_deg": mean + sigma,
                    "source_simulation": path.name,
                })
    path = OUTPUT_DIR / f"{CAMPAIGN_PREFIX}_final_angle_error_bars_mc_summary.json"
    path.write_text(json.dumps({
        "metric": "final angle error",
        "units": "deg",
        "one_sigma_bounds": "mean ± sample standard deviation; lower bound is clipped to zero degrees",
        "results": rows,
    }, indent=2) + "\n")
    return path


def main() -> None:
    configure_ieee_style()
    print(f"Saved {_plot()}")
    print(f"Saved {_write_summary()}")


if __name__ == "__main__":
    main()
