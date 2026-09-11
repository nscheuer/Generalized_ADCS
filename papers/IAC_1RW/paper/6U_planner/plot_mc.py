"""Plot corrected 3+1 boresight Planner convergence with mean +/- 1 sigma bounds."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ADCS
from plot_style import configure_ieee_style

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
DEFAULT_PREFIX = "6u_3mtq_1rw_boresight_planner_mc_10"


def _latest_sim(prefix: str) -> Path:
    candidates = list(OUTPUT_DIR.glob(f"{prefix}_*.sim"))
    if not candidates:
        raise FileNotFoundError(f"No .sim archive matching {prefix!r} in {OUTPUT_DIR}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _angle_error_deg(run, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return body +z boresight-to-goal angle, matching clean-branch error_series()."""
    time = np.asarray(run.time_s, dtype=float)
    q = np.asarray([state.q for state in run.state_hist], dtype=float)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    xyz = np.column_stack((x, y, z))
    body_boresight = np.array([0.0, 0.0, 1.0])
    cross_term = 2.0 * np.cross(xyz, body_boresight)
    boresight_eci = (
        body_boresight
        + w[:, None] * cross_term
        + np.cross(xyz, cross_term)
    )
    goal = np.asarray(config["config"]["goal_vec"], dtype=float)
    goal /= np.linalg.norm(goal)
    angle = np.rad2deg(
        np.arccos(np.clip(boresight_eci @ goal, -1.0, 1.0))
    )
    return time, angle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    args = parser.parse_args()

    configure_ieee_style()
    result_path = _latest_sim(args.prefix)
    results = ADCS.SimulationResults.load(result_path)
    traces = [
        _angle_error_deg(run, config)
        for run, config in zip(results.runs, results.configs or [])
    ]
    if len(traces) != len(results):
        raise ValueError("The archive must contain one config per run")
    time = traces[0][0]
    values = np.vstack(
        [np.interp(time, run_time, angle) for run_time, angle in traces]
    )
    mean = np.mean(values, axis=0)
    sigma = np.std(values, axis=0)

    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    color = "#0072B2"
    time_min = time / 60.0
    ax.plot(time_min, mean, color=color, label="Planner")
    ax.fill_between(
        time_min, np.maximum(mean - sigma, 0.0), mean + sigma,
        color=color, alpha=0.18, linewidth=0.0, label=r"$\pm 1\sigma$",
    )
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Boresight angle error [deg]")
    ax.set_title("6U 3+1 boresight Planner convergence")
    ax.grid(True)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    figure_path = OUTPUT_DIR / f"{args.prefix}_angle_convergence.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Loaded {len(results)} runs from {result_path.name}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
