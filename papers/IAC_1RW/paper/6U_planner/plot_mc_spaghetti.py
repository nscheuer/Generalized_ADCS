"""Plot per-run boresight angle and wheel-momentum spaghetti traces."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ADCS
from plot_style import configure_ieee_style

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
DEFAULT_PREFIX = "6u_3mtq_1rw_boresight_planner_mc_10"
WHEEL_H_MAX_NMS = 15.0e-3


def _latest_sim(prefix: str) -> Path:
    candidates = list(OUTPUT_DIR.glob(f"{prefix}_*.sim"))
    if not candidates:
        raise FileNotFoundError(f"No .sim archive matching {prefix!r} in {OUTPUT_DIR}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _angle_error_deg(run, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return body +z boresight-to-goal angle, matching paper/IAC_1RW clean."""
    time = np.asarray(run.time_s, dtype=float)
    q = np.asarray([state.q for state in run.state_hist], dtype=float)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    xyz = np.column_stack((x, y, z))
    boresight = np.array([0.0, 0.0, 1.0])
    cross_term = 2.0 * np.cross(xyz, boresight)
    boresight_eci = boresight + w[:, None] * cross_term + np.cross(xyz, cross_term)
    goal = np.asarray(config["config"]["goal_vec"], dtype=float)
    goal /= np.linalg.norm(goal)
    return time, np.rad2deg(np.arccos(np.clip(boresight_eci @ goal, -1.0, 1.0)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    args = parser.parse_args()

    configure_ieee_style()
    result_path = _latest_sim(args.prefix)
    results = ADCS.SimulationResults.load(result_path)
    configs = results.configs or []
    if len(configs) != len(results):
        raise ValueError("The archive must contain one config per run")

    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(results)))
    fig, (ax_angle, ax_momentum) = plt.subplots(
        2, 1, figsize=(3.45, 4.8), sharex=True,
        gridspec_kw={"height_ratios": (1.0, 1.0)},
        constrained_layout=True,
    )

    for color, run, config in zip(colors, results.runs, configs):
        time, angle = _angle_error_deg(run, config)
        time_min = time / 60.0
        momentum = np.asarray([state.h[0] for state in run.state_hist]) * 1.0e3
        run_id = config.get("config", {}).get("run_id", "?")
        ax_angle.plot(time_min, angle, color=color, lw=0.8, alpha=0.9)
        ax_momentum.plot(time_min, momentum, color=color, lw=0.8, alpha=0.9)

    ax_angle.set_ylabel("Boresight error [deg]")
    ax_angle.set_title("6U 3+1 boresight Planner: individual runs")
    ax_angle.set_ylim(bottom=0.0)
    ax_angle.grid(True)

    ax_momentum.axhline(WHEEL_H_MAX_NMS * 1.0e3, color="#D55E00", ls="--", lw=0.8)
    ax_momentum.axhline(-WHEEL_H_MAX_NMS * 1.0e3, color="#D55E00", ls="--", lw=0.8)
    ax_momentum.fill_between(
        [0.0, max(run.time_s[-1] for run in results.runs) / 60.0],
        -WHEEL_H_MAX_NMS * 1.0e3,
        WHEEL_H_MAX_NMS * 1.0e3,
        color="#D55E00", alpha=0.05, linewidth=0.0,
    )
    ax_momentum.set_xlabel("Time [min]")
    ax_momentum.set_ylabel("Wheel momentum [mN·m·s]")
    ax_momentum.set_ylim(-WHEEL_H_MAX_NMS * 1.0e3 * 1.1, WHEEL_H_MAX_NMS * 1.0e3 * 1.1)
    ax_momentum.grid(True)

    norm = Normalize(vmin=0, vmax=max(len(results) - 1, 1))
    sm = ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    colorbar = fig.colorbar(sm, ax=(ax_angle, ax_momentum), pad=0.02, aspect=35)
    colorbar.set_label("Run index")
    colorbar.set_ticks([0, len(results) - 1])

    figure_path = OUTPUT_DIR / f"{args.prefix}_angle_momentum_spaghetti.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Loaded {len(results)} runs from {result_path.name}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
