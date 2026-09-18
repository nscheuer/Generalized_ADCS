"""Plot co-allocation, mode-switching, and Planner pointing/momentum trade-offs."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
COALLOC_DIR = REPO_ROOT / "papers/IAC_1RW/paper/6U_coallocation"
PAPER_DIR = COALLOC_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(COALLOC_DIR))
sys.path.insert(0, str(PAPER_DIR))

import ADCS
from plot_style import configure_ieee_style
import plot_coallocation as coallocation


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_NAME = "6u_pointing_momentum_tradeoff.png"
TRADEOFF_TIME_S = coallocation.TRADEOFF_TIME_S
MODE_OUTPUT_DIR = PAPER_DIR / "6U_mode_switching_disturbance" / "outputs"
PLANNER_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
MODE_PREFIX = "6u_mode_switching_disturbance_mc_10_1000min"
PLANNER_PREFIX = "6u_3mtq_1rw_boresight_planner_mc_10"


def _latest(directory: Path, prefix: str) -> Path:
    matches = list(directory.glob(f"{prefix}_*.sim"))
    if not matches:
        raise FileNotFoundError(f"No archive matching {prefix!r} in {directory}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def _planner_angle_error(run, config):
    time = np.asarray(run.time_s, dtype=float)
    q = np.asarray([state.q for state in run.state_hist], dtype=float)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    xyz = np.column_stack((x, y, z))
    b = np.array([0.0, 0.0, 1.0])
    t = 2.0 * np.cross(xyz, b)
    b_eci = b + w[:, None] * t + np.cross(xyz, t)
    goal = np.asarray(config["config"]["goal_vec"], dtype=float)
    goal /= np.linalg.norm(goal)
    return time, np.rad2deg(np.arccos(np.clip(b_eci @ goal, -1.0, 1.0)))


def _point(results, *, planner=False):
    angles, momenta = [], []
    for run, config in zip(results.runs, results.configs or []):
        if planner:
            time, angle = _planner_angle_error(run, config)
        else:
            time = np.asarray(run.time_s, dtype=float)
            angle = coallocation._angle_error_deg(run)
        momentum = np.abs(np.asarray([state.h[0] for state in run.state_hist])) / coallocation.WHEEL_MOMENTUM_MAX
        angle_at_time = float(np.interp(TRADEOFF_TIME_S, time, angle))
        momentum_at_time = float(np.interp(TRADEOFF_TIME_S, time, momentum))
        angles.append(angle_at_time)
        momenta.append(momentum_at_time)
    angles, momenta = np.asarray(angles), np.asarray(momenta)
    angle_p10, angle_median, angle_p90 = np.percentile(angles, (10, 50, 90))
    momentum_p90 = np.percentile(momenta, 90)
    return float(momentum_p90), float(angle_median), float(angle_p10), float(angle_p90)


def main() -> None:
    configure_ieee_style()
    points = []
    for gamma in coallocation.GAMMAS:
        results = ADCS.SimulationResults.load(coallocation._latest_sim(float(gamma)))
        x, y, p10, p90 = _point(results)
        points.append((x, y, p10, p90, rf"$\gamma={gamma:g}$"))

    mode = ADCS.SimulationResults.load(_latest(MODE_OUTPUT_DIR, MODE_PREFIX))
    x, y, p10, p90 = _point(mode)
    points.append((x, y, p10, p90, "Mode switching"))

    planner = ADCS.SimulationResults.load(_latest(PLANNER_OUTPUT_DIR, PLANNER_PREFIX))
    x, y, p10, p90 = _point(planner, planner=True)
    points.append((x, y, p10, p90, "Planner"))

    fig, ax = plt.subplots(figsize=(3.45, 3.35))
    colors = plt.colormaps["viridis"](np.linspace(0.05, 0.95, len(coallocation.GAMMAS)))
    gamma_handles = []
    for point, color in zip(points[:len(colors)], colors):
        x, y, p10, p90, label = point
        ax.errorbar(x * 100.0, y, yerr=np.asarray([[y - p10], [p90 - y]]), fmt="o", color=color, mec="black", mew=0.45,
                    ms=5.5, capsize=2.5, elinewidth=0.8, zorder=3)
        gamma_handles.append(Line2D([], [], marker="o", color=color, linestyle="none",
                                     markeredgecolor="black", markeredgewidth=0.45,
                                     markersize=5.5, label=label))
    comparator_handles = []
    for point, marker, color in zip(points[-2:], ("s", "^"), ("#D55E00", "#0072B2")):
        x, y, p10, p90, label = point
        ax.errorbar(x * 100.0, y, yerr=np.asarray([[y - p10], [p90 - y]]), fmt=marker, color=color, mec="black", mew=0.45,
                    ms=6.5, capsize=2.5, elinewidth=0.8, zorder=4)
        comparator_handles.append(Line2D([], [], marker=marker, color=color, linestyle="none",
                                          markeredgecolor="black", markeredgewidth=0.45,
                                          markersize=6.5, label=label))

    ax.set_xlabel(r"90th-percentile $|h|/h_{max}$ at 5444 s [\%]")
    ax.set_ylabel("Median pointing error at 5444 s [deg]")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True)
    ax.set_title("Pointing–momentum trade-off at 5444 s\n(median, 10--90%)")
    fig.legend(
        handles=gamma_handles + comparator_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        fontsize=5.0,
        frameon=False,
        ncol=4,
        handletextpad=0.4,
        labelspacing=0.3,
        columnspacing=0.8,
    )
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.25, top=0.82)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / OUTPUT_NAME
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
