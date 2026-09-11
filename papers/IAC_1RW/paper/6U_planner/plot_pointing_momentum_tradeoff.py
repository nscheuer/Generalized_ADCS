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
STEADY_STATE_FRACTION = coallocation.STEADY_STATE_FRACTION
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
        momentum = np.abs([state.h[0] for state in run.state_hist]) / coallocation.WHEEL_MOMENTUM_MAX
        tail = max(1, int(np.ceil(len(time) * STEADY_STATE_FRACTION)))
        angles.append(float(np.mean(angle[-tail:])))
        momenta.append(float(np.mean(momentum[-tail:])))
    angles, momenta = np.asarray(angles), np.asarray(momenta)
    return float(np.mean(momenta)), float(np.mean(angles)), float(np.std(angles, ddof=1))


def main() -> None:
    configure_ieee_style()
    points = []
    for gamma in coallocation.GAMMAS:
        results = ADCS.SimulationResults.load(coallocation._latest_sim(float(gamma)))
        x, y, yerr = _point(results)
        points.append((x, y, yerr, rf"$\gamma={gamma:g}$"))

    mode = ADCS.SimulationResults.load(_latest(MODE_OUTPUT_DIR, MODE_PREFIX))
    x, y, yerr = _point(mode)
    points.append((x, y, yerr, "Mode switching"))

    planner = ADCS.SimulationResults.load(_latest(PLANNER_OUTPUT_DIR, PLANNER_PREFIX))
    x, y, yerr = _point(planner, planner=True)
    points.append((x, y, yerr, "Planner"))

    fig, ax = plt.subplots(figsize=(4.1, 3.0))
    colors = plt.colormaps["viridis"](np.linspace(0.05, 0.95, len(coallocation.GAMMAS)))
    gamma_handles = []
    for point, color in zip(points[:len(colors)], colors):
        x, y, yerr, label = point
        ax.errorbar(x, y, yerr=yerr, fmt="o", color=color, mec="black", mew=0.45,
                    ms=5.5, capsize=2.5, elinewidth=0.8, zorder=3)
        gamma_handles.append(Line2D([], [], marker="o", color=color, linestyle="none",
                                     markeredgecolor="black", markeredgewidth=0.45,
                                     markersize=5.5, label=label))
    comparator_handles = []
    for point, marker, color in zip(points[-2:], ("s", "^"), ("#D55E00", "#0072B2")):
        x, y, yerr, label = point
        ax.errorbar(x, y, yerr=yerr, fmt=marker, color=color, mec="black", mew=0.45,
                    ms=6.5, capsize=2.5, elinewidth=0.8, zorder=4)
        comparator_handles.append(Line2D([], [], marker=marker, color=color, linestyle="none",
                                          markeredgecolor="black", markeredgewidth=0.45,
                                          markersize=6.5, label=label))

    ax.set_xlabel(r"Final mean $|h|/h_{max}$")
    ax.set_ylabel("Final mean pointing error [deg]")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True)
    ax.set_title("Pointing–momentum trade-off")
    section_handles = [
        Line2D([], [], linestyle="none", label="Co-Allocation"),
        *gamma_handles,
        Line2D([], [], color="#AAAAAA", linewidth=0.7, label="────────────"),
        Line2D([], [], linestyle="none", label="Comparators"),
        *comparator_handles,
    ]
    ax.legend(handles=section_handles, loc="upper right", fontsize=6,
              frameon=True, ncol=1, handletextpad=0.5,
              labelspacing=0.35)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / OUTPUT_NAME
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
