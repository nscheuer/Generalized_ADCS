"""Direct 6U pointing-time comparison from saved Monte Carlo archives.

The figures report the measured fraction of each saved trajectory that meets a
pointing-error bound. They do not extrapolate to an operational duty cycle or
use a surrogate availability formula.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = SCRIPT_DIR / "outputs"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PAPER_DIR))

import ADCS
from ADCS.helpers.math_helpers import rot_mat
from plot_style import configure_ieee_style


SIX_U_MODE_DIR = PAPER_DIR / "6U_mode_switching_disturbance/outputs"
SIX_U_COALLOCATION_DIR = PAPER_DIR / "6U_coallocation/outputs"
SIX_U_PLANNER_DIR = PAPER_DIR / "6U_planner/outputs"
SIX_U_MODE_PREFIX = "6u_mode_switching_disturbance_mc_10_"
SIX_U_COALLOCATION_PREFIX = "6u_coallocation_disturbance_gamma_0p50_"
SIX_U_PLANNER_PREFIX = "6u_3mtq_1rw_boresight_planner_mc_10"


@dataclass(frozen=True)
class MethodResults:
    """Pointing-error histories from one saved 6U Monte Carlo campaign."""

    name: str
    time_s: list[np.ndarray]
    error_deg: list[np.ndarray]


def _latest_archive(directory: Path, prefix: str) -> Path:
    matches = list(directory.glob(f"{prefix}*.sim"))
    if not matches:
        raise FileNotFoundError(
            f"No 6U simulation archive matching {prefix!r} in {directory}"
        )
    return max(matches, key=lambda path: path.stat().st_mtime)


def _full_attitude_error_deg(run) -> np.ndarray:
    quaternion = np.asarray([state.q for state in run.state_hist], dtype=float)
    target = np.asarray(run.target_hist, dtype=float)
    dots = np.clip(np.einsum("ij,ij->i", quaternion, target), -1.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(np.abs(dots)))


def _planner_boresight_error_deg(run, config: dict) -> np.ndarray:
    quaternion = np.asarray([state.q for state in run.state_hist], dtype=float)
    goal = np.asarray(config["config"]["goal_vec"], dtype=float)
    goal /= np.linalg.norm(goal)
    body_z = np.array([0.0, 0.0, 1.0])
    boresight_eci = np.asarray([rot_mat(q) @ body_z for q in quaternion])
    return np.rad2deg(np.arccos(np.clip(boresight_eci @ goal, -1.0, 1.0)))


def _load_method(path: Path, name: str, *, planner: bool = False) -> MethodResults:
    results = ADCS.SimulationResults.load(path)
    if planner and (results.configs is None or len(results.configs) != len(results.runs)):
        raise ValueError("The planner archive must contain one configuration per run.")

    time_s = [np.asarray(run.time_s, dtype=float) for run in results.runs]
    error_deg = [
        _planner_boresight_error_deg(run, results.configs[index])
        if planner else _full_attitude_error_deg(run)
        for index, run in enumerate(results.runs)
    ]
    return MethodResults(name=name, time_s=time_s, error_deg=error_deg)


def load_methods() -> list[MethodResults]:
    """Load the latest saved 6U random-slew campaigns."""
    return [
        _load_method(
            _latest_archive(SIX_U_MODE_DIR, SIX_U_MODE_PREFIX),
            "Reactive 75/25",
        ),
        _load_method(
            _latest_archive(SIX_U_COALLOCATION_DIR, SIX_U_COALLOCATION_PREFIX),
            r"Co-allocation $\gamma=0.5$",
        ),
        _load_method(
            _latest_archive(SIX_U_PLANNER_DIR, SIX_U_PLANNER_PREFIX),
            "Feasibility-aware planner",
            planner=True,
        ),
    ]


def _time_in_bound_s(time_s: np.ndarray, error_deg: np.ndarray, bound_deg: float) -> float:
    """Integrate recorded intervals whose initial sample meets ``error <= bound``."""
    if time_s.size != error_deg.size or time_s.size < 2:
        raise ValueError("Each error history must contain at least two matched time samples.")
    return float(np.sum(np.diff(time_s) * (error_deg[:-1] <= bound_deg)))


def pointing_time_percent(method: MethodResults, bound_deg: float) -> np.ndarray:
    """Return the percentage of each recorded run spent inside the bound."""
    percentages = []
    for time_s, error_deg in zip(method.time_s, method.error_deg):
        duration_s = time_s[-1] - time_s[0]
        percentages.append(100.0 * _time_in_bound_s(time_s, error_deg, bound_deg) / duration_s)
    return np.asarray(percentages)


def plot_pointing_time(methods: list[MethodResults]) -> Path:
    """Plot direct per-run time within both pointing-error requirements."""
    colors = ("#D55E00", "#009E73", "#0072B2")
    positions = np.arange(len(methods))
    width = 0.34

    fig, ax = plt.subplots(figsize=(4.7, 3.0))
    for bound_deg, offset, hatch in ((5.0, -width / 2.0, ""), (1.0, width / 2.0, "//")):
        values = [pointing_time_percent(method, bound_deg) for method in methods]
        means = np.asarray([np.mean(value) for value in values])
        sigmas = np.asarray([
            np.std(value, ddof=1) if len(value) > 1 else 0.0 for value in values
        ])
        bars = ax.bar(
            positions + offset, means, yerr=sigmas, capsize=3.0, width=width,
            color=colors, hatch=hatch, edgecolor="black", linewidth=0.55,
            label=fr"$\leq {bound_deg:g}^\circ$",
        )
        for index, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width() / 2.0,
                    means[index] + sigmas[index] + 2.5,
                    f"{means[index]:.1f}%", ha="center", va="bottom", fontsize=6.0)

    ax.set_xticks(positions, [method.name for method in methods])
    ax.set_ylabel("Time in pointing [% of run]")
    ax.set_ylim(0.0, 108.0)
    ax.set_title("6U time in pointing (mean ± 1σ)")
    ax.grid(True, axis="y")
    ax.legend(title="Pointing bound", frameon=True)
    fig.tight_layout()
    path = OUTPUT_DIR / "time_in_1deg_and_5deg_pointing.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_summary(methods: list[MethodResults]) -> Path:
    """Save the direct 5° and 1° pointing-time measurements."""
    path = OUTPUT_DIR / "pointing_time_summary.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "method", "runs", "mean_time_in_5deg_percent", "std_time_in_5deg_percent",
            "mean_time_in_1deg_percent", "std_time_in_1deg_percent",
        ])
        for method in methods:
            time_5deg = pointing_time_percent(method, 5.0)
            time_1deg = pointing_time_percent(method, 1.0)
            writer.writerow([
                method.name.replace("$", ""),
                len(time_5deg),
                f"{np.mean(time_5deg):.6f}",
                f"{np.std(time_5deg, ddof=1):.6f}",
                f"{np.mean(time_1deg):.6f}",
                f"{np.std(time_1deg, ddof=1):.6f}",
            ])
    return path


def main() -> None:
    configure_ieee_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    methods = load_methods()
    outputs = [
        plot_pointing_time(methods),
        save_summary(methods),
    ]
    print("Direct time-in-pointing measurements from saved 6U archives:")
    for method in methods:
        time_5deg = pointing_time_percent(method, 5.0)
        time_1deg = pointing_time_percent(method, 1.0)
        print(
            f"  {method.name}: ≤5°={np.mean(time_5deg):.1f}% ± {np.std(time_5deg, ddof=1):.1f}%, "
            f"≤1°={np.mean(time_1deg):.1f}% ± {np.std(time_1deg, ddof=1):.1f}%"
        )
    for path in outputs:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
