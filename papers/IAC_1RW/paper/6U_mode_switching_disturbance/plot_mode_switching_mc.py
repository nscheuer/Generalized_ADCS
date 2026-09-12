"""Plot a saved 6U mode-switching Monte Carlo archive."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
PAPER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PAPER_DIR))
sys.path.insert(0, str(PAPER_DIR / "6U_estimator_nodisturbance"))

import ADCS
from plot_style import configure_ieee_style
from _6u_mc_common import WHEEL_MOMENTUM_MAX


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CAMPAIGN_PREFIX = "6u_mode_switching_disturbance_mc_10_1000min"
DESAT_ENTRY = 0.75
DESAT_EXIT = 0.25


def _mode_history(h_fraction: np.ndarray) -> np.ndarray:
    """Reconstruct the hysteretic desaturation mode from wheel momentum."""
    desaturation = False
    modes = np.zeros(len(h_fraction), dtype=bool)
    for index, fraction in enumerate(h_fraction):
        if not desaturation and fraction >= DESAT_ENTRY:
            desaturation = True
        elif desaturation and fraction <= DESAT_EXIT:
            desaturation = False
        modes[index] = desaturation
    return modes


def _angle_error_deg(run) -> np.ndarray:
    states = np.asarray([state.q for state in run.state_hist], dtype=float)
    targets = np.asarray(run.target_hist, dtype=float)
    dots = np.clip(np.abs(np.sum(states * targets, axis=1)), 0.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(dots))


def _fraction_bars(ax, time_min: np.ndarray, fractions: np.ndarray, color: str) -> None:
    """Add a translucent five-minute population-fraction bar background."""
    bin_width_min = 5.0
    edges = np.arange(time_min[0], time_min[-1] + bin_width_min, bin_width_min)
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = np.array([
        np.mean(fractions[(time_min >= left) & (time_min < right)])
        for left, right in zip(edges[:-1], edges[1:])
    ])
    background = ax.twinx()
    background.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_alpha(0.0)
    background.bar(centers, 100.0 * values, width=bin_width_min * 0.96,
                   color=color, alpha=0.16, edgecolor="none", zorder=0)
    background.set_ylim(0.0, 100.0)
    background.set_ylabel("Satellites [%]", color=color)
    background.tick_params(axis="y", colors=color)
    background.grid(False)


def plot_results(results: ADCS.SimulationResults, *, run_index: int = 0,
                 output_path: Path | None = None) -> Path:
    """Plot one highlighted run while computing population bars from all runs."""
    if not results.runs:
        raise ValueError("The simulation archive contains no runs.")
    if not 0 <= run_index < len(results.runs):
        raise ValueError(f"run index must be between 0 and {len(results.runs) - 1}")

    configure_ieee_style()
    time = np.asarray(results.runs[0].time_s, dtype=float) / 60.0
    angles = np.asarray([_angle_error_deg(run) for run in results.runs])
    momenta = np.asarray([
        np.abs(np.asarray([state.h[0] for state in run.state_hist])) / WHEEL_MOMENTUM_MAX
        for run in results.runs
    ])
    modes = np.asarray([_mode_history(fraction) for fraction in momenta])

    fig, (ax_angle, ax_h) = plt.subplots(2, 1, figsize=(6.4, 4.8), sharex=True)
    _fraction_bars(ax_angle, time, np.mean(angles < 5.0, axis=0), "#66A061")
    _fraction_bars(ax_h, time, np.mean(modes, axis=0), "#C62828")

    angle = angles[run_index]
    ax_angle.plot(time, angle, color="#0072B2", lw=0.8, label="Pointing error")
    ax_h.plot(time, momenta[run_index], color="#D55E00", lw=0.8, label="Wheel momentum")

    ax_angle.axhspan(0.0, 5.0, color="#66A061", alpha=0.08, lw=0)
    ax_angle.axhline(5.0, color="#555555", ls="--", lw=0.7)
    ax_angle.set_ylabel("Pointing error [deg]")
    ax_angle.set_title("6U disturbed mode switching Monte Carlo")
    ax_angle.set_ylim(bottom=0.0)
    ax_angle.grid(True)
    ax_angle.legend(handles=[
        Line2D([], [], color="#0072B2", lw=1.1, label="Pointing error"),
        Line2D([], [], color="#555555", ls="--", lw=0.8, label="5° requirement"),
    ], loc="upper right", ncol=2)

    ax_h.axhline(DESAT_ENTRY, color="#C62828", ls="--", lw=0.7)
    ax_h.axhline(DESAT_EXIT, color="#C62828", ls=":", lw=0.7)
    ax_h.set_xlabel("Time [min]")
    ax_h.set_ylabel(r"$|h|/h_{max}$")
    ax_h.set_ylim(0.0, 1.0)
    ax_h.grid(True)
    ax_h.legend(handles=[
        Line2D([], [], color="#D55E00", lw=1.1, label="Wheel momentum"),
        Line2D([], [], color="#C62828", ls="--", lw=0.8, label="75% entry"),
        Line2D([], [], color="#C62828", ls=":", lw=0.8, label="25% restart"),
    ], loc="upper right", ncol=3)

    fig.tight_layout()
    if output_path is None:
        output_path = OUTPUT_DIR / f"{CAMPAIGN_PREFIX}_run{run_index}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _latest_sim() -> Path:
    candidates = sorted(OUTPUT_DIR.glob(f"{CAMPAIGN_PREFIX}_*.sim"))
    if not candidates:
        raise FileNotFoundError(f"No .sim archives matching {CAMPAIGN_PREFIX!r} in {OUTPUT_DIR}")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=None, help="Saved .sim archive (default: latest matching archive).")
    parser.add_argument("--run", type=int, default=4, help="Zero-based run index to highlight (default: 4).")
    parser.add_argument("--output", type=Path, default=None, help="Output image path.")
    args = parser.parse_args()

    result_path = args.file if args.file is not None else _latest_sim()
    results = ADCS.SimulationResults.load(result_path, ephem=ADCS.Ephemeris())
    output_path = plot_results(results, run_index=args.run, output_path=args.output)
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
