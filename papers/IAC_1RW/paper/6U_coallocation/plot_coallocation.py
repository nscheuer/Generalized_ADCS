"""Plot the latest disturbed 6U coallocation gamma sweep."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
PAPER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PAPER_DIR))

import ADCS
from plot_style import configure_ieee_style
from run_coallocation import (
    CAMPAIGN_PREFIX,
    GAMMAS,
    INITIAL_MOMENTUM_FRACTION,
    OUTPUT_DIR,
    gamma_slug,
)
from _6u_mc_common import WHEEL_MOMENTUM_MAX


STEADY_STATE_FRACTION = 0.20
TRADEOFF_TIME_S = 5444.0


def _latest_sim(gamma: float) -> Path:
    pattern = f"{CAMPAIGN_PREFIX}_gamma_{gamma_slug(gamma)}_*.sim"
    matches = list(OUTPUT_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No simulation found matching {OUTPUT_DIR / pattern}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def _angle_error_deg(run) -> np.ndarray:
    states = np.asarray([state.q for state in run.state_hist], dtype=float)
    targets = np.asarray(run.target_hist, dtype=float)
    dots = np.clip(np.abs(np.sum(states * targets, axis=1)), 0.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(dots))


def _histories(results: ADCS.SimulationResults) -> tuple[np.ndarray, np.ndarray]:
    angles = np.asarray([_angle_error_deg(run) for run in results.runs])
    momentum = np.asarray([
        np.abs([state.h[0] for state in run.state_hist]) / WHEEL_MOMENTUM_MAX
        for run in results.runs
    ])
    return angles, momentum


def _percentile_summary(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pointwise 10th percentile, median, and 90th percentile."""
    p10, median, p90 = np.percentile(values, (10, 50, 90), axis=0)
    return p10, median, p90


def main() -> None:
    configure_ieee_style()
    loaded = {
        float(gamma): ADCS.SimulationResults.load(_latest_sim(float(gamma)))
        for gamma in GAMMAS
    }
    histories = {gamma: _histories(results) for gamma, results in loaded.items()}
    time_min = np.asarray(next(iter(loaded.values())).runs[0].time_s) / 60.0
    colors = plt.colormaps["viridis"](np.linspace(0.05, 0.95, len(GAMMAS)))

    # Operational principle: increasing gamma continuously reallocates control
    # effort from attitude pointing to wheel momentum dumping.
    fig, (ax_angle, ax_h) = plt.subplots(2, 1, figsize=(3.45, 4.55), sharex=True)
    for gamma, color in zip(GAMMAS, colors):
        angles, momentum = histories[float(gamma)]
        angle_p10, angle_median, angle_p90 = _percentile_summary(angles)
        h_p10, h_median, h_p90 = _percentile_summary(momentum)
        label = rf"$\gamma={gamma:g}$"
        ax_angle.plot(time_min, angle_median, color=color, label=label)
        ax_angle.fill_between(
            time_min,
            angle_p10,
            angle_p90,
            color=color,
            alpha=0.10,
            linewidth=0.0,
        )
        ax_h.plot(time_min, h_median, color=color, label=label)
        ax_h.fill_between(
            time_min,
            h_p10,
            h_p90,
            color=color,
            alpha=0.10,
            linewidth=0.0,
        )

    ax_angle.axhspan(0.0, 5.0, color="#66A061", alpha=0.08, linewidth=0.0)
    ax_angle.axhline(5.0, color="#555555", linestyle="--", linewidth=0.7)
    ax_angle.set_ylabel("Median pointing error [deg]")
    ax_angle.set_ylim(bottom=0.0)
    ax_angle.set_title("Pointing/desaturation trade-off\n(median, 10--90%)")
    ax_angle.grid(True)
    ax_angle.legend(title=r"$\gamma$", loc="upper right", ncol=3,
                   fontsize=5.2, title_fontsize=5.5)

    ax_h.axhline(
        INITIAL_MOMENTUM_FRACTION,
        color="#555555",
        linestyle=":",
        linewidth=0.7,
        label="Initial momentum",
    )
    ax_h.axhline(
        1.0,
        color="#C62828",
        linestyle="--",
        linewidth=0.7,
        label=r"Wheel capacity, $h_{max}$",
    )
    ax_h.set_xlabel("Time [min]")
    ax_h.set_ylabel(r"Median $|h|/h_{max}$")
    ax_h.set_ylim(bottom=0.0)
    ax_h.grid(True)
    handles, labels = ax_h.get_legend_handles_labels()
    ax_h.legend(handles[-2:], labels[-2:], loc="upper right", ncol=1,
                fontsize=5.2)
    fig.tight_layout()
    operational_path = OUTPUT_DIR / "6u_coallocation_operational_principle.png"
    fig.savefig(operational_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    # Evaluate the trade-off at one common time instead of averaging a tail.
    # Each marker is the MC median and the error bar is the run-to-run
    # 10th--90th percentile interval at t=5444 s.  Marker colour gives the
    # 90th-percentile wheel momentum at the same instant, which highlights
    # saturation risk without being dominated by one extreme run.
    pointing_medians = []
    pointing_p10 = []
    pointing_p90 = []
    momentum_p90s = []
    for gamma in GAMMAS:
        angles, momentum = histories[float(gamma)]
        sample_index = int(np.argmin(np.abs(
            np.asarray(next(iter(loaded.values())).runs[0].time_s) - TRADEOFF_TIME_S
        )))
        per_run_angle = angles[:, sample_index]
        per_run_momentum = momentum[:, sample_index]
        angle_p10, angle_median, angle_p90 = np.percentile(
            per_run_angle, (10, 50, 90)
        )
        pointing_medians.append(angle_median)
        pointing_p10.append(angle_p10)
        pointing_p90.append(angle_p90)
        momentum_p90s.append(np.percentile(per_run_momentum, 90))

    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    ax.errorbar(
        GAMMAS,
        pointing_medians,
        yerr=(
            np.asarray(pointing_medians) - np.asarray(pointing_p10),
            np.asarray(pointing_p90) - np.asarray(pointing_medians),
        ),
        fmt="none",
        ecolor="#555555",
        elinewidth=0.8,
        capsize=2.5,
        zorder=1,
    )
    scatter = ax.scatter(
        GAMMAS,
        pointing_medians,
        c=100.0 * np.asarray(momentum_p90s),
        cmap="RdYlGn_r",
        s=35,
        edgecolor="black",
        linewidth=0.45,
        zorder=2,
    )
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.03)
    colorbar.set_label(r"90th-percentile $|h|/h_{max}$ at 5444 s [%]")
    ax.set_xlabel(r"Desaturation priority, $\gamma$")
    ax.set_ylabel("Median pointing error at 5444 s [deg]")
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(bottom=0.0)
    ax.grid(True)
    ax.set_title("Coallocation performance at 5444 s\n(median, 10--90%)")
    fig.tight_layout()
    tradeoff_path = OUTPUT_DIR / "6u_coallocation_gamma_vs_pointing.png"
    fig.savefig(tradeoff_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved operational plot: {operational_path}")
    print(f"Saved trade-off plot: {tradeoff_path}")
    print("\nResults at t=5444 s:")
    print(" gamma   pointing [deg]       |h|/hmax")
    for gamma, median, p10, p90, h_p90 in zip(
        GAMMAS, pointing_medians, pointing_p10, pointing_p90, momentum_p90s
    ):
        print(f" {gamma:5.2f}   {median:8.3f} [{p10:7.3f}, {p90:7.3f}]    {h_p90:8.3f}")


if __name__ == "__main__":
    main()
