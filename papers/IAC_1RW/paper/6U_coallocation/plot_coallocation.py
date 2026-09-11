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


def _mean_sigma(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.mean(values, axis=0), np.std(values, axis=0, ddof=1)


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
    fig, (ax_angle, ax_h) = plt.subplots(2, 1, figsize=(6.4, 4.8), sharex=True)
    for gamma, color in zip(GAMMAS, colors):
        angles, momentum = histories[float(gamma)]
        angle_mean, angle_sigma = _mean_sigma(angles)
        h_mean, h_sigma = _mean_sigma(momentum)
        label = rf"$\gamma={gamma:g}$"
        ax_angle.plot(time_min, angle_mean, color=color, label=label)
        ax_angle.fill_between(
            time_min,
            np.maximum(0.0, angle_mean - angle_sigma),
            angle_mean + angle_sigma,
            color=color,
            alpha=0.10,
            linewidth=0.0,
        )
        ax_h.plot(time_min, h_mean, color=color, label=label)
        ax_h.fill_between(
            time_min,
            np.maximum(0.0, h_mean - h_sigma),
            h_mean + h_sigma,
            color=color,
            alpha=0.10,
            linewidth=0.0,
        )

    ax_angle.axhspan(0.0, 5.0, color="#66A061", alpha=0.08, linewidth=0.0)
    ax_angle.axhline(5.0, color="#555555", linestyle="--", linewidth=0.7)
    ax_angle.set_ylabel("Pointing error [deg]")
    ax_angle.set_ylim(bottom=0.0)
    ax_angle.set_title("6U coallocation: pointing/desaturation trade-off (mean ± 1σ)")
    ax_angle.grid(True)
    ax_angle.legend(loc="upper right", ncol=3)

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
    ax_h.set_ylabel(r"$|h|/h_{max}$")
    ax_h.set_ylim(bottom=0.0)
    ax_h.grid(True)
    handles, labels = ax_h.get_legend_handles_labels()
    ax_h.legend(handles[-2:], labels[-2:], loc="upper right", ncol=2)
    fig.tight_layout()
    operational_path = OUTPUT_DIR / "6u_coallocation_operational_principle.png"
    fig.savefig(operational_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    # The final fifth of every run is treated as steady-state.  Each marker is
    # the MC mean and the error bar is the run-to-run 1-sigma spread.  Marker
    # colour gives the corresponding final wheel momentum, keeping the trade-off
    # visible without adding a second ordinate.
    pointing_means = []
    pointing_sigmas = []
    momentum_means = []
    for gamma in GAMMAS:
        angles, momentum = histories[float(gamma)]
        tail = max(1, int(np.ceil(angles.shape[1] * STEADY_STATE_FRACTION)))
        per_run_angle = np.mean(angles[:, -tail:], axis=1)
        per_run_momentum = np.mean(momentum[:, -tail:], axis=1)
        pointing_means.append(np.mean(per_run_angle))
        pointing_sigmas.append(np.std(per_run_angle, ddof=1))
        momentum_means.append(np.mean(per_run_momentum))

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.errorbar(
        GAMMAS,
        pointing_means,
        yerr=pointing_sigmas,
        fmt="none",
        ecolor="#555555",
        elinewidth=0.8,
        capsize=2.5,
        zorder=1,
    )
    scatter = ax.scatter(
        GAMMAS,
        pointing_means,
        c=100.0 * np.asarray(momentum_means),
        cmap="magma_r",
        s=35,
        edgecolor="black",
        linewidth=0.45,
        zorder=2,
    )
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.03)
    colorbar.set_label(r"Final mean $|h|/h_{max}$ [%]")
    ax.set_xlabel(r"Desaturation priority, $\gamma$")
    ax.set_ylabel("Final mean pointing error [deg]")
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(bottom=0.0)
    ax.grid(True)
    ax.set_title("Coallocation performance trade-off (mean ± 1σ)")
    fig.tight_layout()
    tradeoff_path = OUTPUT_DIR / "6u_coallocation_gamma_vs_pointing.png"
    fig.savefig(tradeoff_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved operational plot: {operational_path}")
    print(f"Saved trade-off plot: {tradeoff_path}")
    print("\nSteady-state results (final 20%):")
    print(" gamma   pointing [deg]       |h|/hmax")
    for gamma, mean, sigma, h_mean in zip(
        GAMMAS, pointing_means, pointing_sigmas, momentum_means
    ):
        print(f" {gamma:5.2f}   {mean:8.3f} ± {sigma:7.3f}    {h_mean:8.3f}")


if __name__ == "__main__":
    main()
