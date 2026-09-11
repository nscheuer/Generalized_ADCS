"""Run and plot a 10-run disturbed 6U mode-switching Monte Carlo campaign."""

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
from ADCS.CONOPS.goals import No_Goal
from ADCS.helpers.math_helpers import normalize
from plot_style import configure_ieee_style
from _6u_mc_common import (
    KP,
    KD,
    MOMENTUM_GAIN,
    WHEEL_MOMENTUM_MAX,
    controller_for,
    make_satellite,
    random_400km_97deg_orbit,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CAMPAIGN_NAME = "6u_mode_switching_disturbance_mc_10_1000min"
DT_S = 5.0
DESAT_ENTRY = 0.75
DESAT_EXIT = 0.25


class HystereticModeController:
    """Route the LP controller between pointing and No_Goal desaturation."""

    def __init__(self, controller, *, momentum_max: float) -> None:
        self.controller = controller
        self.momentum_max = float(momentum_max)
        self.mode = "pointing"

    def find_u(self, *, x_hat, sens, est_sat, os_hat, goal):
        h_fraction = abs(float(np.asarray(x_hat.h).reshape(-1)[0])) / self.momentum_max
        if self.mode == "pointing" and h_fraction >= DESAT_ENTRY:
            self.mode = "desaturation"
        elif self.mode == "desaturation" and h_fraction <= DESAT_EXIT:
            self.mode = "pointing"
        active_goal = No_Goal() if self.mode == "desaturation" else goal
        return self.controller.find_u(
            x_hat=x_hat, sens=sens, est_sat=est_sat, os_hat=os_hat, goal=active_goal
        )


def _angle_error_deg(run) -> np.ndarray:
    states = np.asarray([state.q for state in run.state_hist], dtype=float)
    targets = np.asarray(run.target_hist, dtype=float)
    dots = np.clip(np.abs(np.sum(states * targets, axis=1)), 0.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(dots))


def _mode_history(h_fraction: np.ndarray) -> np.ndarray:
    """Reconstruct the controller's hysteretic mode from the saved true state."""
    desaturation = False
    modes = np.zeros(len(h_fraction), dtype=bool)
    for index, fraction in enumerate(h_fraction):
        if not desaturation and fraction >= DESAT_ENTRY:
            desaturation = True
        elif desaturation and fraction <= DESAT_EXIT:
            desaturation = False
        modes[index] = desaturation
    return modes


def _fraction_bars(ax, time_min: np.ndarray, fractions: np.ndarray, color: str) -> None:
    """Add a translucent, five-minute population-fraction bar background."""
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


def _plot(results: ADCS.SimulationResults) -> Path:
    configure_ieee_style()
    time = np.asarray(results.runs[0].time_s, dtype=float) / 60.0
    fig, (ax_angle, ax_h) = plt.subplots(2, 1, figsize=(6.4, 4.8), sharex=True)

    all_angles = []
    all_modes = []
    for run in results.runs:
        angle = _angle_error_deg(run)
        h_fraction = np.abs(np.asarray([state.h[0] for state in run.state_hist])) / WHEEL_MOMENTUM_MAX
        modes = _mode_history(h_fraction)
        all_angles.append(angle)
        all_modes.append(modes)

        # Each run is split by the pointing requirement: green means valid
        # pointing (<5 deg), grey means outside the requirement.
        ax_angle.plot(time, np.where(angle < 5.0, angle, np.nan), color="#66A061", alpha=0.65, lw=0.65)
        ax_angle.plot(time, np.where(angle >= 5.0, angle, np.nan), color="#9E9E9E", alpha=0.55, lw=0.65)

        # Black is pointing mode; red is active B-dot/desaturation mode.
        ax_h.plot(time, np.where(~modes, h_fraction, np.nan), color="black", alpha=0.55, lw=0.65)
        ax_h.plot(time, np.where(modes, h_fraction, np.nan), color="#C62828", alpha=0.70, lw=0.75)

    _fraction_bars(ax_angle, time, np.mean(np.asarray(all_angles) < 5.0, axis=0), "#66A061")
    _fraction_bars(ax_h, time, np.mean(np.asarray(all_modes), axis=0), "#C62828")

    ax_angle.axhspan(0.0, 5.0, color="#66A061", alpha=0.08, lw=0)
    ax_angle.axhline(5.0, color="#555555", ls="--", lw=0.7)
    ax_angle.set_ylabel("Pointing error [deg]")
    ax_angle.set_title("6U disturbed mode switching Monte Carlo (N=10, 1000 min)")
    ax_angle.set_ylim(bottom=0.0)
    ax_angle.grid(True)
    ax_angle.legend(handles=[
        Line2D([], [], color="#66A061", lw=1.1, label="Pointing error < 5°"),
        Line2D([], [], color="#9E9E9E", lw=1.1, label="Pointing error ≥ 5°"),
        Line2D([], [], color="#555555", ls="--", lw=0.8, label="5° requirement"),
    ], loc="upper right", ncol=3)

    ax_h.axhline(DESAT_ENTRY, color="#C62828", ls="--", lw=0.7, label="Desaturation entry: 75%")
    ax_h.axhline(DESAT_EXIT, color="#C62828", ls=":", lw=0.7, label="Pointing restart: 25%")
    ax_h.set_xlabel("Time [min]")
    ax_h.set_ylabel(r"$|h|/h_{max}$")
    ax_h.set_ylim(0.0, 1.0)
    ax_h.grid(True)
    ax_h.legend(handles=[
        Line2D([], [], color="black", lw=1.1, label="Pointing mode"),
        Line2D([], [], color="#C62828", lw=1.1, label="Desaturation mode"),
        Line2D([], [], color="#C62828", ls="--", lw=0.8, label="75% entry"),
        Line2D([], [], color="#C62828", ls=":", lw=0.8, label="25% restart"),
    ], loc="upper right", ncol=4)

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{CAMPAIGN_NAME}.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--tf", type=float, default=1000.0 * 60.0)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()

    satellite = make_satellite(1, estimated=False, disturbances=True)
    controller = HystereticModeController(
        controller_for("lp", satellite, 1, kp=KP, kd=KD, c_gain=MOMENTUM_GAIN),
        momentum_max=WHEEL_MOMENTUM_MAX,
    )
    state = ADCS.State(w=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]), h=np.array([0.5 * WHEEL_MOMENTUM_MAX]))
    mc_config = ADCS.MCConfig(
        q=lambda rng: normalize(rng.standard_normal(4)),
        h=lambda rng: np.array([rng.uniform(0.50, 0.72) * WHEEL_MOMENTUM_MAX]),
        goal=lambda rng: ADCS.goals.Fixed_Attitude_Goal(normalize(rng.standard_normal(4))),
        orbit=random_400km_97deg_orbit,
    )
    results = ADCS.simulate_mc(
        x=state,
        satellite=satellite,
        est_satellite=satellite,
        controller=controller,
        estimator=None,
        goal=ADCS.goals.Fixed_Attitude_Goal(np.array([1.0, 0.0, 0.0, 0.0])),
        os0=random_400km_97deg_orbit(np.random.default_rng(args.seed)),
        dt=DT_S,
        tf=args.tf,
        mc_config=mc_config,
        num_runs=args.runs,
        max_workers=args.workers,
        base_seed=args.seed,
    )
    sim_path = results.save(CAMPAIGN_NAME, out_dir=OUTPUT_DIR)
    figure_path = _plot(results)
    print(f"Saved simulation: {sim_path}")
    print(f"Saved figure: {figure_path}")
    for run_id, run in zip(results.run_ids or range(len(results.runs)), results.runs):
        h_fraction = np.abs(np.asarray([state.h[0] for state in run.state_hist])) / WHEEL_MOMENTUM_MAX
        modes = _mode_history(h_fraction)
        switches = int(np.count_nonzero(np.diff(modes.astype(int))))
        print(f"Run {run_id}: {switches} mode switches, h range {h_fraction.min():.2f}–{h_fraction.max():.2f}")


if __name__ == "__main__":
    main()
