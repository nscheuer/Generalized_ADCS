"""Representative disturbed 6U controller-only mode-switching simulation."""

from __future__ import annotations

import argparse
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
CAMPAIGN_NAME = "6u_mode_switching_disturbance_representative"
REPRESENTATIVE_DURATION_MIN = 1500.0
DT_S = 5.0
DESAT_ENTRY = 0.75
DESAT_EXIT = 0.25


class HystereticModeController:
    """Route the LP controller between pointing and No_Goal desaturation."""

    def __init__(self, controller, *, momentum_max: float) -> None:
        self.controller = controller
        self.momentum_max = float(momentum_max)
        self.mode = "pointing"
        self.mode_history: list[str] = []

    def find_u(self, *, x_hat, sens, est_sat, os_hat, goal):
        momentum_fraction = abs(float(np.asarray(x_hat.h).reshape(-1)[0])) / self.momentum_max
        if self.mode == "pointing" and momentum_fraction >= DESAT_ENTRY:
            self.mode = "desaturation"
        elif self.mode == "desaturation" and momentum_fraction <= DESAT_EXIT:
            self.mode = "pointing"

        self.mode_history.append(self.mode)
        active_goal = No_Goal() if self.mode == "desaturation" else goal
        return self.controller.find_u(
            x_hat=x_hat, sens=sens, est_sat=est_sat, os_hat=os_hat, goal=active_goal
        )


def _angle_error_deg(results: ADCS.SimulationResults) -> np.ndarray:
    run = results.runs[0]
    states = np.asarray([state.q for state in run.state_hist], dtype=float)
    targets = np.asarray(run.target_hist, dtype=float)
    dots = np.clip(np.abs(np.sum(states * targets, axis=1)), 0.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(dots))


def _spans(time: np.ndarray, condition: np.ndarray) -> list[tuple[float, float]]:
    """Convert a boolean sample mask into continuous plotting spans."""
    condition = np.asarray(condition, dtype=bool)
    changes = np.diff(np.r_[False, condition, False].astype(int))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [(float(time[start]), float(time[end])) for start, end in zip(starts, ends)]


def _plot(results: ADCS.SimulationResults, mode_controller: HystereticModeController) -> Path:
    configure_ieee_style()
    run = results.runs[0]
    time = np.asarray(run.time_s, dtype=float)
    angle = _angle_error_deg(results)
    h_fraction = np.abs(np.asarray([state.h[0] for state in run.state_hist])) / WHEEL_MOMENTUM_MAX
    modes = np.asarray(mode_controller.mode_history)
    desaturating = modes == "desaturation"

    fig, (ax_angle, ax_h, ax_mode) = plt.subplots(
        3, 1, figsize=(6.2, 5.0), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.4, 0.55]},
    )

    for start, end in _spans(time, angle < 5.0):
        ax_angle.axvspan(start / 60.0, end / 60.0, color="#66A061", alpha=0.20, lw=0)
    for start, end in _spans(time, desaturating):
        for axis in (ax_h, ax_mode):
            axis.axvspan(start / 60.0, end / 60.0, color="#E69F00", alpha=0.16, lw=0)

    ax_angle.plot(time / 60.0, angle, color="#0072B2", lw=0.9, label="Pointing error")
    ax_angle.axhline(5.0, color="#555555", ls="--", lw=0.7, label="5° requirement")
    ax_angle.set_ylabel("Angle error [deg]")
    ax_angle.set_title("6U disturbed mode switching: pointing and wheel desaturation")
    ax_angle.set_ylim(bottom=0.0)
    ax_angle.legend(loc="upper right", ncol=2)
    ax_angle.grid(True)

    ax_h.plot(time / 60.0, h_fraction, color="#D55E00", lw=0.9, label=r"$|h|/h_{max}$")
    ax_h.axhline(DESAT_ENTRY, color="#8C2D04", ls="--", lw=0.7, label="Desat. entry: 75%")
    ax_h.axhline(DESAT_EXIT, color="#8C2D04", ls=":", lw=0.7, label="Pointing restart: 25%")
    ax_h.set_ylabel(r"$|h|/h_{max}$")
    ax_h.set_ylim(0.0, 1.0)
    ax_h.legend(loc="upper right", ncol=3)
    ax_h.grid(True)

    mode_values = (modes == "desaturation").astype(float)
    ax_mode.step(time / 60.0, mode_values, where="post", color="#7B3294", lw=1.0)
    ax_mode.set_yticks([0.0, 1.0], ["Pointing", "Desaturation"])
    ax_mode.set_xlabel("Time [min]")
    ax_mode.set_ylim(-0.2, 1.2)
    ax_mode.grid(True, axis="x")

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{CAMPAIGN_NAME}.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tf", type=float, default=REPRESENTATIVE_DURATION_MIN * 60.0,
        help=f"Simulation duration in seconds (default: {REPRESENTATIVE_DURATION_MIN:g} min).",
    )
    parser.add_argument("--seed", type=int, default=20260911, help="Random scenario seed.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    satellite = make_satellite(1, estimated=False, disturbances=True)
    # The controller receives the true state directly; no estimator is used.
    est_satellite = make_satellite(1, estimated=False, disturbances=True)
    # Start with a modest, repeatable pointing error so the plot includes a
    # normal pointing transient before the wheel reaches the desaturation limit.
    q_initial = normalize(np.array([0.98480775, 0.17364818, 0.0, 0.0]))
    state = ADCS.State(
        w=np.zeros(3), q=q_initial,
        h=np.array([0.70 * WHEEL_MOMENTUM_MAX]),
    )
    goal = ADCS.goals.Fixed_Attitude_Goal(np.array([1.0, 0.0, 0.0, 0.0]))
    base_controller = controller_for(
        "lp", est_satellite, 1, kp=KP, kd=KD, c_gain=MOMENTUM_GAIN
    )
    mode_controller = HystereticModeController(base_controller, momentum_max=WHEEL_MOMENTUM_MAX)

    results = ADCS.simulate(
        x=state,
        satellite=satellite,
        est_satellite=est_satellite,
        controller=mode_controller,
        goal=goal,
        os0=random_400km_97deg_orbit(rng),
        dt=DT_S,
        tf=args.tf,
    )
    sim_path = results.save(CAMPAIGN_NAME, out_dir=OUTPUT_DIR)
    figure_path = _plot(results, mode_controller)
    print(f"Saved simulation: {sim_path}")
    print(f"Saved figure: {figure_path}")
    print(f"Desaturation samples: {sum(mode == 'desaturation' for mode in mode_controller.mode_history)} / {len(mode_controller.mode_history)}")


if __name__ == "__main__":
    main()
