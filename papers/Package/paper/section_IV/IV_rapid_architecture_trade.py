"""Monte Carlo actuator-architecture trade study using native GenADCS results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import ADCS
from ADCS.helpers.math_helpers import normalize
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window
from ADCS.helpers.plotting_mc.plot_controller_mc import _rot_mat_vec
from ADCS.mc.mcconfig import MCConfig

OUTPUT_DIR = Path(__file__).with_name("outputs")
NUM_RUNS = 100
TF_S = 1000.0
DT_S = 2.0
BODY_BORESIGHT = np.array([0.0, 1.0, 0.0])
ARCHITECTURES = ("3+0 (MTQ-only)", "3+1 (MTQ + 1 RW)", "3+3 (MTQ + 3 RW)")
COLORS = dict(zip(ARCHITECTURES, ("tab:blue", "tab:orange", "tab:green")))
OUTPUT_NAMES = {
    ARCHITECTURES[0]: "rapid_architecture_trade_3mtq_0rw_mc_10",
    ARCHITECTURES[1]: "rapid_architecture_trade_3mtq_1rw_mc_10",
    ARCHITECTURES[2]: "rapid_architecture_trade_3mtq_3rw_mc_10",
}
PLOT_NAMES = {
    "error": "rapid_architecture_trade_pointing_performance.png",
    "final": "rapid_architecture_trade_final_error_distribution.png",
    "control": "rapid_architecture_trade_control_effort.png",
}


def _make_satellite(architecture: str) -> ADCS.Satellite:
    mtqs = [ADCS.MTQ(axis, max_torque=0.4) for axis in np.eye(3)]
    wheel_kwargs = {"max_torque": 0.23e-3, "J": 5.7e-6, "h": 0.0, "h_max": 3.5e-3}
    if architecture == ARCHITECTURES[0]:
        actuators = mtqs
    elif architecture == ARCHITECTURES[1]:
        actuators = mtqs + [ADCS.RW(axis=np.array([0.0, 0.0, 1.0]), **wheel_kwargs)]
    elif architecture == ARCHITECTURES[2]:
        actuators = mtqs + [ADCS.RW(axis=axis, **wheel_kwargs) for axis in np.eye(3)]
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    return ADCS.Satellite(
        mass=4.0, J_0=np.diag([0.03, 0.03, 0.01]), actuators=actuators,
        sensors=[ADCS.MTM(axis) for axis in np.eye(3)], boresight=BODY_BORESIGHT,
    )


def _mc_config(n_rw: int) -> MCConfig:
    return MCConfig(
        w=lambda rng: normalize(rng.standard_normal(3)) * (rng.uniform(0.1, 1.0) * np.pi / 180.0),
        q=lambda rng: normalize(rng.standard_normal(4)),
        h=lambda rng: rng.uniform(-0.0001, 0.0001, size=n_rw),
        goal=lambda rng: ADCS.goals.ECI_Goal(normalize(rng.standard_normal(3))),
    )


def _pointing_errors(results: ADCS.SimulationResults) -> tuple[np.ndarray, np.ndarray]:
    errors = []
    time = None
    for run in results:
        state = ADCS.State.stack(run.state_hist)
        goal = np.asarray(run.target_hist)
        goal = goal[:, 1:4] if goal.shape[1] == 4 else goal
        n = min(len(state), len(goal), len(run.time_s))
        bore = np.einsum("nij,j->ni", _rot_mat_vec(state[:n, 3:7]), BODY_BORESIGHT)
        bore /= np.linalg.norm(bore, axis=1, keepdims=True)
        goal = goal[:n] / np.linalg.norm(goal[:n], axis=1, keepdims=True)
        errors.append(np.rad2deg(np.arccos(np.clip(np.sum(bore * goal, axis=1), -1, 1))))
        if time is None:
            time = np.asarray(run.time_s)[:n]
    return (time, np.vstack(errors)) if errors else (np.empty(0), np.empty((0, 0)))


def plot_comparisons(all_results: dict[str, ADCS.SimulationResults]) -> None:
    fig_error, ax_error = plt.subplots(figsize=(10, 5))
    fig_final, ax_final = plt.subplots(figsize=(8, 5))
    fig_control, ax_control = plt.subplots(figsize=(10, 5))
    for architecture, results in all_results.items():
        color = COLORS[architecture]
        time, errors = _pointing_errors(results)
        if not errors.size:
            continue
        ax_error.fill_between(time, np.percentile(errors, 10, axis=0), np.percentile(errors, 90, axis=0), color=color, alpha=0.18)
        ax_error.plot(time, np.mean(errors, axis=0), color=color, label=architecture)
        final = errors[:, -1]
        ax_final.hist(final, bins=np.linspace(0, max(5.0, np.max(final)), 25), alpha=0.45, color=color,
                      label=f"{architecture} (mean {np.mean(final):.2f}°, <1° {100 * np.mean(final < 1):.0f}%)")
        control_norm = [np.linalg.norm(np.vstack(run.control_hist), axis=1) for run in results]
        ax_control.plot(np.asarray(results.first().time_s), np.mean(control_norm, axis=0), color=color, label=architecture)
    for axis, title, xlabel, ylabel in (
        (ax_error, "Monte Carlo pointing performance", "Time [s]", "Pointing error [deg]"),
        (ax_final, "Final pointing-error distribution", "Final pointing error [deg]", "Number of trials"),
        (ax_control, "Mean control effort", "Time [s]", "||u||"),
    ):
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend()
    for figure, key in ((fig_error, "error"), (fig_final, "final"), (fig_control, "control")):
        figure.tight_layout()
        path = OUTPUT_DIR / PLOT_NAMES[key]
        figure.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {path}")
    create_close_all_button_window()


def main() -> dict[str, ADCS.SimulationResults]:
    all_results: dict[str, ADCS.SimulationResults] = {}
    os0 = ADCS.Orbital_State(ephem=ADCS.Ephemeris(), J2000=0.22,
                             R=7000.0 * np.array([0.0, np.sqrt(2) / 2, np.sqrt(2) / 2]), V=np.array([8.0, 0.0, 0.0]))
    for architecture in ARCHITECTURES:
        satellite = _make_satellite(architecture)
        state = ADCS.State(w=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]), h=np.zeros(satellite.number_RW))
        controller = (
            ADCS.controller.MTQ_Lovera(est_sat=satellite, p_gain=1e-3, d_gain=5e-3, eps=1.0)
            if architecture == ARCHITECTURES[0]
            else ADCS.controller.MTQ_w_RW_LP(est_sat=satellite, p_gain=5e-5, d_gain=2e-3, c_gain=1e-3, h_target=np.zeros(3))
        )
        results = ADCS.simulate_mc(x=state, satellite=satellite, controller=controller,
            goal=ADCS.goals.ECI_Goal(np.array([1.0, 0.0, 0.0])), os0=os0, dt=DT_S, tf=TF_S,
            mc_config=_mc_config(satellite.number_RW), num_runs=NUM_RUNS, max_workers=4, base_seed=1000)
        all_results[architecture] = results
        path = results.save(OUTPUT_NAMES[architecture], out_dir=OUTPUT_DIR)
        print(f"Saved {architecture} results to {path}")
    plot_comparisons(all_results)
    return all_results


if __name__ == "__main__":
    main()
