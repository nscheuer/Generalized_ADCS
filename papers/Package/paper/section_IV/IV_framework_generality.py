"""Monte Carlo framework-generality study using native GenADCS results."""

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
NUM_RUNS = 10
TF_S = 1000.0
DT_S = 2.0
BODY_BORESIGHT = np.array([0.0, 1.0, 0.0])
SATELLITES = ("1U (MTQ-only)", "3U (3+1)", "6U (3+3)", "Large (3+3+thrusters)")
COLORS = dict(zip(SATELLITES, ("tab:blue", "tab:orange", "tab:green", "tab:red")))
OUTPUT_NAMES = {
    SATELLITES[0]: "framework_generality_1u_mtq_only_mc_10",
    SATELLITES[1]: "framework_generality_3u_3mtq_1rw_mc_10",
    SATELLITES[2]: "framework_generality_6u_3mtq_3rw_mc_10",
    SATELLITES[3]: "framework_generality_large_3mtq_3rw_thrusters_mc_10",
}
PLOT_NAMES = {"error": "framework_generality_pointing_performance.png", "final": "framework_generality_final_error_distribution.png", "control": "framework_generality_control_effort.png"}
LARGE_CONTROLLER_GAINS = {"p_gain": 5.0e-2, "d_gain": 2.5, "c_gain": 1.0e-2}


def _small_mtqs() -> list[ADCS.MTQ]:
    return [ADCS.MTQ(axis, max_torque=0.2) for axis in np.eye(3)]


def _small_rws() -> list[ADCS.RW]:
    kwargs = {"max_torque": 0.23e-3, "J": 5.7e-6, "h": 0.0, "h_max": 3.5e-3}
    return [ADCS.RW(axis=axis, **kwargs) for axis in np.eye(3)]


def _make_satellite(name: str) -> ADCS.Satellite:
    sensors = [ADCS.MTM(axis) for axis in np.eye(3)]
    if name == SATELLITES[0]:
        return ADCS.Satellite(mass=1.33, J_0=np.diag([0.002, 0.002, 0.002]), actuators=_small_mtqs(), sensors=sensors, boresight=BODY_BORESIGHT)
    if name == SATELLITES[1]:
        return ADCS.satellite_factory.create_beavercube2_cubesat()
    if name == SATELLITES[2]:
        return ADCS.Satellite(mass=8.0, J_0=np.diag([0.08, 0.06, 0.06]), actuators=_small_mtqs() + _small_rws(), sensors=sensors, boresight=BODY_BORESIGHT)
    if name == SATELLITES[3]:
        mtqs = [ADCS.MTQ(axis, max_torque=1.0) for axis in np.eye(3)]
        rws = [ADCS.RW(axis=axis, max_torque=0.01, J=1.0e-3, h=0.0, h_max=1.0) for axis in np.eye(3)]
        return ADCS.Satellite(mass=500.0, J_0=np.diag([50.0, 150.0, 150.0]), actuators=mtqs + rws, sensors=sensors, boresight=BODY_BORESIGHT)
    raise ValueError(f"Unknown satellite: {name}")


def _mc_config(n_rw: int) -> MCConfig:
    return MCConfig(w=lambda rng: normalize(rng.standard_normal(3)) * (rng.uniform(0.1, 1.0) * np.pi / 180.0), q=lambda rng: normalize(rng.standard_normal(4)), h=lambda rng: rng.uniform(-0.0001, 0.0001, size=n_rw), goal=lambda rng: ADCS.goals.ECI_Goal(normalize(rng.standard_normal(3))))


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
    fig_error, ax_error = plt.subplots(figsize=(10, 5)); fig_final, ax_final = plt.subplots(figsize=(8, 5)); fig_control, ax_control = plt.subplots(figsize=(10, 5))
    for name, results in all_results.items():
        time, errors = _pointing_errors(results)
        if not errors.size: continue
        color = COLORS[name]
        ax_error.fill_between(time, np.percentile(errors, 10, axis=0), np.percentile(errors, 90, axis=0), color=color, alpha=0.18)
        ax_error.plot(time, np.mean(errors, axis=0), color=color, label=name)
        final = errors[:, -1]
        ax_final.hist(final, bins=np.linspace(0, max(5.0, np.max(final)), 25), alpha=0.4, color=color, label=f"{name} (mean {np.mean(final):.2f}°, <1° {100 * np.mean(final < 1):.0f}%)")
        control_norm = [np.linalg.norm(np.vstack(run.control_hist), axis=1) for run in results]
        ax_control.plot(time, np.mean(control_norm, axis=0), color=color, label=name)
    for axis, title, xlabel, ylabel in ((ax_error, "Framework generality: pointing performance", "Time [s]", "Pointing error [deg]"), (ax_final, "Framework generality: final pointing error", "Final pointing error [deg]", "Number of trials"), (ax_control, "Framework generality: mean control effort", "Time [s]", "||u||")):
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel); axis.grid(True, alpha=0.3); axis.legend()
    for figure, key in ((fig_error, "error"), (fig_final, "final"), (fig_control, "control")):
        figure.tight_layout(); path = OUTPUT_DIR / PLOT_NAMES[key]; figure.savefig(path, dpi=300, bbox_inches="tight"); print(f"Saved plot to {path}")
    create_close_all_button_window()


def main() -> dict[str, ADCS.SimulationResults]:
    all_results: dict[str, ADCS.SimulationResults] = {}
    os0 = ADCS.Orbital_State(ephem=ADCS.Ephemeris(), J2000=0.22, R=7000.0 * np.array([0.0, np.sqrt(2) / 2, np.sqrt(2) / 2]), V=np.array([8.0, 0.0, 0.0]))
    for name in SATELLITES:
        satellite = _make_satellite(name)
        state = ADCS.State(w=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]), h=np.zeros(satellite.number_RW))
        if name == SATELLITES[0]:
            controller = ADCS.controller.MTQ_Lovera(est_sat=satellite, p_gain=1e-3, d_gain=5e-3, eps=1.0)
        elif name == SATELLITES[3]:
            controller = ADCS.controller.MTQ_w_RW_LP(est_sat=satellite, **LARGE_CONTROLLER_GAINS, h_target=np.zeros(3))
        else:
            controller = ADCS.controller.MTQ_w_RW_LP(est_sat=satellite, p_gain=5e-5, d_gain=2e-3, c_gain=1e-3, h_target=np.zeros(3))
        results = ADCS.simulate_mc(x=state, satellite=satellite, controller=controller, goal=ADCS.goals.ECI_Goal(np.array([1.0, 0.0, 0.0])), os0=os0, dt=DT_S, tf=TF_S, mc_config=_mc_config(satellite.number_RW), num_runs=NUM_RUNS, max_workers=4, base_seed=1000)
        all_results[name] = results
        path = results.save(OUTPUT_NAMES[name], out_dir=OUTPUT_DIR)
        print(f"Saved {name} results to {path}")
    plot_comparisons(all_results)
    return all_results


if __name__ == "__main__":
    main()
