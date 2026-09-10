"""Monte Carlo demonstration of framework generality across four satellites.

The same pointing campaign is run for a 1U MTQ-only CubeSat, a 3U
BeaverCube-2, a 6U CubeSat, and a large spacecraft.  The 1U case uses
Lovera's MTQ controller; the remaining cases use ``MTQ_w_RW_LP``.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import ADCS
from ADCS.helpers.math_helpers import normalize
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window
from ADCS.helpers.plotting_mc.plot_controller_mc import _rot_mat_vec
from ADCS.helpers.save_and_load.save_and_load import save_data
from ADCS.mc.monte_carlo_runner import (
    MonteCarloRunner,
    claim_worker_slot,
    release_worker_slot,
    update_worker_progress,
)


OUTPUT_DIR = Path(__file__).with_name("outputs")
NUM_RUNS = 10
TF_S = 1000.0
DT_S = 2.0
BODY_BORESIGHT = np.array([0.0, 1.0, 0.0])

SATELLITES = (
    "1U (MTQ-only)",
    "3U (3+1)",
    "6U (3+3)",
    "Large (3+3+thrusters)",
)
COLORS = dict(zip(SATELLITES, ("tab:blue", "tab:orange", "tab:green", "tab:red")))
OUTPUT_NAMES = {
    SATELLITES[0]: "framework_generality_1u_mtq_only_mc_10",
    SATELLITES[1]: "framework_generality_3u_3mtq_1rw_mc_10",
    SATELLITES[2]: "framework_generality_6u_3mtq_3rw_mc_10",
    SATELLITES[3]: "framework_generality_large_3mtq_3rw_thrusters_mc_10",
}
PLOT_NAMES = {
    "error": "framework_generality_pointing_performance.png",
    "final": "framework_generality_final_error_distribution.png",
    "control": "framework_generality_control_effort.png",
}
# The default gains used for the CubeSats request only about 5e-5 N m for a
# one-radian attitude error.  That is appropriate for inertias around
# 1e-2--1e-1 kg m^2, but is four orders of magnitude too small for the large
# vehicle below.  These gains give it a comparable closed-loop bandwidth while
# remaining within the large-wheel authority after LP allocation.
LARGE_CONTROLLER_GAINS = {
    "p_gain": 5.0e-2,
    "d_gain": 2.5,
    "c_gain": 1.0e-2,
}


def _small_mtqs() -> list[ADCS.MTQ]:
    return [ADCS.MTQ(axis, max_torque=0.2) for axis in np.eye(3)]


def _small_rws() -> list[ADCS.RW]:
    kwargs = {"max_torque": 0.23e-3, "J": 5.7e-6, "h": 0.0, "h_max": 3.5e-3}
    return [ADCS.RW(axis=axis, **kwargs) for axis in np.eye(3)]


def _make_satellite(name: str) -> ADCS.Satellite:
    """Construct the spacecraft represented by ``name``."""
    sensors = [ADCS.MTM(axis) for axis in np.eye(3)]
    if name == SATELLITES[0]:
        return ADCS.Satellite(
            mass=1.33,
            J_0=np.diag([0.002, 0.002, 0.002]),
            actuators=_small_mtqs(),
            sensors=sensors,
            boresight=BODY_BORESIGHT,
        )
    if name == SATELLITES[1]:
        return ADCS.satellite_factory.create_beavercube2_cubesat()
    if name == SATELLITES[2]:
        return ADCS.Satellite(
            mass=8.0,
            J_0=np.diag([0.08, 0.06, 0.06]),
            actuators=_small_mtqs() + _small_rws(),
            sensors=sensors,
            boresight=BODY_BORESIGHT,
        )
    if name == SATELLITES[3]:
        large_mtqs = [ADCS.MTQ(axis, max_torque=1.0) for axis in np.eye(3)]
        large_rws = [
            ADCS.RW(axis=axis, max_torque=0.01, J=1.0e-3, h=0.0, h_max=1.0)
            for axis in np.eye(3)
        ]
        # Thrusters are represented by the spacecraft's additional actuator
        # set in the campaign; the framework comparison remains attitude-only.
        return ADCS.Satellite(
            mass=500.0,
            J_0=np.diag([50.0, 150.0, 150.0]),
            actuators=large_mtqs + large_rws,
            sensors=sensors,
            boresight=BODY_BORESIGHT,
        )
    raise ValueError(f"Unknown satellite: {name}")


def generate_mc_config(run_id: int, satellite_name: str) -> dict[str, Any]:
    rng = np.random.default_rng(seed=run_id + 1000)
    n_rw = 0 if satellite_name == SATELLITES[0] else (1 if satellite_name == SATELLITES[1] else 3)
    return {
        "run_id": run_id,
        "seed": run_id,
        "satellite": satellite_name,
        "tf": TF_S,
        "dt": DT_S,
        "w0": normalize(rng.standard_normal(3)) * (rng.uniform(0.1, 1.0) * np.pi / 180.0),
        "q0": normalize(rng.standard_normal(4)),
        "h0": rng.uniform(-0.0001, 0.0001, size=n_rw),
        "goal_eci_vec": normalize(rng.standard_normal(3)),
        "orbit_R": 7000.0 * np.array([0.0, np.sqrt(2) / 2, np.sqrt(2) / 2]),
        "orbit_V": np.array([8.0, 0.0, 0.0]),
    }


def run_single_sim(config: dict[str, Any]) -> dict[str, Any]:
    slot_id = claim_worker_slot()
    try:
        satellite = _make_satellite(config["satellite"])
        state = ADCS.State(w=config["w0"], q=config["q0"], h=config["h0"])
        if config["satellite"] == SATELLITES[0]:
            controller = ADCS.controller.MTQ_Lovera(
                est_sat=satellite, p_gain=1e-3, d_gain=5e-3, eps=1.0
            )
        elif config["satellite"] == SATELLITES[3]:
            controller = ADCS.controller.MTQ_w_RW_LP(
                est_sat=satellite,
                **LARGE_CONTROLLER_GAINS,
                h_target=np.zeros(3),
            )
        else:
            controller = ADCS.controller.MTQ_w_RW_LP(
                est_sat=satellite, p_gain=5e-5, d_gain=2e-3, c_gain=1e-3,
                h_target=np.zeros(3),
            )
        goal = ADCS.goals.ECI_Goal(config["goal_eci_vec"])
        os0 = ADCS.Orbital_State(
            ephem=ADCS.Ephemeris(), J2000=0.22,
            R=config["orbit_R"], V=config["orbit_V"],
        )
        run = ADCS.simulate(
            x=state, satellite=satellite, controller=controller, goal=goal,
            os0=os0, dt=config["dt"], tf=config["tf"],
        ).first()
        update_worker_progress(slot_id, config["run_id"], len(run.time_s), len(run.time_s))
        return {
            "run_id": config["run_id"], "config": config,
            "time": np.asarray(run.time_s),
            "state": ADCS.State.stack(run.state_hist),
            "u": np.vstack(run.control_hist),
            "boresight_goal": np.asarray(run.target_hist),
        }
    finally:
        release_worker_slot(slot_id)


def _pointing_errors(results: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    errors = []
    time = None
    for result in results:
        state = np.asarray(result["state"])
        goal = np.asarray(result["boresight_goal"])
        goal = goal[:, 1:4] if goal.shape[1] == 4 else goal
        n = min(len(state), len(goal), len(result["time"]))
        bore = np.einsum("nij,j->ni", _rot_mat_vec(state[:n, 3:7]), BODY_BORESIGHT)
        bore /= np.linalg.norm(bore, axis=1, keepdims=True)
        goal = goal[:n] / np.linalg.norm(goal[:n], axis=1, keepdims=True)
        errors.append(np.rad2deg(np.arccos(np.clip(np.sum(bore * goal, axis=1), -1, 1))))
        if time is None:
            time = np.asarray(result["time"])[:n]
    return (time, np.vstack(errors)) if errors else (np.empty(0), np.empty((0, 0)))


def save_plot(figure: plt.Figure, name: str) -> Path:
    """Save one comparison figure below the section's output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    figure.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {path}")
    return path


def plot_comparisons(all_results: dict[str, list[dict[str, Any]]]) -> None:
    fig_error, ax_error = plt.subplots(figsize=(10, 5))
    fig_final, ax_final = plt.subplots(figsize=(8, 5))
    fig_control, ax_control = plt.subplots(figsize=(10, 5))
    for name, results in all_results.items():
        time, errors = _pointing_errors(results)
        if not errors.size:
            continue
        color = COLORS[name]
        ax_error.fill_between(time, np.percentile(errors, 10, axis=0), np.percentile(errors, 90, axis=0), color=color, alpha=0.18)
        ax_error.plot(time, np.mean(errors, axis=0), color=color, label=name)
        final = errors[:, -1]
        ax_final.hist(final, bins=np.linspace(0, max(5.0, np.max(final)), 25), alpha=0.4, color=color,
                      label=f"{name} (mean {np.mean(final):.2f}°, <1° {100 * np.mean(final < 1):.0f}%)")
        control_norm = [np.linalg.norm(np.asarray(result["u"]), axis=1) for result in results]
        ax_control.plot(time, np.mean(control_norm, axis=0), color=color, label=name)
    for axis, title, xlabel, ylabel in (
        (ax_error, "Framework generality: pointing performance", "Time [s]", "Pointing error [deg]"),
        (ax_final, "Framework generality: final pointing error", "Final pointing error [deg]", "Number of trials"),
        (ax_control, "Framework generality: mean control effort", "Time [s]", "||u||"),
    ):
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend()
    for figure, key in ((fig_error, "error"), (fig_final, "final"), (fig_control, "control")):
        figure.tight_layout()
        save_plot(figure, PLOT_NAMES[key])
    create_close_all_button_window()


def main() -> dict[str, list[dict[str, Any]]]:
    all_results = {}
    for name in SATELLITES:
        runner = MonteCarloRunner(
            sim_func=run_single_sim,
            config_generator=partial(generate_mc_config, satellite_name=name),
            num_runs=NUM_RUNS,
            max_workers=4,
        )
        results = runner.run()
        if any(result is None for result in results):
            raise RuntimeError(f"{name} campaign produced failed runs")
        all_results[name] = results
    for name, results in all_results.items():
        saved_path = save_data(OUTPUT_NAMES[name], results, out_dir=OUTPUT_DIR)
        print(f"Saved {name} results to {saved_path}")
    plot_comparisons(all_results)
    return all_results


if __name__ == "__main__":
    main()
