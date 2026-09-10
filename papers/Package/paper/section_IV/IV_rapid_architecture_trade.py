"""Monte Carlo actuator-architecture trade study.

Runs the same pointing campaign for 3+0, 3+1, and 3+3 actuator
architectures, saves the results below ``section_IV/outputs``, and plots them.
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
NUM_RUNS = 100
TF_S = 1000.0
DT_S = 2.0
# The archived architecture-trade data used the BeaverCube-2 body boresight.
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
    """Build one of the three actuator configurations."""
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
        mass=4.0,
        J_0=np.diag([0.03, 0.03, 0.01]),
        actuators=actuators,
        sensors=[ADCS.MTM(axis) for axis in np.eye(3)],
        boresight=BODY_BORESIGHT,
    )


def generate_mc_config(run_id: int, architecture: str) -> dict[str, Any]:
    """Create reproducible, paired initial conditions for all architectures."""
    rng = np.random.default_rng(seed=run_id + 1000)
    n_rw = {
        ARCHITECTURES[0]: 0,
        ARCHITECTURES[1]: 1,
        ARCHITECTURES[2]: 3,
    }[architecture]
    # Draw the initial attitude and target independently.  This gives each
    # trial a random initial and final pointing direction, rather than forcing
    # the target to be opposite the initial boresight.
    q0 = normalize(rng.standard_normal(4))
    goal_eci_vec = normalize(rng.standard_normal(3))

    return {
        "run_id": run_id,
        "seed": run_id,
        "architecture": architecture,
        "tf": TF_S,
        "dt": DT_S,
        "w0": normalize(rng.standard_normal(3)) * (rng.uniform(0.1, 1.0) * np.pi / 180.0),
        "q0": q0,
        "h0": rng.uniform(-0.0001, 0.0001, size=n_rw),
        "goal_eci_vec": goal_eci_vec,
        "orbit_R": 7000.0 * np.array([0.0, np.sqrt(2) / 2, np.sqrt(2) / 2]),
        "orbit_V": np.array([8.0, 0.0, 0.0]),
    }


def run_single_sim(config: dict[str, Any]) -> dict[str, Any]:
    """Run one case and return a save_data-compatible record."""
    slot_id = claim_worker_slot()
    try:
        satellite = _make_satellite(config["architecture"])
        state = ADCS.State(w=config["w0"], q=config["q0"], h=config["h0"])
        if config["architecture"] == ARCHITECTURES[0]:
            controller = ADCS.controller.MTQ_Lovera(
                est_sat=satellite, p_gain=1e-3, d_gain=5e-3, eps=1.0
            )
        else:
            controller = ADCS.controller.MTQ_w_RW_LP(
                est_sat=satellite, p_gain=5e-5, d_gain=2e-3, c_gain=1e-3,
                h_target=np.zeros(3),
            )
        goal = ADCS.goals.ECI_Goal(config["goal_eci_vec"])
        os0 = ADCS.Orbital_State(
            ephem=ADCS.Ephemeris(),
            J2000=0.22,
            R=config["orbit_R"],
            V=config["orbit_V"],
        )
        simulation = ADCS.simulate(
            x=state,
            satellite=satellite,
            controller=controller,
            goal=goal,
            os0=os0,
            dt=config["dt"],
            tf=config["tf"],
        )
        run = simulation.first()
        update_worker_progress(slot_id, config["run_id"], len(run.time_s), len(run.time_s))
        return {
            "run_id": config["run_id"],
            "config": config,
            "time": np.asarray(run.time_s),
            "state": ADCS.State.stack(run.state_hist),
            "u": np.vstack(run.control_hist),
            "boresight_goal": np.asarray(run.target_hist),
        }
    finally:
        release_worker_slot(slot_id)


def _pointing_errors(results: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Calculate instantaneous boresight-to-goal errors for all runs."""
    errors = []
    time = None
    body_boresight = BODY_BORESIGHT / np.linalg.norm(BODY_BORESIGHT)
    for result in results:
        state = np.asarray(result["state"])
        goal = np.asarray(result["boresight_goal"])
        goal = goal[:, 1:4] if goal.shape[1] == 4 else goal
        n = min(len(state), len(goal), len(result["time"]))
        bore = np.einsum("nij,j->ni", _rot_mat_vec(state[:n, 3:7]), body_boresight)
        bore /= np.linalg.norm(bore, axis=1, keepdims=True)
        goal = goal[:n] / np.linalg.norm(goal[:n], axis=1, keepdims=True)
        errors.append(np.rad2deg(np.arccos(np.clip(np.sum(bore * goal, axis=1), -1, 1))))
        if time is None:
            time = np.asarray(result["time"])[:n]
    return (time, np.vstack(errors)) if errors else (np.empty(0), np.empty((0, 0)))


def plot_comparisons(all_results: dict[str, list[dict[str, Any]]]) -> None:
    """Plot and save pointing, final-error, and control-effort comparisons."""
    fig_error, ax_error = plt.subplots(figsize=(10, 5))
    fig_final, ax_final = plt.subplots(figsize=(8, 5))
    fig_control, ax_control = plt.subplots(figsize=(10, 5))
    for architecture, results in all_results.items():
        color = COLORS[architecture]
        time, errors = _pointing_errors(results)
        if errors.size == 0:
            continue
        ax_error.fill_between(time, np.percentile(errors, 10, axis=0), np.percentile(errors, 90, axis=0), color=color, alpha=0.18)
        ax_error.plot(time, np.mean(errors, axis=0), color=color, label=architecture)
        final = errors[:, -1]
        ax_final.hist(final, bins=np.linspace(0, max(5.0, np.max(final)), 25), alpha=0.45, color=color,
                      label=f"{architecture} (mean {np.mean(final):.2f}°, <1° {100 * np.mean(final < 1):.0f}%)")
        control_norm = [np.linalg.norm(np.asarray(result["u"]), axis=1) for result in results]
        ax_control.plot(np.asarray(results[0]["time"]), np.mean(control_norm, axis=0), color=color, label=architecture)

    for axis, title, xlabel, ylabel in (
        (ax_error, "Monte Carlo pointing performance", "Time [s]", "Pointing error [deg]"),
        (ax_final, "Final pointing-error distribution", "Final pointing error [deg]", "Number of trials"),
        (ax_control, "Mean control effort", "Time [s]", "||u||"),
    ):
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig_error.tight_layout()
    fig_final.tight_layout()
    fig_control.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for figure, key in (
        (fig_error, "error"),
        (fig_final, "final"),
        (fig_control, "control"),
    ):
        path = OUTPUT_DIR / PLOT_NAMES[key]
        figure.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {path}")

    # This helper calls ``plt.show()`` internally.  It must be called after
    # creating the comparison figures; otherwise its blocking show prevents
    # the plots below it from ever being created.
    create_close_all_button_window()


def main() -> dict[str, list[dict[str, Any]]]:
    """Run, save, and plot all three architecture campaigns."""
    all_results = {}
    for architecture in ARCHITECTURES:
        runner = MonteCarloRunner(
            sim_func=run_single_sim,
            config_generator=partial(generate_mc_config, architecture=architecture),
            num_runs=NUM_RUNS,
            max_workers=4,
        )
        results = runner.run()
        failed = sum(result is None for result in results)
        if failed:
            raise RuntimeError(
                f"{architecture} campaign produced {failed} failed runs; "
                "no incomplete results will be saved or plotted."
            )
        all_results[architecture] = results

    for architecture, results in all_results.items():
        saved_path = save_data(
            OUTPUT_NAMES[architecture], results, out_dir=OUTPUT_DIR
        )
        print(f"Saved {architecture} results to {saved_path}")
    plot_comparisons(all_results)
    return all_results


if __name__ == "__main__":
    main()
