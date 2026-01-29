"""
Monte Carlo: 3MTQ+1RW ALTRO+TVLQR Planner - Full Attitude (180° Quaternion Slew).

Uses BC2 satellite configuration with trajectory planner.
Same ICs as LP full test for fair comparison - 180° rotation about axis perpendicular to boresight.
"""
import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Any

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.CONOPS.goals import Fixed_Attitude_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.helpers import PlannerSettings

# Import good settings
from mc_planner_settings import create_good_planner_settings
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.helpers.math_helpers import normalize, quat_mult
from ADCS.helpers.save_and_load.save_and_load import save_data, load_data
from ADCS.helpers.plotting_mc.plot_controller_mc import plot_target_tracking_mc, plot_convergence_histogram_mc
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window
from ADCS.helpers.mc.monte_carlo_runner import (
    MonteCarloRunner, claim_worker_slot, release_worker_slot, update_worker_progress
)

BODY_BORESIGHT = np.array([0, 1, 0])

_CACHED_ORBIT = None
_CACHED_ORBIT_KEY = None


def run_single_sim(config: Dict[str, Any]) -> Dict[str, Any]:
    global _CACHED_ORBIT, _CACHED_ORBIT_KEY

    slot_id = claim_worker_slot()
    run_id = config["run_id"]

    try:
        tf = config.get("tf", 1000)
        dt = config.get("dt", 2)
        dt_planning = config.get("dt_planning", 1)
        N = int(tf / dt)

        radius_km = config.get("radius_km", 7000.0)
        orbit_key = (slot_id, radius_km, tf, dt_planning)

        if _CACHED_ORBIT is None or _CACHED_ORBIT_KEY != orbit_key:
            rng_state = np.random.get_state()
            try:
                np.random.seed(100_000 + int(slot_id))
                _CACHED_ORBIT = create_random_circular_orbit(
                    radius_km=radius_km, dt=dt_planning, tf=tf, use_J2=True, fast=True
                )
                _CACHED_ORBIT_KEY = orbit_key
            finally:
                np.random.set_state(rng_state)

        orb = _CACHED_ORBIT
        np.random.seed(config["seed"])

        real_sat = create_beavercube2_cubesat(estimated=False)
        rws = real_sat.rw_actuators

        x0 = np.concatenate([config["w0"], config["q0"], config["h0"]])
        for i, rw in enumerate(rws):
            rw.h = config["h0"][i]

        # Use well-conditioned normalized settings
        planner_settings = create_good_planner_settings(real_sat, dt_planning=dt_planning, has_rw=True)

        controller = Plan_and_Track_LQR(est_sat=real_sat, planner_settings=planner_settings)

        goals = GoalList({0.22: Fixed_Attitude_Goal(config["q_goal"])})
        os0 = orb.get_os(0.22)

        try:
            traj = controller.calculate_trajectory(
                t_start=0.22, duration=tf, x_0=x0, os_0=os0, goals=goals, verbose=False
            )
            controller.set_active_trajectory(traj)
            traj_valid = True
        except Exception as e:
            return {"run_id": run_id, "config": config, "error": str(e), "traj_valid": False}

        time_hist = np.zeros(N)
        state_hist = np.zeros((N, len(x0)))
        u_hist = np.zeros((N, len(real_sat.actuators)))
        q_goal_hist = np.zeros((N, 4))

        for i, rw in enumerate(rws):
            rw.h = config["h0"][i]

        x = x0.copy()
        t = 0
        sec2cent = TimeConstants.sec2cent

        for i in range(N):
            if i % 10 == 0:
                update_worker_progress(slot_id, run_id, i, N)

            J2000 = 0.22 + t * sec2cent
            os_state = orb.get_os(J2000=J2000)
            sens = real_sat.sensor_readings(x=x, os=os_state)
            u = controller.find_u(x_hat=x, sens=sens, est_sat=real_sat, os_hat=os_state)

            time_hist[i] = t
            state_hist[i, :] = x
            u_hist[i, :] = u
            q_goal_hist[i, :] = config["q_goal"]

            t += dt
            os_next = orb.get_os(0.22 + t * sec2cent)
            out = solve_ivp(
                real_sat.dynamics_for_solver, (0, dt), x, method="RK45",
                args=(u, os_state, os_next), rtol=1e-6, atol=1e-6
            )
            x = out.y[:, -1]
            x[3:7] = normalize(x[3:7])

        update_worker_progress(slot_id, run_id, N, N)

        return {
            "run_id": run_id, "config": config, "traj_valid": True,
            "time": time_hist, "state": state_hist, "u": u_hist,
            "q_goal": q_goal_hist, "goal_type": "full_attitude"
        }
    finally:
        release_worker_slot(slot_id)


def generate_mc_config(run_id: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed=run_id + 1000)
    
    q0 = normalize(rng.standard_normal(4))
    
    # 180° rotation about axis perpendicular to boresight (in XZ plane)
    rand_angle = rng.uniform(0, 2 * np.pi)
    axis_body = np.array([np.cos(rand_angle), 0, np.sin(rand_angle)])
    q_180_body = np.array([0, axis_body[0], axis_body[1], axis_body[2]])
    q_goal = quat_mult(q0, q_180_body)
    q_goal = normalize(q_goal)
    
    return {
        "run_id": run_id,
        "seed": run_id,
        "tf": 1000,
        "dt": 2,
        "dt_planning": 1,
        "radius_km": 7000.0,
        "w0": normalize(rng.standard_normal(3)) * (rng.uniform(0.1, 1.0) * np.pi / 180.0),
        "q0": q0,
        "q_goal": q_goal,
        "h0": rng.uniform(-0.0001, 0.0001, size=1),
    }


if __name__ == "__main__":
    RUN_MC = True
    OUTPUT_DIR = "papers/Planner/output_data"
    NUM_RUNS = 5  # Quick test

    if RUN_MC:
        runner = MonteCarloRunner(
            sim_func=run_single_sim,
            config_generator=generate_mc_config,
            num_runs=NUM_RUNS,
            max_workers=4
        )
        full_results = runner.run()
        
        valid = [r for r in full_results if r and r.get("traj_valid", False)]
        print(f"\n--- Monte Carlo Complete: {len(valid)}/{len(full_results)} valid ---")
        save_data(f"3MTQ+1RW_plan_full180_mc_{NUM_RUNS}", full_results, out_dir=OUTPUT_DIR)
        create_close_all_button_window()
    else:
        results = load_data(f"{OUTPUT_DIR}/3MTQ+1RW_plan_full180_mc_{NUM_RUNS}")
        full_results = results[0] if isinstance(results, tuple) else results
        create_close_all_button_window()
