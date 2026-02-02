"""
Monte Carlo: 3MTQ+0RW Lovera Controller - Reduced Attitude (180° Boresight Slew).

Uses BC2 satellite configuration with Lovera (MTQ-only) controller.
Same setup as 3+1 reduced for fair comparison - goal is 180° boresight slew.
"""
import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Any, Tuple, Optional

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.CONOPS.goals import ECI_Goal
from ADCS.controller import MTQ_Lovera
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.helpers.math_helpers import normalize, rot_mat
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
        N = int(tf / dt)

        radius_km = config.get("radius_km", 7000.0)
        orbit_key = (slot_id, radius_km, tf, dt)

        if _CACHED_ORBIT is None or _CACHED_ORBIT_KEY != orbit_key:
            rng_state = np.random.get_state()
            try:
                np.random.seed(100_000 + int(slot_id))
                _CACHED_ORBIT = create_random_circular_orbit(
                    radius_km=radius_km, dt=dt, tf=tf, use_J2=True, fast=True
                )
                _CACHED_ORBIT.populate_environment(compute_B=True, compute_S=True)
                _CACHED_ORBIT_KEY = orbit_key
            finally:
                np.random.set_state(rng_state)

        orb = _CACHED_ORBIT
        np.random.seed(config["seed"])

        real_sat = create_beavercube2_cubesat(estimated=False)
        
        x = np.concatenate([config["w0"], config["q0"], config["h0"]])
        for i, rw in enumerate(real_sat.rw_actuators):
            rw.h = config["h0"][i]

        # Lovera controller (MTQ-only)
        controller = MTQ_Lovera(est_sat=real_sat, p_gain=0.001, d_gain=0.005, eps=1.0)
        goal = ECI_Goal(config["goal_eci_vec"])

        time_hist = np.zeros(N)
        state_hist = np.zeros((N, len(x)))
        u_hist = np.zeros((N, len(real_sat.actuators)))
        boresight_hist = np.zeros((N, 3))

        t = 0
        sec2cent = TimeConstants.sec2cent
        for i in range(N):
            if i % 10 == 0:
                update_worker_progress(slot_id, run_id, i, N)

            J2000 = 0.22 + t * sec2cent
            os_state = orb.get_os(J2000=J2000)
            sens = real_sat.sensor_readings(x=x, os=os_state)
            u = controller.find_u(x_hat=x, sens=sens, est_sat=real_sat, os_hat=os_state, goal=goal)

            time_hist[i] = t
            state_hist[i, :] = x
            u_hist[i, :] = u
            eci_goal_ref, _ = goal.to_ref(os0=os_state)
            boresight_hist[i, :] = eci_goal_ref

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
            "run_id": run_id, "config": config,
            "time": time_hist, "state": state_hist, "u": u_hist,
            "boresight_goal": boresight_hist
        }
    finally:
        release_worker_slot(slot_id)


def generate_mc_config(run_id: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed=run_id + 1000)
    
    # Random initial quaternion
    q0 = normalize(rng.standard_normal(4))
    
    # Compute initial boresight direction in ECI
    R0 = rot_mat(q0)
    initial_boresight_eci = R0 @ BODY_BORESIGHT
    
    # Goal is opposite direction (true 180° boresight slew)
    goal_eci_vec = -initial_boresight_eci
    
    return {
        "run_id": run_id,
        "seed": run_id,
        "tf": 1000,
        "dt": 2,
        "radius_km": 7000.0,
        "w0": normalize(rng.standard_normal(3)) * (rng.uniform(0.1, 1.0) * np.pi / 180.0),
        "q0": q0,
        "h0": rng.uniform(-0.0001, 0.0001, size=1),
        "goal_eci_vec": goal_eci_vec,
    }


if __name__ == "__main__":
    RUN_MC = True
    OUTPUT_DIR = "papers/Planner/output_data"
    NUM_RUNS = 100

    if RUN_MC:
        runner = MonteCarloRunner(
            sim_func=run_single_sim,
            config_generator=generate_mc_config,
            num_runs=NUM_RUNS,
            max_workers=12
        )
        full_results = runner.run()
        print(f"\n--- Monte Carlo Complete: {len(full_results)} runs ---")
        save_data(f"3MTQ+0RW_Lovera_reduced_mc_{NUM_RUNS}", full_results, out_dir=OUTPUT_DIR)
        plot_target_tracking_mc(full_results, body_boresight=BODY_BORESIGHT, title=f"3MTQ+0RW Lovera Reduced N={NUM_RUNS}")
        plot_convergence_histogram_mc(full_results, body_boresight=BODY_BORESIGHT, title=f"3MTQ+0RW Lovera Reduced")
        #create_close_all_button_window()  # Disabled for batch runs
    else:
        results = load_data(f"{OUTPUT_DIR}/3MTQ+0RW_Lovera_reduced_mc_{NUM_RUNS}")
        full_results = results[0] if isinstance(results, tuple) else results
        plot_target_tracking_mc(full_results, body_boresight=BODY_BORESIGHT, title=f"3MTQ+0RW Lovera Reduced")
        plot_convergence_histogram_mc(full_results, body_boresight=BODY_BORESIGHT, title=f"3MTQ+0RW Lovera Reduced")
        #create_close_all_button_window()  # Disabled for batch runs
