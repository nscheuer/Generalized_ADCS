"""
Monte Carlo for 3MTQ + 0RW with LP (PD) Controller - Reduced Attitude 180° Slew.

This script generates Monte Carlo simulations for:
- MTQ-only configuration (3 MTQs, no reaction wheels)
- LP-based PD controller (Lovera-style allocation)
- Reduced attitude goal (ECI vector pointing)
- 180° slew from initial boresight direction

Uses fast orbit propagation with batch B/S computation for efficiency.
"""
import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Any, Tuple, Optional

# --- Path Setup ---
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

# --- ADCS Imports ---
from ADCS.CONOPS.goals import ECI_Goal
from ADCS.controller import MTQ_w_RW_LP
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import MTQ
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize, rot_mat
from ADCS.helpers.save_and_load.save_and_load import save_data, load_data
from ADCS.helpers.plotting_mc.plot_controller_mc import plot_target_tracking_mc, plot_convergence_histogram_mc
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window

# --- MC Runner Imports ---
from ADCS.helpers.mc.monte_carlo_runner import (
    MonteCarloRunner, 
    claim_worker_slot, 
    release_worker_slot, 
    update_worker_progress
)

# --- GLOBAL WORKER CACHE ---
_CACHED_ORBIT: Optional[Orbit] = None
_CACHED_ORBIT_KEY: Optional[Tuple] = None


def run_single_sim(config: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function for single MC simulation."""
    global _CACHED_ORBIT, _CACHED_ORBIT_KEY

    slot_id = claim_worker_slot()
    run_id = config["run_id"]

    try:
        tf = config.get("tf", 2000)
        dt = config.get("dt", 2)
        t0 = 0
        N = int((tf - t0) / dt)

        # --- Orbit Retrieval (Cached with fast propagation + batch B/S) ---
        radius_km = config.get("radius_km", 7000.0)
        orbit_key = (slot_id, radius_km, tf, dt)
        
        if _CACHED_ORBIT is None or _CACHED_ORBIT_KEY != orbit_key:
            rng_state = np.random.get_state()
            try:
                np.random.seed(100_000 + int(slot_id))
                _CACHED_ORBIT = create_random_circular_orbit(
                    radius_km=radius_km,
                    dt=dt,
                    tf=tf,
                    use_J2=True,
                    fast=True,  # Fast propagation
                )
                # Batch compute B and S vectors
                _CACHED_ORBIT.populate_environment(compute_B=True, compute_S=True)
                _CACHED_ORBIT_KEY = orbit_key
            finally:
                np.random.set_state(rng_state)
        
        orb = _CACHED_ORBIT

        # Setup Randomness (per run)
        np.random.seed(config["seed"])

        # Hardware Setup - MTQ only
        mtq_max = 0.4
        acts = [MTQ(axis=j, max_torque=mtq_max) for j in MathConstants.unitvecs]
        mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
        
        real_sat = Satellite(
            mass=1.2, 
            J_0=np.diagflat([0.022, 0.022, 0.004]), 
            actuators=acts, 
            sensors=mtms, 
            boresight=np.array([0, 0, 1])
        )

        # Initial Conditions
        x = np.concatenate([config["w0"], config["q0"]])

        # Controller - LP (PD-based)
        controller = MTQ_w_RW_LP(
            est_sat=real_sat, 
            p_gain=0.00005, 
            d_gain=0.001, 
            c_gain=0.001, 
            h_target=np.array([0.0, 0.0, 0.0])
        )
        
        # Reduced attitude goal: ECI vector for 180° slew
        R_b2i = rot_mat(config["q0"])
        initial_boresight_eci = R_b2i @ np.array([0, 0, 1])
        goal_eci_vec = -initial_boresight_eci  # Opposite direction = 180° slew
        goal = ECI_Goal(goal_eci_vec)

        # Arrays
        time_hist = np.zeros(N)
        state_hist = np.zeros((N, len(x)))
        u_hist = np.zeros((N, len(acts)))
        boresight_hist = np.zeros((N, 3))

        # Simulation loop
        t = t0
        orb_get_os = orb.get_os
        sat_sensor_readings = real_sat.sensor_readings
        ctrl_find_u = controller.find_u
        goal_to_ref = goal.to_ref
        sat_dynamics = real_sat.dynamics_for_solver
        sec2cent = TimeConstants.sec2cent

        for i in range(N):
            if i % 10 == 0:
                update_worker_progress(slot_id, run_id, i, N)

            J2000 = 0.22 + t * sec2cent
            os_state = orb_get_os(J2000=J2000)
            sens = sat_sensor_readings(x=x, os=os_state)
            u = ctrl_find_u(x_hat=x, sens=sens, est_sat=real_sat, os_hat=os_state, goal=goal)

            time_hist[i] = t
            state_hist[i, :] = x
            u_hist[i, :] = u
            eci_goal_ref, _ = goal_to_ref(os0=os_state)
            boresight_hist[i, :] = eci_goal_ref

            t += dt
            prev_os = os_state
            os_next = orb_get_os(0.22 + t * sec2cent)
            
            out = solve_ivp(
                fun=sat_dynamics, t_span=(0, dt), y0=x, method="RK45", 
                args=(u, prev_os, os_next), rtol=1e-6, atol=1e-6
            )
            x = out.y[:, -1]
            x[3:7] = normalize(x[3:7])

        update_worker_progress(slot_id, run_id, N, N)

        return {
            "run_id": config["run_id"],
            "config": config,
            "time": time_hist,
            "state": state_hist,
            "u": u_hist,
            "boresight_goal": boresight_hist
        }

    finally:
        release_worker_slot(slot_id)


def generate_mc_config(run_id: int) -> Dict[str, Any]:
    """Generate config for 180° slew MC run."""
    rng = np.random.default_rng(seed=run_id)
    
    q0 = normalize(rng.standard_normal(4))
    
    return {
        "run_id": run_id,
        "seed": run_id,
        "tf": 2000,  # Longer time for 180° slew with MTQ-only
        "dt": 2,
        "radius_km": 7000.0,
        "w0": normalize(rng.standard_normal(3)) * (rng.uniform(0.1, 1.0) * np.pi / 180.0),
        "q0": q0,
    }


if __name__ == "__main__":
    RUN_MC: bool = True
    OUTPUT_DIR = "papers/Planner/output_data"
    NUM_RUNS = 100

    if RUN_MC:
        runner = MonteCarloRunner(
            sim_func=run_single_sim,
            config_generator=generate_mc_config,
            num_runs=NUM_RUNS
        )
        full_results = runner.run()
        
        print(f"\n--- Monte Carlo Complete: {len(full_results)} runs ---")
        save_data(f"3MTQ+0RW_LP_reduced_180slew_mc_{NUM_RUNS}", full_results, out_dir=OUTPUT_DIR)
        
        plot_target_tracking_mc(full_results=full_results, title=f"3MTQ+0RW LP Reduced 180° Slew N={NUM_RUNS}")
        plot_convergence_histogram_mc(full_results=full_results, title=f"3MTQ+0RW LP Reduced 180° Slew N={NUM_RUNS}")
        create_close_all_button_window()
    else:
        results = load_data(f"{OUTPUT_DIR}/3MTQ+0RW_LP_reduced_180slew_mc_{NUM_RUNS}")
        full_results = results[0] if isinstance(results, tuple) else results
        plot_target_tracking_mc(full_results=full_results, title=f"3MTQ+0RW LP Reduced 180° Slew")
        plot_convergence_histogram_mc(full_results=full_results, title=f"3MTQ+0RW LP Reduced 180° Slew")
        create_close_all_button_window()
