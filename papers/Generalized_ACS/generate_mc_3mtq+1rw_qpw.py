import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Any, Tuple, Optional

# --- Path Setup ---
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

# --- ADCS Imports ---
from ADCS.CONOPS.goals import ECI_Goal
from ADCS.controller.mtq_w_rw_QPW import MTQ_w_RW_QPW
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize
from ADCS.helpers.save_and_load.save_and_load import save_data, load_data
from ADCS.helpers.plotting_mc.plot_controller_mc import plot_target_tracking_mc
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window

# --- MC Runner Imports ---
from ADCS.mc.monte_carlo_runner import (
    MonteCarloRunner, 
    claim_worker_slot, 
    release_worker_slot, 
    update_worker_progress
)

# --- GLOBAL WORKER CACHE ---
_CACHED_ORBIT: Optional[Orbit] = None
_CACHED_ORBIT_KEY: Optional[Tuple] = None

def run_single_sim(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function. Checks global cache for orbit before running sim.
    Uses generic MonteCarloRunner helpers to update the terminal UI.
    """
    global _CACHED_ORBIT, _CACHED_ORBIT_KEY

    # 1. UI Setup: Claim a slot
    slot_id = claim_worker_slot()
    run_id = config["run_id"]

    try:
        # 2. Setup Randomness
        np.random.seed(config["seed"])
        tf = config.get("tf", 500)
        dt = config.get("dt", 2)
        t0 = 0
        N = int((tf - t0) / dt)

        # 3. Hardware Setup
        mtq_max = 0.4
        rw_max = 7e-3
        rw_hmax = 16.2e-3
        
        acts = [MTQ(axis=j, max_moment=mtq_max) for j in MathConstants.unitvecs]
        rws = [RW(axis=j, max_torque=rw_max, J=1e-3, h=0, h_max=rw_hmax) for j in MathConstants.unitvecs]
        rws.pop()
        rws.pop()
        acts.extend(rws)
        mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
        
        real_sat = Satellite(
            mass=1.2, 
            J_0=np.diagflat([0.022, 0.022, 0.004]), 
            actuators=acts, 
            sensors=mtms, 
            boresight=np.array([0, 0, 1])
        )

        # 4. Initial Conditions
        x = np.concatenate([config["w0"], config["q0"], config["h0"]])
        for i, rw in enumerate(rws):
            rw.h = config["h0"][i]

        # 5. Orbit Retrieval (Cached)
        orbit_key = (tuple(config["orbit_R"]), tuple(config["orbit_V"]), tf, dt)
        if _CACHED_ORBIT is None or _CACHED_ORBIT_KEY != orbit_key:
            ephem = Ephemeris()
            start_time = 0.22 - 1 * TimeConstants.sec2cent
            os0 = Orbital_State(ephem=ephem, J2000=start_time, R=config["orbit_R"], V=config["orbit_V"])
            _CACHED_ORBIT = Orbit(os0=os0, end_time=0.22 + tf * TimeConstants.sec2cent, dt=dt, use_J2=True, fast=False, verbose=False)
            _CACHED_ORBIT_KEY = orbit_key
        orb = _CACHED_ORBIT

        # 6. Controller
        controller = MTQ_w_RW_QPW(est_sat=real_sat, p_gain=0.00005, d_gain=0.001, c_gain=0.001, h_target=np.array([0.004, 0.0, 0.0]))
        goal = ECI_Goal(config["goal_eci_vec"])

        # 7. Arrays
        time_hist = np.zeros(N)
        state_hist = np.zeros((N, len(x)))
        u_hist = np.zeros((N, len(acts)))
        boresight_hist = np.zeros((N, 3))

        t = t0
        ind = 0
        steps = int((tf - t0) / dt)

        # 8. Loop
        orb_get_os = orb.get_os
        sat_sensor_readings = real_sat.sensor_readings
        ctrl_find_u = controller.find_u
        goal_to_ref = goal.to_ref
        sat_dynamics = real_sat.dynamics_for_solver
        sec2cent = TimeConstants.sec2cent

        for i in range(steps):
            # --- UI UPDATE ---
            if i % 10 == 0: # Update every 10 steps to keep IPC light
                update_worker_progress(slot_id, run_id, i, steps)

            # Physics
            J2000 = 0.22 + t * sec2cent
            os_state = orb_get_os(J2000=J2000)
            sens = sat_sensor_readings(x=x, os=os_state)
            u = ctrl_find_u(x_hat=x, sens=sens, est_sat=real_sat, os_hat=os_state, goal=goal)

            time_hist[ind] = t
            state_hist[ind, :] = x
            u_hist[ind, :] = u
            eci_goal_ref, _ = goal_to_ref(os0=os_state)
            boresight_hist[ind, :] = eci_goal_ref

            ind += 1
            t += dt
            prev_os = os_state
            os_next = orb_get_os(0.22 + (t - t0) * sec2cent)
            
            out = solve_ivp(
                fun=sat_dynamics, t_span=(0, dt), y0=x, method="RK45", 
                args=(u, prev_os, os_next), rtol=1e-6, atol=1e-6
            )
            x = out.y[:, -1]
            x[3:7] = normalize(x[3:7])

        # Final UI update
        update_worker_progress(slot_id, run_id, steps, steps)

        return {
            "run_id": config["run_id"],
            "config": config,
            "time": time_hist,
            "state": state_hist,
            "u": u_hist,
            "boresight_goal": boresight_hist
        }

    finally:
        # Important: Release the slot even if an error occurs
        release_worker_slot(slot_id)

# --- Config Generator ---

def generate_mc_config(run_id: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed=run_id)
    return {
        "run_id": run_id,
        "seed": run_id,
        "tf": 1000,
        "dt": 2,
        "w0": normalize(rng.standard_normal(3)) * (rng.uniform(0.1, 2.0) * np.pi / 180.0),
        "q0": normalize(rng.standard_normal(4)),
        "h0": rng.uniform(-0.005, 0.005, size=1),
        "goal_eci_vec": normalize(rng.standard_normal(3)),
        "orbit_R": 7000 * np.array([0, np.sqrt(2)/2, np.sqrt(2)/2]),
        "orbit_V": np.array([8, 0, 0])
    }

if __name__ == "__main__":
    RUN_MC: bool = True
    OUTPUT_DIR = "papers/3MTQ+1RW/output_data"

    if RUN_MC:
        runner = MonteCarloRunner(
            sim_func=run_single_sim,
            config_generator=generate_mc_config,
            num_runs=100
        )
        full_results = runner.run()
        
        print(f"\n--- Monte Carlo Complete: Generated {len(full_results)} histories ---")
        save_data("3MTQ+1RW_QPW_mc_100", full_results, out_dir=OUTPUT_DIR)
        
        plot_target_tracking_mc(full_results=full_results, title="3 MTQ + 1 RW QPW MC:100")
        create_close_all_button_window()
    else:
        results_qpw = load_data("papers/3MTQ+1RW/output_data/3MTQ+1RW_QPW_mc_100_20260110_161705")
        results_qpw = results_qpw[0]
        results_lp = load_data("papers/3MTQ+1RW/output_data/3MTQ+2RW_LP_mc_100_20260110_154413")
        results_lp = results_lp[0]
        plot_target_tracking_mc(results_qpw, title="3 MTQ + 1 RW QPW MC:100")
        plot_target_tracking_mc(results_qpw, title="3 MTQ + 2 RW LP MC:100")
        create_close_all_button_window()