import sys
import os
import numpy as np
from ADCS.state import State
from scipy.integrate import solve_ivp
from typing import Dict, Any, Tuple, Optional

# --- Path Setup ---
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

# --- ADCS Imports ---
from ADCS.CONOPS.goals import ECI_Goal
from ADCS.controller import MTQ_Lovera
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants, EarthConstants
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize
from ADCS.helpers.save_and_load.save_and_load import save_data, load_data
from ADCS.helpers.plotting_mc.plot_controller_mc import plot_target_tracking_mc, plot_convergence_histogram_mc
from ADCS.helpers.plotting_mc.plot_controller_compare_mc import (
    plot_target_tracking_mc_compare,
    plot_convergence_histogram_mc_compare,
    plot_h_tracking_mc_compare,
)
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window

# --- MC Runner Imports ---
from ADCS.mc.monte_carlo_runner import (
    MonteCarloRunner,
    claim_worker_slot,
    release_worker_slot,
    update_worker_progress,
)

"""
Comparison between 3 MTQ + 1 RW LP control vs 3 MTQ Lovera control for ECI target tracking.
- Uses BC2 configuration
- Uses GG, SRP, and GG Disturbances
- Initial RW momentum randomized between +/- 0.0001 Nms
- Randomized circular orbit position per worker
"""

# --- GLOBAL WORKER CACHE ---
_CACHED_ORBIT: Optional[Orbit] = None
_CACHED_ORBIT_KEY: Optional[Tuple] = None

def run_single_sim(config: Dict[str, Any]) -> Dict[str, Any]:
    global _CACHED_ORBIT, _CACHED_ORBIT_KEY

    # 1. UI Setup: Claim a slot
    slot_id = claim_worker_slot()
    run_id = config["run_id"]

    try:
        tf = config.get("tf", 500)
        dt = config.get("dt", 2)
        t0 = 0
        N = int((tf - t0) / dt)

        # 2. Orbit Retrieval (Cached)  -> ONE RANDOM CIRCULAR ORBIT PER WORKER SLOT
        #    We create it before per-run seeding, and we isolate its RNG so it doesn't
        #    affect per-run randomness.
        radius_km = float(config.get("radius_km", 7000.0))
        orbit_key = (slot_id, radius_km, tf, dt, True, False)

        if _CACHED_ORBIT is None or _CACHED_ORBIT_KEY != orbit_key:
            rng_state = np.random.get_state()
            try:
                np.random.seed(100_000 + int(slot_id))  # deterministic per worker/core
                _CACHED_ORBIT = create_random_circular_orbit(
                    radius_km=radius_km,
                    dt=dt,
                    tf=tf,
                    use_J2=True,
                    fast=False,
                )
                _CACHED_ORBIT_KEY = orbit_key
            finally:
                np.random.set_state(rng_state)

        orb = _CACHED_ORBIT

        # 3. Setup Randomness (per run)
        np.random.seed(config["seed"])

        real_sat = create_beavercube2_cubesat(estimated=False)

        # 4. Initial Conditions
        x = State(w=config["w0"], q=config["q0"], h=config["h0"])
        for i, rw in enumerate(real_sat.rw_actuators):
            rw.h = config["h0"][i]

        # 5. Controller
        controller = MTQ_Lovera(est_sat=real_sat, p_gain=0.001, d_gain=0.005, eps=1.0)
        goal = ECI_Goal(config["goal_eci_vec"])

        # 6. Arrays
        time_hist = np.zeros(N)
        state_hist = np.zeros((N, len(x)))
        u_hist = np.zeros((N, len(real_sat.actuators)))
        boresight_hist = np.zeros((N, 4))
        t = t0
        ind = 0
        steps = int((tf - t0) / dt)

        # 7. Loop
        orb_get_os = orb.get_os
        sat_sensor_readings = real_sat.sensor_readings
        ctrl_find_u = controller.find_u
        goal_to_ref = goal.to_ref
        sat_dynamics = real_sat.dynamics_for_solver
        sec2cent = TimeConstants.sec2cent

        for i in range(steps):
            # --- UI UPDATE ---
            if i % 10 == 0:  # Update every 10 steps to keep IPC light
                update_worker_progress(slot_id, run_id, i, steps)

            # Physics
            J2000 = 0.22 + t * sec2cent
            os_state = orb_get_os(J2000=J2000)
            sens = sat_sensor_readings(x=x, os=os_state)
            u = ctrl_find_u(x_hat=x, sens=sens, est_sat=real_sat, os_hat=os_state, goal=goal)

            time_hist[ind] = t
            state_hist[ind, :] = x.as_array()
            u_hist[ind, :] = u
            eci_goal_ref, _ = goal_to_ref(os0=os_state)
            boresight_hist[ind, :] = eci_goal_ref

            ind += 1
            t += dt
            prev_os = os_state
            os_next = orb_get_os(0.22 + (t - t0) * sec2cent)

            out = solve_ivp(
                fun=sat_dynamics,
                t_span=(0, dt),
                y0=x.as_array(),
                method="RK45",
                args=(u, prev_os, os_next),
                rtol=1e-6,
                atol=1e-6,
            )
            x = State.from_array(out.y[:, -1])
            x = x.normalized()

        # Final UI update
        update_worker_progress(slot_id, run_id, steps, steps)

        return {
            "run_id": config["run_id"],
            "config": config,
            "time": time_hist,
            "state": state_hist,
            "u": u_hist,
            "boresight_goal": boresight_hist,
        }

    finally:
        # Important: Release the slot even if an error occurs
        release_worker_slot(slot_id)


# --- Config Generator ---
def generate_mc_config(run_id: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed=run_id + 1000)
    return {
        "run_id": run_id,
        "seed": run_id,
        "tf": 1000,
        "dt": 2,
        "radius_km": 7000.0,  # circular orbit radius; each core gets a different random position/plane
        "w0": normalize(rng.standard_normal(3)) * (rng.uniform(0.1, 1.0) * np.pi / 180.0),
        "q0": normalize(rng.standard_normal(4)),
        "h0": rng.uniform(-0.0001, 0.0001, size=1),
        "goal_eci_vec": normalize(rng.standard_normal(3)),
    }


if __name__ == "__main__":
    RUN_MC: bool = True
    OUTPUT_DIR = "papers/3MTQ+1RW/output_data"

    if RUN_MC:
        runner = MonteCarloRunner(
            sim_func=run_single_sim,
            config_generator=generate_mc_config,
            num_runs=100,
            max_workers=24
        )
        full_results = runner.run()

        print(f"\n--- Monte Carlo Complete: Generated {len(full_results)} histories ---")
        save_data("3MTQ+1RW_Lovera_mc_100_1000s", full_results, out_dir=OUTPUT_DIR)

        plot_target_tracking_mc(full_results=full_results, body_boresight=np.array([0, 1, 0]), title="3 MTQ + 1 RW Lovera MC:100")
        plot_convergence_histogram_mc(full_results=full_results, body_boresight=np.array([0, 1, 0]), title="3 MTQ + 1 RW Lovera")

        create_close_all_button_window()
    else:
        results = load_data("papers/3MTQ+1RW/output_data/3MTQ+1RW_LP_mc_100_20260118_013735")
        full_results = results[0]
        plot_target_tracking_mc(full_results=full_results, body_boresight=np.array([0, 1, 0]))
        plot_convergence_histogram_mc(full_results=full_results, body_boresight=np.array([0, 1, 0]), title="P = 40")
        # results = load_data("papers/3MTQ+1RW/output_data/3MTQ+1RW_LP_mc_36_20260118_013308")
        # full_results = results[0]
        # plot_target_tracking_mc(full_results=full_results)
        # plot_convergence_histogram_mc(full_results=full_results, title="P = 45")
        create_close_all_button_window()
        print("Done plotting loaded data.")
