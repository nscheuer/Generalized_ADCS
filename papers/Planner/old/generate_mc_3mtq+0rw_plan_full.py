import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Any, Tuple, Optional

# --- Path Setup ---
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

# --- ADCS Imports ---
from ADCS.CONOPS.goals import Fixed_Attitude_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.plan_and_track import PlannerSettings
from ADCS.controller.plan_and_track.planner_subsettings import CostWeights
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube1_cubesat
from ADCS.helpers.math_helpers import normalize
from ADCS.helpers.save_and_load.save_and_load import save_data, load_data
from ADCS.helpers.plotting_mc.plot_controller_mc import plot_target_tracking_mc, plot_convergence_histogram_mc
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window

# --- MC Runner Imports ---
from ADCS.mc.monte_carlo_runner import (
    MonteCarloRunner,
    claim_worker_slot,
    release_worker_slot,
    update_worker_progress,
)

"""
Monte Carlo Simulation for ALTRO (Plan_and_Track_LQR) Controller.
- Configuration matches debug_altro settings (tf=500, dt=1).
- Uses specific cost weights and planner settings defined in the debug file.
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
        dt = config.get("dt", 1)
        t0 = 0
        steps = int((tf - t0) / dt)

        # 2. Orbit Retrieval (Cached)
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

        # Create Satellite (using exact factory from debug)
        real_sat = create_beavercube1_cubesat(estimated=False)

        # 4. Initial Conditions
        x = np.concatenate([config["w0"], config["q0"]])

        # 5. Controller Setup (Exact settings from debug_altro)
        planner_settings = PlannerSettings(
            est_sat=real_sat,
            bdot_on=0,  # Skip bdot initial guess
            dt_tp=50,
            dt_tvlqr=1,
        )
        
        # --- Apply Debug Gains and Costs ---
        planner_settings.verbosity = False
        planner_settings.cost_main.use_full_cost_hessian = True
        planner_settings.pass1.regularization.use_dynamics_hess = 1
        planner_settings.init_traj.bdot_gain = 500
        planner_settings.pass1.aug_lag.penalty_init = 1e-3
        planner_settings.pass1.aug_lag.penalty_scale = 10
        planner_settings.pass1.convergence.max_outer_iter = 15
        planner_settings.pass1.convergence.max_inner_iter = 40
        planner_settings.pass2.aug_lag.penalty_init = 1e5
        planner_settings.pass2.aug_lag.penalty_scale = 10
        planner_settings.pass2.convergence.max_outer_iter = 8
        planner_settings.pass2.convergence.max_inner_iter = 20

        planner_settings.cost_main = CostWeights(
                angle=1e1,
                angle_N=1e1,   # 10x running cost
                ang_vel=1e5,
                ang_vel_N=1e5, # 10x running cost
                ang_vel_err_dir=1e2,
                ang_vel_err_dir_N=0.0,
                ang_vel_mag=0.0,
                ang_vel_mag_N=0.0,
                control_mult=1.0,
                ang_cost_func_type=2,
            )
        
        planner_settings.cost_second = planner_settings.cost_main
        
        planner_settings.cost_tvlqr = CostWeights(
                angle=1e5,
                angle_N=1e6,
                ang_vel=1e6,
                ang_vel_N=1e8,
                ang_vel_mag=0.0,
                ang_vel_mag_N=0.0,
                control_mult=1.0,
                ang_cost_func_type=2,
            )

        controller = Plan_and_Track_LQR(est_sat=real_sat, planner_settings=planner_settings)

        # 6. Goals and Trajectory Planning
        start_time_j2000 = 0.22  # Matching debug file start time
        goals = GoalList({start_time_j2000: Fixed_Attitude_Goal(config["q0_goal"])})
        
        # Get initial orbital state for planning
        os0 = orb.get_os(J2000=start_time_j2000)

        # Calculate Trajectory (The "Plan" part)
        traj = controller.calculate_trajectory(
            t_start=start_time_j2000,
            duration=(tf - t0),
            x_0=x,
            os_0=os0,
            goals=goals,
            verbose=False
        )
        
        controller.set_active_trajectory(traj)

        # 7. Simulation Arrays
        time_hist = np.zeros(steps)
        state_hist = np.zeros((steps, len(x)))
        u_hist = np.zeros((steps, len(real_sat.actuators)))
        q_goal_hist = np.zeros((steps, 4))
        
        t = t0
        ind = 0
        
        # Optimization vars
        orb_get_os = orb.get_os
        sat_sensor_readings = real_sat.sensor_readings
        ctrl_find_u = controller.find_u
        sat_dynamics = real_sat.dynamics_for_solver
        sec2cent = TimeConstants.sec2cent

        # 8. Loop (The "Track" part)
        for i in range(steps):
            # --- UI UPDATE ---
            if i % 10 == 0:
                update_worker_progress(slot_id, run_id, i, steps)

            # Physics
            current_j2000 = start_time_j2000 + t * sec2cent
            os_state = orb_get_os(J2000=current_j2000)
            
            sens = sat_sensor_readings(x=x, os=os_state)
            
            # Find U (Tracking the active trajectory)
            u = ctrl_find_u(x_hat=x, sens=sens, est_sat=real_sat, os_hat=os_state)

            time_hist[ind] = t
            state_hist[ind, :] = x
            u_hist[ind, :] = u
            
            # Log the goal (reference) at this timestep
            q_goal_ref, _ = goals.to_ref(t=current_j2000, os0=os_state)
            q_goal_hist[ind, :] = q_goal_ref

            ind += 1
            t += dt
            
            prev_os = os_state
            os_next = orb_get_os(start_time_j2000 + (t - t0) * sec2cent)

            # Integration (using rtol=1e-7 per debug file)
            out = solve_ivp(
                fun=sat_dynamics,
                t_span=(0, dt),
                y0=x,
                method="RK45",
                args=(u, prev_os, os_next),
                rtol=1e-7,
                atol=1e-7,
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
            "q_goal": q_goal_hist,
        }

    finally:
        release_worker_slot(slot_id)


# --- Config Generator ---
def generate_mc_config(run_id: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed=run_id + 238)
    
    # Matching initial condition randomization from debug/template
    return {
        "run_id": run_id,
        "seed": run_id,
        "tf": 1000,  # Explicitly requested 500s
        "dt": 1,    # Explicitly requested 1s
        "radius_km": 7000.0,
        "w0": normalize(rng.standard_normal(3)) * (rng.uniform(0.1, 1.0) * np.pi / 180.0),
        "q0": normalize(rng.standard_normal(4)),
        "q0_goal": normalize(rng.standard_normal(4)),
    }


if __name__ == "__main__":
    RUN_MC: bool = True
    OUTPUT_DIR = "papers/Planner/output_data" # Adjusted folder name

    if RUN_MC:
        runner = MonteCarloRunner(
            sim_func=run_single_sim,
            config_generator=generate_mc_config,
            num_runs=100,
            max_workers=24
        )
        full_results = runner.run()

        print(f"\n--- Monte Carlo Complete: Generated {len(full_results)} histories ---")
        save_data("3MTQ+0RW_ALTRO_100_1000s_full", full_results, out_dir=OUTPUT_DIR)

        plot_target_tracking_mc(full_results=full_results, body_boresight=np.array([0, 1, 0]), title="ALTRO Trajectory Tracking MC:100")
        plot_convergence_histogram_mc(full_results=full_results, body_boresight=np.array([0, 1, 0]), title="ALTRO Convergence")

        create_close_all_button_window()
