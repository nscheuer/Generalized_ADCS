"""
Monte Carlo for 3MTQ + 0RW with Trajectory Planner - Multi-Goal Scenario.

This script generates Monte Carlo simulations for:
- MTQ-only configuration (3 MTQs, no reaction wheels)
- ALTRO trajectory planner with TVLQR tracking
- Multi-goal scenario with 3 sequential goals (as in dissertation planning chapter)

Uses fast orbit propagation with batch B/S computation for efficiency.
"""
import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Any, Tuple, Optional, List

# --- Path Setup ---
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

# --- ADCS Imports ---
from ADCS.CONOPS.goals import ECI_Goal, No_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.helpers import PlannerSettings, Trajectory
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import MTQ
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize
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
    """Worker function for trajectory-based MC simulation."""
    global _CACHED_ORBIT, _CACHED_ORBIT_KEY

    slot_id = claim_worker_slot()
    run_id = config["run_id"]

    try:
        tf = config.get("tf", 3000)
        dt = config.get("dt", 2)
        dt_planning = config.get("dt_planning", 1)
        t0 = 0
        N = int((tf - t0) / dt)

        # --- Orbit Retrieval (Cached with fast propagation + batch B/S) ---
        radius_km = config.get("radius_km", 7000.0)
        orbit_key = (slot_id, radius_km, tf, dt_planning)
        
        if _CACHED_ORBIT is None or _CACHED_ORBIT_KEY != orbit_key:
            rng_state = np.random.get_state()
            try:
                np.random.seed(100_000 + int(slot_id))
                _CACHED_ORBIT = create_random_circular_orbit(
                    radius_km=radius_km,
                    dt=dt_planning,
                    tf=tf,
                    use_J2=True,
                    fast=True,
                )
                _CACHED_ORBIT.populate_environment(compute_B=True, compute_S=True)
                _CACHED_ORBIT_KEY = orbit_key
            finally:
                np.random.set_state(rng_state)
        
        orb = _CACHED_ORBIT

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
        x0 = np.concatenate([config["w0"], config["q0"]])

        # Planner Setup
        planner_settings = PlannerSettings(
            est_sat=real_sat,
            bdot_on=0,
            dt_tp=10,
            dt_tvlqr=dt_planning,
        )
        planner_settings.verbosity = False
        planner_settings.pass1.convergence.max_outer_iter = 8
        planner_settings.pass1.convergence.max_inner_iter = 30
        planner_settings.pass2.convergence.max_outer_iter = 4
        planner_settings.pass2.convergence.max_inner_iter = 15

        controller = Plan_and_Track_LQR(
            est_sat=real_sat,
            planner_settings=planner_settings,
        )

        # Multi-goal scenario: 3 sequential ECI pointing goals
        t_start = 0.22
        sec2cent = TimeConstants.sec2cent
        
        t_goal1 = t_start + 200 * sec2cent
        t_goal2 = t_start + 1000 * sec2cent
        t_goal3 = t_start + 2000 * sec2cent
        
        goals = GoalList({
            t_start: No_Goal(),
            t_goal1: ECI_Goal(config["goal1_eci"]),
            t_goal2: ECI_Goal(config["goal2_eci"]),
            t_goal3: ECI_Goal(config["goal3_eci"]),
        })
        os0 = orb.get_os(0.22)

        # Generate Trajectory
        try:
            traj: Trajectory = controller.calculate_trajectory(
                t_start=t_start,
                duration=tf - t0,
                x_0=x0,
                os_0=os0,
                goals=goals,
                verbose=False,
            )
            controller.set_active_trajectory(traj)
            traj_valid = True
        except Exception as e:
            return {
                "run_id": run_id,
                "config": config,
                "error": str(e),
                "traj_valid": False,
            }

        # Arrays
        time_hist = np.zeros(N)
        state_hist = np.zeros((N, len(x0)))
        u_hist = np.zeros((N, len(acts)))
        boresight_hist = np.zeros((N, 3))
        goal_index_hist = np.zeros(N)

        # Simulation loop with TVLQR tracking
        x = x0.copy()
        t = t0
        orb_get_os = orb.get_os
        sat_sensor_readings = real_sat.sensor_readings
        sat_dynamics = real_sat.dynamics_for_solver

        for i in range(N):
            if i % 20 == 0:
                update_worker_progress(slot_id, run_id, i, N)

            J2000 = t_start + t * sec2cent
            os_state = orb_get_os(J2000=J2000)

            sens = sat_sensor_readings(x=x, os=os_state)
            u = controller.find_u(x_hat=x, sens=sens, est_sat=real_sat, os_hat=os_state)

            time_hist[i] = t
            state_hist[i, :] = x
            u_hist[i, :] = u
            
            eci_goal_ref, _ = goals.to_ref(t=J2000, os0=os_state)
            boresight_hist[i, :] = eci_goal_ref
            
            if J2000 < t_goal1:
                goal_index_hist[i] = 0
            elif J2000 < t_goal2:
                goal_index_hist[i] = 1
            elif J2000 < t_goal3:
                goal_index_hist[i] = 2
            else:
                goal_index_hist[i] = 3

            t += dt
            prev_os = os_state
            os_next = orb_get_os(t_start + t * sec2cent)

            out = solve_ivp(
                fun=sat_dynamics,
                t_span=(0, dt),
                y0=x,
                method="RK45",
                args=(u, prev_os, os_next),
                rtol=1e-6,
                atol=1e-6,
            )
            x = out.y[:, -1]
            x[3:7] = normalize(x[3:7])

        update_worker_progress(slot_id, run_id, N, N)

        return {
            "run_id": run_id,
            "config": config,
            "traj_valid": True,
            "time": time_hist,
            "state": state_hist,
            "u": u_hist,
            "boresight_goal": boresight_hist,
            "goal_index": goal_index_hist,
        }

    finally:
        release_worker_slot(slot_id)


def generate_mc_config(run_id: int) -> Dict[str, Any]:
    """Generate config for multi-goal MC run."""
    rng = np.random.default_rng(seed=run_id + 1000)
    
    q0 = normalize(rng.standard_normal(4))
    goal1 = normalize(rng.standard_normal(3))
    goal2 = normalize(rng.standard_normal(3))
    goal3 = normalize(rng.standard_normal(3))
    
    return {
        "run_id": run_id,
        "seed": run_id,
        "tf": 3000,
        "dt": 2,
        "dt_planning": 1,
        "radius_km": 7000.0,
        "w0": normalize(rng.standard_normal(3)) * (rng.uniform(0.5, 2.0) * np.pi / 180.0),
        "q0": q0,
        "goal1_eci": goal1,
        "goal2_eci": goal2,
        "goal3_eci": goal3,
    }


if __name__ == "__main__":
    RUN_MC: bool = True
    OUTPUT_DIR = "papers/Planner/output_data"
    NUM_RUNS = 100

    if RUN_MC:
        runner = MonteCarloRunner(
            sim_func=run_single_sim,
            config_generator=generate_mc_config,
            num_runs=NUM_RUNS,
            max_workers=4,
        )
        full_results = runner.run()
        
        valid_results = [r for r in full_results if r and r.get("traj_valid", False)]
        print(f"\n--- Monte Carlo Complete: {len(valid_results)}/{len(full_results)} valid ---")
        save_data(f"3MTQ+0RW_PLAN_multigoal_mc_{NUM_RUNS}", full_results, out_dir=OUTPUT_DIR)
        
        plot_target_tracking_mc(full_results=valid_results, title=f"3MTQ+0RW Planner Multi-Goal N={len(valid_results)}")
        plot_convergence_histogram_mc(full_results=valid_results, title=f"3MTQ+0RW Planner Multi-Goal N={len(valid_results)}")
        create_close_all_button_window()
    else:
        results = load_data(f"{OUTPUT_DIR}/3MTQ+0RW_PLAN_multigoal_mc_{NUM_RUNS}")
        full_results = results[0] if isinstance(results, tuple) else results
        valid_results = [r for r in full_results if r and r.get("traj_valid", False)]
        plot_target_tracking_mc(full_results=valid_results, title=f"3MTQ+0RW Planner Multi-Goal")
        plot_convergence_histogram_mc(full_results=valid_results, title=f"3MTQ+0RW Planner Multi-Goal")
        create_close_all_button_window()
