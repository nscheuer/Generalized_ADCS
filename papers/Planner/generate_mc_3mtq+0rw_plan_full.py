"""
Monte Carlo: 3MTQ+0RW ALTRO Planner - Full Attitude (180° Quaternion Slew).

Uses BC1 satellite configuration with MTQ-only trajectory planner.
Same ICs as Lovera full test for fair comparison.

Supports different tracking modes:
- TVLQR: Standard time-varying LQR feedback (default)
- MPC: MPC-TVLQR hybrid tracking (RECOMMENDED for MTQ-only - 10x better!)

MPC tracking is especially beneficial for MTQ-only systems because MTQ torque
depends on B-field which depends on attitude. TVLQR uses the planned B-field
which diverges from actual when attitude drifts. MPC uses actual B-field.
"""
import os
os.environ.setdefault('DISPLAY', ':0')  # WSLg display
os.environ.setdefault('MPLBACKEND', 'TkAgg')  # Must be set before any matplotlib import
import sys
import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Any
import time as time_module

# --- Path Setup (works from Generalized_ADCS root directory) ---
_this_file = os.path.abspath(__file__)
_this_dir = os.path.dirname(_this_file)
_root_dir = os.path.abspath(os.path.join(_this_dir, "../../.."))
sys.path.insert(0, _root_dir)  # Add root for ADCS imports
sys.path.insert(0, _this_dir)  # Add local dir for local imports (e.g., mc_planner_settings)

from ADCS.CONOPS.goals import Fixed_Attitude_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.plan_and_track_mpc import Plan_and_Track_MPC
from ADCS.controller.plan_and_track_python_alilqr import Plan_and_Track_PythonALILQR
from ADCS.controller.helpers import PlannerSettings

# Import good settings
from mc_planner_settings import create_optimized_planner_settings, create_mc_planner_and_factory

# ============================================================================
# CONFIGURATION: Choose tracking mode
# ============================================================================
# Options:
#   "tvlqr" - Standard TVLQR feedback (original)
#   "mpc"   - MPC-TVLQR hybrid (RECOMMENDED for MTQ-only - 10x better tracking!)
TRACKING_MODE = "tvlqr"
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube1_cubesat
from ADCS.helpers.math_helpers import normalize, quat_mult
from ADCS.helpers.save_and_load.save_and_load import save_data, load_data
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window
from ADCS.helpers.plotting_mc.plot_controller_mc import (
    plot_single_run, plot_mc_summary, plot_quaternion_error_mc, plot_quaternion_histogram_mc,
    plot_planned_trajectory, create_planner_diagnostic_callback
)
from ADCS.helpers.mc.monte_carlo_runner import (
    MonteCarloRunner, claim_worker_slot, release_worker_slot, update_worker_progress
)
import argparse

BODY_BORESIGHT = np.array([0, 1, 0])

_CACHED_ORBIT = None
_CACHED_ORBIT_KEY = None
TF_OVERRIDE = None
DT_OVERRIDE = None
DT_PLANNING_OVERRIDE = None


def run_single_sim(config: Dict[str, Any]) -> Dict[str, Any]:
    global _CACHED_ORBIT, _CACHED_ORBIT_KEY

    slot_id = claim_worker_slot()
    run_id = config["run_id"]

    try:
        tf = config.get("tf", 1000)
        dt = config.get("dt", 1)
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
                _CACHED_ORBIT.populate_environment(compute_B=True, compute_S=True)
                _CACHED_ORBIT_KEY = orbit_key
            finally:
                np.random.set_state(rng_state)

        orb = _CACHED_ORBIT
        np.random.seed(config["seed"])

        real_sat = create_beavercube1_cubesat(estimated=False)

        x0 = np.concatenate([config["w0"], config["q0"]])

        # Use auto_refine with settings factory for robust MTQ-only planning
        planner_settings, settings_factory = create_mc_planner_and_factory(
            real_sat, tf=tf, dt_planning=10.0, has_rw=False
        )

        # Choose controller based on tracking mode and visualization
        visualize = config.get("visualize", False)
        tracking_mode = config.get("tracking_mode", TRACKING_MODE)
        if visualize:
            controller = Plan_and_Track_PythonALILQR(est_sat=real_sat, planner_settings=planner_settings)
        elif tracking_mode == "mpc":
            controller = Plan_and_Track_MPC(est_sat=real_sat, planner_settings=planner_settings)
        else:
            controller = Plan_and_Track_LQR(
                est_sat=real_sat, planner_settings=planner_settings,
                settings_factory=settings_factory
            )

        goals = GoalList({0.22: Fixed_Attitude_Goal(config["q_goal"])})
        os0 = orb.get_os(0.22)

        try:
            t_plan_start = time_module.perf_counter()
            if visualize:
                callback = create_planner_diagnostic_callback(config, BODY_BORESIGHT, tf)
                controller.set_iteration_callback(callback)
                traj = controller.calculate_trajectory(
                    t_start=0.22, duration=tf, x_0=x0, os_0=os0, goals=goals,
                    verbose=False, visualize=True, viz_save_path="/tmp/planner_viz_final.png",
                    skip_pass2=False, orbit=orb
                )
            else:
                traj = controller.calculate_trajectory(
                    t_start=0.22, duration=tf, x_0=x0, os_0=os0, goals=goals,
                    verbose=False, orbit=orb
                )
            plan_time_s = time_module.perf_counter() - t_plan_start
            controller.set_active_trajectory(traj)
            traj_valid = True
            if visualize:
                plot_planned_trajectory(
                    traj, config, BODY_BORESIGHT,
                    title_prefix="3MTQ+0RW Planner Full: Planned Trajectory"
                )
        except Exception as e:
            return {"run_id": run_id, "config": config, "error": str(e), "traj_valid": False}

        time_hist = np.zeros(N)
        state_hist = np.zeros((N, len(x0)))
        u_hist = np.zeros((N, len(real_sat.actuators)))
        q_goal_hist = np.zeros((N, 4))

        x = x0.copy()
        t = 0
        sec2cent = TimeConstants.sec2cent

        t_sim_start = time_module.perf_counter()
        for i in range(N):
            if i % 10 == 0:
                update_worker_progress(slot_id, run_id, i, N)

            J2000 = 0.22 + t * sec2cent
            os_state = orb.get_os(J2000=J2000)
            sens = real_sat.sensor_readings(x=x, os=os_state)
            
            # MPC needs B_body; TVLQR computes it internally
            if tracking_mode == "mpc":
                from scipy.spatial.transform import Rotation
                q = x[3:7]
                R_mat = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
                B_body = R_mat.T @ os_state.B
                u = controller.find_u(x_hat=x, sens=sens, est_sat=real_sat, os_hat=os_state, B_body=B_body)
            else:
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

        sim_time_s = time_module.perf_counter() - t_sim_start
        update_worker_progress(slot_id, run_id, N, N)

        # Extract trajectory data for plotting (convert from column-major to row-major)
        traj_times_sec = (traj.times - traj.times[0]) * 36525 * 24 * 3600  # Convert J2000 centuries to seconds
        traj_state = traj.states.T  # (N_traj, n_state)
        traj_u = traj.controls.T    # (N_traj-1, n_control)

        return {
            "run_id": run_id, "config": config, "traj_valid": True,
            "time": time_hist, "state": state_hist, "u": u_hist,
            "q_goal": q_goal_hist, "goal_type": "full_attitude",
            "plan_time_s": plan_time_s, "sim_time_s": sim_time_s,
            # Trajectory data for comparison plotting
            "traj_time": traj_times_sec, "traj_state": traj_state, "traj_u": traj_u
        }
    finally:
        release_worker_slot(slot_id)


def generate_mc_config(run_id: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed=run_id + 1000)
    
    q0 = normalize(rng.standard_normal(4))
    
    rand_angle = rng.uniform(0, 2 * np.pi)
    axis_body = np.array([np.cos(rand_angle), 0, np.sin(rand_angle)])
    q_180_body = np.array([0, axis_body[0], axis_body[1], axis_body[2]])
    q_goal = quat_mult(q0, q_180_body)
    q_goal = normalize(q_goal)
    
    tf = TF_OVERRIDE if TF_OVERRIDE is not None else 1000
    dt = DT_OVERRIDE if DT_OVERRIDE is not None else 1
    dt_planning = DT_PLANNING_OVERRIDE if DT_PLANNING_OVERRIDE is not None else 1

    return {
        "run_id": run_id,
        "seed": run_id,
        "tf": tf,
        "dt": dt,
        "dt_planning": dt_planning,
        "radius_km": 7000.0,
        "w0": normalize(rng.standard_normal(3)) * (rng.uniform(0.1, 1.0) * np.pi / 180.0),
        "q0": q0,
        "q_goal": q_goal,
        "tracking_mode": TRACKING_MODE,  # Include tracking mode in config
    }


if __name__ == "__main__":
    RUN_MC = True
    OUTPUT_DIR = "papers/Planner/output_data"
    NUM_RUNS = 100  # Production run
    
    # Include tracking mode in filename for differentiation
    tracking_suffix = f"_{TRACKING_MODE}" if TRACKING_MODE != "tvlqr" else ""
    output_name = f"3MTQ+0RW_plan_full180{tracking_suffix}_mc_{NUM_RUNS}"

    # --- Command-line argument parsing ---
    parser = argparse.ArgumentParser(description="Run Monte Carlo simulations")
    parser.add_argument("-t", "--test", action="store_true", 
                        help="Run single test simulation (no multiprocessing, with visualization)")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="Seed for test mode (default: 0)")
    parser.add_argument("-n", "--num-runs", type=int, default=None,
                        help="Override number of runs")
    parser.add_argument("--tf", type=float, default=None, help="Override planning duration [s]")
    parser.add_argument("--dt", type=float, default=None, help="Override sim dt [s]")
    parser.add_argument("--dt-planning", type=float, default=None, help="Override planning dt [s]")
    parser.add_argument("--plot", action="store_true", help="Show plots after MC (default: just save data)")
    args = parser.parse_args()
    
    TEST_MODE = args.test
    if args.num_runs is not None:
        NUM_RUNS = args.num_runs
    TF_OVERRIDE = args.tf
    DT_OVERRIDE = args.dt
    DT_PLANNING_OVERRIDE = args.dt_planning
    
    if TEST_MODE:
        # Single run test mode - no multiprocessing, with visualization
        test_seed = args.seed
        print(f"=== TEST MODE: Single run (seed={test_seed}), no multiprocessing ===")
        config = generate_mc_config(test_seed)
        config["visualize"] = True
        result = run_single_sim(config)
        full_results = [result]
        valid = [r for r in full_results if r and r.get("traj_valid", False)]
        if valid:
            print(f"Test run completed successfully")
            # Plot single run results
            plot_single_run(result, body_boresight=BODY_BORESIGHT, title_prefix="3MTQ+0RW Planner Full Test")
            create_close_all_button_window()
            import matplotlib.pyplot as plt
            plt.show()
        else:
            print(f"Test run failed: {result.get('error', 'Unknown error')}")

    elif RUN_MC:
        print(f"Running with tracking_mode={TRACKING_MODE}")
        runner = MonteCarloRunner(
            sim_func=run_single_sim,
            config_generator=generate_mc_config,
            num_runs=NUM_RUNS,
            max_workers=4
        )
        full_results = runner.run()

        valid = [r for r in full_results if r and r.get("traj_valid", False)]
        print(f"\n--- Monte Carlo Complete: {len(valid)}/{len(full_results)} valid ---")
        print(f"Tracking mode: {TRACKING_MODE}")
        save_data(output_name, full_results, out_dir=OUTPUT_DIR)
        if args.plot:
            plot_mc_summary(valid, body_boresight=BODY_BORESIGHT, title_prefix="3MTQ+0RW Planner Full")
            create_close_all_button_window()
            import matplotlib.pyplot as plt
            plt.show()
    else:
        results = load_data(f"{OUTPUT_DIR}/{output_name}")
        full_results = results[0] if isinstance(results, tuple) else results
        valid = [r for r in full_results if r and r.get("traj_valid", False)]
        # Plot loaded MC results
        plot_mc_summary(valid, body_boresight=BODY_BORESIGHT, title_prefix="3MTQ+0RW Planner Full")
        create_close_all_button_window()
        import matplotlib.pyplot as plt
        plt.show()
