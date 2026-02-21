"""
Monte Carlo: 3MTQ+1RW ALTRO+TVLQR Planner - Reduced Attitude (180° Boresight Slew).

Uses BC2 satellite configuration with trajectory planner.
Runs both open-loop (raw trajectory) and closed-loop (TVLQR tracking).
Same ICs as LP test for fair comparison.
"""
import os
os.environ.setdefault('DISPLAY', ':0')  # WSLg display
os.environ.setdefault('MPLBACKEND', 'TkAgg')  # Must be set before any matplotlib import
import sys
import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Any
import time as time_module
from typing import Tuple, Optional, List

# --- Path Setup (works from Generalized_ADCS root directory) ---
_this_file = os.path.abspath(__file__)
_this_dir = os.path.dirname(_this_file)
_root_dir = os.path.abspath(os.path.join(_this_dir, "../../.."))
sys.path.insert(0, _root_dir)  # Add root for ADCS imports
sys.path.insert(0, _this_dir)  # Add local dir for local imports (e.g., mc_planner_settings)

from ADCS.CONOPS.goals import ECI_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.plan_and_track_python_alilqr import Plan_and_Track_PythonALILQR
from ADCS.controller.helpers import PlannerSettings

# Import good settings
from mc_planner_settings import create_optimized_planner_settings, create_mc_planner_and_factory
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.helpers.math_helpers import normalize, rot_mat
from ADCS.helpers.save_and_load.save_and_load import save_data, load_data
from ADCS.helpers.plotting_mc.plot_controller_mc import (
    plot_target_tracking_mc, plot_convergence_histogram_mc, plot_single_run, plot_mc_summary,
    plot_planned_trajectory, create_planner_diagnostic_callback
)
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window
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


def _rot_mat_vec(q: np.ndarray) -> np.ndarray:
    """
    Vectorized conversion of Scalar-First Quaternions (w, x, y, z)
    to Rotation Matrices (Body -> Inertial).
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x*y - z*w)
    R[:, 0, 2] = 2 * (x*z + y*w)
    R[:, 1, 0] = 2 * (x*y + z*w)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y*z - x*w)
    R[:, 2, 0] = 2 * (x*z - y*w)
    R[:, 2, 1] = 2 * (y*z + x*w)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)
    return R


def compute_tracking_metrics(result: Dict[str, Any], body_boresight: np.ndarray) -> Dict[str, float]:
    state = result["state"]
    goal = result["boresight_goal"]
    u_hist = result["u"]
    time_hist = result["time"]

    v_bore_body = body_boresight / np.linalg.norm(body_boresight)
    q_hist = state[:, 3:7]
    R_b2i = _rot_mat_vec(q_hist)
    v_bore_eci = np.einsum("nij,j->ni", R_b2i, v_bore_body)
    v_bore_eci /= np.linalg.norm(v_bore_eci, axis=1, keepdims=True)
    v_goal = goal / np.linalg.norm(goal, axis=1, keepdims=True)
    dot_prod = np.sum(v_bore_eci * v_goal, axis=1)
    dot_prod = np.clip(dot_prod, -1.0, 1.0)
    error_deg = np.rad2deg(np.arccos(dot_prod))

    u_norm = np.linalg.norm(u_hist, axis=1)

    settle_1deg = np.nan
    below = error_deg < 1.0
    if np.any(below):
        for i in range(len(error_deg)):
            if np.all(error_deg[i:] < 1.0):
                settle_1deg = float(time_hist[i])
                break

    return {
        "err_deg_max": float(np.max(error_deg)),
        "err_deg_mean": float(np.mean(error_deg)),
        "err_deg_rms": float(np.sqrt(np.mean(error_deg**2))),
        "err_deg_p95": float(np.percentile(error_deg, 95)),
        "err_deg_final": float(error_deg[-1]),
        "settle_1deg_s": float(settle_1deg),
        "u_norm_max": float(np.max(u_norm)),
        "u_norm_mean": float(np.mean(u_norm)),
        "u_norm_rms": float(np.sqrt(np.mean(u_norm**2))),
    }


def print_tracking_metrics(metrics: Dict[str, float]) -> None:
    print(
        "METRICS: tracking_error_deg "
        f"max={metrics['err_deg_max']:.4f} "
        f"rms={metrics['err_deg_rms']:.4f} "
        f"mean={metrics['err_deg_mean']:.4f} "
        f"p95={metrics['err_deg_p95']:.4f} "
        f"final={metrics['err_deg_final']:.4f} "
        f"settle_1deg_s={metrics['settle_1deg_s']:.2f}"
    )
    print(
        "METRICS: control_norm "
        f"max={metrics['u_norm_max']:.4e} "
        f"rms={metrics['u_norm_rms']:.4e} "
        f"mean={metrics['u_norm_mean']:.4e}"
    )


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

        real_sat = create_beavercube2_cubesat(estimated=False)
        rws = real_sat.rw_actuators

        x0 = np.concatenate([config["w0"], config["q0"], config["h0"]])
        for i, rw in enumerate(rws):
            rw.h = config["h0"][i]

        # Use auto_refine with settings factory for robust planning across dt values
        planner_settings, settings_factory = create_mc_planner_and_factory(
            real_sat, tf=tf, dt_planning=10.0, has_rw=True
        )

        visualize = config.get("visualize", False)
        if visualize:
            controller = Plan_and_Track_PythonALILQR(est_sat=real_sat, planner_settings=planner_settings)
        else:
            controller = Plan_and_Track_LQR(
                est_sat=real_sat, planner_settings=planner_settings,
                settings_factory=settings_factory
            )

        goals = GoalList({0.22: ECI_Goal(config["goal_eci_vec"])})
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
                    title_prefix="3MTQ+1RW Planner Reduced: Planned Trajectory"
                )
        except Exception as e:
            return {"run_id": run_id, "config": config, "error": str(e), "traj_valid": False}

        # Arrays for LQR tracking
        time_hist = np.zeros(N)
        state_hist = np.zeros((N, len(x0)))
        u_hist = np.zeros((N, len(real_sat.actuators)))
        boresight_hist = np.zeros((N, 3))

        # Reset RW state
        for i, rw in enumerate(rws):
            rw.h = config["h0"][i]

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
            u = controller.find_u(x_hat=x, sens=sens, est_sat=real_sat, os_hat=os_state)

            time_hist[i] = t
            state_hist[i, :] = x
            u_hist[i, :] = u
            eci_goal_ref, _ = goals.to_ref(t=J2000, os0=os_state)
            boresight_hist[i, :] = eci_goal_ref

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
            "boresight_goal": boresight_hist,
            "plan_time_s": plan_time_s, "sim_time_s": sim_time_s,
            # Trajectory data for comparison plotting
            "traj_time": traj_times_sec, "traj_state": traj_state, "traj_u": traj_u
        }
    finally:
        release_worker_slot(slot_id)


def generate_mc_config(run_id: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed=run_id + 1000)
    
    q0 = normalize(rng.standard_normal(4))
    R0 = rot_mat(q0)
    initial_boresight_eci = R0 @ BODY_BORESIGHT
    goal_eci_vec = -initial_boresight_eci  # 180° boresight slew
    
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
        "h0": rng.uniform(-0.0001, 0.0001, size=1),
        "goal_eci_vec": goal_eci_vec,
    }


if __name__ == "__main__":
    RUN_MC = True
    OUTPUT_DIR = "papers/Planner/output_data"
    NUM_RUNS = 100  # Production run

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
        config["visualize"] = False
        result = run_single_sim(config)
        full_results = [result]
        valid = [r for r in full_results if r and r.get("traj_valid", False)]
        if valid:
            print(f"Test run completed successfully")
            metrics = compute_tracking_metrics(valid[0], BODY_BORESIGHT)
            print_tracking_metrics(metrics)
            # Plot single run results
            plot_single_run(result, body_boresight=BODY_BORESIGHT, title_prefix="3MTQ+1RW Planner Reduced Test")
            create_close_all_button_window()
            import matplotlib.pyplot as plt
            plt.show()
        else:
            print(f"Test run failed: {result.get('error', 'Unknown error')}")

    elif RUN_MC:
        runner = MonteCarloRunner(
            sim_func=run_single_sim,
            config_generator=generate_mc_config,
            num_runs=NUM_RUNS,
            max_workers=4, per_run_timeout=600  # 10min per seed
        )
        full_results = runner.run()

        valid = [r for r in full_results if r and r.get("traj_valid", False)]
        print(f"\n--- Monte Carlo Complete: {len(valid)}/{len(full_results)} valid ---")
        save_data(f"3MTQ+1RW_plan_reduced_mc_{NUM_RUNS}", full_results, out_dir=OUTPUT_DIR)
        if args.plot:
            plot_mc_summary(valid, body_boresight=BODY_BORESIGHT, title_prefix="3MTQ+1RW Planner Reduced")
            create_close_all_button_window()
            import matplotlib.pyplot as plt
            plt.show()
    else:
        results = load_data(f"{OUTPUT_DIR}/3MTQ+1RW_plan_reduced_mc_{NUM_RUNS}")
        full_results = results[0] if isinstance(results, tuple) else results
        valid = [r for r in full_results if r and r.get("traj_valid", False)]
        # Plot loaded MC results
        plot_mc_summary(valid, body_boresight=BODY_BORESIGHT, title_prefix="3MTQ+1RW Planner Reduced")
        create_close_all_button_window()
        import matplotlib.pyplot as plt
        plt.show()
