"""
Monte Carlo: 3MTQ+1RW ALTRO Planner - Full Attitude (180° Quaternion Slew).

Uses BC2 satellite configuration with trajectory planner.
Same ICs as LP full test for fair comparison - 180° rotation about axis perpendicular to boresight.

Supports different tracking modes:
- TVLQR: Standard time-varying LQR feedback (default)
- MPC: MPC-TVLQR hybrid tracking (better for MTQ-only, also improves RW systems)
"""
import os
os.environ.setdefault('DISPLAY', ':0')  # WSLg display
os.environ['MPLBACKEND'] = 'TkAgg'  # Must be set before any matplotlib import
import sys
import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Any

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
from mc_planner_settings import create_optimized_planner_settings

# ============================================================================
# CONFIGURATION: Choose tracking mode
# ============================================================================
# Options:
#   "tvlqr" - Standard TVLQR feedback (original)
#   "mpc"   - MPC-TVLQR hybrid (better tracking, especially for MTQ-only)
TRACKING_MODE = "tvlqr"
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.helpers.math_helpers import normalize, quat_mult
from ADCS.helpers.save_and_load.save_and_load import save_data, load_data
from ADCS.helpers.plotting_mc.plot_controller_mc import (
    plot_target_tracking_mc, plot_convergence_histogram_mc, plot_single_run, plot_mc_summary,
    plot_planned_trajectory
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
                # CRITICAL: Compute B field for closed-loop simulation
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

        # Use well-conditioned normalized settings
        # Use "fast_slew" tuning for better RW usage (~72%) with good accuracy (<1° error)
        planner_settings = create_optimized_planner_settings(
            real_sat, duration=tf, dt_planning=dt_planning, tuning="fast_slew"
        )
        planner_settings.verbosity = False  # Disable C++ planner verbose output
        # Single-goal: skip Pass 2 optimization to avoid wound trajectories
        planner_settings.dt_tp = 10
        planner_settings.skip_pass2_optimization = True

        # Choose controller based on tracking mode and verbosity
        tracking_mode = config.get("tracking_mode", TRACKING_MODE)
        verbose = config.get("verbose", False)
        visualize = config.get("visualize", False)
        
        if visualize:
            # Use Python planner with live visualization for test mode
            controller = Plan_and_Track_PythonALILQR(est_sat=real_sat, planner_settings=planner_settings)
        elif tracking_mode == "mpc":
            controller = Plan_and_Track_MPC(est_sat=real_sat, planner_settings=planner_settings)
        else:
            controller = Plan_and_Track_LQR(est_sat=real_sat, planner_settings=planner_settings)

        goals = GoalList({0.22: Fixed_Attitude_Goal(config["q_goal"])})
        os0 = orb.get_os(0.22)

        try:
            if visualize:
                # Python planner supports visualize parameter
                save_viz = config.get("save_viz", False)
                viz_save_path = "/tmp/planner_viz_final.png"  # Always save
                
                # Track iteration count for periodic saving
                iter_count = [0]  # Use list to allow modification in nested function

                
                # Add a custom callback to print angle error and save figures periodically
                def diagnostic_callback(iter_data):
                    # Compute angle error across entire trajectory
                    Xset = iter_data.Xset
                    q_goal = config["q_goal"]
                    q_goal_inv = np.array([q_goal[0], -q_goal[1], -q_goal[2], -q_goal[3]])
                    
                    N = Xset.shape[1]
                    angles = np.zeros(N)
                    for k in range(N):
                        qk = Xset[3:7, k]
                        qerr_w = q_goal_inv[0]*qk[0] - np.dot(q_goal_inv[1:], qk[1:])
                        angles[k] = np.degrees(2 * np.arccos(np.clip(np.abs(qerr_w), 0, 1)))
                    
                    # Find max angle (spike detection)
                    max_angle = np.max(angles)
                    max_idx = np.argmax(angles)
                    
                    # Max in second half (where spikes typically happen after initial progress)
                    half_N = N // 2
                    second_half_angles = angles[half_N:]
                    max_2nd_half = np.max(second_half_angles)
                    max_2nd_idx = half_N + np.argmax(second_half_angles)  # Index in full array
                    
                    print(f"  [{iter_data.pass_label}] O:{iter_data.outer_iter} I:{iter_data.inner_iter} "
                          f"Cost:{iter_data.LA:.2e} Cmax:{iter_data.cmax:.2e} rho:{iter_data.rho:.1e} "
                          f"Angle[start:{angles[0]:.0f}° max:{max_angle:.0f}°@{max_idx} spike:{max_2nd_half:.0f}°@{max_2nd_idx} mean:{np.mean(angles):.0f}° end:{angles[-1]:.0f}°]")
                    
                    # Save figure at key iterations: 0, 5, 10, 20, 33 of each outer loop
                    iter_count[0] += 1
                    save_iters = [1, 5, 10, 20, 34, 50, 70, 100, 150, 200, 250, 300]
                    if iter_count[0] in save_iters:
                        import matplotlib.pyplot as plt
                        
                        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                        times = np.arange(N) * (1000.0 / N)  # Approximate times
                        
                        # Angle error over trajectory
                        axes[0, 0].plot(times, angles, 'b-', linewidth=1.5)
                        axes[0, 0].axhline(90, color='r', linestyle='--', alpha=0.5, label='90°')
                        axes[0, 0].set_xlabel('Time (s)')
                        axes[0, 0].set_ylabel('Angle Error (deg)')
                        axes[0, 0].set_title(f'Angle Error - Iter {iter_count[0]} [{iter_data.pass_label}]')
                        axes[0, 0].set_ylim(0, 200)
                        axes[0, 0].grid(True)
                        axes[0, 0].legend()
                        
                        # Quaternion components
                        axes[0, 1].plot(times, Xset[3, :], label='q0')
                        axes[0, 1].plot(times, Xset[4, :], label='q1')
                        axes[0, 1].plot(times, Xset[5, :], label='q2')
                        axes[0, 1].plot(times, Xset[6, :], label='q3')
                        axes[0, 1].set_xlabel('Time (s)')
                        axes[0, 1].set_ylabel('Quaternion')
                        axes[0, 1].set_title('Quaternion Components')
                        axes[0, 1].legend()
                        axes[0, 1].grid(True)
                        
                        # Angular velocity
                        axes[1, 0].plot(times, np.degrees(Xset[0, :]), label='ωx')
                        axes[1, 0].plot(times, np.degrees(Xset[1, :]), label='ωy')
                        axes[1, 0].plot(times, np.degrees(Xset[2, :]), label='ωz')
                        axes[1, 0].set_xlabel('Time (s)')
                        axes[1, 0].set_ylabel('Angular Velocity (deg/s)')
                        axes[1, 0].set_title('Angular Velocity')
                        axes[1, 0].legend()
                        axes[1, 0].grid(True)
                        
                        # Controls
                        Uset = iter_data.Uset
                        ctrl_times = times[:Uset.shape[1]]
                        for i in range(Uset.shape[0]):
                            axes[1, 1].plot(ctrl_times, Uset[i, :], label=f'u{i}')
                        axes[1, 1].set_xlabel('Time (s)')
                        axes[1, 1].set_ylabel('Control')
                        axes[1, 1].set_title('Control Inputs')
                        axes[1, 1].legend()
                        axes[1, 1].grid(True)
                        
                        plt.tight_layout()
                        plt.savefig(f'/tmp/planner_iter_{iter_count[0]:03d}.png', dpi=100)
                        plt.close(fig)
                        print(f"    -> Saved /tmp/planner_iter_{iter_count[0]:03d}.png")
                
                controller.set_iteration_callback(diagnostic_callback)
                
                traj = controller.calculate_trajectory(
                    t_start=0.22, duration=tf, x_0=x0, os_0=os0, goals=goals, 
                    verbose=False, visualize=True, viz_save_path=viz_save_path,
                    skip_pass2=False  # Always do pass 2 for best result
                )
                
                # Save an additional diagnostic figure
                if save_viz and hasattr(controller, 'pass1_result'):
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                    
                    # Plot final trajectory angle error over time
                    Xset = traj.states if hasattr(traj, 'states') else controller.pass1_result.Xset
                    N_traj = Xset.shape[1]
                    times = np.linspace(0, tf, N_traj)
                    
                    # Compute angle error at each timestep
                    q_goal = config["q_goal"]
                    q_goal_inv = np.array([q_goal[0], -q_goal[1], -q_goal[2], -q_goal[3]])
                    angle_errors = []
                    for k in range(N_traj):
                        qk = Xset[3:7, k]
                        qerr_w = q_goal_inv[0]*qk[0] - np.dot(q_goal_inv[1:], qk[1:])
                        angle_rad = 2 * np.arccos(np.clip(np.abs(qerr_w), 0, 1))
                        angle_errors.append(np.degrees(angle_rad))
                    
                    axes[0, 0].plot(times, angle_errors)
                    axes[0, 0].set_xlabel('Time (s)')
                    axes[0, 0].set_ylabel('Angle Error (deg)')
                    axes[0, 0].set_title('Trajectory Angle Error')
                    axes[0, 0].grid(True)
                    
                    # Plot quaternion
                    axes[0, 1].plot(times, Xset[3, :], label='q0')
                    axes[0, 1].plot(times, Xset[4, :], label='q1')
                    axes[0, 1].plot(times, Xset[5, :], label='q2')
                    axes[0, 1].plot(times, Xset[6, :], label='q3')
                    axes[0, 1].axhline(q_goal[0], color='k', linestyle='--', alpha=0.5)
                    axes[0, 1].set_xlabel('Time (s)')
                    axes[0, 1].set_ylabel('Quaternion')
                    axes[0, 1].set_title(f'Quaternion (goal: [{q_goal[0]:.2f}, {q_goal[1]:.2f}, {q_goal[2]:.2f}, {q_goal[3]:.2f}])')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True)
                    
                    # Plot angular velocity
                    axes[1, 0].plot(times, np.degrees(Xset[0, :]), label='ωx')
                    axes[1, 0].plot(times, np.degrees(Xset[1, :]), label='ωy')
                    axes[1, 0].plot(times, np.degrees(Xset[2, :]), label='ωz')
                    axes[1, 0].set_xlabel('Time (s)')
                    axes[1, 0].set_ylabel('Angular Velocity (deg/s)')
                    axes[1, 0].set_title('Angular Velocity')
                    axes[1, 0].legend()
                    axes[1, 0].grid(True)
                    
                    # Plot controls
                    Uset = traj.controls if hasattr(traj, 'controls') else controller.pass1_result.Uset
                    ctrl_times = times[:Uset.shape[1]]
                    for i in range(Uset.shape[0]):
                        axes[1, 1].plot(ctrl_times, Uset[i, :], label=f'u{i}')
                    axes[1, 1].set_xlabel('Time (s)')
                    axes[1, 1].set_ylabel('Control')
                    axes[1, 1].set_title('Control Inputs')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True)
                    
                    plt.tight_layout()
                    plt.savefig('/tmp/planner_diagnostic.png', dpi=150)
                    print(f"Saved diagnostic plot to /tmp/planner_diagnostic.png", flush=True)
                    plt.close(fig)
            else:
                print('hi')
                # C++ planner only supports verbose
                traj = controller.calculate_trajectory(
                    t_start=0.22, duration=tf, x_0=x0, os_0=os0, goals=goals, verbose=verbose
                )
            controller.set_active_trajectory(traj)
            traj_valid = True

            # Show planned trajectory non-blocking before sim starts
            if config.get("visualize", False):
                plot_planned_trajectory(
                    traj, config, BODY_BORESIGHT,
                    title_prefix="3MTQ+1RW Planner Full: Planned Trajectory"
                )

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

        import time as time_module
        sim_start_time = time_module.time()
        print(f"Starting simulation: {N} steps, dt={dt}s, tf={tf}s", flush=True)
        
        for i in range(N):
            t0_step = time_module.time()
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
            # Clamp to avoid overshooting max orbit time on last step
            t_next_clamped = min(t, tf - 0.01)
            os_next = orb.get_os(0.22 + t_next_clamped * sec2cent)
            out = solve_ivp(
                real_sat.dynamics_for_solver, (0, dt), x, method="RK45",
                args=(u, os_state, os_next), rtol=1e-6, atol=1e-6
            )
            x = out.y[:, -1]
            x[3:7] = normalize(x[3:7])
            if visualize:
                if i < 10:
                    omega_deg = np.linalg.norm(x[0:3]) * 180/np.pi
                    print(f"    Step {i}: u={u}, |ω|={omega_deg:.2f}°/s, t={t:.1f}s", flush=True)
                elif i % 50 == 0:
                    omega_deg = np.linalg.norm(x[0:3]) * 180/np.pi
                    print(f"    Step {i} took {time_module.time()-t0_step:.3f}s, nfev={out.nfev}, |u|={np.linalg.norm(u):.4f}, |ω|={omega_deg:.2f}°/s", flush=True)

        update_worker_progress(slot_id, run_id, N, N)
        sim_elapsed = time_module.time() - sim_start_time
        print(f"Simulation complete: {N} steps in {sim_elapsed:.1f}s ({sim_elapsed/N*1000:.1f}ms/step)", flush=True)

        # Extract trajectory data for plotting (convert from column-major to row-major)
        traj_times_sec = (traj.times - traj.times[0]) * 36525 * 24 * 3600  # Convert J2000 centuries to seconds
        traj_state = traj.states.T  # (N_traj, n_state)
        traj_u = traj.controls.T    # (N_traj-1, n_control)

        return {
            "run_id": run_id, "config": config, "traj_valid": True,
            "time": time_hist, "state": state_hist, "u": u_hist,
            "q_goal": q_goal_hist, "goal_type": "full_attitude",
            # Trajectory data for comparison plotting
            "traj_time": traj_times_sec, "traj_state": traj_state, "traj_u": traj_u
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
        "h0": rng.uniform(-0.0001, 0.0001, size=1),
        "tracking_mode": TRACKING_MODE,  # Include tracking mode in config
    }


if __name__ == "__main__":
    RUN_MC = True
    OUTPUT_DIR = "papers/Planner/output_data"
    NUM_RUNS = 100  # Production run
    
    # Include tracking mode in filename for differentiation
    tracking_suffix = f"_{TRACKING_MODE}" if TRACKING_MODE != "tvlqr" else ""
    output_name = f"3MTQ+1RW_plan_full180{tracking_suffix}_mc_{NUM_RUNS}"

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
    args = parser.parse_args()
    
    TEST_MODE = args.test
    if args.num_runs is not None:
        NUM_RUNS = args.num_runs
    TF_OVERRIDE = args.tf
    DT_OVERRIDE = args.dt
    DT_PLANNING_OVERRIDE = args.dt_planning
    
    if TEST_MODE:
        # Single run test mode - no multiprocessing
        test_seed = args.seed
        print(f"=== TEST MODE: Single run (seed={test_seed}), no multiprocessing ===")
        config = generate_mc_config(test_seed)
        config["verbose"] = False  # Disable verbose text output
        config["visualize"] = False  # Use C++ planner (Plan_and_Track_LQR)
        config["save_viz"] = False
        config["plot_planned_traj"] = True  # Plot planned trajectory before tracking
        result = run_single_sim(config)
        full_results = [result]
        valid = [r for r in full_results if r and r.get("traj_valid", False)]
        if valid:
            print(f"Test run completed successfully")
            # Plot single run results
            plot_single_run(result, body_boresight=BODY_BORESIGHT, title_prefix="3MTQ+1RW Planner Full Test")
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
        # Plot MC summary
        plot_mc_summary(valid, body_boresight=BODY_BORESIGHT, title_prefix="3MTQ+1RW Planner Full")
        create_close_all_button_window()
        import matplotlib.pyplot as plt
        plt.show()
    else:
        results = load_data(f"{OUTPUT_DIR}/{output_name}")
        full_results = results[0] if isinstance(results, tuple) else results
        valid = [r for r in full_results if r and r.get("traj_valid", False)]
        # Plot loaded MC results
        plot_mc_summary(valid, body_boresight=BODY_BORESIGHT, title_prefix="3MTQ+1RW Planner Full")
        create_close_all_button_window()
        import matplotlib.pyplot as plt
        plt.show()
