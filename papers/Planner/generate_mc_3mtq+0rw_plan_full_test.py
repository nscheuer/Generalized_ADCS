"""
Monte Carlo: 3MTQ+0RW ALTRO Planner - Full Attitude (180° Quaternion Slew).

Uses BC1 satellite configuration (MTQ-only, no reaction wheel).
Same structure as 3MTQ+1RW test for fair comparison.

Supports different tracking modes:
- TVLQR: Standard time-varying LQR feedback (default)
- MPC: MPC-TVLQR hybrid tracking (better for MTQ-only)
"""
import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Any

# --- Path Setup (works from Generalized_ADCS root directory) ---
_this_file = os.path.abspath(__file__)
_this_dir = os.path.dirname(_this_file)
_root_dir = os.path.abspath(os.path.join(_this_dir, "../.."))
os.chdir(_root_dir)
sys.path.insert(0, _root_dir)  # Add root for ADCS imports
sys.path.insert(0, _this_dir)  # Add local dir for local imports (e.g., mc_planner_settings)

from ADCS.CONOPS.goals import Fixed_Attitude_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.plan_and_track_mpc import Plan_and_Track_MPC
from ADCS.controller.plan_and_track_python_alilqr import Plan_and_Track_PythonALILQR
from ADCS.controller.helpers import PlannerSettings, create_planner_settings
from ADCS.controller.helpers.normalized_settings import (
    NormalizedPlannerConfig, NormalizedActuatorCosts, NormalizedStateCosts,
    NormalizedConstraints,
)

# Import good settings from mc_planner_settings (same as 3+1)
from mc_planner_settings import create_optimized_planner_settings

# ============================================================================
# CONFIGURATION: Choose tracking mode
# ============================================================================
# Options:
#   "tvlqr" - Standard TVLQR feedback (original)
#   "mpc"   - MPC-TVLQR hybrid (better tracking for MTQ-only)
TRACKING_MODE = "tvlqr"
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube1_cubesat
from ADCS.helpers.math_helpers import normalize, quat_mult
from ADCS.helpers.save_and_load.save_and_load import save_data, load_data
from ADCS.helpers.plotting_mc.plot_controller_mc import plot_target_tracking_mc, plot_convergence_histogram_mc
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window
from ADCS.helpers.mc.monte_carlo_runner import (
    MonteCarloRunner, claim_worker_slot, release_worker_slot, update_worker_progress
)
import argparse

BODY_BORESIGHT = np.array([0, 1, 0])

_CACHED_ORBIT = None
_CACHED_ORBIT_KEY = None


def create_mtq_only_settings(sat, dt_planning: float = 1.0):
    """
    Create well-conditioned planner settings for MTQ-only (no RW).
    
    Uses the EXACT same tuning as our 3MTQ+1RW settings from mc_planner_settings.py.
    """
    config = NormalizedPlannerConfig(
        actuator_costs=NormalizedActuatorCosts(
            mtq_cost=0.1,            # 100x lower - MTQ-only needs cheap control
            rw_torque_cost=1.0,      # Not used for MTQ-only
            rw_momentum_cost=2.0,    # Not used for MTQ-only
            rw_stiction_cost=1.0,    # Not used for MTQ-only
            use_torque_effective_mtq_scaling=False,  # Disabled - no B-field scaling
            expected_B_field_uT=30.0,
        ),
        state_costs=NormalizedStateCosts(
            angle_cost=1000.0,          # 10x increase for better convergence (same as 3+1)
            angle_terminal_cost=10000.0,
            ang_vel_cost=100.0,
            ang_vel_terminal_cost=1000.0,
            use_scale_normalization=True,
            angle_scale_deg=90.0,
            ang_vel_scale_deg_s=20.0,
        ).set_cross_term_auto(0.75),  # Add cross-term at 75% of PSD limit (same as 3+1)
        constraints=NormalizedConstraints(
            max_angular_velocity_deg_s=20.0,
            control_margin=0.25,
            rw_momentum_margin=0.9,
        ),
    )
    
    settings = create_planner_settings(sat, config)
    
    # Key settings (same as 3+1)
    settings.bdot_on = 0  # Random init, not B-dot (better for slew maneuvers)
    settings.dt_tp = 10   # Coarse planning timestep
    settings.dt_tvlqr = dt_planning
    settings.verbosity = False
    
    # Set ang_cost_func_type = 4 for stronger 180° penalty (1-cos²(θ/2))
    settings.cost_main.ang_cost_func_type = 4
    settings.cost_second.ang_cost_func_type = 4
    settings.cost_tvlqr.ang_cost_func_type = 4
    
    # Convergence settings - more iterations for MTQ-only (harder problem)
    settings.pass1.convergence.max_outer_iter = 25
    settings.pass1.convergence.max_inner_iter = 100
    settings.pass2.convergence.max_outer_iter = 15
    settings.pass2.convergence.max_inner_iter = 100
    
    # Augmented Lagrangian settings - balance cost and constraint enforcement
    settings.pass1.aug_lag.penalty_init = 1e-4  # Middle ground
    settings.pass1.aug_lag.penalty_max = 1e8
    settings.pass2.aug_lag.penalty_init = 1e-2
    settings.pass2.aug_lag.penalty_max = 1e18
    
    # State-space regularization (in addition to control-space) (same as 3+1)
    # reg_mode: 0=control-space only, 1=state-space only, 2=both
    settings.pass1.regularization.reg_mode = 2
    settings.pass2.regularization.reg_mode = 2
    
    # reg_min_cond: 0=no minimum regularization enforcement (same as 3+1)
    settings.pass1.regularization.reg_min_cond = 0
    settings.pass2.regularization.reg_min_cond = 0
    
    return settings


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
                # CRITICAL: Compute B field for closed-loop simulation
                _CACHED_ORBIT.populate_environment(compute_B=True, compute_S=True)
                _CACHED_ORBIT_KEY = orbit_key
            finally:
                np.random.set_state(rng_state)

        orb = _CACHED_ORBIT
        np.random.seed(config["seed"])

        # BC1 satellite - MTQ only, no reaction wheels
        real_sat = create_beavercube1_cubesat(estimated=False)

        # State is just [w0, q0] - no RW momentum
        x0 = np.concatenate([config["w0"], config["q0"]])

        # Use same optimized settings as 3+1 (auto-detects MTQ-only)
        planner_settings = create_optimized_planner_settings(
            real_sat, duration=tf, dt_planning=dt_planning, tuning="fast_slew"
        )
        
        # Use defaults from create_optimized_planner_settings - no overrides
        # Strong path length cost to discourage going the long way
        planner_settings.cost_main.ang_vel_mag = planner_settings.cost_main.angle * 1.0
        planner_settings.cost_second.ang_vel_mag = planner_settings.cost_second.angle * 1.0
        
        # Finer timestep for MTQ - more B-field samples = more control flexibility
        planner_settings.dt_tp = 2
        
        # PD initialization instead of random - may give better starting trajectory
        planner_settings.bdot_on = 4
        
        # Fewer iterations for testing
        planner_settings.pass1.convergence.max_inner_iter = 50
        planner_settings.pass1.convergence.max_outer_iter = 5
        planner_settings.verbosity = False

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
                    
                    # Save figure at key iterations
                    iter_count[0] += 1
                    save_iters = [1, 5, 10, 20, 34, 50, 70, 100, 150, 200, 250, 300]
                    if iter_count[0] in save_iters:
                        import matplotlib
                        matplotlib.use('Agg')
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
                            axes[1, 1].plot(ctrl_times, Uset[i, :], label=f'MTQ{i}')
                        axes[1, 1].set_xlabel('Time (s)')
                        axes[1, 1].set_ylabel('Control (A·m²)')
                        axes[1, 1].set_title('MTQ Dipole Moments')
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
                    skip_pass2=True  # Skip pass2 for faster iteration during tuning
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
                        axes[1, 1].plot(ctrl_times, Uset[i, :], label=f'MTQ{i}')
                    axes[1, 1].set_xlabel('Time (s)')
                    axes[1, 1].set_ylabel('Control (A·m²)')
                    axes[1, 1].set_title('MTQ Dipole Moments')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True)
                    
                    plt.tight_layout()
                    plt.savefig('/tmp/planner_diagnostic.png', dpi=150)
                    print(f"Saved diagnostic plot to /tmp/planner_diagnostic.png")
                    plt.close(fig)
            else:
                # C++ planner only supports verbose
                traj = controller.calculate_trajectory(
                    t_start=0.22, duration=tf, x_0=x0, os_0=os0, goals=goals, verbose=verbose
                )
            controller.set_active_trajectory(traj)
            traj_valid = True
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"run_id": run_id, "config": config, "error": str(e), "traj_valid": False}

        time_hist = np.zeros(N)
        state_hist = np.zeros((N, len(x0)))
        u_hist = np.zeros((N, len(real_sat.actuators)))
        q_goal_hist = np.zeros((N, 4))

        x = x0.copy()
        t = 0
        sec2cent = TimeConstants.sec2cent

        for i in range(N - 1):  # N-1 to avoid orbit time overflow on last step
            if i % 10 == 0:
                update_worker_progress(slot_id, run_id, i, N)
            if verbose and i % 50 == 0:
                print(f"  Sim step {i}/{N} (t={t:.1f}s)")

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

        # Fill last entry
        time_hist[N-1] = t
        state_hist[N-1, :] = x
        u_hist[N-1, :] = u_hist[N-2, :]
        q_goal_hist[N-1, :] = config["q_goal"]

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
    
    # Random rotation (typically ~90-120°, rarely near 180°)
    q_goal = normalize(rng.standard_normal(4))
    # Ensure shortest path (same hemisphere)
    if np.dot(q0, q_goal) < 0:
        q_goal = -q_goal
    
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
        # No h0 for MTQ-only
        "tracking_mode": TRACKING_MODE,
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
    args = parser.parse_args()
    
    TEST_MODE = args.test
    if args.num_runs is not None:
        NUM_RUNS = args.num_runs
    
    if TEST_MODE:
        # Single run test mode - no multiprocessing, with visualization
        test_seed = args.seed
        print(f"=== TEST MODE: Single run (seed={test_seed}), no multiprocessing, with planner visualization ===")
        config = generate_mc_config(test_seed)
        config["verbose"] = False  # Disable verbose text output
        config["visualize"] = True  # Enable live planner visualization
        config["save_viz"] = True  # Save visualization figures
        result = run_single_sim(config)
        full_results = [result]
        valid = [r for r in full_results if r and r.get("traj_valid", False)]
        if valid:
            print(f"Test run completed successfully")
            # Plot results
            plot_target_tracking_mc(full_results=valid, title="Test Run - 3MTQ+0RW")
            plot_convergence_histogram_mc(full_results=valid, title="Test Run - 3MTQ+0RW")
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
    else:
        results = load_data(f"{OUTPUT_DIR}/{output_name}")
        full_results = results[0] if isinstance(results, tuple) else results
