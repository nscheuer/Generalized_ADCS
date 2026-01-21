#!/usr/bin/env python3
"""
Debug script for TinyMPC-based Plan and Track controllers.

Tests both the Pure Python and C++ TinyMPC implementations for trajectory tracking.
Compares tracking performance against the standard TVLQR controller.

Usage:
    python debug_plan_and_track_tinympc.py [--cpp] [--compare] [--duration SECONDS]

Options:
    --cpp       Also test C++ TinyMPC (requires rebuilt pytinympc module)
    --compare   Run TVLQR comparison
    --duration  Simulation duration in seconds (default: 100)
"""
from __future__ import annotations

import argparse
import time as time_module
import os as os_module
import sys

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Add parent paths
sys.path.insert(0, os_module.path.dirname(os_module.path.dirname(os_module.path.dirname(
    os_module.path.dirname(os_module.path.abspath(__file__))))))

from ADCS.helpers.math_helpers import rot_mat, quat_diff, quat_to_vec3, normalize
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.orbit import Orbit
from ADCS.CONOPS.goallist import GoalList
from ADCS.CONOPS.goals import ECI_Goal
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.controller import Plan_and_Track_LQR
from ADCS.controller.helpers import PlannerSettings, CostWeights, Trajectory
from ADCS.controller.helpers.tinympc_settings import TinyMPCSettings

# Try to import TinyMPC controllers
try:
    from ADCS.controller import Plan_and_Track_TinyMPC_Py
    _HAS_PY_TINYMPC = True
except ImportError as e:
    print(f"Warning: Could not import Plan_and_Track_TinyMPC_Py: {e}")
    _HAS_PY_TINYMPC = False

try:
    from ADCS.controller import Plan_and_Track_TinyMPC_Cpp
    _HAS_CPP_TINYMPC = True
except ImportError as e:
    print(f"Warning: Could not import Plan_and_Track_TinyMPC_Cpp: {e}")
    _HAS_CPP_TINYMPC = False


# ===========================================================================
# Test Configuration
# ===========================================================================

def create_test_planner_settings(est_sat, dt: float = 1.0, dt_planning: float = 50.0) -> PlannerSettings:
    """Create planner settings for testing.

    Uses settings tuned from debug_plan_and_track_bc2.py which achieved good tracking.

    CRITICAL: TVLQR/TinyMPC gains are computed from cost_tvlqr settings.
    If control_mult is too low, gains will be too aggressive and cause instability.
    cost_tvlqr.control_mult should be ~1e4 (much higher than cost_main.control_mult).

    Args:
        est_sat: Estimated satellite model
        dt: Control timestep (dt_tvlqr)
        dt_planning: Planning timestep (dt_tp) - use 50 for ALTRO convergence
    """
    settings = PlannerSettings(
        est_sat=est_sat,
        bdot_on=2,  # Smart bdot initial guess
        dt_tp=dt_planning,
        dt_tvlqr=dt,
    )

    # ALTRO planning cost function (for trajectory optimization)
    # Match BC2 tuning: higher angle costs for early convergence
    settings.cost_main.ang_cost_func_type = 0  # Linear - best convergence
    settings.cost_main.angle = 1e7       # Higher than before for early convergence
    settings.cost_main.angle_N = 1e8
    settings.cost_main.ang_vel = 1e4
    settings.cost_main.ang_vel_N = 1e5
    settings.cost_main.control_mult = 1e-2  # Low for ALTRO (encourages control usage)
    settings.cost_main.use_raw_control_cost = True
    settings.cost_main.use_full_cost_hessian = True  # Faster convergence

    # Hessian settings for faster convergence (from BC2)
    settings.pass1.regularization.use_dynamics_hess = 1
    settings.pass1.convergence.max_outer_iter = 8
    settings.pass1.convergence.max_inner_iter = 50
    settings.pass2.convergence.max_outer_iter = 5
    settings.pass2.convergence.max_inner_iter = 20
    settings.init_traj.bdot_gain = 500
    settings.pass1.aug_lag.penalty_init = 100

    # TVLQR/TinyMPC tracking cost function
    # CRITICAL: control_mult must be MUCH HIGHER than cost_main to avoid aggressive gains
    settings.cost_tvlqr.ang_cost_func_type = 0
    settings.cost_tvlqr.angle = 1e6
    settings.cost_tvlqr.angle_N = 1e7
    settings.cost_tvlqr.ang_vel = 1e4
    settings.cost_tvlqr.ang_vel_N = 1e5
    settings.cost_tvlqr.control_mult = 1e4  # HIGH for TVLQR (prevents aggressive gains)
    settings.cost_tvlqr.use_raw_control_cost = True

    return settings


def create_test_tinympc_settings() -> TinyMPCSettings:
    """Create TinyMPC settings for testing.

    Tuned for faster convergence:
    - Higher rho (100) for faster ADMM convergence
    - Looser tolerances (1e-3) since tracking doesn't need extreme precision
    - More iterations (100) to ensure convergence
    - Shorter horizon (5) for faster computation
    """
    return TinyMPCSettings(
        # ADMM solver settings - tuned for convergence
        max_iter=100,
        abs_tol=1e-3,       # Looser tolerance for tracking
        rel_tol=1e-3,
        rho=100.0,          # Higher rho for faster ADMM convergence
        rho_min=10.0,
        rho_max=1000.0,
        adaptive_rho=True,
        check_interval=5,
        # MPC horizon - shorter for speed
        track_horizon=5,
        track_dt=1.0,
        # Re-planning thresholds
        replan_enabled=True,
        replan_attitude_threshold=np.deg2rad(15.0),  # 15 degrees
        replan_angvel_threshold=np.deg2rad(10.0),    # 10 deg/s
        replan_min_interval=30.0,  # seconds
        # Debug
        verbose=1,
    )


# ===========================================================================
# Simulation Functions
# ===========================================================================

def run_simulation(
    controller,
    sat,
    orbit,
    goals,
    x_0: NDArray,
    t_start_cent: float,
    duration_sec: float,
    dt_ctrl: float = 1.0,
    controller_name: str = "Controller"
) -> dict:
    """
    Run closed-loop simulation with the given controller.

    Uses the satellite's dynamics_for_solver method which properly handles
    orbital state interpolation and disturbances.

    Returns dict with time, states, controls, and metrics.
    """
    print(f"\n{'='*60}")
    print(f"Running simulation with {controller_name}")
    print(f"{'='*60}")

    # Storage
    time_hist = [0.0]
    state_hist = [x_0.copy()]
    control_hist = []
    tracking_error_hist = []
    solve_time_hist = []
    B_eci_hist = []

    n_steps = int(duration_sec / dt_ctrl)

    x_current = x_0.copy()
    t_current_sec = 0.0

    for step in range(n_steps):
        t_cent = t_start_cent + t_current_sec * TimeConstants.sec2cent

        # Get orbital state
        os_t = orbit.get_os(t_cent)
        B_eci_hist.append(os_t.B.copy())

        # Normalize quaternion
        x_current[3:7] = normalize(x_current[3:7])

        # Get goal vector from GoalList
        goal_vec_eci, _ = goals.to_ref(t=t_cent, os0=os_t)

        # Get sensor readings for controller
        sens = sat.sensor_readings(x=x_current, os=os_t)

        # Compute control with timing
        t_solve_start = time_module.perf_counter()
        try:
            u = controller.find_u(x_current, sens, sat, os_t, goal_vec_eci)
        except Exception as e:
            print(f"Step {step}: Control error: {e}")
            u = np.zeros(controller.ctrl_dim)
        t_solve_end = time_module.perf_counter()

        solve_time_ms = (t_solve_end - t_solve_start) * 1000
        solve_time_hist.append(solve_time_ms)
        control_hist.append(u.copy())

        # Compute tracking error using pointing error (more robust than quaternion error)
        # This measures how far the body boresight is from the goal direction
        if hasattr(controller, 'active_trajectory') and controller.active_trajectory is not None:
            try:
                # Compute pointing error: angle between body Z-axis and goal vector
                q = x_current[3:7]
                w, x_q, y_q, z_q = q
                R = np.array([
                    [1 - 2*(y_q**2 + z_q**2), 2*(x_q*y_q - z_q*w), 2*(x_q*z_q + y_q*w)],
                    [2*(x_q*y_q + z_q*w), 1 - 2*(x_q**2 + z_q**2), 2*(y_q*z_q - x_q*w)],
                    [2*(x_q*z_q - y_q*w), 2*(y_q*z_q + x_q*w), 1 - 2*(x_q**2 + y_q**2)]
                ])
                body_boresight = np.array([0, 0, 1])
                eci_boresight = R @ body_boresight
                goal_vec_eci_t, _ = goals.to_ref(t=t_cent, os0=os_t)
                att_err_rad = np.arccos(np.clip(np.dot(eci_boresight, goal_vec_eci_t), -1, 1))
                tracking_error_hist.append(np.rad2deg(att_err_rad))
            except Exception:
                tracking_error_hist.append(np.nan)
        else:
            tracking_error_hist.append(np.nan)

        # Get next orbital state for dynamics interpolation
        t_next_cent = t_start_cent + (t_current_sec + dt_ctrl) * TimeConstants.sec2cent
        os_next = orbit.get_os(t_next_cent)

        # Integrate one step using satellite's dynamics
        sol = solve_ivp(
            fun=sat.dynamics_for_solver,
            t_span=(0, dt_ctrl),
            y0=x_current,
            method='RK45',
            args=(u, os_t, os_next),
            rtol=1e-7,
            atol=1e-7,
        )

        if sol.success:
            x_current = sol.y[:, -1]
            x_current[3:7] = normalize(x_current[3:7])
        else:
            print(f"Step {step}: Integration failed")
            break

        t_current_sec += dt_ctrl
        time_hist.append(t_current_sec)
        state_hist.append(x_current.copy())

        # Progress
        if step % 20 == 0:
            avg_solve = np.mean(solve_time_hist[-20:]) if len(solve_time_hist) >= 20 else np.mean(solve_time_hist)
            print(f"  Step {step}/{n_steps}: t={t_current_sec:.1f}s, "
                  f"solve={avg_solve:.2f}ms, track_err={tracking_error_hist[-1]:.2f}deg")

    # Convert to arrays
    time_arr = np.array(time_hist)
    state_arr = np.array(state_hist)
    control_arr = np.array(control_hist)
    tracking_err_arr = np.array(tracking_error_hist)
    solve_time_arr = np.array(solve_time_hist)
    B_eci_arr = np.array(B_eci_hist)

    # Compute final metrics
    final_att_err = tracking_err_arr[-1] if len(tracking_err_arr) > 0 else np.nan
    avg_att_err = np.nanmean(tracking_err_arr)
    max_att_err = np.nanmax(tracking_err_arr)
    avg_solve_time = np.mean(solve_time_arr)

    print(f"\nResults for {controller_name}:")
    print(f"  Final attitude error: {final_att_err:.3f} deg")
    print(f"  Average attitude error: {avg_att_err:.3f} deg")
    print(f"  Max attitude error: {max_att_err:.3f} deg")
    print(f"  Average solve time: {avg_solve_time:.2f} ms")

    return {
        'name': controller_name,
        'time': time_arr,
        'state': state_arr,
        'control': control_arr,
        'tracking_error': tracking_err_arr,
        'solve_time': solve_time_arr,
        'B_eci': B_eci_arr,
    }


# ===========================================================================
# Plotting Functions
# ===========================================================================

def plot_comparison(results_list: list[dict], save_dir: str = None):
    """Plot comparison of multiple controller results."""

    n_controllers = len(results_list)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_controllers, 3)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Tracking error
    ax = axes[0, 0]
    for i, results in enumerate(results_list):
        ax.plot(results['time'][:-1], results['tracking_error'],
                label=results['name'], color=colors[i], alpha=0.8)
    ax.set_ylabel('Attitude Error [deg]')
    ax.set_title('Tracking Error vs Time')
    ax.legend()
    ax.grid(True)
    ax.set_xlim([0, results_list[0]['time'][-1]])

    # Angular velocity magnitude
    ax = axes[0, 1]
    for i, results in enumerate(results_list):
        w = results['state'][:, :3]
        w_mag = np.rad2deg(np.linalg.norm(w, axis=1))
        ax.plot(results['time'], w_mag, label=results['name'], color=colors[i], alpha=0.8)
    ax.set_ylabel('|ω| [deg/s]')
    ax.set_title('Angular Velocity Magnitude')
    ax.legend()
    ax.grid(True)

    # Solve time
    ax = axes[1, 0]
    for i, results in enumerate(results_list):
        ax.plot(results['time'][:-1], results['solve_time'],
                label=results['name'], color=colors[i], alpha=0.8)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Solve Time [ms]')
    ax.set_title('Control Solve Time')
    ax.legend()
    ax.grid(True)

    # Control effort (MTQ)
    ax = axes[1, 1]
    for i, results in enumerate(results_list):
        u = results['control']
        mtq_mag = np.linalg.norm(u[:, :3], axis=1)
        ax.plot(results['time'][:-1], mtq_mag, label=results['name'], color=colors[i], alpha=0.8)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('MTQ Magnitude [A·m²]')
    ax.set_title('Control Effort (MTQ)')
    ax.legend()
    ax.grid(True)

    fig.suptitle('TinyMPC vs TVLQR Tracking Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout()

    if save_dir:
        os_module.makedirs(save_dir, exist_ok=True)
        path = os_module.path.join(save_dir, "tinympc_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")

    return fig


def plot_detailed_results(results: dict, save_dir: str = None):
    """Plot detailed results for a single controller."""

    time = results['time']
    state = results['state']
    control = results['control']
    name = results['name']

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Angular velocity components
    ax = axes[0, 0]
    w = state[:, :3]
    ax.plot(time, np.rad2deg(w[:, 0]), label='ωx', alpha=0.8)
    ax.plot(time, np.rad2deg(w[:, 1]), label='ωy', alpha=0.8)
    ax.plot(time, np.rad2deg(w[:, 2]), label='ωz', alpha=0.8)
    ax.set_ylabel('Angular Velocity [deg/s]')
    ax.set_title(f'{name}: Angular Velocity')
    ax.legend()
    ax.grid(True)

    # Quaternion components
    ax = axes[0, 1]
    q = state[:, 3:7]
    ax.plot(time, q[:, 0], label='q0 (scalar)', alpha=0.8)
    ax.plot(time, q[:, 1], label='q1', alpha=0.8)
    ax.plot(time, q[:, 2], label='q2', alpha=0.8)
    ax.plot(time, q[:, 3], label='q3', alpha=0.8)
    ax.set_ylabel('Quaternion')
    ax.set_title(f'{name}: Quaternion')
    ax.legend()
    ax.grid(True)

    # Tracking error
    ax = axes[0, 2]
    if 'tracking_error' in results and len(results['tracking_error']) > 0:
        ax.plot(time[:-1], results['tracking_error'], 'b-', alpha=0.8)
        ax.axhline(y=np.nanmean(results['tracking_error']), color='r', linestyle='--',
                   label=f'Mean: {np.nanmean(results["tracking_error"]):.2f}°')
    ax.set_ylabel('Attitude Error [deg]')
    ax.set_title(f'{name}: Tracking Error')
    ax.legend()
    ax.grid(True)

    # MTQ commands
    ax = axes[1, 0]
    ax.plot(time[:-1], control[:, 0], label='mx', alpha=0.8)
    ax.plot(time[:-1], control[:, 1], label='my', alpha=0.8)
    ax.plot(time[:-1], control[:, 2], label='mz', alpha=0.8)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('MTQ Dipole [A·m²]')
    ax.set_title(f'{name}: MTQ Commands')
    ax.legend()
    ax.grid(True)

    # RW commands (if present)
    ax = axes[1, 1]
    if control.shape[1] > 3:
        ax.plot(time[:-1], control[:, 3], 'b-', label='RW torque', alpha=0.8)
        ax.set_ylabel('RW Torque [N·m]')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No RW', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Time [s]')
    ax.set_title(f'{name}: Reaction Wheel')
    ax.grid(True)

    # RW momentum (if present)
    ax = axes[1, 2]
    if state.shape[1] > 7:
        ax.plot(time, state[:, 7], 'r-', alpha=0.8)
        ax.set_ylabel('RW Momentum [N·m·s]')
    else:
        ax.text(0.5, 0.5, 'No RW state', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Time [s]')
    ax.set_title(f'{name}: RW Momentum')
    ax.grid(True)

    fig.tight_layout()

    if save_dir:
        os_module.makedirs(save_dir, exist_ok=True)
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
        path = os_module.path.join(save_dir, f"{safe_name}_detailed.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")

    return fig


# ===========================================================================
# Main Test Functions
# ===========================================================================

def test_tinympc_python(sat, orbit, goals, x_0, t_start_cent, duration_sec, planner_settings, tinympc_settings):
    """Test the pure Python TinyMPC controller."""

    if not _HAS_PY_TINYMPC:
        print("Skipping Python TinyMPC test - import failed")
        return None

    print("\n" + "="*60)
    print("Creating Plan_and_Track_TinyMPC_Py controller...")
    print("="*60)

    # Create controller
    controller = Plan_and_Track_TinyMPC_Py(
        est_sat=sat,
        planner_settings=planner_settings,
        tinympc_settings=tinympc_settings
    )

    # Plan trajectory
    os_0 = orbit.get_os(t_start_cent)
    print(f"\nPlanning trajectory for {duration_sec}s...")

    t_plan_start = time_module.perf_counter()
    traj = controller.calculate_trajectory(
        t_start=t_start_cent,
        duration=duration_sec,
        x_0=x_0,
        os_0=os_0,
        goals=goals,
        verbose=True
    )
    t_plan_end = time_module.perf_counter()
    print(f"Trajectory planning took {t_plan_end - t_plan_start:.2f}s")

    controller.active_trajectory = traj

    # Run simulation
    results = run_simulation(
        controller=controller,
        sat=sat,
        orbit=orbit,
        goals=goals,
        x_0=x_0,
        t_start_cent=t_start_cent,
        duration_sec=duration_sec,
        dt_ctrl=planner_settings.dt_tvlqr,
        controller_name="TinyMPC (Python)"
    )

    return results


def test_tinympc_cpp(sat, orbit, goals, x_0, t_start_cent, duration_sec, planner_settings, tinympc_settings):
    """Test the C++ TinyMPC controller."""

    if not _HAS_CPP_TINYMPC:
        print("Skipping C++ TinyMPC test - pytinympc module not available")
        return None

    print("\n" + "="*60)
    print("Creating Plan_and_Track_TinyMPC_Cpp controller...")
    print("="*60)

    try:
        # Create controller
        controller = Plan_and_Track_TinyMPC_Cpp(
            est_sat=sat,
            planner_settings=planner_settings,
            tinympc_settings=tinympc_settings
        )
    except ImportError as e:
        print(f"Could not create C++ TinyMPC controller: {e}")
        return None

    # Plan trajectory
    os_0 = orbit.get_os(t_start_cent)
    print(f"\nPlanning trajectory for {duration_sec}s...")

    t_plan_start = time_module.perf_counter()
    traj = controller.calculate_trajectory(
        t_start=t_start_cent,
        duration=duration_sec,
        x_0=x_0,
        os_0=os_0,
        goals=goals,
        verbose=True
    )
    t_plan_end = time_module.perf_counter()
    print(f"Trajectory planning took {t_plan_end - t_plan_start:.2f}s")

    controller.active_trajectory = traj

    # Run simulation
    results = run_simulation(
        controller=controller,
        sat=sat,
        orbit=orbit,
        goals=goals,
        x_0=x_0,
        t_start_cent=t_start_cent,
        duration_sec=duration_sec,
        dt_ctrl=planner_settings.dt_tvlqr,
        controller_name="TinyMPC (C++)"
    )

    return results


def test_tvlqr_baseline(sat, orbit, goals, x_0, t_start_cent, duration_sec, planner_settings):
    """Test the baseline TVLQR controller for comparison."""

    print("\n" + "="*60)
    print("Creating Plan_and_Track_LQR controller (baseline)...")
    print("="*60)

    # Create controller
    controller = Plan_and_Track_LQR(
        est_sat=sat,
        planner_settings=planner_settings
    )

    # Plan trajectory
    os_0 = orbit.get_os(t_start_cent)
    print(f"\nPlanning trajectory for {duration_sec}s...")

    t_plan_start = time_module.perf_counter()
    traj = controller.calculate_trajectory(
        t_start=t_start_cent,
        duration=duration_sec,
        x_0=x_0,
        os_0=os_0,
        goals=goals,
        verbose=True
    )
    t_plan_end = time_module.perf_counter()
    print(f"Trajectory planning took {t_plan_end - t_plan_start:.2f}s")

    controller.active_trajectory = traj

    # Run simulation
    results = run_simulation(
        controller=controller,
        sat=sat,
        orbit=orbit,
        goals=goals,
        x_0=x_0,
        t_start_cent=t_start_cent,
        duration_sec=duration_sec,
        dt_ctrl=planner_settings.dt_tvlqr,
        controller_name="TVLQR (Baseline)"
    )

    return results


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Test TinyMPC Plan and Track controllers")
    parser.add_argument('--cpp', action='store_true', help='Also test C++ TinyMPC')
    parser.add_argument('--compare', action='store_true', help='Run TVLQR comparison')
    parser.add_argument('--duration', type=float, default=500.0, help='Simulation duration (seconds)')
    parser.add_argument('--save', type=str, default=None, help='Directory to save plots')
    args = parser.parse_args()

    print("="*60)
    print("TinyMPC Plan and Track Controller Test")
    print("="*60)

    # Use deterministic seed like BC2 test
    np.random.seed(42)

    # Default save directory
    save_dir = args.save
    if save_dir is None:
        save_dir = os_module.path.join(os_module.path.dirname(__file__), "debug_plots")

    # Create satellite
    print("\nCreating BeaverCube2 satellite...")
    sat = create_beavercube2_cubesat()

    # Create orbit (ISS-like)
    print("Setting up orbit...")
    ephem = Ephemeris()

    # Start time
    t_start_cent = 0.22 - 1 * TimeConstants.sec2cent

    # ISS-like position/velocity in ECI
    R = 7000e3 * np.array([0, np.sqrt(2) / 2, np.sqrt(2) / 2])  # meters
    V = np.array([8000, 0, 0])  # m/s

    os_0 = Orbital_State(ephem=ephem, J2000=t_start_cent, R=R, V=V)
    end_time = t_start_cent + (args.duration + 50) * TimeConstants.sec2cent
    orbit = Orbit(os0=os_0, end_time=end_time, dt=1.0, use_J2=True, fast=True)

    # Create goal (nadir pointing)
    print("Setting up goals...")
    # Point body Z toward nadir (ECI -Z is roughly nadir for this orbit)
    goal_vec = normalize(np.array([0.0, 0.0, 1.0]))
    goal = ECI_Goal(goal_vec)
    goals = GoalList({t_start_cent: goal})

    # Initial state - use random initial conditions like BC2 (with deterministic seed)
    n_rw = 1  # BeaverCube2 has 1 reaction wheel
    state_dim = 7 + n_rw
    w0 = np.random.randn(3) * 0.01  # ~0.5 deg/s typical
    q0 = normalize(np.random.randn(4))
    h0 = np.array([0.0])  # RW momentum
    x_0 = np.concatenate([w0, q0, h0])
    print(f"Initial angular velocity: {np.rad2deg(np.linalg.norm(w0)):.2f} deg/s")
    print(f"Initial quaternion: {q0}")

    # Settings
    planner_settings = create_test_planner_settings(sat, dt=1.0, dt_planning=50.0)
    tinympc_settings = create_test_tinympc_settings()

    duration_sec = args.duration

    # Collect results
    all_results = []

    # Test Python TinyMPC
    results_py = test_tinympc_python(
        sat, orbit, goals, x_0, t_start_cent, duration_sec,
        planner_settings, tinympc_settings
    )
    if results_py is not None:
        all_results.append(results_py)

    # Test C++ TinyMPC (optional)
    if args.cpp:
        results_cpp = test_tinympc_cpp(
            sat, orbit, goals, x_0, t_start_cent, duration_sec,
            planner_settings, tinympc_settings
        )
        if results_cpp is not None:
            all_results.append(results_cpp)

    # Test TVLQR baseline (optional)
    if args.compare:
        results_tvlqr = test_tvlqr_baseline(
            sat, orbit, goals, x_0, t_start_cent, duration_sec,
            planner_settings
        )
        if results_tvlqr is not None:
            all_results.append(results_tvlqr)

    # Plot results
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)

    for results in all_results:
        plot_detailed_results(results, save_dir)

    if len(all_results) > 1:
        plot_comparison(all_results, save_dir)

    plt.show()

    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)


if __name__ == "__main__":
    main()
