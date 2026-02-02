"""
ALTRO BC2 Tuning Script

This script is designed for iterative tuning of ALTRO planner settings.
It saves plots and raw data to disk for analysis.
"""
import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Dict, Any, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
from ADCS.CONOPS.goals import Goal, ECI_Goal, Coordinate_Goal, No_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.helpers import PlannerSettings, Trajectory
from ADCS.controller.helpers.planner_subsettings import CostWeights
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.helpers.math_helpers import random_n_unit_vec, normalize

# Output directory for tuning runs
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "tuning_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_tracking_error(state_hist: np.ndarray, boresight_hist: np.ndarray, 
                           body_boresight: np.ndarray = np.array([0, 1, 0])) -> np.ndarray:
    """Compute pointing error in degrees at each timestep."""
    errors = np.zeros(len(state_hist))
    for i in range(len(state_hist)):
        q = state_hist[i, 3:7]
        q = normalize(q)
        
        # Rotation matrix from body to ECI
        q0, q1, q2, q3 = q
        R = np.array([
            [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
        ])
        
        # Body boresight in ECI
        body_in_eci = R @ body_boresight
        
        # Angle to target
        target = boresight_hist[i]
        if np.linalg.norm(target) > 0:
            target = normalize(target)
            dot = np.clip(np.dot(body_in_eci, target), -1, 1)
            errors[i] = np.arccos(dot) * 180 / np.pi
        else:
            errors[i] = np.nan
    return errors


def compute_angular_velocity_magnitude(state_hist: np.ndarray) -> np.ndarray:
    """Compute angular velocity magnitude in deg/s at each timestep."""
    w = state_hist[:, 0:3]
    return np.linalg.norm(w, axis=1) * 180 / np.pi


def run_altro_tuning(
    settings: Dict[str, Any],
    tf: float = 500,
    dt: float = 1,
    seed: int = 42,
    verbose: bool = False,
    run_name: str = "test",
    use_simple_ic: bool = True,  # Use simple ICs like original debug script
) -> Dict[str, Any]:
    """
    Run a single ALTRO tuning test with given settings.
    
    Returns dictionary with results and metadata.
    """
    np.random.seed(seed)
    t0 = 0
    N = int((tf - t0) / dt)
    
    # Create satellite
    real_sat = create_beavercube2_cubesat()
    
    if use_simple_ic:
        # Simple ICs from original debug_altro_bc2_3+1.py
        w0 = np.array([0, 0, 0])
        q0 = normalize(np.array([1, 0, 0, 0]))  # Identity quaternion
        h0 = np.array([0.0001])
        x0 = np.concatenate([w0, q0, h0])
        goal_vec = np.array([0, 0, 1])  # Point body Y at +Z ECI (90° slew)
        
        # Simple orbit
        ephem = Ephemeris()
        R = 7000 * np.array([0, np.sqrt(2)/2, np.sqrt(2)/2])
        V = np.array([8, 0, 0])
        os0 = Orbital_State(ephem=ephem, J2000=0.22, R=R, V=V)
    else:
        # Random ICs
        rw_h0 = np.random.uniform(-0.0001, 0.0001)
        w0 = random_n_unit_vec(3) * np.random.uniform(0.5, 2) * np.pi / 180.0
        q0 = normalize(random_n_unit_vec(4))
        h0 = np.array([rw_h0])
        x0 = np.concatenate([w0, q0, h0])
        goal_vec = normalize(random_n_unit_vec(3))
        
        ephem = Ephemeris()
        R = 7000 * normalize(random_n_unit_vec(3))
        V = np.cross(R, random_n_unit_vec(3))
        V = normalize(V) * np.sqrt(398600.4418 / np.linalg.norm(R))
        os0 = Orbital_State(ephem=ephem, J2000=0.22, R=R, V=V)
    
    end_time = 0.22 + (tf - t0) * TimeConstants.sec2cent
    orb = Orbit(os0=os0, end_time=end_time, dt=dt, use_J2=True, fast=True)
    
    # Build planner with settings
    dt_tp = settings.get("dt_tp", 10)
    planner_settings = PlannerSettings(
        est_sat=real_sat,
        bdot_on=settings.get("bdot_on", 0),
        dt_tp=dt_tp,
        dt_tvlqr=settings.get("dt_tvlqr", 1),
    )
    planner_settings.verbosity = verbose
    
    # Control weights
    planner_settings.mtq_control_weight = settings.get("mtq_control_weight", 1e3)
    planner_settings.rw_control_weight = settings.get("rw_control_weight", 1e7)
    planner_settings.rw_AM_weight = settings.get("rw_AM_weight", 1e4)
    planner_settings.rw_stic_weight = settings.get("rw_stic_weight", 1e0)
    
    # Cost weights
    planner_settings.cost_main = CostWeights(
        angle=settings.get("angle", 1e3),
        angle_N=settings.get("angle_N", 1e6),
        ang_vel=settings.get("ang_vel", 1e3),
        ang_vel_N=settings.get("ang_vel_N", 1e5),
        ang_vel_mag=settings.get("ang_vel_mag", 0.0),
        ang_vel_mag_N=settings.get("ang_vel_mag_N", 0.0),
        control_mult=settings.get("control_mult", 1.0),
        ang_cost_func_type=settings.get("ang_cost_func_type", 2),
    )
    planner_settings.cost_main.use_full_cost_hessian = settings.get("use_full_cost_hessian", True)
    
    planner_settings.cost_tvlqr = CostWeights(
        angle=settings.get("tvlqr_angle", 1e2),
        angle_N=settings.get("tvlqr_angle_N", 1e3),
        ang_vel=settings.get("tvlqr_ang_vel", 1e6),
        ang_vel_N=settings.get("tvlqr_ang_vel_N", 1e8),
        ang_vel_mag=0.0,
        ang_vel_mag_N=0.0,
        control_mult=1.0,
        ang_cost_func_type=2,
    )
    
    # Convergence settings
    planner_settings.pass1.convergence.max_outer_iter = settings.get("max_outer_iter", 8)
    planner_settings.pass1.convergence.max_inner_iter = settings.get("max_inner_iter", 40)
    planner_settings.pass2.convergence.max_outer_iter = settings.get("pass2_max_outer", 5)
    planner_settings.pass2.convergence.max_inner_iter = settings.get("pass2_max_inner", 15)
    
    # Aug lag settings
    planner_settings.pass1.aug_lag.penalty_init = settings.get("penalty_init", 100)
    
    # Regularization / Hessian settings
    planner_settings.pass1.regularization.use_dynamics_hess = settings.get("use_dynamics_hess", 1)
    planner_settings.pass1.regularization.use_constraint_hess = settings.get("use_constraint_hess", 0)
    
    controller = Plan_and_Track_LQR(est_sat=real_sat, planner_settings=planner_settings)
    
    # Goal
    goals = GoalList({0.22: ECI_Goal(goal_vec)})
    
    # Compute initial error
    body_boresight = np.array([0, 1, 0])
    q_init = normalize(x0[3:7])
    q0_, q1_, q2_, q3_ = q_init
    R_init = np.array([
        [1 - 2*(q2_**2 + q3_**2), 2*(q1_*q2_ - q0_*q3_), 2*(q1_*q3_ + q0_*q2_)],
        [2*(q1_*q2_ + q0_*q3_), 1 - 2*(q1_**2 + q3_**2), 2*(q2_*q3_ - q0_*q1_)],
        [2*(q1_*q3_ - q0_*q2_), 2*(q2_*q3_ + q0_*q1_), 1 - 2*(q1_**2 + q2_**2)]
    ])
    body_in_eci_init = R_init @ body_boresight
    initial_error = np.arccos(np.clip(np.dot(body_in_eci_init, goal_vec), -1, 1)) * 180 / np.pi
    
    # Plan trajectory
    t_plan_start = time.perf_counter()
    try:
        traj: Trajectory = controller.calculate_trajectory(
            t_start=0.22,
            duration=tf - t0,
            x_0=x0,
            os_0=os0,
            goals=goals,
            verbose=verbose
        )
        controller.set_active_trajectory(traj)
        traj_valid = True
        plan_error = None
    except Exception as e:
        traj_valid = False
        plan_error = str(e)
        traj = None
    
    t_plan_end = time.perf_counter()
    plan_time = t_plan_end - t_plan_start
    
    if not traj_valid:
        return {
            "run_name": run_name,
            "settings": settings,
            "seed": seed,
            "tf": tf,
            "dt": dt,
            "traj_valid": False,
            "error": plan_error,
            "plan_time": plan_time,
            "initial_error": initial_error,
        }
    
    # Arrays for results
    time_hist = np.zeros(N)
    state_hist_traj = traj.states.T  # Planned trajectory
    u_hist_traj = traj.controls.T
    time_hist_traj = (traj.times - 0.22) / TimeConstants.sec2cent
    
    state_hist_sim = np.zeros((N, len(x0)))
    u_hist_sim = np.zeros((N, len(real_sat.actuators)))
    boresight_hist = np.zeros((N, 4))
    
    # Simulate with TVLQR tracking
    x = x0.copy()
    t = t0
    
    for i in range(N):
        J2000 = 0.22 + t * TimeConstants.sec2cent
        os = orb.get_os(J2000=J2000)
        
        sens = real_sat.sensor_readings(x=x, os=os)
        u = controller.find_u(x_hat=x, sens=sens, est_sat=real_sat, os_hat=os)
        
        time_hist[i] = t
        state_hist_sim[i, :] = x
        u_hist_sim[i, :] = u
        eci_goal, _ = goals.to_ref(t=J2000, os0=os)
        boresight_hist[i, :] = eci_goal
        
        t += dt
        prev_os = os.copy()
        os_next = orb.get_os(0.22 + t * TimeConstants.sec2cent)
        
        out = solve_ivp(
            fun=real_sat.dynamics_for_solver, 
            t_span=(0, dt), 
            y0=x, 
            method="RK45", 
            args=(u, prev_os, os_next), 
            rtol=1e-7, 
            atol=1e-7
        )
        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])
    
    # Compute metrics
    errors_sim = compute_tracking_error(state_hist_sim, boresight_hist)
    w_mag_sim = compute_angular_velocity_magnitude(state_hist_sim)
    
    # Also compute for planned trajectory
    boresight_hist_traj = np.zeros((len(traj.times), 3))
    for i, t_j2000 in enumerate(traj.times):
        os = orb.get_os(J2000=t_j2000)
        eci_goal, _ = goals.to_ref(t=t_j2000, os0=os)
        boresight_hist_traj[i, :] = eci_goal
    errors_traj = compute_tracking_error(state_hist_traj, boresight_hist_traj)
    w_mag_traj = compute_angular_velocity_magnitude(state_hist_traj)
    
    # Convergence metrics
    final_error_sim = errors_sim[-1] if not np.isnan(errors_sim[-1]) else 999
    final_error_traj = errors_traj[-1] if not np.isnan(errors_traj[-1]) else 999
    final_w_sim = w_mag_sim[-1]
    final_w_traj = w_mag_traj[-1]
    
    # Time to settle to < 1 degree (and stay there)
    settle_time_sim = tf
    for i, err in enumerate(errors_sim):
        if err < 1.0:
            if all(errors_sim[i:] < 1.0):
                settle_time_sim = time_hist[i]
                break
    
    settle_time_traj = tf
    for i, err in enumerate(errors_traj):
        if err < 1.0:
            if all(errors_traj[i:] < 1.0):
                settle_time_traj = time_hist_traj[i]
                break
    
    # Time to < 0.5 degree
    settle_05_sim = tf
    for i, err in enumerate(errors_sim):
        if err < 0.5:
            if all(errors_sim[i:] < 0.5):
                settle_05_sim = time_hist[i]
                break
                
    settle_05_traj = tf
    for i, err in enumerate(errors_traj):
        if err < 0.5:
            if all(errors_traj[i:] < 0.5):
                settle_05_traj = time_hist_traj[i]
                break
    
    # Check for oscillation (max error after reaching < 5 deg)
    max_error_after_converge_traj = 0
    converged = False
    for i, err in enumerate(errors_traj):
        if err < 5.0:
            converged = True
        if converged:
            max_error_after_converge_traj = max(max_error_after_converge_traj, err)
    
    return {
        "run_name": run_name,
        "settings": settings,
        "seed": seed,
        "tf": tf,
        "dt": dt,
        "traj_valid": True,
        "plan_time": plan_time,
        "rtf": plan_time / tf,
        # Raw data
        "time_sim": time_hist,
        "state_sim": state_hist_sim,
        "u_sim": u_hist_sim,
        "errors_sim": errors_sim,
        "w_mag_sim": w_mag_sim,
        "time_traj": time_hist_traj,
        "state_traj": state_hist_traj,
        "u_traj": u_hist_traj,
        "errors_traj": errors_traj,
        "w_mag_traj": w_mag_traj,
        "boresight": boresight_hist,
        # Metrics
        "initial_error": initial_error,
        "final_error_sim": final_error_sim,
        "final_error_traj": final_error_traj,
        "final_w_sim": final_w_sim,
        "final_w_traj": final_w_traj,
        "settle_time_sim": settle_time_sim,
        "settle_time_traj": settle_time_traj,
        "settle_05_sim": settle_05_sim,
        "settle_05_traj": settle_05_traj,
        "max_error_after_converge_traj": max_error_after_converge_traj,
        # Initial conditions
        "x0": x0,
        "goal_vec": goal_vec,
    }


def save_results(results: Dict[str, Any], run_name: str):
    """Save results to disk."""
    run_dir = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save numpy arrays
    if results.get("traj_valid", False):
        np.savez(
            os.path.join(run_dir, "data.npz"),
            time_sim=results["time_sim"],
            state_sim=results["state_sim"],
            u_sim=results["u_sim"],
            errors_sim=results["errors_sim"],
            w_mag_sim=results["w_mag_sim"],
            time_traj=results["time_traj"],
            state_traj=results["state_traj"],
            u_traj=results["u_traj"],
            errors_traj=results["errors_traj"],
            w_mag_traj=results["w_mag_traj"],
            boresight=results["boresight"],
            x0=results["x0"],
            goal_vec=results["goal_vec"],
        )
    
    # Save metadata as JSON
    metadata = {
        "run_name": results["run_name"],
        "settings": results["settings"],
        "seed": results["seed"],
        "tf": results["tf"],
        "dt": results["dt"],
        "traj_valid": results["traj_valid"],
        "plan_time": results.get("plan_time", None),
        "rtf": results.get("rtf", None),
        "initial_error": float(results.get("initial_error", 0)),
        "final_error_sim": float(results.get("final_error_sim", 999)),
        "final_error_traj": float(results.get("final_error_traj", 999)),
        "final_w_sim": float(results.get("final_w_sim", 999)),
        "final_w_traj": float(results.get("final_w_traj", 999)),
        "settle_time_sim": float(results.get("settle_time_sim", results["tf"])),
        "settle_time_traj": float(results.get("settle_time_traj", results["tf"])),
        "settle_05_sim": float(results.get("settle_05_sim", results["tf"])),
        "settle_05_traj": float(results.get("settle_05_traj", results["tf"])),
        "max_error_after_converge_traj": float(results.get("max_error_after_converge_traj", 0)),
        "error": results.get("error", None),
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Results saved to {run_dir}")


def plot_results(results: Dict[str, Any], run_name: str, show: bool = False):
    """Generate and save plots for the results."""
    if not results.get("traj_valid", False):
        print(f"Cannot plot - trajectory invalid: {results.get('error', 'Unknown error')}")
        return
    
    run_dir = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    time_sim = results["time_sim"]
    time_traj = results["time_traj"]
    errors_sim = results["errors_sim"]
    errors_traj = results["errors_traj"]
    w_mag_sim = results["w_mag_sim"]
    w_mag_traj = results["w_mag_traj"]
    state_sim = results["state_sim"]
    state_traj = results["state_traj"]
    u_sim = results["u_sim"]
    u_traj = results["u_traj"]
    
    # Figure 1: Pointing Error
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(time_traj, errors_traj, 'b-', label='Planned Trajectory', alpha=0.7, linewidth=2)
    ax1.plot(time_sim, errors_sim, 'r--', label='Simulated (TVLQR)', linewidth=1.5)
    ax1.axhline(y=1.0, color='g', linestyle=':', label='1° threshold')
    ax1.axhline(y=0.5, color='orange', linestyle=':', label='0.5° threshold')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Pointing Error [deg]')
    ax1.set_title(f'Pointing Error - {run_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, min(100, max(errors_sim.max(), errors_traj.max()) * 1.1)])
    
    # Add metrics text
    metrics_text = (
        f"Initial error: {results['initial_error']:.1f}°\n"
        f"Final error (traj): {results['final_error_traj']:.2f}°\n"
        f"Final error (sim): {results['final_error_sim']:.2f}°\n"
        f"Settle <1° (traj): {results['settle_time_traj']:.0f}s\n"
        f"Final ω (traj): {results['final_w_traj']:.2f}°/s\n"
        f"Plan time: {results['plan_time']:.1f}s"
    )
    ax1.text(0.98, 0.98, metrics_text, transform=ax1.transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
    
    fig1.tight_layout()
    fig1.savefig(os.path.join(run_dir, "pointing_error.png"), dpi=150)
    
    # Figure 2: Angular Velocity
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(time_traj, w_mag_traj, 'b-', label='Planned Trajectory', alpha=0.7, linewidth=2)
    ax2.plot(time_sim, w_mag_sim, 'r--', label='Simulated (TVLQR)', linewidth=1.5)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Angular Velocity Magnitude [deg/s]')
    ax2.set_title(f'Angular Velocity - {run_name}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig2.tight_layout()
    fig2.savefig(os.path.join(run_dir, "angular_velocity.png"), dpi=150)
    
    # Figure 3: Control Effort
    fig3, axes3 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # MTQ control
    ax3a = axes3[0]
    for i in range(min(3, u_traj.shape[1])):
        ax3a.plot(time_traj, u_traj[:, i]*1000, alpha=0.7, label=f'MTQ {i+1} (traj)')
    ax3a.set_ylabel('MTQ Dipole [mAm²]')
    ax3a.set_title(f'Control Effort - {run_name}')
    ax3a.legend(loc='upper right', fontsize=8)
    ax3a.grid(True, alpha=0.3)
    
    # RW control (if present)
    ax3b = axes3[1]
    if u_traj.shape[1] > 3:
        ax3b.plot(time_traj, u_traj[:, 3]*1000, 'b-', alpha=0.7, label='RW (traj)')
        ax3b.plot(time_sim, u_sim[:, 3]*1000, 'r--', alpha=0.7, label='RW (sim)')
    ax3b.set_xlabel('Time [s]')
    ax3b.set_ylabel('RW Torque [mNm]')
    ax3b.legend()
    ax3b.grid(True, alpha=0.3)
    
    fig3.tight_layout()
    fig3.savefig(os.path.join(run_dir, "control_effort.png"), dpi=150)
    
    # Figure 4: RW Momentum
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    if state_traj.shape[1] > 7:
        ax4.plot(time_traj, state_traj[:, 7] * 1000, 'b-', label='Planned Trajectory', alpha=0.7, linewidth=2)
        ax4.plot(time_sim, state_sim[:, 7] * 1000, 'r--', label='Simulated (TVLQR)', linewidth=1.5)
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('RW Momentum [mNms]')
    ax4.set_title(f'Reaction Wheel Momentum - {run_name}')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    fig4.tight_layout()
    fig4.savefig(os.path.join(run_dir, "rw_momentum.png"), dpi=150)
    
    plt.close('all')
    print(f"Plots saved to {run_dir}")


def print_summary(results: Dict[str, Any]):
    """Print summary of results."""
    print("\n" + "="*70)
    print(f"RUN: {results['run_name']}")
    print("="*70)
    
    if not results.get("traj_valid", False):
        print(f"❌ FAILED: {results.get('error', 'Unknown error')}")
        print(f"Plan time: {results.get('plan_time', 0):.2f}s")
        return
    
    print(f"✅ Trajectory valid | Plan time: {results['plan_time']:.1f}s")
    print(f"Initial error: {results['initial_error']:.1f}°")
    print(f"\nPlanned trajectory:")
    print(f"  Final error: {results['final_error_traj']:.3f}° | Final ω: {results['final_w_traj']:.3f}°/s")
    print(f"  Settle <1°: {results['settle_time_traj']:.0f}s | Settle <0.5°: {results['settle_05_traj']:.0f}s")
    if results.get('max_error_after_converge_traj', 0) > 5:
        print(f"  ⚠️  Max error after converge: {results['max_error_after_converge_traj']:.1f}° (OSCILLATING)")
    print(f"\nSimulated (TVLQR):")
    print(f"  Final error: {results['final_error_sim']:.3f}° | Final ω: {results['final_w_sim']:.3f}°/s")
    print(f"  Settle <1°: {results['settle_time_sim']:.0f}s | Settle <0.5°: {results['settle_05_sim']:.0f}s")
    print("="*70)


def print_trajectory_profile(results: Dict[str, Any]):
    """Print error profile over time."""
    if not results.get("traj_valid", False):
        return
    
    errors_traj = results["errors_traj"]
    w_mag_traj = results["w_mag_traj"]
    time_traj = results["time_traj"]
    
    print("\nTrajectory profile:")
    for t in [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
        if t > results["tf"]:
            break
        idx = np.argmin(np.abs(time_traj - t))
        print(f"  t={t:3.0f}s: err={errors_traj[idx]:6.1f}° ω={w_mag_traj[idx]:.2f}°/s")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf", type=float, default=500, help="Duration in seconds")
    parser.add_argument("--name", type=str, default="test", help="Run name")
    args = parser.parse_args()
    
    # Test configuration
    settings = {
        "name": args.name,
        # Timestep settings
        "dt_tp": 10,
        "dt_tvlqr": 1,
        "bdot_on": 3,  # Use bdot initial guess
        
        # Control weights (actuator preference)
        "mtq_control_weight": 1e3,
        "rw_control_weight": 1e7,
        "rw_AM_weight": 1e4,
        "rw_stic_weight": 1e0,
        
        # Cost weights
        "angle": 1e4,
        "angle_N": 1e6,
        "ang_vel": 1e3,
        "ang_vel_N": 1e5,
        "control_mult": 1.0,
        
        # Hessian settings
        "use_full_cost_hessian": True,
        "use_dynamics_hess": 1,
        "use_constraint_hess": 0,
        
        # Iteration settings
        "max_outer_iter": 10,
        "max_inner_iter": 40,
        "pass2_max_outer": 5,
        "pass2_max_inner": 15,
        "penalty_init": 100,
    }
    
    results = run_altro_tuning(
        settings=settings,
        tf=args.tf,
        dt=1,
        seed=42,
        verbose=False,
        run_name=args.name,
        use_simple_ic=True,
    )
    
    print_summary(results)
    print_trajectory_profile(results)
    save_results(results, args.name)
    plot_results(results, args.name)
