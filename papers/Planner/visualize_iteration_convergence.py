#!/usr/bin/env python3
"""
Visualize iteration-by-iteration convergence of the trajectory planner.

This script demonstrates the Python ALILQR wrapper which allows:
- Real-time monitoring of optimization progress
- Collecting data at each iteration
- Generating convergence plots

Run:
    python papers/Planner/visualize_iteration_convergence.py
"""

import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/pmckeen/Generalized_ADCS')

from ADCS.CONOPS.goals import Fixed_Attitude_Goal, ECI_Goal, No_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_base import PlanAndTrackBase
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.helpers.math_helpers import rot_mat
from ADCS.controller.plan_and_track_python_alilqr import Plan_and_Track_PythonALILQR
from ADCS.controller.helpers import (
    PlannerSettings, create_planner_settings,
    NormalizedPlannerConfig, NormalizedActuatorCosts, NormalizedStateCosts,
    IterationData, LivePlannerViz,
)
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.helpers.math_helpers import normalize, quat_mult, quat_diff


def quat_error_angle(q1, q2):
    """Compute angle between two quaternions in degrees."""
    q_err = quat_diff(q1, q2)
    return 2 * np.arccos(np.clip(np.abs(q_err[0]), 0, 1)) * 180 / np.pi


def plot_convergence(iterations: List[IterationData], q_goal: np.ndarray, save_path: str = None):
    """
    Create a comprehensive convergence plot.
    
    Automatically detects Pass 1/Pass 2 from pass_label field and colors them differently.
    """
    if not iterations:
        print("No iteration data to plot!")
        return
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Trajectory Planner Convergence Analysis', fontsize=14, fontweight='bold')
    
    # Extract data
    outer_iters = [it.outer_iter for it in iterations]
    inner_iters = [it.inner_iter for it in iterations]
    total_iters = list(range(len(iterations)))
    
    costs = [it.LA for it in iterations]
    costs_nc = [it.LA_nc for it in iterations]
    cmaxs = [it.cmax for it in iterations]
    grads = [it.grad for it in iterations]
    
    # Detect Pass 1/Pass 2 boundary from pass_label
    pass_labels = [getattr(it, 'pass_label', '') for it in iterations]
    pass1_mask = [('Pass1' in pl or 'pass1' in pl.lower() if pl else True) for pl in pass_labels]
    pass2_start = None
    for i, pl in enumerate(pass_labels):
        if pl and ('Pass2' in pl or 'pass2' in pl.lower()):
            pass2_start = i
            break
    
    # Compute angle errors
    angle_errors = []
    for it in iterations:
        q_final = it.Xset[3:7, -1]
        q_final = q_final / np.linalg.norm(q_final)
        angle_errors.append(quat_error_angle(q_final, q_goal))
    
    # Helper to plot with Pass1/Pass2 coloring
    def plot_with_passes(ax, y_data, ylabel, title, use_semilogy=True, add_offset=0):
        y = [v + add_offset for v in y_data]
        if pass2_start is not None:
            # Two-pass case
            x1, y1 = total_iters[:pass2_start], y[:pass2_start]
            x2, y2 = total_iters[pass2_start:], y[pass2_start:]
            if use_semilogy:
                ax.semilogy(x1, y1, 'b-', linewidth=1.5, label='Pass 1 (coarse)')
                ax.semilogy(x2, y2, 'r-', linewidth=1.5, label='Pass 2 (fine)')
            else:
                ax.plot(x1, y1, 'b-', linewidth=1.5, label='Pass 1 (coarse)')
                ax.plot(x2, y2, 'r-', linewidth=1.5, label='Pass 2 (fine)')
            ax.axvline(pass2_start, color='gray', linestyle='--', linewidth=2, alpha=0.7)
            ax.legend(loc='upper right', fontsize=8)
        else:
            # Single pass
            if use_semilogy:
                ax.semilogy(total_iters, y, 'b-', linewidth=1.5)
            else:
                ax.plot(total_iters, y, 'b-', linewidth=1.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    # Plot 1: Cost vs iteration
    ax1 = axes[0, 0]
    plot_with_passes(ax1, costs, 'Cost', 'Cost Convergence')
    
    # Plot 2: Constraint violation
    ax2 = axes[0, 1]
    plot_with_passes(ax2, cmaxs, 'Max Constraint Violation', 'Constraint Satisfaction', add_offset=1e-10)
    ax2.axhline(0.002, color='green', linestyle=':', alpha=0.7, label='cmax target')
    
    # Plot 3: Gradient (convergence indicator)
    ax3 = axes[1, 0]
    plot_with_passes(ax3, grads, 'Gradient Norm', 'Gradient Convergence', add_offset=1e-10)
    
    # Plot 4: Angle error
    ax4 = axes[1, 1]
    plot_with_passes(ax4, angle_errors, 'Angle Error (deg)', 'Pointing Error Evolution', add_offset=1e-3)
    ax4.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='1° target')
    ax4.legend(loc='upper right', fontsize=8)
    
    # Plot 5: Final trajectory states
    ax5 = axes[2, 0]
    final_it = iterations[-1]
    times = np.arange(final_it.Xset.shape[1]) * 1.0  # Assume dt=1
    
    # Angular velocity
    for i, label in enumerate(['ωx', 'ωy', 'ωz']):
        ax5.plot(times, final_it.Xset[i, :] * 180/np.pi, label=label)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Angular Velocity (deg/s)')
    ax5.set_title('Final Trajectory: Angular Velocity')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Final trajectory controls
    ax6 = axes[2, 1]
    n_ctrl = final_it.Uset.shape[0]
    n_ctrl_times = final_it.Uset.shape[1]
    times_ctrl = np.arange(n_ctrl_times) * 1.0
    ctrl_labels = ['MTQ_x', 'MTQ_y', 'MTQ_z', 'RW'] if n_ctrl == 4 else [f'u{i}' for i in range(n_ctrl)]
    for i in range(min(n_ctrl, 4)):
        ax6.plot(times_ctrl, final_it.Uset[i, :], label=ctrl_labels[i])
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Control')
    ax6.set_title('Final Trajectory: Controls')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved convergence plot to: {save_path}")
    
    plt.show()


def boresight_error(q, goal_eci):
    """Compute boresight pointing error in degrees."""
    if np.linalg.norm(goal_eci) < 0.1:
        return np.nan
    q = q / np.linalg.norm(q)
    R = rot_mat(q)
    bore = R @ np.array([0, 1, 0])
    g = goal_eci / np.linalg.norm(goal_eci)
    return np.degrees(np.arccos(np.clip(np.dot(bore, g), -1, 1)))


class EnvHelper(PlanAndTrackBase):
    """Helper to access _propagate_environment."""
    def __init__(self, est_sat, planner_settings):
        self.est_sat = est_sat
        self.planner_settings = planner_settings
        self.planner = None
    def find_u(self, *a, **kw): pass
    def calculate_trajectory(self, *a, **kw): pass


def main():
    print("="*70)
    print("3MTQ+1RW MULTI-GOAL CONVERGENCE VISUALIZATION")
    print("="*70)
    
    # Import optimized settings
    sys.path.insert(0, '/home/pmckeen/Generalized_ADCS/papers/Planner')
    from mc_planner_settings import create_optimized_planner_settings
    
    # Setup - seed 4 has 70.8% yaw fraction, good for showing RW usage
    run_id = 4
    rng = np.random.default_rng(seed=run_id)  # Use seed directly for reproducibility
    sat = create_beavercube2_cubesat(estimated=False)
    
    tf = 1000
    t_start = 0.22
    sec2cent = TimeConstants.sec2cent
    
    print("\nCreating orbit...")
    np.random.seed(4)  # Seed 4 has 70.8% yaw fraction - good for RW usage
    orb = create_random_circular_orbit(radius_km=7000.0, dt=1, tf=tf+100, use_J2=True, fast=True)
    orb.populate_environment(compute_B=True, compute_S=True)
    os0 = orb.get_os(t_start)
    
    # Initial conditions - start from near-identity to make geometry clear
    # Identity quaternion: boresight (body-Y) points at ECI-Y
    q0 = normalize(np.array([1.0, 0.0, 0.0, 0.0]) + 0.01 * rng.standard_normal(4))
    w0 = np.array([0.001, 0.001, 0.001])  # Small initial rates
    h0 = np.array([0.0])
    
    # Multi-goal: Goal1(0-250s) → No_Goal(250-350s) → Goal2(350-600s) → No_Goal(600-700s) → Goal3(700-1000s)
    # Set goals that require z-axis rotation (yaw) since RW is on z-axis
    # Boresight is body-Y [0,1,0]. For z-axis rotation to help:
    # - Goal in ECI should differ from current boresight mainly in the x-y plane
    # Start with identity quaternion: boresight_eci = [0,1,0]
    # Goal1 = [1,0,0] requires 90° yaw (z-axis rotation)
    # Goal2 = [-1,0,0] requires another 90° yaw
    # Goal3 = [0,-1,0] requires another 90° yaw
    goal1 = np.array([1.0, 0.0, 0.0])  # +X in ECI
    goal2 = np.array([-1.0, 0.0, 0.0])  # -X in ECI (180° from goal1)
    goal3 = np.array([0.0, -1.0, 0.0])  # -Y in ECI
    
    x0 = np.concatenate([w0, q0, h0])
    for i, rw in enumerate(sat.rw_actuators):
        rw.h = h0[i]
    
    initial_error = boresight_error(q0, goal1)
    print(f"\nInitial boresight error to Goal1: {initial_error:.1f}°")
    print(f"Goal1: {goal1}")
    print(f"Goal2: {goal2}")
    print(f"Goal3: {goal3}")
    
    # Create settings
    settings = create_optimized_planner_settings(sat, duration=tf, tuning="balanced")
    
    # Increase wmax to allow faster slews (60 deg/s)
    settings.wmax = np.radians(60)
    
    # Make RW MUCH cheaper - try 0.0001x MTQ cost
    # RW cost is now properly scaled in C++ by NONMTQ_TORQ_SCALE²
    # Use same weight ratio as MTQ for equal cost-per-torque
    settings.rw_control_weight = settings.mtq_control_weight * 0.1  # Favor RW slightly
    
    # Disable momentum cost and allow more momentum capacity
    settings.rw_AM_weight = 0.0
    settings.rw_stic_weight = 0.0
    settings.RWh_max_mult = 0.9  # Allow 90% of physical momentum capacity
    
    print(f"RW control weight: {settings.rw_control_weight}")
    print(f"MTQ control weight: {settings.mtq_control_weight}")
    print(f"Ratio MTQ/RW: {settings.mtq_control_weight / settings.rw_control_weight:.0f}x")
    
    # Try bdot_on=4 (PD control with RW) 
    settings.bdot_on = 4
    print(f"Using bdot_on={settings.bdot_on} (PD control init with RW)")
    
    # Reduce Pass1 iterations for quick testing
    settings.pass1.convergence.max_iters = 10  # was 20
    settings.pass1.aug_lag.outer_loop_max = 3  # was 10
    
    # Create multi-goal GoalList (100s No_Goal gaps instead of 50s)
    goals = GoalList({
        t_start: ECI_Goal(goal1),
        t_start + 250*sec2cent: No_Goal(),      # Goal1 ends at 250s
        t_start + 350*sec2cent: ECI_Goal(goal2), # Goal2 starts at 350s (100s gap)
        t_start + 600*sec2cent: No_Goal(),      # Goal2 ends at 600s
        t_start + 700*sec2cent: ECI_Goal(goal3), # Goal3 starts at 700s (100s gap)
    })
    
    # Get time-varying goal vectors (E) for visualization
    env = EnvHelper(sat, settings)
    dt_tp = settings.dt_tp
    N_pass1 = int(tf / dt_tp) + 1
    t_end = t_start + tf * sec2cent
    vecsPy = env._propagate_environment(os0, t_start, t_end, dt_tp, N_pass1, goals)
    goal_vectors = vecsPy[6]  # E vectors shape (3, N)
    
    print(f"\nGoal vectors shape: {goal_vectors.shape}")
    
    # Collect iteration data
    all_iterations: List[IterationData] = []
    
    # Create live visualization with time-varying goal vectors
    live_viz = LivePlannerViz(
        goal_vector_eci=goal_vectors,  # (3, N) time-varying!
        body_vector=np.array([0, 1, 0]),
        dt=dt_tp,
        update_interval=1,
        figsize=(14, 10),
        actuator_names=['MTQ_x', 'MTQ_y', 'MTQ_z', 'RW'],
        umax=settings.umax  # For normalizing control plot
    )
    live_viz.start()
    
    def iteration_callback(iter_data: IterationData):
        all_iterations.append(iter_data)
        live_viz.update(iter_data)
        
        # Save screenshot every 20 iterations so we can see what's happening
        if len(all_iterations) % 20 == 1:
            live_viz.save(f"/tmp/live_viz_iter_{len(all_iterations):04d}.png")
        
        q_final = iter_data.Xset[3:7, -1]
        err = boresight_error(q_final, goal3)
        
        # Debug: Check RW control values (normalized by limit)
        if iter_data.Uset.shape[0] > 3:
            rw_ctrl = iter_data.Uset[3, :]
            rw_max_physical = np.abs(rw_ctrl).max()
            # Normalize by RW limit for display (same as MTQ normalization)
            rw_limit = sat.rw_actuators[0].u_max * settings.control_limit_scale
            rw_normalized = rw_max_physical / rw_limit if rw_limit > 0 else 0
            rw_info = f" RW={rw_normalized:.1%}"
        else:
            rw_info = ""
        
        pass_str = f"[{iter_data.pass_label}] " if iter_data.pass_label else ""
        print(f"  {pass_str}Outer {iter_data.outer_iter}, Inner {iter_data.inner_iter:2d}: "
              f"cost={iter_data.LA:.2e}, cmax={iter_data.cmax:.2e}, "
              f"grad={iter_data.grad:.2e}, error={err:.2f}°{rw_info}")
    
    # Create controller with Python ALILQR
    controller = Plan_and_Track_PythonALILQR(
        est_sat=sat, 
        planner_settings=settings,
        verbose=False
    )
    controller.set_iteration_callback(iteration_callback)
    
    # Run optimization
    print("\nRunning optimization...")
    print("-"*70)
    
    t0 = time.perf_counter()
    traj = controller.calculate_trajectory(
        t_start=t_start, duration=tf, x_0=x0, os_0=os0, 
        goals=goals, verbose=False
    )
    elapsed = time.perf_counter() - t0
    
    print("-"*70)
    
    # Results
    q_final = traj.states[3:7, -1]
    final_error = boresight_error(q_final, goal3)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print("="*70)
    print(f"Total time: {elapsed:.2f}s")
    print(f"Total iterations: {len(all_iterations)}")
    print(f"Final boresight error to Goal3: {final_error:.4f}°")
    print(f"Improvement: {initial_error:.1f}° → {final_error:.4f}°")
    
    # Save live viz
    live_viz.save("/home/pmckeen/Generalized_ADCS/papers/Planner/figures/live_viz_multigoal.png")
    print("\nLive visualization saved.")
    
    # Keep live viz open
    print("\nClose the plot window to exit...")
    live_viz.finish(block=True)
    
    return all_iterations


if __name__ == '__main__':
    iterations = main()
