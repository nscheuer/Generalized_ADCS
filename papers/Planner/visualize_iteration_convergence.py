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

from ADCS.CONOPS.goals import Fixed_Attitude_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_python_alilqr import Plan_and_Track_PythonALILQR
from ADCS.controller.helpers import (
    PlannerSettings, create_planner_settings,
    NormalizedPlannerConfig, NormalizedActuatorCosts, NormalizedStateCosts,
    IterationData,
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
    
    # Compute angle errors
    angle_errors = []
    for it in iterations:
        q_final = it.Xset[3:7, -1]
        q_final = q_final / np.linalg.norm(q_final)
        angle_errors.append(quat_error_angle(q_final, q_goal))
    
    # Plot 1: Cost vs iteration
    ax1 = axes[0, 0]
    ax1.semilogy(total_iters, costs, 'b-', linewidth=1.5, label='Augmented Lagrangian')
    ax1.semilogy(total_iters, costs_nc, 'g--', linewidth=1.5, label='Cost (no constraints)')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost')
    ax1.set_title('Cost Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Mark outer iteration boundaries
    outer_starts = [0]
    for i in range(1, len(outer_iters)):
        if outer_iters[i] != outer_iters[i-1]:
            outer_starts.append(i)
            ax1.axvline(i, color='red', linestyle=':', alpha=0.5)
    
    # Plot 2: Constraint violation
    ax2 = axes[0, 1]
    ax2.semilogy(total_iters, [c + 1e-10 for c in cmaxs], 'r-', linewidth=1.5)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Max Constraint Violation')
    ax2.set_title('Constraint Satisfaction')
    ax2.grid(True, alpha=0.3)
    for s in outer_starts[1:]:
        ax2.axvline(s, color='red', linestyle=':', alpha=0.5)
    
    # Plot 3: Gradient (convergence indicator)
    ax3 = axes[1, 0]
    ax3.semilogy(total_iters, [g + 1e-10 for g in grads], 'm-', linewidth=1.5)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Gradient Norm')
    ax3.set_title('Gradient Convergence')
    ax3.grid(True, alpha=0.3)
    for s in outer_starts[1:]:
        ax3.axvline(s, color='red', linestyle=':', alpha=0.5)
    
    # Plot 4: Angle error
    ax4 = axes[1, 1]
    ax4.semilogy(total_iters, [e + 1e-3 for e in angle_errors], 'c-', linewidth=1.5)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Angle Error (deg)')
    ax4.set_title('Pointing Error Evolution')
    ax4.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='1° target')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    for s in outer_starts[1:]:
        ax4.axvline(s, color='red', linestyle=':', alpha=0.5)
    
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


def main():
    print("="*70)
    print("ITERATION-BY-ITERATION CONVERGENCE VISUALIZATION")
    print("="*70)
    
    # Setup
    np.random.seed(42)
    sat = create_beavercube2_cubesat(estimated=False)
    
    print("\nCreating orbit...")
    orb = create_random_circular_orbit(radius_km=7000.0, dt=1, tf=150, use_J2=True, fast=True)
    orb.populate_environment(compute_B=True, compute_S=True)
    os0 = orb.get_os(0.22)
    
    # Initial conditions (fixed seed for reproducibility)
    rng = np.random.default_rng(seed=1000)
    q0 = normalize(rng.standard_normal(4))
    w0 = normalize(rng.standard_normal(3)) * (0.5 * np.pi / 180.0)  # 0.5 deg/s
    h0 = rng.uniform(-0.0001, 0.0001, size=1)
    
    # 90 degree slew
    half_angle = 45 * np.pi / 180
    q_rot = np.array([np.cos(half_angle), np.sin(half_angle), 0, 0])
    q_goal = normalize(quat_mult(q0, q_rot))
    
    x0 = np.concatenate([w0, q0, h0])
    for i, rw in enumerate(sat.rw_actuators):
        rw.h = h0[i]
    
    initial_error = quat_error_angle(q0, q_goal)
    print(f"\nInitial pointing error: {initial_error:.1f}°")
    print(f"Goal: 90° slew in {120}s")
    
    # Create planner settings
    print("\nUsing well-conditioned normalized settings...")
    config = NormalizedPlannerConfig(
        actuator_costs=NormalizedActuatorCosts(mtq_cost=1.0, rw_torque_cost=5.0),
        state_costs=NormalizedStateCosts(
            angle_cost=1000.0, angle_terminal_cost=1000000.0,
            ang_vel_cost=1000.0, ang_vel_terminal_cost=100000.0,
        ),
    )
    settings = create_planner_settings(sat, config)
    settings.rw_AM_weight = 1e4
    settings.RWh_ok_mult = 0.5
    settings.bdot_on = 0  # IMPORTANT: Use random init, not B-dot for slew maneuvers
    settings.pass1.convergence.max_outer_iter = 5
    settings.pass1.convergence.max_inner_iter = 20
    settings.pass2.convergence.max_outer_iter = 3
    settings.pass2.convergence.max_inner_iter = 10
    
    # Collect iteration data
    all_iterations: List[IterationData] = []
    
    def iteration_callback(iter_data: IterationData):
        all_iterations.append(iter_data)
        
        # Print progress
        q_final = iter_data.Xset[3:7, -1]
        q_final = q_final / np.linalg.norm(q_final)
        err = quat_error_angle(q_final, q_goal)
        
        print(f"  Outer {iter_data.outer_iter}, Inner {iter_data.inner_iter:2d}: "
              f"cost={iter_data.LA:.2e}, cmax={iter_data.cmax:.2e}, "
              f"grad={iter_data.grad:.2e}, error={err:.2f}°")
    
    # Create controller with Python ALILQR
    controller = Plan_and_Track_PythonALILQR(
        est_sat=sat, 
        planner_settings=settings,
        verbose=False  # We'll print our own progress
    )
    controller.set_iteration_callback(iteration_callback)
    
    # Create goal
    goals = GoalList({0.22: Fixed_Attitude_Goal(q_goal)})
    
    # Run optimization
    print("\nRunning optimization...")
    print("-"*70)
    
    t0 = time.perf_counter()
    traj = controller.calculate_trajectory(
        t_start=0.22, duration=120, x_0=x0, os_0=os0, 
        goals=goals, verbose=False
    )
    elapsed = time.perf_counter() - t0
    
    print("-"*70)
    
    # Results
    q_final = traj.states[3:7, -1]
    final_error = quat_error_angle(q_final, q_goal)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print("="*70)
    print(f"Total time: {elapsed:.2f}s")
    print(f"Total iterations: {len(all_iterations)}")
    print(f"Final pointing error: {final_error:.4f}°")
    print(f"Improvement: {initial_error:.1f}° → {final_error:.4f}°")
    
    # Generate convergence plot
    print("\nGenerating convergence plot...")
    save_path = "/home/pmckeen/Generalized_ADCS/papers/Planner/figures/iteration_convergence.png"
    plot_convergence(all_iterations, q_goal, save_path)
    
    return all_iterations


if __name__ == '__main__':
    iterations = main()
