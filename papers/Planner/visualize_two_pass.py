#!/usr/bin/env python3
"""
Two-Pass Trajectory Optimization Visualization.

Demonstrates the Python ALILQR v2 wrapper with:
- Pass 1: Coarse timestep (dt_tp) exploration
- Pass 2: Fine timestep (dt_tvlqr) refinement with high penalty

This matches the C++ trajOpt behavior exactly.
"""

import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from typing import List
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/pmckeen/Generalized_ADCS')

from ADCS.CONOPS.goals import Fixed_Attitude_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.helpers import PlannerSettings
from ADCS.controller.helpers.build_csat import build_cpp_satellite
from ADCS.controller.helpers.python_alilqr_v2 import PythonALILQRv2, IterationData
from ADCS.controller.helpers.live_planner_viz import LivePlannerViz
from ADCS.controller.plan_and_track_base import PlanAndTrackBase
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.helpers.math_helpers import normalize, quat_mult, quat_diff
import trajectory_planner.build.tplaunch as tplaunch


def quat_error_angle(q1, q2):
    """Compute angle between two quaternions in degrees."""
    q_err = quat_diff(q1, q2)
    return 2 * np.arccos(np.clip(np.abs(q_err[0]), 0, 1)) * 180 / np.pi


class EnvironmentHelper(PlanAndTrackBase):
    """Helper class to use _propagate_environment."""
    def __init__(self, est_sat, planner_settings, planner):
        self.est_sat = est_sat
        self.planner_settings = planner_settings
        self.planner = planner
    
    def find_u(self, *args, **kwargs):
        pass
    
    def calculate_trajectory(self, *args, **kwargs):
        pass


def main():
    print("="*70)
    print("TWO-PASS TRAJECTORY OPTIMIZATION")
    print("="*70)
    
    # =========================================================================
    # SETUP
    # =========================================================================
    print("\nCreating orbit...")
    np.random.seed(42)
    
    orb = create_random_circular_orbit(radius_km=7000.0, dt=1, tf=200, use_J2=True, fast=True)
    orb.populate_environment(compute_B=True, compute_S=True)
    
    # Create satellite
    sat = create_beavercube2_cubesat(estimated=False)
    
    # Create planner settings
    settings = PlannerSettings(est_sat=sat, bdot_on=0)
    
    # Configure Pass 1: Exploration (coarse timestep)
    settings.pass1.aug_lag.penalty_init = 1.0
    settings.pass1.aug_lag.penalty_max = 1e6
    settings.pass1.convergence.max_outer_iter = 10
    settings.pass1.convergence.max_inner_iter = 15
    
    # Configure Pass 2: Refinement (fine timestep, high penalty)
    settings.pass2.aug_lag.penalty_init = 1e4
    settings.pass2.aug_lag.penalty_max = 1e16
    settings.pass2.convergence.max_outer_iter = 8
    settings.pass2.convergence.max_inner_iter = 15
    
    # Initial state - 90 degree slew
    rng = np.random.default_rng(seed=1000)
    q0 = normalize(rng.standard_normal(4))
    w0 = normalize(rng.standard_normal(3)) * (0.5 * np.pi / 180.0)
    h0 = rng.uniform(-0.0001, 0.0001, size=1)
    
    # Goal: 90 degree rotation about X axis
    half_angle = 45 * np.pi / 180
    q_rot = np.array([np.cos(half_angle), np.sin(half_angle), 0, 0])
    q_goal = normalize(quat_mult(q0, q_rot))
    
    x0 = np.concatenate([w0, q0, h0])
    for i, rw in enumerate(sat.rw_actuators):
        rw.h = h0[i]
    
    initial_error = quat_error_angle(q0, q_goal)
    print(f"\nInitial pointing error: {initial_error:.1f}°")
    print(f"Goal: 90° slew in 120s")
    
    # =========================================================================
    # BUILD PLANNER
    # =========================================================================
    print("\nBuilding C++ planner...")
    csat = build_cpp_satellite(est_sat=sat, planner_settings=settings)
    planner = tplaunch.Planner(
        csat,
        settings.systemSettings(),
        settings.mainAlilqrSettings(),
        settings.secondAlilqrSettings(),
        settings.initTrajSettings(),
        settings.optMainCostSettings(),
        settings.optSecondCostSettings(),
        settings.optTVLQRCostSettings(tracking_LQR_formulation=0)
    )
    planner.setquaternionTo3VecMode(2)
    
    # Create environment helper
    env_helper = EnvironmentHelper(sat, settings, planner)
    
    # =========================================================================
    # SETUP VISUALIZATION
    # =========================================================================
    all_iterations: List[IterationData] = []
    
    # Create live visualization
    live_viz = LivePlannerViz(
        goal_vector_eci=q_goal,
        dt=settings.dt_tvlqr,  # Use fine timestep for display
        update_interval=1,
        figsize=(14, 10)
    )
    live_viz.start()
    
    def iteration_callback(iter_data: IterationData):
        """Called at each iteration."""
        all_iterations.append(iter_data)
        live_viz.update(iter_data)
        
        # Print progress
        q_final = iter_data.Xset[3:7, -1]
        q_final = q_final / np.linalg.norm(q_final)
        err = quat_error_angle(q_final, q_goal)
        
        pass_str = f"[{iter_data.pass_label}]" if iter_data.pass_label else ""
        print(f"  {pass_str} Outer {iter_data.outer_iter}, Inner {iter_data.inner_iter:2d}: "
              f"cost={iter_data.LA:.2e}, cmax={iter_data.cmax:.2e}, "
              f"grad={iter_data.grad:.2e}, error={err:.2f}°")
    
    # Create Python ALILQR v2
    py_alilqr = PythonALILQRv2(planner, debug_callback=iteration_callback, verbose=False)
    
    # =========================================================================
    # PASS 1: COARSE EXPLORATION
    # =========================================================================
    duration = 120.0
    t_start = 0.22
    dt_coarse = settings.dt_tp
    dt_fine = settings.dt_tvlqr
    t_end = t_start + duration * TimeConstants.sec2cent
    
    print(f"\n{'='*70}")
    print(f"PASS 1: Exploration (dt={dt_coarse}s)")
    print(f"{'='*70}")
    
    os0 = orb.get_os(t_start)
    goals = GoalList({t_start: Fixed_Attitude_Goal(q_goal)})
    x0_clean = np.copy(x0.astype(np.float64).flatten(), order='C')
    
    # Propagate environment at coarse timestep
    N_coarse = int(np.ceil(duration / dt_coarse)) + 1
    vecsPy_coarse = env_helper._propagate_environment(os0, t_start, t_end, dt_coarse, N_coarse, goals)
    
    # Get initial trajectory
    initial_result_1 = planner.prepareForAlilqr(
        vecsPy_coarse, dt_coarse, t_start, t_end, x0_clean, 0
    )
    initial_traj_1, vecs_dt_coarse, _ = initial_result_1
    
    # Run Pass 1
    t0 = time.perf_counter()
    result1 = py_alilqr.optimize(
        dt=dt_coarse,
        initial_traj=initial_traj_1,
        vecs=vecs_dt_coarse,
        cost_settings=settings.optMainCostSettings(),
        alilqr_settings=settings.mainAlilqrSettings(),
        is_first_search=True,
        collect_all=True,
        pass_label="Pass1"
    )
    t1 = time.perf_counter()
    
    print(f"\nPass 1 complete in {t1-t0:.2f}s")
    print(f"  Iterations: {result1.total_inner_iters}")
    print(f"  Final cost: {result1.final_cost:.4e}")
    print(f"  Final cmax: {result1.final_cmax:.4e}")
    
    # =========================================================================
    # PASS 2: FINE REFINEMENT
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"PASS 2: Refinement (dt={dt_fine}s, penalty_init={settings.pass2.aug_lag.penalty_init})")
    print(f"{'='*70}")
    
    # Propagate environment at fine timestep
    N_fine = int(np.ceil(duration / dt_fine)) + 1
    vecsPy_fine = env_helper._propagate_environment(os0, t_start, t_end, dt_fine, N_fine, goals)
    
    # Get initial trajectory for fine timestep
    initial_result_2 = planner.prepareForAlilqr(
        vecsPy_fine, dt_fine, t_start, t_end, x0_clean, 0
    )
    initial_traj_2, vecs_dt_fine, _ = initial_result_2
    
    # Run Pass 2
    t0 = time.perf_counter()
    result2 = py_alilqr.optimize(
        dt=dt_fine,
        initial_traj=initial_traj_2,
        vecs=vecs_dt_fine,
        cost_settings=settings.optSecondCostSettings(),
        alilqr_settings=settings.secondAlilqrSettings(),
        is_first_search=False,
        collect_all=True,
        pass_label="Pass2"
    )
    t2 = time.perf_counter()
    
    print(f"\nPass 2 complete in {t2-t0:.2f}s")
    print(f"  Iterations: {result2.total_inner_iters}")
    print(f"  Final cost: {result2.final_cost:.4e}")
    print(f"  Final cmax: {result2.final_cmax:.4e}")
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    q_final = result2.Xset[3:7, -1]
    q_final = q_final / np.linalg.norm(q_final)
    final_error = quat_error_angle(q_final, q_goal)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Total time: {(t1-t0) + (t2-t0):.2f}s")
    print(f"Total iterations: {result1.total_inner_iters + result2.total_inner_iters}")
    print(f"Final pointing error: {final_error:.4f}°")
    print(f"Improvement: {initial_error:.1f}° → {final_error:.4f}°")
    print(f"Final cmax: {result2.final_cmax:.2e}")
    
    # Save visualization
    live_viz.save("/home/pmckeen/Generalized_ADCS/papers/Planner/figures/two_pass_viz.png")
    print("\nLive visualization saved.")
    
    # =========================================================================
    # CONVERGENCE HISTORY PLOT (separate window)
    # =========================================================================
    print("\nGenerating convergence history plot...")
    
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('Two-Pass Optimization Convergence History', fontsize=14, fontweight='bold')
    
    # Combine all iterations
    all_iters = all_iterations
    total_iters = list(range(len(all_iters)))
    
    # Find Pass1/Pass2 boundary
    pass1_end = result1.total_inner_iters
    
    # Extract data
    costs = [it.LA for it in all_iters]
    cmaxs = [max(it.cmax, 1e-16) for it in all_iters]
    grads = [max(it.grad, 1e-16) for it in all_iters]
    mus = [it.mu for it in all_iters]
    
    # Compute angle errors
    angle_errors = []
    for it in all_iters:
        q_f = it.Xset[3:7, -1]
        q_f = q_f / np.linalg.norm(q_f)
        angle_errors.append(quat_error_angle(q_f, q_goal))
    
    # Plot 1: Cost
    ax = axes[0, 0]
    ax.semilogy(total_iters[:pass1_end], costs[:pass1_end], 'b-', linewidth=1.5, label='Pass 1')
    ax.semilogy(total_iters[pass1_end:], costs[pass1_end:], 'r-', linewidth=1.5, label='Pass 2')
    ax.axvline(x=pass1_end, color='gray', linestyle='--', alpha=0.5, label='Pass boundary')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost (Augmented Lagrangian)')
    ax.set_title('Cost Convergence')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Constraint violation
    ax = axes[0, 1]
    ax.semilogy(total_iters[:pass1_end], cmaxs[:pass1_end], 'b-', linewidth=1.5, label='Pass 1')
    ax.semilogy(total_iters[pass1_end:], cmaxs[pass1_end:], 'r-', linewidth=1.5, label='Pass 2')
    ax.axvline(x=pass1_end, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.002, color='green', linestyle=':', alpha=0.7, label='cmax target')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max Constraint Violation')
    ax.set_title('Constraint Satisfaction')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Pointing error
    ax = axes[1, 0]
    ax.semilogy(total_iters[:pass1_end], angle_errors[:pass1_end], 'b-', linewidth=1.5, label='Pass 1')
    ax.semilogy(total_iters[pass1_end:], angle_errors[pass1_end:], 'r-', linewidth=1.5, label='Pass 2')
    ax.axvline(x=pass1_end, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1.0, color='green', linestyle=':', alpha=0.7, label='1° target')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Pointing Error (deg)')
    ax.set_title('Pointing Error Convergence')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Penalty (mu)
    ax = axes[1, 1]
    ax.semilogy(total_iters[:pass1_end], mus[:pass1_end], 'b-', linewidth=1.5, label='Pass 1')
    ax.semilogy(total_iters[pass1_end:], mus[pass1_end:], 'r-', linewidth=1.5, label='Pass 2')
    ax.axvline(x=pass1_end, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Penalty (μ)')
    ax.set_title('Augmented Lagrangian Penalty')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save convergence plot
    conv_path = "/home/pmckeen/Generalized_ADCS/papers/Planner/figures/two_pass_convergence.png"
    fig2.savefig(conv_path, dpi=150, bbox_inches='tight')
    print(f"Convergence plot saved to {conv_path}")
    
    # Show both figures
    print("\nClose the figures to exit.")
    plt.show()


if __name__ == "__main__":
    main()
