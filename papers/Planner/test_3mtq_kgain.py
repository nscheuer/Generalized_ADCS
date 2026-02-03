#!/usr/bin/env python
"""
Test 3MTQ+0RW with K-gain warm-start and live visualization.

Uses the same settings we've been tuning for 3MTQ+1RW.
"""
import sys
import os
import numpy as np
import argparse

# --- Path Setup ---
_this_file = os.path.abspath(__file__)
_this_dir = os.path.dirname(_this_file)
_root_dir = os.path.abspath(os.path.join(_this_dir, "../.."))
os.chdir(_root_dir)
sys.path.insert(0, _root_dir)
sys.path.insert(0, _this_dir)

from ADCS.CONOPS.goals import Fixed_Attitude_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_python_alilqr import Plan_and_Track_PythonALILQR
from ADCS.controller.helpers import PlannerSettings, create_planner_settings
from ADCS.controller.helpers.normalized_settings import (
    NormalizedPlannerConfig, NormalizedActuatorCosts, NormalizedStateCosts,
    NormalizedConstraints, PlannerPresets
)
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube1_cubesat
from ADCS.helpers.math_helpers import normalize


def create_mtq_only_settings(sat, dt_planning: float = 1.0):
    """
    Create well-conditioned planner settings for MTQ-only (no RW).
    
    Uses the same tuning philosophy as our 3MTQ+1RW settings.
    """
    config = NormalizedPlannerConfig(
        actuator_costs=NormalizedActuatorCosts(
            mtq_cost=10.0,
            rw_torque_cost=1.0,      # Not used for MTQ-only
            rw_momentum_cost=1.0,    # Not used for MTQ-only
            rw_stiction_cost=1.0,    # Not used for MTQ-only
            use_torque_effective_mtq_scaling=False,
            expected_B_field_uT=30.0,
        ),
        state_costs=NormalizedStateCosts(
            angle_cost=1000.0,          # High angle cost
            angle_terminal_cost=10000.0,
            ang_vel_cost=100.0,
            ang_vel_terminal_cost=1000.0,
            use_scale_normalization=True,
            angle_scale_deg=90.0,
            ang_vel_scale_deg_s=20.0,
        ).set_cross_term_auto(0.25),  # Add cross-term at 25% of PSD limit
        constraints=NormalizedConstraints(
            max_angular_velocity_deg_s=20.0,
            control_margin=0.25,
            rw_momentum_margin=0.9,
        ),
    )
    
    settings = create_planner_settings(sat, config)
    
    # Key settings
    settings.bdot_on = 0  # Random init
    settings.dt_tp = 10   # Coarse planning timestep
    settings.dt_tvlqr = dt_planning
    settings.verbosity = False
    
    # Convergence settings
    settings.pass1.convergence.max_outer_iter = 10
    settings.pass1.convergence.max_inner_iter = 30
    settings.pass2.convergence.max_outer_iter = 8
    settings.pass2.convergence.max_inner_iter = 30
    
    # Augmented Lagrangian settings
    settings.pass1.aug_lag.penalty_init = 10.0
    settings.pass1.aug_lag.penalty_max = 1e8
    settings.pass2.aug_lag.penalty_init = 1e5
    settings.pass2.aug_lag.penalty_max = 1e18
    
    # State-space regularization (in addition to control-space)
    settings.pass1.regularization.reg_mode = 2
    settings.pass2.regularization.reg_mode = 2
    settings.pass1.regularization.reg_min_cond = 0
    settings.pass2.regularization.reg_min_cond = 0
    
    return settings


def run_test(seed: int = 0, tf: float = 1000, visualize: bool = True, 
             warm_start: str = "kgain", goal_angle_deg: float = 180.0):
    """Run a single test."""
    dt = 2
    dt_planning = 1
    
    print(f"=== Test: seed={seed}, goal_angle={goal_angle_deg}°, warm_start={warm_start} ===")
    
    # Set warm-start mode via env var
    os.environ["PY_ALILQR_WARMSTART"] = warm_start
    
    # Create orbit
    np.random.seed(100_000 + seed)
    orb = create_random_circular_orbit(
        radius_km=7000.0, dt=dt_planning, tf=tf, use_J2=True, fast=True
    )
    orb.populate_environment(compute_B=True, compute_S=True)
    
    # Create satellite (MTQ-only)
    real_sat = create_beavercube1_cubesat(estimated=False)
    
    # Random initial state
    np.random.seed(seed)
    w0 = (np.random.rand(3) - 0.5) * 0.1  # ±0.05 rad/s
    q0 = np.random.randn(4)
    q0 = q0 / np.linalg.norm(q0)
    if q0[0] < 0:
        q0 = -q0
    
    x0 = np.concatenate([w0, q0])
    
    # Create goal based on angle
    if goal_angle_deg == 180:
        # Identity quaternion (180° from random start)
        q_goal = np.array([1.0, 0.0, 0.0, 0.0])
    elif goal_angle_deg == 90:
        # 90° rotation from initial
        axis = np.array([0, 0, 1])  # Around z-axis
        angle_rad = np.radians(90)
        q_rot = np.array([np.cos(angle_rad/2), 
                         axis[0]*np.sin(angle_rad/2),
                         axis[1]*np.sin(angle_rad/2), 
                         axis[2]*np.sin(angle_rad/2)])
        from ADCS.helpers.math_helpers import quat_mult
        q_goal = quat_mult(q_rot, q0)
        q_goal = q_goal / np.linalg.norm(q_goal)
        if q_goal[0] < 0:
            q_goal = -q_goal
    else:
        # General angle rotation
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        angle_rad = np.radians(goal_angle_deg)
        q_rot = np.array([np.cos(angle_rad/2), 
                         axis[0]*np.sin(angle_rad/2),
                         axis[1]*np.sin(angle_rad/2), 
                         axis[2]*np.sin(angle_rad/2)])
        from ADCS.helpers.math_helpers import quat_mult
        q_goal = quat_mult(q_rot, q0)
        q_goal = q_goal / np.linalg.norm(q_goal)
        if q_goal[0] < 0:
            q_goal = -q_goal
    
    print(f"  Initial q: [{q0[0]:.3f}, {q0[1]:.3f}, {q0[2]:.3f}, {q0[3]:.3f}]")
    print(f"  Goal q:    [{q_goal[0]:.3f}, {q_goal[1]:.3f}, {q_goal[2]:.3f}, {q_goal[3]:.3f}]")
    
    # Compute initial angle error
    q_goal_inv = np.array([q_goal[0], -q_goal[1], -q_goal[2], -q_goal[3]])
    qerr_w = q_goal_inv[0]*q0[0] - np.dot(q_goal_inv[1:], q0[1:])
    init_angle = np.degrees(2 * np.arccos(np.clip(np.abs(qerr_w), 0, 1)))
    print(f"  Initial angle error: {init_angle:.1f}°")
    
    # Create planner settings
    planner_settings = create_mtq_only_settings(real_sat, dt_planning=dt_planning)
    
    # Create controller with Python ALILQR for live viz
    controller = Plan_and_Track_PythonALILQR(
        est_sat=real_sat,
        planner_settings=planner_settings,
        use_v2=True
    )
    
    # Set up goals
    goals = GoalList({0.22: Fixed_Attitude_Goal(q_goal)})
    os0 = orb.get_os(0.22)
    
    # Diagnostic callback
    iter_count = [0]
    
    def diagnostic_callback(iter_data):
        Xset = iter_data.Xset
        q_goal_inv_local = np.array([q_goal[0], -q_goal[1], -q_goal[2], -q_goal[3]])
        
        N = Xset.shape[1]
        angles = np.zeros(N)
        for k in range(N):
            qk = Xset[3:7, k]
            qerr_w = q_goal_inv_local[0]*qk[0] - np.dot(q_goal_inv_local[1:], qk[1:])
            angles[k] = np.degrees(2 * np.arccos(np.clip(np.abs(qerr_w), 0, 1)))
        
        max_angle = np.max(angles)
        max_idx = np.argmax(angles)
        half_N = N // 2
        max_2nd_half = np.max(angles[half_N:])
        max_2nd_idx = half_N + np.argmax(angles[half_N:])
        
        print(f"  [{iter_data.pass_label}] O:{iter_data.outer_iter} I:{iter_data.inner_iter} "
              f"Cost:{iter_data.LA:.2e} Cmax:{iter_data.cmax:.2e} rho:{iter_data.rho:.1e} "
              f"Angle[start:{angles[0]:.0f}° max:{max_angle:.0f}°@{max_idx} "
              f"spike:{max_2nd_half:.0f}°@{max_2nd_idx} mean:{np.mean(angles):.0f}° end:{angles[-1]:.0f}°]")
        
        iter_count[0] += 1
    
    controller.set_iteration_callback(diagnostic_callback)
    
    # Calculate trajectory
    viz_save_path = f"/tmp/mtq_test_seed{seed}.png" if visualize else None
    
    try:
        traj = controller.calculate_trajectory(
            t_start=0.22, duration=tf, x_0=x0, os_0=os0, goals=goals,
            verbose=True, visualize=visualize, viz_save_path=viz_save_path,
            skip_pass2=False
        )
        
        # Compute final angle error
        q_final = traj.X[:, -1][3:7]
        qerr_w = q_goal_inv[0]*q_final[0] - np.dot(q_goal_inv[1:], q_final[1:])
        final_angle = np.degrees(2 * np.arccos(np.clip(np.abs(qerr_w), 0, 1)))
        
        print(f"\n=== RESULT ===")
        print(f"  Final angle error: {final_angle:.2f}°")
        print(f"  Total iterations: {iter_count[0]}")
        if viz_save_path:
            print(f"  Saved: {viz_save_path}")
        
        return {"success": True, "final_angle": final_angle, "iterations": iter_count[0]}
        
    except Exception as e:
        import traceback
        print(f"\n=== FAILED ===")
        print(f"  Error: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test 3MTQ with K-gain warm-start")
    parser.add_argument("-s", "--seed", type=int, default=0, help="Random seed")
    parser.add_argument("-a", "--angle", type=float, default=90.0, 
                        help="Goal angle in degrees (default: 90)")
    parser.add_argument("-w", "--warmstart", type=str, default="kgain",
                        choices=["foh", "slerp", "lsq", "kgain"],
                        help="Warm-start mode (default: kgain)")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    args = parser.parse_args()
    
    result = run_test(
        seed=args.seed,
        goal_angle_deg=args.angle,
        warm_start=args.warmstart,
        visualize=not args.no_viz
    )
