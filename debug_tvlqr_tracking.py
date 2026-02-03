#!/usr/bin/env python3
"""
Deep dive diagnostic for TVLQR tracking issues.

This script isolates the TVLQR tracking controller and checks:
1. K-gain shapes and ordering (C++ vs Python)
2. State diff computation
3. Control computation at each step
4. Signs and scaling
"""

import numpy as np
from scipy.spatial.transform import Rotation

# Setup path
import sys
sys.path.insert(0, '/home/pmckeen/Generalized_ADCS')

from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.CONOPS.goals.attitude_goals.fixed_attitude_goal import Fixed_Attitude_Goal
from ADCS.CONOPS.goallist.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.helpers.trajectory import Trajectory
from ADCS.helpers.math_helpers import normalize, quat_diff, quat_to_vec3
from papers.Planner.mc_planner_settings import create_optimized_planner_settings

np.set_printoptions(precision=6, suppress=True, linewidth=200)

def main():
    print("="*70)
    print("TVLQR TRACKING DIAGNOSTIC")
    print("="*70)
    
    # Create satellite and orbit
    np.random.seed(42)
    real_sat = create_beavercube2_cubesat(estimated=False)
    
    tf = 100  # Shorter for debugging
    dt = 2
    
    orb = create_random_circular_orbit(radius_km=7000.0, dt=1, tf=tf, use_J2=True, fast=True)
    orb.populate_environment(compute_B=True, compute_S=True)
    
    # Initial state: random orientation, small angular velocity
    q0 = normalize(np.random.randn(4))
    w0 = np.array([0.01, 0.01, 0.01])  # Small initial rate
    h0 = np.array([0.0])  # Zero RW momentum
    x0 = np.concatenate([w0, q0, h0])
    
    # Goal: just use q0 as goal (converge to initial) for simplicity
    q_goal = q0.copy()
    
    print(f"\nInitial state:")
    print(f"  w0 = {np.degrees(w0)} deg/s")
    print(f"  q0 = {q0}")
    print(f"  q_goal = {q_goal}")
    
    # Create planner and controller
    planner_settings = create_optimized_planner_settings(
        real_sat, duration=tf, dt_planning=1, tuning="fast_slew"
    )
    planner_settings.verbosity = True
    
    controller = Plan_and_Track_LQR(est_sat=real_sat, planner_settings=planner_settings)
    
    goals = GoalList({0.22: Fixed_Attitude_Goal(q_goal)})
    os0 = orb.get_os(0.22)
    
    print("\n" + "="*70)
    print("PLANNING TRAJECTORY")
    print("="*70)
    
    traj = controller.calculate_trajectory(
        t_start=0.22, duration=tf, x_0=x0, os_0=os0, goals=goals, verbose=True
    )
    controller.set_active_trajectory(traj)
    
    print(f"\nTrajectory created:")
    print(f"  states shape: {traj.states.shape}")
    print(f"  controls shape: {traj.controls.shape}")
    print(f"  gains shape: {traj.gains.shape}")
    print(f"  times: {traj.times[0]:.6f} to {traj.times[-1]:.6f} (J2000 centuries)")
    print(f"  n_steps: {traj.n_steps}")
    print(f"  state_dim: {traj.state_dim}, ctrl_dim: {traj.ctrl_dim}")
    
    # Check trajectory endpoint
    x_end = traj.states[:, -1]
    w_end = x_end[0:3]
    q_end = x_end[3:7]
    print(f"\nTrajectory endpoint:")
    print(f"  w_end = {np.degrees(w_end)} deg/s")
    print(f"  q_end = {q_end}")
    angle_end = 2 * np.arccos(np.clip(np.abs(np.dot(q_end, q_goal)), 0, 1))
    print(f"  angle from goal = {np.degrees(angle_end):.2f} deg")
    
    print("\n" + "="*70)
    print("K-GAIN ANALYSIS")
    print("="*70)
    
    # Check K-gain format
    K = traj.gains
    print(f"\nK-gains array shape: {K.shape}")
    print(f"  ndim: {K.ndim}")
    
    # Get K at t=0
    t0 = traj.times[0]
    K0 = traj.get_gain_at(t0)
    print(f"\nK at t=0:")
    print(f"  shape: {K0.shape}")
    print(f"  K0 =\n{K0}")
    
    # Check for reasonable K magnitude
    print(f"\n  K0 norms by row (control):")
    for i in range(K0.shape[0]):
        print(f"    K0[{i},:] norm = {np.linalg.norm(K0[i,:]):.6f}")
    
    # Check K at middle
    t_mid = (traj.times[0] + traj.times[-1]) / 2
    K_mid = traj.get_gain_at(t_mid)
    print(f"\nK at t_mid:")
    print(f"  K_mid =\n{K_mid}")
    
    print("\n" + "="*70)
    print("STATE DIFF ANALYSIS")
    print("="*70)
    
    # Get reference state at t=0
    x_ref = traj.get_state_at(t0)
    print(f"\nx_ref at t=0: {x_ref}")
    print(f"x0 (actual):  {x0}")
    
    # Compute state diff manually
    dx_manual = np.zeros(6 + 1)  # 6 + n_rw
    dx_manual[0:3] = x0[0:3] - x_ref[0:3]  # omega error
    q_err = quat_diff(x_ref[3:7], x0[3:7])
    dx_manual[3:6] = quat_to_vec3(q_err, mode=0)  # attitude error (MRP mode 0)
    dx_manual[6:] = x0[7:] - x_ref[7:]  # RW momentum error
    
    print(f"\nManual dx computation:")
    print(f"  q_ref = {x_ref[3:7]}")
    print(f"  q_curr = {x0[3:7]}")
    print(f"  q_err = {q_err}")
    print(f"  quat_to_vec3(q_err, mode=0) = {quat_to_vec3(q_err, mode=0)}")
    print(f"  dx_manual = {dx_manual}")
    
    # Use trajectory's _state_diff
    dx_traj = traj._state_diff(x0, x_ref)
    print(f"\nTrajectory._state_diff result:")
    print(f"  dx_traj = {dx_traj}")
    print(f"  diff from manual: {np.max(np.abs(dx_manual - dx_traj)):.2e}")
    
    print("\n" + "="*70)
    print("CONTROL COMPUTATION ANALYSIS")
    print("="*70)
    
    # Get reference control
    u_ref = traj.get_control_at(t0)
    print(f"\nu_ref at t=0: {u_ref}")
    
    # Compute control manually
    # u = u_ref - K @ dx
    u_manual = u_ref - K0 @ dx_traj
    print(f"\nManual control computation:")
    print(f"  u = u_ref - K @ dx")
    print(f"  K @ dx = {K0 @ dx_traj}")
    print(f"  u_manual = {u_manual}")
    
    # Use controller's find_u
    os_state = orb.get_os(0.22)
    sens = np.zeros(10)  # dummy
    u_controller = controller.find_u(x_hat=x0, sens=sens, est_sat=real_sat, os_hat=os_state)
    print(f"\nController.find_u result:")
    print(f"  u_controller = {u_controller}")
    
    # Compare
    print(f"\nDifference: {np.max(np.abs(u_manual - u_controller)):.2e}")
    
    print("\n" + "="*70)
    print("TRACKING SIMULATION (10 steps)")
    print("="*70)
    
    x = x0.copy()
    t = 0
    sec2cent = TimeConstants.sec2cent
    
    from scipy.integrate import solve_ivp
    
    for i in range(10):
        J2000 = 0.22 + t * sec2cent
        os_state = orb.get_os(J2000=J2000)
        
        # Get trajectory reference at this time
        x_ref = traj.get_state_at(J2000)
        u_ref = traj.get_control_at(J2000)
        K = traj.get_gain_at(J2000)
        
        # Compute dx
        dx = traj._state_diff(x, x_ref)
        
        # Compute control
        u = controller.find_u(x_hat=x, sens=np.zeros(10), est_sat=real_sat, os_hat=os_state)
        
        # Angle from goal
        q = x[3:7]
        q_ref = x_ref[3:7]
        angle = np.degrees(2 * np.arccos(np.clip(np.abs(np.dot(q, q_goal)), 0, 1)))
        angle_ref = np.degrees(2 * np.arccos(np.clip(np.abs(np.dot(q_ref, q_goal)), 0, 1)))
        tracking_error = np.degrees(2 * np.arccos(np.clip(np.abs(np.dot(q, q_ref)), 0, 1)))
        
        print(f"\nStep {i}: t={t:.1f}s")
        print(f"  x[w] = {np.degrees(x[0:3])} deg/s")
        print(f"  x_ref[w] = {np.degrees(x_ref[0:3])} deg/s")
        print(f"  dx = {dx}")
        print(f"  K[0,:] = {K[0,:]}")
        print(f"  K @ dx = {K @ dx}")
        print(f"  u_ref = {u_ref}")
        print(f"  u = {u}")
        print(f"  ACTUAL angle from goal = {angle:.2f}°")
        print(f"  REF angle from goal = {angle_ref:.2f}°")  
        print(f"  TRACKING error = {tracking_error:.2f}°")
        
        # Propagate
        t_next = t + dt
        os_next = orb.get_os(0.22 + t_next * sec2cent)
        out = solve_ivp(
            real_sat.dynamics_for_solver, (0, dt), x, method="RK45",
            args=(u, os_state, os_next), rtol=1e-6, atol=1e-6
        )
        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])
        t = t_next
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)

if __name__ == "__main__":
    main()
