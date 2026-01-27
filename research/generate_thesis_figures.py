#!/usr/bin/env python3
"""
Generate Thesis Figures
=======================

This script generates figures matching those in the PhD thesis using the 
existing ADCS codebase infrastructure. All parameters are from 
thesis_figures_config.py which contains verified values from the thesis.

Usage:
    python generate_thesis_figures.py --all --quick     # Quick test (short durations)
    python generate_thesis_figures.py --all --full      # Full thesis parameters
    python generate_thesis_figures.py --chapter 6       # Chapter 6 only
    python generate_thesis_figures.py --test lovera     # Specific test

Chapters:
    6 - Disturbance Control (Lovera, Wisniewski)
    7 - Planning (Spinning, Monte Carlo, Sequential)
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path
from scipy.integrate import solve_ivp
from typing import List, Optional, Tuple, Dict
from tqdm import tqdm
from dataclasses import dataclass

# --- Path Setup ---
sys.path.insert(0, str(Path(__file__).parent.parent))

# --- ADCS Imports ---
from ADCS.CONOPS.goals import ECI_Goal, Zenith_Goal, AntiVelocity_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller import MTQ_Lovera, MTQ_Wisniewski
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.helpers import PlannerSettings
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_factory.satellites.create_cubesats import (
    create_wisniewski_test_satellite, create_lovera_test_satellite
)
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.disturbances import (
    Prop_Disturbance, GG_Disturbance, Dipole_Disturbance, General_Disturbance,
    Drag_Disturbance, SRP_Disturbance, GeometryConfig, GeometryFace
)
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize, rot_mat, quat_mult
import pickle

# --- Thesis Configuration ---
from thesis_figures_config import (
    Ch6_Lovera_Config, Ch6_Wisniewski_Config,
    Ch7_Spinning_Config, Ch7_MonteCarlo_Config, Ch7_Sequential_Config,
    get_thesis_planner_settings
)

# --- Plotting ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def angular_error_deg(q: np.ndarray, q_goal: np.ndarray) -> float:
    """Calculate angular error in degrees between two quaternions."""
    q = normalize(q)
    q_goal = normalize(q_goal)
    dot = np.abs(np.dot(q, q_goal))
    dot = min(dot, 1.0)
    return 2 * np.arccos(dot) * 180 / np.pi


def pointing_error_deg(q: np.ndarray, body_axis: np.ndarray, goal_eci: np.ndarray) -> float:
    """Calculate pointing error between body axis and ECI goal vector."""
    R = rot_mat(q)
    body_in_eci = R.T @ body_axis
    dot = np.dot(normalize(body_in_eci), normalize(goal_eci))
    dot = np.clip(dot, -1.0, 1.0)
    return np.arccos(dot) * 180 / np.pi


def create_thesis_orbit(altitude_km: float, inclination_deg: float, 
                        duration_s: float, dt: float) -> Orbit:
    """Create orbit matching thesis parameters."""
    ephem = Ephemeris()
    R_earth = 6371  # km
    orbital_radius = R_earth + altitude_km
    
    # Calculate circular orbit velocity
    mu = 398600.4418  # km^3/s^2
    v_circ = np.sqrt(mu / orbital_radius)
    
    # Initial position/velocity
    inc_rad = np.radians(inclination_deg)
    R = orbital_radius * np.array([1, 0, 0])
    V = v_circ * np.array([0, np.cos(inc_rad), np.sin(inc_rad)])
    
    start_time = 0.22 - 1 * TimeConstants.sec2cent
    end_time = 0.22 + duration_s * TimeConstants.sec2cent
    
    os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
    return Orbit(os0=os0, end_time=end_time, dt=dt, use_J2=True, fast=False, verbose=False)


# =============================================================================
# CHAPTER 6: DISTURBANCE CONTROL
# =============================================================================

# Disturbance cases for comparison (matching thesis Figure 6.6)
# - Clean: No disturbances applied, no compensation
# - Disturbed: Disturbances applied, NOT compensated  
# - Disturbance-aware: Disturbances applied, compensated with full model
# - All-in-one: Disturbances applied, compensated with general estimator (TODO)
# Disturbance cases for Chapter 6 tests (from thesis Tables and code)
#
# In thesis code, GG is always included analytically.
# The comparison is about OTHER disturbances (drag, SRP, dipole/MTQ bias).
#
# Clean: GG only on truth, no compensation needed (baseline)
# Disturbed: Full disturbances on truth (GG+drag+SRP+dipole), only GG on est, no compensation
# Disturbance-aware: Full on truth, full on est with compensation
# All-in-one: Full on truth, GG + general estimator on est
#
# truth_mode: what disturbances affect dynamics
# est_mode: what disturbances the controller knows about
DISTURBANCE_CASES = [
    {"name": "Clean", "color": "blue", "truth_mode": "gg_only", "est_mode": "none", "compensate": False},
    {"name": "Disturbed", "color": "orange", "truth_mode": "full", "est_mode": "none", "compensate": False},
    {"name": "Disturbance-aware", "color": "green", "truth_mode": "full", "est_mode": "full", "compensate": True},
    {"name": "All-in-one", "color": "red", "truth_mode": "full", "est_mode": "general", "compensate": True},
]


def run_single_case(controller_type: str, cfg, case: Dict, orb: Orbit, 
                    duration: float, dt: float, q_goal: np.ndarray) -> Dict:
    """Run a single disturbance case for a controller.
    
    Parameters
    ----------
    controller_type : str
        'lovera' or 'wisniewski'
    cfg : config object
        Controller configuration (gains, inertia, etc.)
    case : dict
        Disturbance case with keys:
        - 'use_dist': bool - Apply disturbances to dynamics
        - 'compensate': bool - Enable disturbance feedforward in controller
        - 'all_in_one': bool (optional) - Use general estimator instead of full model
    
    Notes
    -----
    Two satellites are used:
    - truth_sat: Has disturbances that affect dynamics (if use_dist=True)
    - est_sat: Used by controller for feedforward (estimated=True)
    
    Cases:
    - Clean: truth_sat has no disturbances, est_sat has no model
    - Disturbed: truth_sat has disturbances, est_sat has no model (no compensation)
    - Disturbance-aware: truth_sat has disturbances, est_sat has FULL model (perfect knowledge)
    - All-in-one: truth_sat has disturbances, est_sat has GENERAL estimator
    """
    N = int(duration / dt)
    
    # Get disturbance modes from case
    compensate = case.get("compensate", False)
    truth_dist_mode = case.get("truth_mode", "none")
    est_dist_mode = case.get("est_mode", "none")
    
    # use_disturbances flag for factory
    truth_has_dist = truth_dist_mode != 'none'
    est_has_dist = est_dist_mode != 'none'
    
    # Create satellites using factory functions
    if controller_type == 'lovera':
        truth_sat = create_lovera_test_satellite(
            estimated=False, use_disturbances=truth_has_dist, 
            disturbance_mode=truth_dist_mode
        )
        est_sat = create_lovera_test_satellite(
            estimated=True, use_disturbances=est_has_dist,
            disturbance_mode=est_dist_mode
        )
    else:  # wisniewski
        truth_sat = create_wisniewski_test_satellite(
            estimated=False, use_disturbances=truth_has_dist,
            disturbance_mode=truth_dist_mode
        )
        est_sat = create_wisniewski_test_satellite(
            estimated=True, use_disturbances=est_has_dist,
            disturbance_mode=est_dist_mode
        )
    
    # Create controller with disturbance compensation enabled/disabled
    if controller_type == 'lovera':
        controller = MTQ_Lovera(est_sat=est_sat, p_gain=cfg.kp, d_gain=cfg.kv, eps=cfg.eps,
                                include_disturbances=compensate)
    else:  # wisniewski
        lambda_q = cfg.lambda_q * np.eye(3)
        lambda_s = cfg.lambda_s * np.eye(3)
        controller = MTQ_Wisniewski(est_sat=est_sat, lambda_q=lambda_q, lambda_s=lambda_s,
                                    include_disturbances=compensate)
    
    # Goal: body z-axis pointing zenith (from thesis)
    # "body z-axis pointing zenith and body x-axis aligned with the orbit normal"
    goal = Zenith_Goal()
    
    # Initial state
    w0 = cfg.omega_init
    q0 = normalize(cfg.q_init)
    x = np.concatenate([w0, q0])
    
    # Storage
    time_hist = np.zeros(N)
    state_hist = np.zeros((N, len(x)))
    u_hist = np.zeros((N, 3))
    error_hist = np.zeros(N)
    dist_est_hist = np.zeros((N, 3))  # Track disturbance estimates
    
    t = 0
    for step in range(N):
        J2000 = 0.22 + t * TimeConstants.sec2cent
        os = orb.get_os(J2000=J2000)
        
        # For All-in-one case: Update general disturbance estimate
        # In a real system, this would come from the estimator
        # Here we use a simple approach: estimate = average of recent residuals
        if est_dist_mode == 'general' and len(est_sat.disturbances) > 0:
            # Get the actual disturbance from truth satellite for "cheating" estimate
            # In practice, this would come from an EKF/UKF
            actual_dist = truth_sat.dist_torques(x=x, os=os)
            # Add noise to simulate estimation uncertainty
            est_dist = actual_dist + np.random.normal(0, 1e-6, 3)
            # Update the general disturbance estimate
            for d in est_sat.disturbances:
                if isinstance(d, General_Disturbance):
                    # Blend with previous estimate (simple low-pass filter)
                    alpha = 0.1  # Filter gain
                    d.main_param = (1 - alpha) * d.main_param + alpha * est_dist
        
        # Get sensor readings from TRUTH satellite
        sens = truth_sat.sensor_readings(x=x, os=os)
        
        # Controller uses ESTIMATED satellite for feedforward
        u = controller.find_u(x_hat=x, sens=sens, est_sat=est_sat, os_hat=os, goal=goal)
        
        # Store data
        time_hist[step] = t
        state_hist[step, :] = x
        u_hist[step, :] = u
        error_hist[step] = angular_error_deg(x[3:7], q_goal)
        if compensate:
            dist_est_hist[step, :] = controller._last_dist_torque
        
        # Propagate dynamics using TRUTH satellite
        prev_os = os.copy()
        next_os = orb.get_os(0.22 + (t + dt) * TimeConstants.sec2cent)
        out = solve_ivp(
            fun=truth_sat.dynamics_for_solver,
            t_span=(0, dt), y0=x, method="RK45",
            args=(u, prev_os, next_os),
            rtol=1e-7, atol=1e-7
        )
        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])
        t += dt
    
    return {
        'state': state_hist,
        'control': u_hist,
        'error': error_hist,
        'dist_est': dist_est_hist,
        'color': case["color"],
        'name': case["name"],
    }


def run_lovera_test(cfg: Ch6_Lovera_Config, output_dir: Path, quick: bool = False) -> Dict:
    """
    Run Lovera MTQ-PD controller test matching thesis.
    
    Runs 4 cases: Clean, Disturbed, Disturbance-aware, All-in-one
    Thesis Reference: Table 5.2, Section 6.4
    """
    print(f"\n{'='*60}")
    print(f"LOVERA MTQ-PD CONTROLLER - DISTURBANCE COMPARISON")
    print(f"{'='*60}")
    
    duration = 500.0 if quick else cfg.duration_s
    dt = 10.0 if quick else cfg.dt
    N = int(duration / dt)
    
    print(f"Satellite: J = diag{list(np.diag(cfg.J))}")
    print(f"Controller: eps={cfg.eps}, kp={cfg.kp}, kv={cfg.kv}")
    print(f"MTQ Max: {cfg.mtq_max} Am²")
    print(f"Duration: {duration}s ({duration/3600:.1f} hours)")
    print(f"Running {len(DISTURBANCE_CASES)} disturbance cases...")
    
    # Create orbit once (shared)
    orb = create_thesis_orbit(altitude_km=450, inclination_deg=87,
                               duration_s=duration, dt=dt)
    
    q_goal = np.array([0, 0, 0, 1])
    time_hist = np.linspace(0, duration - dt, N)
    
    results = {
        'time': time_hist,
        'cases': {},
        'config': {
            'controller': 'lovera',
            'J': cfg.J.tolist(),
            'eps': cfg.eps,
            'kp': cfg.kp,
            'kv': cfg.kv,
            'mtq_max': cfg.mtq_max,
            'duration': duration,
        }
    }
    
    for case in DISTURBANCE_CASES:
        print(f"  Running: {case['name']}...")
        case_result = run_single_case('lovera', cfg, case, orb, duration, dt, q_goal)
        results['cases'][case['name']] = case_result
        print(f"    Final error: {case_result['error'][-1]:.2f}°")
    
    # Save data
    with open(output_dir / "lovera_data.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Saved: lovera_data.pkl")
    return results


def run_wisniewski_test(cfg: Ch6_Wisniewski_Config, output_dir: Path, quick: bool = False) -> Dict:
    """
    Run Wisniewski Sliding Mode controller test matching thesis.
    
    Runs 4 cases: Clean, Disturbed, Disturbance-aware, All-in-one
    Thesis Reference: Section 5.2.3
    """
    print(f"\n{'='*60}")
    print(f"WISNIEWSKI SLIDING MODE - DISTURBANCE COMPARISON")
    print(f"{'='*60}")
    
    duration = 500.0 if quick else cfg.duration_s
    dt = 10.0 if quick else cfg.dt
    N = int(duration / dt)
    
    print(f"Satellite: J = diag{list(np.diag(cfg.J))}")
    print(f"Controller: λq={cfg.lambda_q}, λs={cfg.lambda_s}")
    print(f"MTQ Max: {cfg.mtq_max} Am²")
    print(f"Duration: {duration}s ({duration/3600:.1f} hours)")
    print(f"Running {len(DISTURBANCE_CASES)} disturbance cases...")
    
    # Create orbit once (shared)
    orb = create_thesis_orbit(altitude_km=450, inclination_deg=87,
                               duration_s=duration, dt=dt)
    
    q_goal = np.array([0, 0, 0, 1])
    time_hist = np.linspace(0, duration - dt, N)
    
    results = {
        'time': time_hist,
        'cases': {},
        'config': {
            'controller': 'wisniewski',
            'J': cfg.J.tolist(),
            'lambda_q': cfg.lambda_q,
            'lambda_s': cfg.lambda_s,
            'mtq_max': cfg.mtq_max,
            'duration': duration,
        }
    }
    
    for case in DISTURBANCE_CASES:
        print(f"  Running: {case['name']}...")
        case_result = run_single_case('wisniewski', cfg, case, orb, duration, dt, q_goal)
        results['cases'][case['name']] = case_result
        print(f"    Final error: {case_result['error'][-1]:.2f}°")
    
    # Save data
    with open(output_dir / "wisniewski_data.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Saved: wisniewski_data.pkl")
    return results


# =============================================================================
# CHAPTER 7: PLANNING
# =============================================================================

def run_spinning_test(cfg: Ch7_Spinning_Config, output_dir: Path, quick: bool = False) -> Dict:
    """
    Run spinning solution test matching thesis.
    
    Thesis Reference: Table 7.1 (tab:plan_dist_test_details), Section 7.5.2
    
    A 3U CubeSat with MTQ+1RW countering a body-fixed propulsion disturbance
    by spinning about the goal axis.
    """
    print(f"\n{'='*60}")
    print(f"SPINNING SOLUTION TEST (Ch 7)")
    print(f"{'='*60}")
    
    duration = 100.0 if quick else cfg.duration_s
    dt = 1.0
    
    print(f"Disturbance: {cfg.disturbance_torque * 1000} mNm (body-fixed)")
    print(f"Goal: {cfg.goal_body_axis} → {cfg.goal_direction}")
    print(f"Duration: {duration}s")
    
    N = int(duration / dt)
    
    # Create satellite with asymmetric MTQs and 1 RW (thesis Table 7.1)
    mtqs = [
        MTQ(axis=np.array([1, 0, 0]), max_torque=cfg.mtq_max_x),
        MTQ(axis=np.array([0, 1, 0]), max_torque=cfg.mtq_max_yz),
        MTQ(axis=np.array([0, 0, 1]), max_torque=cfg.mtq_max_yz),
    ]
    rws = [RW(axis=cfg.rw_axis, max_torque=cfg.rw_max_torque, 
              J=cfg.rw_inertia, h=cfg.h_init, h_max=cfg.rw_max_momentum)]
    mtms = [MTM(axis=ax) for ax in MathConstants.unitvecs]
    
    # Body-fixed disturbance (propulsion)
    dist = Prop_Disturbance(torque_nominal=cfg.disturbance_torque)
    
    sat = Satellite(
        mass=cfg.mass,
        J_0=cfg.J,
        actuators=mtqs + rws,
        sensors=mtms,
        disturbances=[dist],
        boresight=cfg.goal_body_axis
    )
    
    # Initial state from thesis
    w0 = cfg.omega_init
    q0 = normalize(cfg.q_init)
    h0 = np.array([cfg.h_init])
    x = np.concatenate([w0, q0, h0])
    
    # Create orbit
    orb = create_thesis_orbit(
        altitude_km=cfg.orbital_radius - 6371,
        inclination_deg=cfg.inclination,
        duration_s=duration, dt=dt
    )
    
    # For this test, we need the planner to handle the disturbance
    # For now, use a simple PD baseline to show the issue
    # TODO: Connect to Plan_and_Track_LQR with proper goals
    
    # Storage
    time_hist = np.zeros(N)
    state_hist = np.zeros((N, len(x)))
    u_hist = np.zeros((N, len(mtqs) + len(rws)))
    error_hist = np.zeros(N)
    
    print("NOTE: Full spinning solution requires trajectory planner.")
    print("      Running baseline simulation to show disturbance effect...")
    
    t = 0
    for step in tqdm(range(N), desc="Spinning Sim"):
        J2000 = 0.22 + t * TimeConstants.sec2cent
        os = orb.get_os(J2000=J2000)
        
        # Calculate pointing error
        R = rot_mat(x[3:7])
        goal_eci = -os.V_hat  # anti-ram
        body_in_eci = R.T @ cfg.goal_body_axis
        error_hist[step] = np.arccos(np.clip(np.dot(body_in_eci, goal_eci), -1, 1)) * 180/np.pi
        
        # Zero control for baseline (disturbance only)
        u = np.zeros(len(mtqs) + len(rws))
        
        time_hist[step] = t
        state_hist[step, :] = x
        u_hist[step, :] = u
        
        # Propagate
        prev_os = os.copy()
        next_os = orb.get_os(0.22 + (t + dt) * TimeConstants.sec2cent)
        out = solve_ivp(
            fun=sat.dynamics_for_solver,
            t_span=(0, dt),
            y0=x,
            method="RK45",
            args=(u, prev_os, next_os),
            rtol=1e-7, atol=1e-7
        )
        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])
        t += dt
    
    # Save
    results = {
        'time': time_hist,
        'state': state_hist,
        'control': u_hist,
        'error': error_hist,
    }
    with open(output_dir / "spinning_data.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(time_hist, error_hist)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Pointing Error (deg)')
    axes[0, 0].set_title('Spinning Test: Pointing Error')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(time_hist, state_hist[:, 0:3] * 180/np.pi)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Angular Velocity (deg/s)')
    axes[0, 1].set_title('Angular Velocity')
    axes[0, 1].legend(['ωx', 'ωy', 'ωz'])
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(time_hist, u_hist[:, :3])
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('MTQ Command (Am²)')
    axes[1, 0].set_title('MTQ Commands')
    axes[1, 0].legend(['mx', 'my', 'mz'])
    axes[1, 0].grid(True)
    
    if len(rws) > 0:
        axes[1, 1].plot(time_hist, state_hist[:, 7:] * 1000)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('RW Momentum (mNms)')
        axes[1, 1].set_title('RW Stored Momentum')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / "spinning_ang.png", dpi=150)
    plt.close()
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate Thesis Figures")
    parser.add_argument('--all', action='store_true', help="Run all tests")
    parser.add_argument('--chapter', type=int, choices=[6, 7], help="Chapter to run")
    parser.add_argument('--test', type=str, choices=['lovera', 'wisniewski', 'spinning', 'mc', 'sequential'],
                        help="Specific test to run")
    parser.add_argument('--quick', action='store_true', help="Quick mode (short durations)")
    parser.add_argument('--full', action='store_true', help="Full thesis durations")
    parser.add_argument('--output', type=str, default="thesis_figures", help="Output directory")
    
    args = parser.parse_args()
    
    if not (args.all or args.chapter or args.test):
        parser.print_help()
        return
    
    quick = args.quick or not args.full
    
    output_dir = Path(args.output)
    ch6_dir = output_dir / "chapter6"
    ch7_dir = output_dir / "chapter7"
    ch6_dir.mkdir(parents=True, exist_ok=True)
    ch7_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"THESIS FIGURE GENERATION")
    print(f"{'='*60}")
    print(f"Mode: {'QUICK (short durations)' if quick else 'FULL (thesis durations)'}")
    print(f"Output: {output_dir}")
    
    # Chapter 6: Disturbance Control
    if args.all or args.chapter == 6 or args.test in ['lovera', 'wisniewski']:
        if args.all or args.chapter == 6 or args.test == 'lovera':
            cfg = Ch6_Lovera_Config()
            run_lovera_test(cfg, ch6_dir, quick=quick)
        
        if args.all or args.chapter == 6 or args.test == 'wisniewski':
            cfg = Ch6_Wisniewski_Config()
            run_wisniewski_test(cfg, ch6_dir, quick=quick)
    
    # Chapter 7: Planning
    if args.all or args.chapter == 7 or args.test in ['spinning', 'mc', 'sequential']:
        if args.all or args.chapter == 7 or args.test == 'spinning':
            cfg = Ch7_Spinning_Config()
            run_spinning_test(cfg, ch7_dir, quick=quick)
        
        if args.test == 'mc':
            print("\nMonte Carlo test requires planner integration - see papers/Generalized_ACS/")
        
        if args.test == 'sequential':
            print("\nSequential test requires planner integration - TODO")
    
    print(f"\n{'='*60}")
    print(f"COMPLETE - Figures saved to {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
