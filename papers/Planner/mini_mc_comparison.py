#!/usr/bin/env python3
"""
Mini Monte Carlo comparison: Legacy vs Normalized planner settings.

Runs 10 random cases for each configuration and compares:
- Planning time
- Final pointing error
- Control smoothness (RW sign changes)
- Convergence success rate
"""

import numpy as np
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/pmckeen/Generalized_ADCS')

from ADCS.CONOPS.goals import Fixed_Attitude_Goal, Nadir_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.helpers import (
    PlannerSettings, create_planner_settings,
    NormalizedPlannerConfig, NormalizedActuatorCosts, NormalizedStateCosts,
)
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.helpers.math_helpers import normalize, quat_mult, quat_diff


def quat_error_angle(q1, q2):
    """Compute angle between two quaternions in degrees."""
    q_err = quat_diff(q1, q2)
    return 2 * np.arccos(np.clip(np.abs(q_err[0]), 0, 1)) * 180 / np.pi


@dataclass
class MCResult:
    """Results from a single MC run."""
    seed: int
    time_s: float
    error_deg: float
    rw_sign_changes_pct: float
    converged: bool
    initial_error_deg: float
    

def create_legacy_settings(sat) -> PlannerSettings:
    """Create legacy (default) planner settings."""
    settings = PlannerSettings(est_sat=sat, bdot_on=0)
    return settings


def create_normalized_settings(sat) -> PlannerSettings:
    """Create well-conditioned normalized settings."""
    config = NormalizedPlannerConfig(
        actuator_costs=NormalizedActuatorCosts(
            mtq_cost=1.0,
            rw_torque_cost=5.0,  # Good conditioning
            rw_momentum_cost=10.0,
            rw_stiction_cost=1.0,
        ),
        state_costs=NormalizedStateCosts(
            angle_cost=1000.0,
            angle_terminal_cost=1000000.0,
            ang_vel_cost=1000.0,
            ang_vel_terminal_cost=100000.0,
        ),
    )
    settings = create_planner_settings(sat, config)
    settings.rw_AM_weight = 1e4
    settings.RWh_ok_mult = 0.5
    return settings


def run_single_case(
    sat,
    orb,
    seed: int,
    settings: PlannerSettings,
    duration: float = 120.0,
    slew_angle_deg: float = 90.0,
) -> MCResult:
    """Run a single MC case."""
    rng = np.random.default_rng(seed=seed)
    
    # Random initial attitude
    q0 = normalize(rng.standard_normal(4))
    w0 = normalize(rng.standard_normal(3)) * (0.5 * np.pi / 180.0)  # 0.5 deg/s
    h0 = rng.uniform(-0.0001, 0.0001, size=len(sat.rw_actuators))
    
    # Random slew direction
    axis = normalize(rng.standard_normal(3))
    half_angle = slew_angle_deg / 2 * np.pi / 180
    q_rot = np.concatenate([[np.cos(half_angle)], np.sin(half_angle) * axis])
    q_goal = normalize(quat_mult(q0, q_rot))
    
    initial_error = quat_error_angle(q0, q_goal)
    
    # Set initial RW momentum
    x0 = np.concatenate([w0, q0, h0])
    for i, rw in enumerate(sat.rw_actuators):
        rw.h = h0[i]
    
    # Create goal and orbital state
    t_start = 0.22
    goals = GoalList({t_start: Fixed_Attitude_Goal(q_goal)})
    os0 = orb.get_os(t_start)
    
    # Create controller and run
    controller = Plan_and_Track_LQR(est_sat=sat, planner_settings=settings)
    
    try:
        t0 = time.perf_counter()
        traj = controller.calculate_trajectory(
            t_start=t_start, duration=duration, x_0=x0, os_0=os0, 
            goals=goals, verbose=False
        )
        elapsed = time.perf_counter() - t0
        
        # Compute metrics
        q_final = traj.states[3:7, -1]
        error = quat_error_angle(q_final, q_goal)
        
        # RW control smoothness
        if traj.controls.shape[0] > 3:
            rw_ctrl = traj.controls[3, :]
            sign_changes = np.sum(np.diff(np.sign(rw_ctrl)) != 0)
            sign_change_pct = sign_changes / max(1, len(rw_ctrl) - 1) * 100
        else:
            sign_change_pct = 0.0
        
        converged = error < 5.0  # Less than 5 degrees is "converged"
        
        return MCResult(
            seed=seed,
            time_s=elapsed,
            error_deg=error,
            rw_sign_changes_pct=sign_change_pct,
            converged=converged,
            initial_error_deg=initial_error,
        )
        
    except Exception as e:
        print(f"  Seed {seed} failed: {e}")
        return MCResult(
            seed=seed,
            time_s=float('nan'),
            error_deg=float('nan'),
            rw_sign_changes_pct=float('nan'),
            converged=False,
            initial_error_deg=initial_error,
        )


def run_mc_suite(
    name: str,
    sat,
    orb,
    settings: PlannerSettings,
    n_cases: int = 10,
    duration: float = 120.0,
) -> List[MCResult]:
    """Run a Monte Carlo suite."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    
    results = []
    for i in range(n_cases):
        seed = 1000 + i
        print(f"  Case {i+1}/{n_cases} (seed={seed})...", end=" ", flush=True)
        result = run_single_case(sat, orb, seed, settings, duration)
        results.append(result)
        
        if result.converged:
            print(f"✓ {result.time_s:.2f}s, {result.error_deg:.2f}°")
        else:
            print(f"✗ {result.time_s:.2f}s, {result.error_deg:.2f}°")
    
    return results


def summarize_results(name: str, results: List[MCResult]) -> Dict:
    """Compute summary statistics."""
    valid = [r for r in results if not np.isnan(r.time_s)]
    
    if not valid:
        return {'name': name, 'n_valid': 0}
    
    times = [r.time_s for r in valid]
    errors = [r.error_deg for r in valid]
    sign_changes = [r.rw_sign_changes_pct for r in valid]
    converged = [r.converged for r in valid]
    
    return {
        'name': name,
        'n_valid': len(valid),
        'n_converged': sum(converged),
        'time_mean': np.mean(times),
        'time_std': np.std(times),
        'error_mean': np.mean(errors),
        'error_std': np.std(errors),
        'error_median': np.median(errors),
        'sign_change_mean': np.mean(sign_changes),
        'converge_rate': sum(converged) / len(valid) * 100,
    }


def print_comparison(summaries: List[Dict]):
    """Print comparison table."""
    print("\n" + "="*80)
    print("MONTE CARLO RESULTS COMPARISON")
    print("="*80)
    
    headers = ['Config', 'N', 'Conv%', 'Time(s)', 'Error(°)', 'Err Med', 'RW Chatter']
    widths = [25, 5, 8, 12, 12, 10, 12]
    
    # Header
    header_str = ""
    for h, w in zip(headers, widths):
        header_str += f"{h:<{w}}"
    print(header_str)
    print("-" * 80)
    
    # Data rows
    for s in summaries:
        if s['n_valid'] == 0:
            print(f"{s['name']:<25} No valid results")
            continue
        
        row = [
            s['name'],
            f"{s['n_valid']}",
            f"{s['converge_rate']:.0f}%",
            f"{s['time_mean']:.2f}±{s['time_std']:.2f}",
            f"{s['error_mean']:.2f}±{s['error_std']:.2f}",
            f"{s['error_median']:.2f}",
            f"{s['sign_change_mean']:.1f}%",
        ]
        
        row_str = ""
        for val, w in zip(row, widths):
            row_str += f"{val:<{w}}"
        print(row_str)


def main():
    print("="*80)
    print("MINI MONTE CARLO: Legacy vs Normalized Planner Settings")
    print("="*80)
    print()
    print("Configuration:")
    print("  - Satellite: BeaverCube2 (3 MTQ + 1 RW)")
    print("  - Slew: 90° random axis")
    print("  - Horizon: 120s")
    print("  - Cases per config: 10")
    print()
    
    # Setup
    np.random.seed(42)
    sat = create_beavercube2_cubesat(estimated=False)
    
    print("Creating orbit...")
    orb = create_random_circular_orbit(radius_km=7000.0, dt=1, tf=200, use_J2=True, fast=True)
    orb.populate_environment(compute_B=True, compute_S=True)
    
    n_cases = 10
    duration = 120.0
    
    all_summaries = []
    
    # 1. Legacy settings
    legacy_settings = create_legacy_settings(sat)
    legacy_results = run_mc_suite("Legacy (default)", sat, orb, legacy_settings, n_cases, duration)
    all_summaries.append(summarize_results("Legacy (default)", legacy_results))
    
    # 2. Normalized well-conditioned
    norm_settings = create_normalized_settings(sat)
    norm_results = run_mc_suite("Normalized (well-cond)", sat, orb, norm_settings, n_cases, duration)
    all_summaries.append(summarize_results("Normalized (well-cond)", norm_results))
    
    # 3. Normalized with higher RW cost (more aggressive)
    aggressive_config = NormalizedPlannerConfig(
        actuator_costs=NormalizedActuatorCosts(
            mtq_cost=1.0,
            rw_torque_cost=20.0,  # Higher = prefer MTQ more
        ),
        state_costs=NormalizedStateCosts(
            angle_cost=1000.0,
            angle_terminal_cost=1000000.0,
            ang_vel_cost=1000.0,
            ang_vel_terminal_cost=100000.0,
        ),
    )
    aggressive_settings = create_planner_settings(sat, aggressive_config)
    aggressive_settings.rw_AM_weight = 1e4
    aggressive_settings.RWh_ok_mult = 0.5
    aggressive_results = run_mc_suite("Normalized (MTQ-prefer)", sat, orb, aggressive_settings, n_cases, duration)
    all_summaries.append(summarize_results("Normalized (MTQ-prefer)", aggressive_results))
    
    # Print comparison
    print_comparison(all_summaries)
    
    # Compute speedup
    if all_summaries[0]['n_valid'] > 0 and all_summaries[1]['n_valid'] > 0:
        speedup = all_summaries[0]['time_mean'] / all_summaries[1]['time_mean']
        print()
        print(f"Speedup (Normalized vs Legacy): {speedup:.2f}x")
    
    print()
    print("="*80)
    print("DONE")
    print("="*80)


if __name__ == '__main__':
    main()
