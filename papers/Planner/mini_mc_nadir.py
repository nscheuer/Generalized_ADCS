#!/usr/bin/env python3
"""
Mini Monte Carlo for Nadir pointing goal (vector goal type).
"""

import numpy as np
import sys
import time
from dataclasses import dataclass
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/pmckeen/Generalized_ADCS')

from ADCS.CONOPS.goals import Nadir_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.helpers import (
    PlannerSettings, create_planner_settings,
    NormalizedPlannerConfig, NormalizedActuatorCosts, NormalizedStateCosts,
)
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.helpers.math_helpers import normalize


@dataclass
class MCResult:
    seed: int
    time_s: float
    final_alignment_deg: float
    rw_sign_changes_pct: float
    converged: bool


def run_single_case(sat, orb, seed: int, settings: PlannerSettings, duration: float = 60.0) -> MCResult:
    """Run a single MC case with Nadir goal."""
    rng = np.random.default_rng(seed=seed)
    
    # Random initial attitude and rates
    q0 = normalize(rng.standard_normal(4))
    w0 = normalize(rng.standard_normal(3)) * (1.0 * np.pi / 180.0)  # 1 deg/s
    h0 = rng.uniform(-0.0001, 0.0001, size=len(sat.rw_actuators))
    
    x0 = np.concatenate([w0, q0, h0])
    for i, rw in enumerate(sat.rw_actuators):
        rw.h = h0[i]
    
    t_start = 0.22
    goals = GoalList({t_start: Nadir_Goal()})
    os0 = orb.get_os(t_start)
    
    controller = Plan_and_Track_LQR(est_sat=sat, planner_settings=settings)
    
    try:
        t0 = time.perf_counter()
        traj = controller.calculate_trajectory(
            t_start=t_start, duration=duration, x_0=x0, os_0=os0, 
            goals=goals, verbose=False
        )
        elapsed = time.perf_counter() - t0
        
        # For nadir goal, we'd need to compute alignment with nadir vector
        # For simplicity, use final angular velocity as proxy for "settled"
        w_final = np.linalg.norm(traj.states[0:3, -1]) * 180 / np.pi
        
        # RW control smoothness
        if traj.controls.shape[0] > 3:
            rw_ctrl = traj.controls[3, :]
            sign_changes = np.sum(np.diff(np.sign(rw_ctrl)) != 0)
            sign_change_pct = sign_changes / max(1, len(rw_ctrl) - 1) * 100
        else:
            sign_change_pct = 0.0
        
        converged = w_final < 0.5  # Less than 0.5 deg/s
        
        return MCResult(
            seed=seed,
            time_s=elapsed,
            final_alignment_deg=w_final,  # Using angular rate as proxy
            rw_sign_changes_pct=sign_change_pct,
            converged=converged,
        )
        
    except Exception as e:
        print(f"  Seed {seed} failed: {e}")
        return MCResult(seed=seed, time_s=float('nan'), final_alignment_deg=float('nan'),
                       rw_sign_changes_pct=float('nan'), converged=False)


def main():
    print("="*70)
    print("MINI MC: Nadir Pointing (Vector Goal)")
    print("="*70)
    
    np.random.seed(42)
    sat = create_beavercube2_cubesat(estimated=False)
    
    print("Creating orbit...")
    orb = create_random_circular_orbit(radius_km=7000.0, dt=1, tf=120, use_J2=True, fast=True)
    orb.populate_environment(compute_B=True, compute_S=True)
    
    n_cases = 10
    duration = 60.0
    
    configs = [
        ("Legacy", PlannerSettings(est_sat=sat, bdot_on=0)),
    ]
    
    # Normalized well-conditioned
    norm_config = NormalizedPlannerConfig(
        actuator_costs=NormalizedActuatorCosts(mtq_cost=1.0, rw_torque_cost=5.0),
        state_costs=NormalizedStateCosts(
            angle_cost=1000.0, angle_terminal_cost=1000000.0,
            ang_vel_cost=1000.0, ang_vel_terminal_cost=100000.0,
        ),
    )
    norm_settings = create_planner_settings(sat, norm_config)
    norm_settings.rw_AM_weight = 1e4
    norm_settings.RWh_ok_mult = 0.5
    configs.append(("Normalized", norm_settings))
    
    all_results = {}
    
    for name, settings in configs:
        print(f"\n--- {name} ---")
        results = []
        for i in range(n_cases):
            seed = 2000 + i
            print(f"  Case {i+1}/{n_cases}...", end=" ", flush=True)
            result = run_single_case(sat, orb, seed, settings, duration)
            results.append(result)
            status = "✓" if result.converged else "✗"
            print(f"{status} {result.time_s:.2f}s, ω={result.final_alignment_deg:.3f}°/s")
        all_results[name] = results
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Config':<15} {'Time (s)':<15} {'Final ω (°/s)':<15} {'Conv%':<10}")
    print("-"*60)
    
    for name, results in all_results.items():
        valid = [r for r in results if not np.isnan(r.time_s)]
        if valid:
            time_mean = np.mean([r.time_s for r in valid])
            time_std = np.std([r.time_s for r in valid])
            w_mean = np.mean([r.final_alignment_deg for r in valid])
            conv_rate = sum(r.converged for r in valid) / len(valid) * 100
            print(f"{name:<15} {time_mean:.2f}±{time_std:.2f}      {w_mean:.3f}            {conv_rate:.0f}%")


if __name__ == '__main__':
    main()
