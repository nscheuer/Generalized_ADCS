#!/usr/bin/env python3
"""
Benchmark script for ALTRO trajectory planner performance.

Run this before and after optimizations to measure impact.
Usage: python benchmark_altro.py [--iterations N]
"""

import sys
import os
import time
import argparse
import numpy as np

# Add project to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ADCS.CONOPS.goals import ECI_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.helpers import PlannerSettings
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.helpers.math_helpers import normalize
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants


def setup_test_case():
    """Create a standard test case for benchmarking."""
    np.random.seed(42)
    
    # Satellite
    sat = create_beavercube2_cubesat(estimated=False)
    sat.rw_actuators[0].h = 0.0
    
    # Initial state
    w0 = np.array([0.01, -0.005, 0.008])  # Small angular velocity
    q0 = normalize(np.array([0.9, 0.1, 0.2, 0.3]))
    h0 = np.array([0.0])
    x0 = np.concatenate([w0, q0, h0])
    
    # Orbit
    ephem = Ephemeris()
    start_time = 0.22
    R = 7000 * np.array([0, np.sqrt(2)/2, np.sqrt(2)/2])
    V = np.array([8, 0, 0])
    os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
    
    orb = Orbit(
        os0=os0,
        end_time=start_time + 200 * TimeConstants.sec2cent,
        dt=1,
        use_J2=True,
        fast=True,
        verbose=False
    )
    
    # Goal
    goal_vec = normalize(np.array([0, 0, 1]))
    goal = ECI_Goal(goal_vec)
    goals = GoalList({start_time: goal})
    
    return sat, x0, orb, goals, start_time


def run_benchmark(sat, x0, orb, goals, start_time, duration=60, iterations=5):
    """Run trajectory planning benchmark."""
    
    # Planner settings - optimized for speed
    planner_settings = PlannerSettings(
        est_sat=sat,
        bdot_on=0,      # Skip bdot for speed
        dt_tp=10,       # Must be <= 20 for N >= 4
        dt_tvlqr=1,
    )
    # Use moderate iteration counts for meaningful benchmark
    planner_settings.pass1.convergence.max_outer_iter = 8
    planner_settings.pass1.convergence.max_inner_iter = 40
    planner_settings.pass2.convergence.max_outer_iter = 5
    planner_settings.pass2.convergence.max_inner_iter = 15
    
    print(f"Creating controller...")
    controller = Plan_and_Track_LQR(est_sat=sat, planner_settings=planner_settings)
    
    os0_for_traj = orb.get_os(start_time)
    
    times = []
    
    print(f"\nRunning {iterations} iterations with duration={duration}s...")
    print("-" * 50)
    
    for i in range(iterations):
        t_start = time.perf_counter()
        
        traj = controller.calculate_trajectory(
            t_start=start_time,
            duration=duration,
            x_0=x0,
            os_0=os0_for_traj,
            goals=goals,
            verbose=False,
        )
        
        t_end = time.perf_counter()
        elapsed = t_end - t_start
        times.append(elapsed)
        
        # Verify trajectory is valid
        assert traj is not None, "Trajectory is None"
        assert not np.any(np.isnan(traj.states)), "NaN in trajectory states"
        
        print(f"  Iteration {i+1}: {elapsed:.3f}s")
    
    return times


def print_statistics(times, label=""):
    """Print timing statistics."""
    times = np.array(times)
    
    print(f"\n{'='*50}")
    print(f"BENCHMARK RESULTS {label}")
    print(f"{'='*50}")
    print(f"  Iterations:    {len(times)}")
    print(f"  Mean time:     {times.mean():.3f}s")
    print(f"  Std dev:       {times.std():.3f}s")
    print(f"  Min time:      {times.min():.3f}s")
    print(f"  Max time:      {times.max():.3f}s")
    print(f"  Median time:   {np.median(times):.3f}s")
    print(f"{'='*50}")
    
    return {
        'mean': times.mean(),
        'std': times.std(),
        'min': times.min(),
        'max': times.max(),
        'median': np.median(times)
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark ALTRO trajectory planner')
    parser.add_argument('--iterations', '-n', type=int, default=5,
                        help='Number of iterations (default: 5)')
    parser.add_argument('--duration', '-d', type=int, default=60,
                        help='Trajectory duration in seconds (default: 60)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Save results to JSON file')
    args = parser.parse_args()
    
    print("="*50)
    print("ALTRO TRAJECTORY PLANNER BENCHMARK")
    print("="*50)
    
    # Setup
    print("\nSetting up test case...")
    sat, x0, orb, goals, start_time = setup_test_case()
    
    # Warmup run
    print("\nWarmup run...")
    planner_settings = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=10, dt_tvlqr=1)
    planner_settings.pass1.convergence.max_outer_iter = 2
    planner_settings.pass1.convergence.max_inner_iter = 5
    controller = Plan_and_Track_LQR(est_sat=sat, planner_settings=planner_settings)
    os0_for_traj = orb.get_os(start_time)
    _ = controller.calculate_trajectory(
        t_start=start_time, duration=20, x_0=x0, os_0=os0_for_traj,
        goals=goals, verbose=False
    )
    print("  Warmup complete")
    
    # Benchmark
    times = run_benchmark(sat, x0, orb, goals, start_time,
                          duration=args.duration, iterations=args.iterations)
    
    # Statistics
    stats = print_statistics(times)
    
    # Save results
    if args.output:
        import json
        results = {
            'iterations': args.iterations,
            'duration': args.duration,
            'times': times,
            'statistics': stats,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    return stats


if __name__ == '__main__':
    main()
