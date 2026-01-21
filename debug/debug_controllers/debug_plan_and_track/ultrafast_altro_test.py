#!/usr/bin/env python3
"""
Ultrafast ALTRO Test - Minimal overhead test for C++ solver performance.

This test focuses purely on the C++ ALTRO solver performance by:
1. Creating a planner once
2. Calling trajOpt directly with minimal Python overhead

Use this to measure the true C++ solver performance.

Usage:
    python ultrafast_altro_test.py
    python ultrafast_altro_test.py --iterations 20
"""

import sys
import os
import numpy as np
import time
import argparse

# Add project root to path BEFORE any ADCS imports
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))


def run_ultrafast_test(iterations: int = 10, verbose: int = 0):
    """Run ultrafast ALTRO test - directly calling C++ planner."""
    from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
    from ADCS.orbits.orbit import Orbit
    from ADCS.orbits.orbital_state import Orbital_State
    from ADCS.orbits.ephemeris import Ephemeris
    from ADCS.orbits.universal_constants import TimeConstants
    from ADCS.CONOPS.goallist import GoalList
    from ADCS.CONOPS.goals import ECI_Goal
    from ADCS.controller.helpers import PlannerSettings
    from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
    from ADCS.helpers.math_helpers import normalize

    print("=" * 60)
    print("Ultrafast ALTRO Test")
    print("=" * 60)

    # Setup
    print("\n[1/4] Setting up...")
    real_sat = create_beavercube2_cubesat(estimated=False)
    if hasattr(real_sat, 'rw_actuators') and len(real_sat.rw_actuators) > 0:
        real_sat.rw_actuators[0].h = 0.0

    planner_settings = PlannerSettings(
        est_sat=real_sat,
        bdot_on=2,
        dt_tp=30,
        dt_tvlqr=1,
    )

    # Create orbit
    ephem = Ephemeris()
    start_time = 0.22
    end_time = start_time + 180 * TimeConstants.sec2cent
    R = 7000 * np.array([0, np.sqrt(2)/2, np.sqrt(2)/2])
    V = np.array([8, 0, 0])
    os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
    orb = Orbit(os0=os0, end_time=end_time, dt=1, use_J2=True, fast=False)
    os_0 = orb.get_os(start_time)

    # Controller
    controller = Plan_and_Track_LQR(
        est_sat=real_sat,
        planner_settings=planner_settings,
    )

    goal_vec = normalize(np.array([0, 0, 1]))
    goal = ECI_Goal(goal_vec)
    goals = GoalList({start_time: goal})

    duration = 60.0  # 60 seconds
    print(f"      Duration: {duration}s")

    # Prepare initial state
    omega0 = np.array([0.01, 0.01, 0.01])
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    h0 = np.array([0.0])
    x0_base = np.concatenate([omega0, q0, h0])

    # Warm up
    print("\n[2/4] Warming up (2 iterations)...")
    for _ in range(2):
        controller.calculate_trajectory(
            t_start=start_time,
            duration=duration,
            x_0=x0_base.copy(),
            os_0=os_0,
            goals=goals,
            verbose=0
        )

    # Benchmark
    print(f"\n[3/4] Running {iterations} iterations...")
    solve_times = []

    for i in range(iterations):
        # Small perturbation
        x0 = x0_base.copy()
        x0[:3] += np.random.randn(3) * 0.001

        start = time.perf_counter()
        trajectory = controller.calculate_trajectory(
            t_start=start_time,
            duration=duration,
            x_0=x0,
            os_0=os_0,
            goals=goals,
            verbose=0
        )
        elapsed = time.perf_counter() - start
        solve_times.append(elapsed * 1000)  # Convert to ms

        if (i + 1) % 5 == 0:
            print(f"      Iteration {i+1}: {elapsed*1000:.1f}ms")

        # Quick validation
        if np.any(np.isnan(trajectory.states)):
            print(f"      ERROR: NaN detected at iteration {i+1}")
            return False

    # Results
    print("\n[4/4] Results")
    print("=" * 60)
    print(f"  Iterations: {iterations}")
    print(f"  Mean: {np.mean(solve_times):.1f}ms")
    print(f"  Median: {np.median(solve_times):.1f}ms")
    print(f"  Min: {np.min(solve_times):.1f}ms")
    print(f"  Max: {np.max(solve_times):.1f}ms")
    print(f"  Std: {np.std(solve_times):.1f}ms")
    if np.mean(solve_times) > 0:
        print(f"  Throughput: {1000/np.mean(solve_times):.1f} solves/sec")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(description="Ultrafast ALTRO test")
    parser.add_argument("--iterations", "-n", type=int, default=10,
                        help="Number of iterations")
    parser.add_argument("--verbose", "-v", type=int, default=0,
                        help="Verbosity level (0-4)")
    args = parser.parse_args()

    success = run_ultrafast_test(args.iterations, args.verbose)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
