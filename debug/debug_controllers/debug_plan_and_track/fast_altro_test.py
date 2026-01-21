#!/usr/bin/env python3
"""
Fast ALTRO Test - Multiple trajectory planning iterations.

This test runs multiple trajectory planning iterations to measure performance
and check for consistency.

Usage:
    python fast_altro_test.py
    python fast_altro_test.py --iterations 10  # Run 10 trajectory plans
"""

import sys
import os
import numpy as np
import time
import argparse

# Add project root to path BEFORE any ADCS imports
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))


def run_fast_altro_test(iterations: int = 5, verbose: int = 0):
    """Run fast ALTRO test with multiple iterations."""
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
    print("Fast ALTRO Test")
    print("=" * 60)

    # Setup
    print("\n[1/3] Setting up satellite and orbit...")
    real_sat = create_beavercube2_cubesat(estimated=False)
    if hasattr(real_sat, 'rw_actuators') and len(real_sat.rw_actuators) > 0:
        real_sat.rw_actuators[0].h = 0.0
    print(f"      Satellite: BeaverCube2")
    print(f"      MTQs: {len(real_sat.mtq_actuators)}, RWs: {len(real_sat.rw_actuators)}")

    ephem = Ephemeris()
    start_time = 0.22
    end_time = start_time + 180 * TimeConstants.sec2cent
    R = 7000 * np.array([0, np.sqrt(2)/2, np.sqrt(2)/2])
    V = np.array([8, 0, 0])
    os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
    orb = Orbit(os0=os0, end_time=end_time, dt=1, use_J2=True, fast=False)
    os_0 = orb.get_os(start_time)

    planner_settings = PlannerSettings(
        est_sat=real_sat,
        bdot_on=2,
        dt_tp=30,
        dt_tvlqr=1,
    )
    controller = Plan_and_Track_LQR(
        est_sat=real_sat,
        planner_settings=planner_settings,
    )

    goal_vec = normalize(np.array([0, 0, 1]))
    goal = ECI_Goal(goal_vec)
    goals = GoalList({start_time: goal})

    duration = 60.0  # 60 seconds

    # Run trajectory planning multiple times
    print(f"\n[2/3] Running {iterations} trajectory plan(s)...")
    omega0 = np.array([0.01, 0.01, 0.01])
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    h0 = np.array([0.0])
    x0_base = np.concatenate([omega0, q0, h0])

    solve_times = []
    for i in range(iterations):
        # Add small perturbation to initial state for variety
        x0 = x0_base.copy()
        x0[:3] += np.random.randn(3) * 0.001

        start = time.time()
        try:
            trajectory = controller.calculate_trajectory(
                t_start=start_time,
                duration=duration,
                x_0=x0,
                os_0=os_0,
                goals=goals,
                verbose=verbose
            )
            elapsed = time.time() - start
            solve_times.append(elapsed)

            if i == 0 or (i + 1) % 5 == 0:
                print(f"      Iteration {i+1}: {elapsed:.3f}s")

            # Validate
            if np.any(np.isnan(trajectory.states)) or np.any(np.isinf(trajectory.states)):
                print(f"      ERROR: NaN/Inf detected at iteration {i+1}")
                return False
        except Exception as e:
            print(f"      ERROR at iteration {i+1}: {e}")
            return False

    # Results
    print("\n[3/3] Results")
    print("=" * 60)
    print(f"  Iterations: {iterations}")
    print(f"  Mean solve time: {np.mean(solve_times):.3f}s")
    print(f"  Min solve time: {np.min(solve_times):.3f}s")
    print(f"  Max solve time: {np.max(solve_times):.3f}s")
    print(f"  Std solve time: {np.std(solve_times):.3f}s")
    print(f"  Total time: {sum(solve_times):.2f}s")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(description="Fast ALTRO test")
    parser.add_argument("--iterations", "-n", type=int, default=5,
                        help="Number of trajectory plans to run")
    parser.add_argument("--verbose", "-v", type=int, default=0,
                        help="Verbosity level (0-4)")
    args = parser.parse_args()

    success = run_fast_altro_test(args.iterations, args.verbose)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
