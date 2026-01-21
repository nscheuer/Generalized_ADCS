#!/usr/bin/env python3
"""
Benchmark comparing original ALTRO vs blended dynamics warm-start.

This script tests whether using relaxed physics (α=0) early in optimization
improves convergence compared to the standard cross-product formulation.
"""

import sys
import os
import numpy as np
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))

from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.controller.helpers import PlannerSettings
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.CONOPS.goallist import GoalList
from ADCS.CONOPS.goals import ECI_Goal
from ADCS.helpers.math_helpers import normalize


def create_setup():
    """Create satellite, orbit, and controller."""
    print("Creating shared satellite/orbit...")
    setup_start = time.time()

    # Create satellite
    real_sat = create_beavercube2_cubesat(estimated=False)
    if hasattr(real_sat, 'rw_actuators') and len(real_sat.rw_actuators) > 0:
        real_sat.rw_actuators[0].h = 0.0

    # Create orbit
    ephem = Ephemeris()
    start_time = 0.22
    end_time = start_time + 180 * TimeConstants.sec2cent
    R = 7000 * np.array([0, np.sqrt(2)/2, np.sqrt(2)/2])
    V = np.array([8, 0, 0])
    os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
    orb = Orbit(os0=os0, end_time=end_time, dt=1, use_J2=True, fast=False)
    os_0 = orb.get_os(start_time)

    # Create controller
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

    setup_time = time.time() - setup_start
    print(f"Setup done in {setup_time:.2f}s")

    return real_sat, orb, os_0, controller, start_time


def get_goals(start_time):
    """Create ECI pointing goal."""
    goal_vec = normalize(np.array([0, 0, 1]))
    goal = ECI_Goal(goal_vec)
    goals = GoalList({start_time: goal})
    return goals


def run_benchmark(controller, start_time, os_0, goals, x0, duration, name, verbose=0):
    """Run a single trajectory calculation and time it."""
    print(f"  Running {name}...", end=" ", flush=True)

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

        has_nan = np.any(np.isnan(trajectory.states)) or np.any(np.isinf(trajectory.states))

        # Calculate final error metrics
        final_omega = trajectory.states[:3, -1]
        final_q = trajectory.states[3:7, -1]
        omega_norm = np.linalg.norm(final_omega)

        print(f"DONE ({elapsed:.2f}s)")
        return {
            'name': name,
            'time': elapsed,
            'success': not has_nan,
            'final_omega_norm': omega_norm,
            'trajectory': trajectory
        }
    except Exception as e:
        elapsed = time.time() - start
        print(f"FAILED ({elapsed:.2f}s) - {e}")
        return {
            'name': name,
            'time': elapsed,
            'success': False,
            'error': str(e)
        }


def main():
    print("=" * 60)
    print("Blended Dynamics Benchmark")
    print("=" * 60)
    print()

    real_sat, orb, os_0, controller, start_time = create_setup()
    goals = get_goals(start_time)

    # Test cases with increasing difficulty
    test_cases = [
        ("Low angular velocity", np.array([0.01, 0.01, 0.01])),
        ("Medium angular velocity", np.array([0.05, 0.05, 0.05])),
        ("High angular velocity", np.array([0.1, 0.1, 0.1])),
    ]

    print("\n" + "=" * 60)
    print("Running benchmarks (original ALTRO)")
    print("=" * 60)

    results = []
    for name, omega in test_cases:
        x0 = np.concatenate([omega, np.array([0.0, 0.0, 0.0, 1.0]), np.array([0.0])])
        result = run_benchmark(controller, start_time, os_0, goals, x0, 35.0, name, verbose=0)
        results.append(result)

    # Print summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"{'Test Case':<30} {'Time (s)':<12} {'Success':<10} {'Final ω (rad/s)'}")
    print("-" * 70)
    for r in results:
        success_str = "✓" if r.get('success', False) else "✗"
        omega_str = f"{r.get('final_omega_norm', float('nan')):.6f}" if r.get('success') else "N/A"
        print(f"{r['name']:<30} {r['time']:<12.2f} {success_str:<10} {omega_str}")

    # Note: To benchmark blended vs original, we would need to:
    # 1. Expose alilqrBlended through Python bindings
    # 2. Add a flag to the controller to use blended vs original
    # For now, this benchmark tests the baseline performance.

    print("\n" + "=" * 60)
    print("Note: Blended optimization requires Python binding updates")
    print("to expose alilqrBlended. Current benchmark shows baseline.")
    print("=" * 60)

    return 0 if all(r.get('success', False) for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
