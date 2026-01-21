#!/usr/bin/env python3
"""
Quick Planner Tests - Fast validation suite for trajectory planner.

Uses shared setup to avoid repeated orbit propagation (the slow part).

Usage:
    python quick_planner_tests.py           # Run all tests
    python quick_planner_tests.py --verbose # Run with verbose output
"""

import sys
import os
import numpy as np
import time
import argparse

# Add project root to path BEFORE any ADCS imports
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))


class TestResult:
    def __init__(self, name: str, passed: bool, duration: float, message: str = ""):
        self.name = name
        self.passed = passed
        self.duration = duration
        self.message = message


# Global shared setup - created once, reused across all tests
_shared_setup = None


def get_shared_setup():
    """Get or create the shared test setup (lazy initialization)."""
    global _shared_setup
    if _shared_setup is None:
        _shared_setup = _create_test_setup()
    return _shared_setup


def _create_test_setup():
    """Create satellite, orbit, and controller for testing. Called once."""
    from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
    from ADCS.orbits.orbit import Orbit
    from ADCS.orbits.orbital_state import Orbital_State
    from ADCS.orbits.ephemeris import Ephemeris
    from ADCS.orbits.universal_constants import TimeConstants
    from ADCS.controller.helpers import PlannerSettings
    from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR

    print("  [Setup] Creating shared satellite/orbit...")
    setup_start = time.time()

    # Create satellite
    real_sat = create_beavercube2_cubesat(estimated=False)
    if hasattr(real_sat, 'rw_actuators') and len(real_sat.rw_actuators) > 0:
        real_sat.rw_actuators[0].h = 0.0

    # Create orbit - use_J2=True is required for controller to work properly
    ephem = Ephemeris()
    start_time = 0.22
    end_time = start_time + 120 * TimeConstants.sec2cent  # 120s orbit
    R = 7000e3 * np.array([0, np.sqrt(2)/2, np.sqrt(2)/2])
    V = np.array([8000, 0, 0])
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
    print(f"  [Setup] Done in {setup_time:.2f}s")

    return real_sat, orb, os_0, controller, start_time


def get_default_goals(start_time):
    """Create default ECI pointing goal."""
    from ADCS.CONOPS.goallist import GoalList
    from ADCS.CONOPS.goals import ECI_Goal
    from ADCS.helpers.math_helpers import normalize

    goal_vec = normalize(np.array([0, 0, 1]))
    goal = ECI_Goal(goal_vec)
    goals = GoalList({start_time: goal})
    return goals


def get_initial_state(omega=None, q=None, h=None):
    """Create initial state [omega(3), q(4), h(1)]."""
    if omega is None:
        omega = np.array([0.01, 0.01, 0.01])
    if q is None:
        q = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
    if h is None:
        h = np.array([0.0])
    return np.concatenate([omega, q, h])


def run_test(test_func, name: str, verbose: bool = False) -> TestResult:
    """Run a single test and capture result."""
    print(f"  Running: {name}...", end=" ", flush=True)
    start = time.time()
    try:
        result = test_func(verbose)
        duration = time.time() - start
        if result:
            print(f"PASS ({duration:.2f}s)")
            return TestResult(name, True, duration)
        else:
            print(f"FAIL ({duration:.2f}s)")
            return TestResult(name, False, duration, "Test returned False")
    except Exception as e:
        duration = time.time() - start
        print(f"ERROR ({duration:.2f}s)")
        import traceback
        if verbose:
            traceback.print_exc()
        return TestResult(name, False, duration, str(e))


def test_basic_altro(verbose: bool = False) -> bool:
    """Test basic ALTRO convergence with default settings."""
    real_sat, orb, os_0, controller, start_time = get_shared_setup()
    goals = get_default_goals(start_time)

    x0 = get_initial_state(omega=np.array([0.01, 0.01, 0.01]))

    trajectory = controller.calculate_trajectory(
        t_start=start_time,
        duration=30.0,
        x_0=x0,
        os_0=os_0,
        goals=goals,
        verbose=2 if verbose else 0
    )

    return not (np.any(np.isnan(trajectory.states)) or np.any(np.isinf(trajectory.states)))


def test_high_angular_velocity(verbose: bool = False) -> bool:
    """Test with high initial angular velocity (~6 deg/s)."""
    real_sat, orb, os_0, controller, start_time = get_shared_setup()
    goals = get_default_goals(start_time)

    # High angular velocity: 0.1 rad/s per axis (~6 deg/s)
    x0 = get_initial_state(omega=np.array([0.1, 0.1, 0.1]))

    trajectory = controller.calculate_trajectory(
        t_start=start_time,
        duration=30.0,
        x_0=x0,
        os_0=os_0,
        goals=goals,
        verbose=2 if verbose else 0
    )

    return not (np.any(np.isnan(trajectory.states)) or np.any(np.isinf(trajectory.states)))


def test_90_degree_slew(verbose: bool = False) -> bool:
    """Test with 90 degree initial angle offset."""
    real_sat, orb, os_0, controller, start_time = get_shared_setup()
    goals = get_default_goals(start_time)

    # 90 degree rotation about z-axis
    angle = np.pi / 2
    x0 = get_initial_state(
        omega=np.array([0.01, 0.01, 0.01]),
        q=np.array([0.0, 0.0, np.sin(angle/2), np.cos(angle/2)])
    )

    trajectory = controller.calculate_trajectory(
        t_start=start_time,
        duration=45.0,
        x_0=x0,
        os_0=os_0,
        goals=goals,
        verbose=2 if verbose else 0
    )

    return not (np.any(np.isnan(trajectory.states)) or np.any(np.isinf(trajectory.states)))


def test_zero_initial_omega(verbose: bool = False) -> bool:
    """Test with zero initial angular velocity."""
    real_sat, orb, os_0, controller, start_time = get_shared_setup()
    goals = get_default_goals(start_time)

    x0 = get_initial_state(omega=np.array([0.0, 0.0, 0.0]))

    trajectory = controller.calculate_trajectory(
        t_start=start_time,
        duration=30.0,
        x_0=x0,
        os_0=os_0,
        goals=goals,
        verbose=2 if verbose else 0
    )

    return not (np.any(np.isnan(trajectory.states)) or np.any(np.isinf(trajectory.states)))


def test_trajectory_shape(verbose: bool = False) -> bool:
    """Test that trajectory has expected dimensions."""
    real_sat, orb, os_0, controller, start_time = get_shared_setup()
    goals = get_default_goals(start_time)

    x0 = get_initial_state()

    trajectory = controller.calculate_trajectory(
        t_start=start_time,
        duration=30.0,
        x_0=x0,
        os_0=os_0,
        goals=goals,
        verbose=0
    )

    # Check dimensions
    state_dim = trajectory.states.shape[0]
    ctrl_dim = trajectory.controls.shape[0]
    n_steps = trajectory.n_steps

    if verbose:
        print(f"\n    States: {trajectory.states.shape}, Controls: {trajectory.controls.shape}")

    return state_dim == 8 and ctrl_dim == 4 and n_steps > 0


def main():
    parser = argparse.ArgumentParser(description="Quick planner tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("=" * 60)
    print("Quick Planner Tests")
    print("=" * 60)
    print()

    # Force setup first to time it separately
    print("Initializing shared setup...")
    get_shared_setup()
    print()

    tests = [
        (test_basic_altro, "Basic ALTRO"),
        (test_high_angular_velocity, "High Angular Velocity"),
        (test_90_degree_slew, "90 Degree Slew"),
        (test_zero_initial_omega, "Zero Initial Omega"),
        (test_trajectory_shape, "Trajectory Shape"),
    ]

    results = []
    for test_func, name in tests:
        result = run_test(test_func, name, args.verbose)
        results.append(result)

    # Summary
    print()
    print("=" * 60)
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)

    total_time = sum(r.duration for r in results)
    print(f"Total test time: {total_time:.2f}s (excludes setup)")

    if passed < total:
        print("\nFailed tests:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.message}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
