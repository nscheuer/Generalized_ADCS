#!/usr/bin/env python3
"""
Fast Plan & Track Tests - Quick validation tests for CI/CD.

These tests use shared setup with proper orbit propagation.
Note: Duration must be > dt_tp (30s) to avoid edge case errors.

Run with pytest:
    pytest test_plan_and_track_fast.py -v
    pytest test_plan_and_track_fast.py -v -k "test_basic"
"""

import sys
import os
import numpy as np
import pytest
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


@pytest.fixture(scope="module")
def test_setup():
    """Create satellite, orbit, and controller once for all tests."""
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

    # Create satellite
    real_sat = create_beavercube2_cubesat(estimated=False)
    if hasattr(real_sat, 'rw_actuators') and len(real_sat.rw_actuators) > 0:
        real_sat.rw_actuators[0].h = 0.0

    # Create orbit - use_J2=True required, 180s to support trajectories
    ephem = Ephemeris()
    start_time = 0.22
    end_time = start_time + 180 * TimeConstants.sec2cent
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

    # Create default goals
    goal_vec = normalize(np.array([0, 0, 1]))
    goal = ECI_Goal(goal_vec)
    goals = GoalList({start_time: goal})

    return {
        'sat': real_sat,
        'orb': orb,
        'os_0': os_0,
        'controller': controller,
        'start_time': start_time,
        'goals': goals,
    }


def get_initial_state(omega=None, q=None, h=None):
    """Create initial state [omega(3), q(4), h(1)]."""
    if omega is None:
        omega = np.array([0.01, 0.01, 0.01])
    if q is None:
        q = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
    if h is None:
        h = np.array([0.0])
    return np.concatenate([omega, q, h])


class TestBasicFunctionality:
    """Basic functionality tests."""

    def test_import_tplaunch(self):
        """Test that tplaunch C++ module imports correctly."""
        import tplaunch
        assert hasattr(tplaunch, 'Planner')
        assert hasattr(tplaunch.Planner, 'setVerbosity')

    def test_planner_creation(self, test_setup):
        """Test that planner can be created."""
        assert test_setup['controller'] is not None

    def test_trajectory_planning_basic(self, test_setup):
        """Test basic trajectory planning."""
        x0 = get_initial_state()

        # Duration must be > dt_tp=30 to avoid edge case
        trajectory = test_setup['controller'].calculate_trajectory(
            t_start=test_setup['start_time'],
            duration=35.0,
            x_0=x0,
            os_0=test_setup['os_0'],
            goals=test_setup['goals'],
            verbose=0
        )

        assert trajectory is not None
        assert not np.any(np.isnan(trajectory.states))
        assert not np.any(np.isinf(trajectory.states))

    def test_trajectory_no_nan_inf(self, test_setup):
        """Verify trajectory has no NaN or Inf values."""
        x0 = get_initial_state(omega=np.array([0.05, 0.05, 0.05]))

        trajectory = test_setup['controller'].calculate_trajectory(
            t_start=test_setup['start_time'],
            duration=35.0,
            x_0=x0,
            os_0=test_setup['os_0'],
            goals=test_setup['goals'],
            verbose=0
        )

        assert not np.any(np.isnan(trajectory.states)), "states contains NaN"
        assert not np.any(np.isinf(trajectory.states)), "states contains Inf"
        assert not np.any(np.isnan(trajectory.controls)), "controls contains NaN"
        assert not np.any(np.isinf(trajectory.controls)), "controls contains Inf"


class TestInitialConditions:
    """Test various initial conditions."""

    def test_identity_quaternion(self, test_setup):
        """Test with identity quaternion (no rotation)."""
        x0 = get_initial_state(
            omega=np.array([0.0, 0.0, 0.0]),
            q=np.array([0.0, 0.0, 0.0, 1.0])
        )

        trajectory = test_setup['controller'].calculate_trajectory(
            t_start=test_setup['start_time'],
            duration=35.0,
            x_0=x0,
            os_0=test_setup['os_0'],
            goals=test_setup['goals'],
            verbose=0
        )

        assert not np.any(np.isnan(trajectory.states))

    def test_high_angular_velocity(self, test_setup):
        """Test with high initial angular velocity."""
        x0 = get_initial_state(omega=np.array([0.1, 0.1, 0.1]))  # ~6 deg/s

        trajectory = test_setup['controller'].calculate_trajectory(
            t_start=test_setup['start_time'],
            duration=35.0,
            x_0=x0,
            os_0=test_setup['os_0'],
            goals=test_setup['goals'],
            verbose=0
        )

        assert not np.any(np.isnan(trajectory.states))

    def test_90_degree_offset(self, test_setup):
        """Test with 90 degree initial angle offset."""
        angle = np.pi / 2
        x0 = get_initial_state(
            omega=np.array([0.01, 0.01, 0.01]),
            q=np.array([0.0, 0.0, np.sin(angle/2), np.cos(angle/2)])
        )

        trajectory = test_setup['controller'].calculate_trajectory(
            t_start=test_setup['start_time'],
            duration=45.0,
            x_0=x0,
            os_0=test_setup['os_0'],
            goals=test_setup['goals'],
            verbose=0
        )

        assert not np.any(np.isnan(trajectory.states))


class TestVerbosity:
    """Test verbosity levels."""

    def test_verbosity_silent(self, test_setup):
        """Test that verbosity=0 produces no errors."""
        x0 = get_initial_state()

        trajectory = test_setup['controller'].calculate_trajectory(
            t_start=test_setup['start_time'],
            duration=35.0,
            x_0=x0,
            os_0=test_setup['os_0'],
            goals=test_setup['goals'],
            verbose=0
        )
        assert trajectory is not None

    def test_verbosity_milestone(self, test_setup):
        """Test that verbosity=1 (milestones) works."""
        x0 = get_initial_state()

        trajectory = test_setup['controller'].calculate_trajectory(
            t_start=test_setup['start_time'],
            duration=35.0,
            x_0=x0,
            os_0=test_setup['os_0'],
            goals=test_setup['goals'],
            verbose=1
        )
        assert trajectory is not None


class TestTrajectoryOutput:
    """Test trajectory output properties."""

    def test_trajectory_dimensions(self, test_setup):
        """Test that trajectory has expected dimensions."""
        x0 = get_initial_state()

        trajectory = test_setup['controller'].calculate_trajectory(
            t_start=test_setup['start_time'],
            duration=35.0,
            x_0=x0,
            os_0=test_setup['os_0'],
            goals=test_setup['goals'],
            verbose=0
        )

        # BeaverCube2: 8 states (omega[3], q[4], h[1]), 4 controls (mtq[3], rw[1])
        assert trajectory.states.shape[0] == 8, f"Expected 8 states, got {trajectory.states.shape[0]}"
        assert trajectory.controls.shape[0] == 4, f"Expected 4 controls, got {trajectory.controls.shape[0]}"
        assert trajectory.n_steps > 0

    def test_quaternion_normalization(self, test_setup):
        """Test that quaternion stays normalized throughout trajectory."""
        x0 = get_initial_state()

        trajectory = test_setup['controller'].calculate_trajectory(
            t_start=test_setup['start_time'],
            duration=35.0,
            x_0=x0,
            os_0=test_setup['os_0'],
            goals=test_setup['goals'],
            verbose=0
        )

        # Quaternion is states[3:7]
        q = trajectory.states[3:7, :]
        q_norms = np.linalg.norm(q, axis=0)

        # Allow small numerical tolerance
        assert np.allclose(q_norms, 1.0, atol=0.01), f"Quaternion norms: min={q_norms.min():.4f}, max={q_norms.max():.4f}"


class TestPerformance:
    """Performance-related tests."""

    def test_solve_time_reasonable(self, test_setup):
        """Test that solve time is reasonable (<15s for 60s horizon)."""
        x0 = get_initial_state()

        start = time.time()
        trajectory = test_setup['controller'].calculate_trajectory(
            t_start=test_setup['start_time'],
            duration=60.0,
            x_0=x0,
            os_0=test_setup['os_0'],
            goals=test_setup['goals'],
            verbose=0
        )
        elapsed = time.time() - start

        assert elapsed < 15.0, f"Solve took {elapsed:.1f}s, expected <15s"
        assert not np.any(np.isnan(trajectory.states))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
