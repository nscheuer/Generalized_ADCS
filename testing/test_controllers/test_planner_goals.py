"""
Integration tests for ALTRO trajectory planner with different goal types.

Tests verify that the planner can generate trajectories for:
- ECI_Goal: Point at a fixed direction in inertial frame
- Coordinate_Goal: Point at a ground location (lat/lon)
- No_Goal: Detumbling / rate damping only
"""

import sys
import os
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.controller.helpers.optional_dependencies import (
    trajectory_planner_available,
    trajectory_planner_missing_reason,
)

if not trajectory_planner_available():
    pytest.skip(trajectory_planner_missing_reason(), allow_module_level=True)

from ADCS.CONOPS.goals import ECI_Goal, Coordinate_Goal, No_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.plan_and_track import PlannerSettings, Trajectory
from ADCS.controller.plan_and_track.planner_subsettings import CostWeights
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import RW
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def create_test_satellite():
    """Create a standard RW-only satellite for testing."""
    rw_max_torque = 0.01
    rw_J = 0.001
    rw_h0 = 0.0
    rw_hmax = 0.05
    rws = [RW(axis=j, max_torque=rw_max_torque, J=rw_J, h=rw_h0, h_max=rw_hmax)
           for j in MathConstants.unitvecs]
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    return Satellite(
        mass=4.0,
        J_0=np.diagflat([0.1, 0.1, 0.1]),
        actuators=rws,
        sensors=mtms,
        boresight=np.array([0, 0, 1])
    )


def create_test_orbit(duration: int = 150):
    """Create test orbit with position for coordinate goal testing."""
    ephem = Ephemeris()
    # Position over equator at ~400km altitude
    R = 6778 * np.array([1, 0, 0])  # km, over equator
    V = np.array([0, 7.67, 0])  # km/s, roughly circular orbit velocity
    os0 = Orbital_State(
        ephem=ephem,
        J2000=0.22,
        R=R, V=V,
        B=np.array([0, 0, 0]),  # Zero B-field
        S=np.array([1e5, 0, 0]),
        rho=0.0
    )

    orbs = [os0.copy() for _ in range(duration + 10)]
    for j in range(len(orbs)):
        orbs[j].J2000 = os0.J2000 + j * TimeConstants.sec2cent
    return Orbit(orbs), os0


def create_initial_state():
    """Create standard initial state."""
    w0 = np.array([0.0, 0.0, 0.0])
    q0 = normalize(np.array([0, 0, 0, 1]))  # Identity
    h0 = np.array([0.0, 0.0, 0.0])
    return np.concatenate([w0, q0, h0])


def run_planning_with_goal(goal, duration: float = 60.0,
                           cost_weights: CostWeights = None):
    """Run trajectory planning with specified goal."""
    sat = create_test_satellite()
    x0 = create_initial_state()
    orb, os0 = create_test_orbit(int(duration) + 50)

    planner_settings = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=1.0)
    planner_settings.verbosity = False

    if cost_weights:
        planner_settings.cost_main = cost_weights
        planner_settings.cost_second = cost_weights

    controller = Plan_and_Track_LQR(est_sat=sat, planner_settings=planner_settings)
    goals = GoalList({os0.J2000: goal})

    traj = controller.calculate_trajectory(
        t_start=os0.J2000,
        duration=duration,
        x_0=x0,
        os_0=os0,
        goals=goals,
        verbose=False
    )
    return traj, sat, os0


def compute_pointing_error(q: np.ndarray, goal_vec: np.ndarray,
                          boresight: np.ndarray = np.array([0, 0, 1])) -> float:
    """Compute angle between boresight and goal in radians."""
    r = Rotation.from_quat([q[0], q[1], q[2], q[3]])
    boresight_eci = r.apply(boresight)
    cos_angle = np.clip(np.dot(boresight_eci, goal_vec), -1, 1)
    return np.arccos(cos_angle)


# ==========================================
# ECI_GOAL TESTS
# ==========================================

@pytest.mark.vslow
def test_eci_goal_x_axis():
    """Test pointing at +X direction in ECI frame."""
    goal_vec = np.array([1, 0, 0])
    goal = ECI_Goal(goal_vec)

    traj, sat, _ = run_planning_with_goal(goal, duration=80.0)

    # Verify trajectory was created
    assert traj is not None
    assert traj.states.shape[1] > 1

    # Check that error decreases
    q_init = traj.states[3:7, 0]
    q_final = traj.states[3:7, -1]

    error_init = compute_pointing_error(q_init, goal_vec)
    error_final = compute_pointing_error(q_final, goal_vec)

    assert error_final < error_init, \
        f"Error should decrease: init={np.degrees(error_init):.1f}° final={np.degrees(error_final):.1f}°"


@pytest.mark.vslow
def test_eci_goal_diagonal():
    """Test pointing at a diagonal direction [1,1,1]."""
    goal_vec = normalize(np.array([1, 1, 1]))
    goal = ECI_Goal(goal_vec)

    traj, sat, _ = run_planning_with_goal(goal, duration=80.0)

    # Check final pointing error
    q_final = traj.states[3:7, -1]
    error_final = compute_pointing_error(q_final, goal_vec)

    # Should get reasonably close to goal
    assert error_final < 0.5, \
        f"Final error too large: {np.degrees(error_final):.1f}° (should be < 29°)"


@pytest.mark.vslow
def test_eci_goal_negative_z():
    """Test pointing at -Z direction (180° rotation from identity)."""
    goal_vec = np.array([0, 0, -1])
    goal = ECI_Goal(goal_vec)

    # Use longer duration for 180° rotation
    traj, sat, _ = run_planning_with_goal(goal, duration=120.0)

    # Verify trajectory handles large angle maneuver
    assert traj.states.shape[1] > 1

    # Check error decreases from initial 180°
    q_final = traj.states[3:7, -1]
    error_final = compute_pointing_error(q_final, goal_vec)

    # Initial error is ~180°, should reduce significantly
    assert error_final < 2.5, \
        f"180° maneuver didn't converge: final error = {np.degrees(error_final):.1f}°"


@pytest.mark.vslow
def test_eci_goal_small_offset():
    """Test small angle maneuver (already close to goal)."""
    # Goal is close to boresight [0,0,1]
    goal_vec = normalize(np.array([0.1, 0, 1]))  # ~6° offset
    goal = ECI_Goal(goal_vec)

    traj, sat, _ = run_planning_with_goal(goal, duration=40.0)

    # Should converge well for small angles
    q_final = traj.states[3:7, -1]
    error_final = compute_pointing_error(q_final, goal_vec)

    assert error_final < 0.15, \
        f"Small angle maneuver didn't converge: {np.degrees(error_final):.1f}°"


# ==========================================
# COORDINATE_GOAL TESTS
# ==========================================

@pytest.mark.vslow
def test_coordinate_goal_nadir():
    """Test pointing at nadir (directly below satellite)."""
    # Satellite is at [R, 0, 0], nadir is [-1, 0, 0] direction
    # Use coordinates at 0° lat, 0° lon
    goal = Coordinate_Goal(lat=0, lon=0, alt=0)

    traj, sat, os0 = run_planning_with_goal(goal, duration=80.0)

    # Verify trajectory was created
    assert traj is not None
    assert traj.states.shape[1] > 1

    # Check that satellite orientation changes
    q_init = traj.states[3:7, 0]
    q_final = traj.states[3:7, -1]
    q_diff = np.linalg.norm(q_final - q_init)

    assert q_diff > 0.01, "Quaternion should change for coordinate goal"


@pytest.mark.vslow
def test_coordinate_goal_off_nadir():
    """Test pointing at a ground location not at nadir."""
    # Point at location offset from nadir
    goal = Coordinate_Goal(lat=30, lon=45, alt=0)

    traj, sat, _ = run_planning_with_goal(goal, duration=80.0)

    # Verify trajectory was created with valid data
    assert traj is not None
    assert np.all(np.isfinite(traj.states))
    assert np.all(np.isfinite(traj.controls))


# ==========================================
# NO_GOAL TESTS (DETUMBLING)
# ==========================================

@pytest.mark.vslow
def test_no_goal_rate_damping():
    """Test that No_Goal performs rate damping (reduces angular velocity)."""
    goal = No_Goal()

    # Start with initial rotation
    sat = create_test_satellite()
    w0 = np.array([0.05, -0.03, 0.04])  # Initial angular velocity
    q0 = normalize(np.array([0.1, 0.2, 0.3, 0.9]))
    h0 = np.array([0.0, 0.0, 0.0])
    x0 = np.concatenate([w0, q0, h0])

    orb, os0 = create_test_orbit(100)

    # Use high velocity cost for detumbling
    cost = CostWeights(
        angle=0.0,  # Don't care about orientation
        angle_N=0.0,
        ang_vel=1e4,
        ang_vel_N=1e6,  # High terminal velocity cost
        control_mult=1.0,
    )

    planner_settings = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=1.0)
    planner_settings.verbosity = False
    planner_settings.cost_main = cost
    planner_settings.cost_second = cost

    controller = Plan_and_Track_LQR(est_sat=sat, planner_settings=planner_settings)
    goals = GoalList({os0.J2000: goal})

    traj = controller.calculate_trajectory(
        t_start=os0.J2000,
        duration=80.0,
        x_0=x0,
        os_0=os0,
        goals=goals,
        verbose=False
    )

    # Check velocity reduction
    w_init = traj.states[0:3, 0]
    w_final = traj.states[0:3, -1]

    w_init_norm = np.linalg.norm(w_init)
    w_final_norm = np.linalg.norm(w_final)

    assert w_final_norm < w_init_norm, \
        f"Angular velocity should decrease: init={w_init_norm:.4f} final={w_final_norm:.4f}"


@pytest.mark.vslow
def test_no_goal_maintains_attitude():
    """Test that No_Goal from rest maintains current attitude."""
    goal = No_Goal()

    # Start at rest with non-identity orientation
    sat = create_test_satellite()
    w0 = np.array([0.0, 0.0, 0.0])
    q0 = normalize(np.array([0.3, 0.4, 0.5, 0.7]))  # Non-identity
    h0 = np.array([0.0, 0.0, 0.0])
    x0 = np.concatenate([w0, q0, h0])

    orb, os0 = create_test_orbit(80)

    planner_settings = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=1.0)
    planner_settings.verbosity = False

    controller = Plan_and_Track_LQR(est_sat=sat, planner_settings=planner_settings)
    goals = GoalList({os0.J2000: goal})

    traj = controller.calculate_trajectory(
        t_start=os0.J2000,
        duration=60.0,
        x_0=x0,
        os_0=os0,
        goals=goals,
        verbose=False
    )

    # Attitude should remain relatively stable
    q_init = traj.states[3:7, 0]
    q_final = traj.states[3:7, -1]

    # Compute angle change
    q_diff_angle = 2 * np.arccos(np.clip(np.abs(np.dot(q_init, q_final)), 0, 1))

    # Should not rotate much from initial orientation
    assert q_diff_angle < 0.5, \
        f"Attitude changed too much with No_Goal from rest: {np.degrees(q_diff_angle):.1f}°"


# ==========================================
# GOAL SWITCHING TESTS
# ==========================================

@pytest.mark.vslow
def test_goallist_multiple_goals():
    """Test trajectory planning with GoalList containing multiple goals."""
    sat = create_test_satellite()
    x0 = create_initial_state()
    orb, os0 = create_test_orbit(150)

    # Create goallist with multiple goals at different times
    # Note: Times are in Julian centuries from J2000
    t_start = os0.J2000
    dt_sec = 50  # Goal switch after 50 seconds
    t_switch = t_start + dt_sec * TimeConstants.sec2cent

    goals = GoalList({
        t_start: ECI_Goal(np.array([1, 0, 0])),      # First: point at +X
        t_switch: ECI_Goal(np.array([0, 1, 0])),    # Then: point at +Y
    })

    planner_settings = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=1.0)
    planner_settings.verbosity = False

    controller = Plan_and_Track_LQR(est_sat=sat, planner_settings=planner_settings)

    traj = controller.calculate_trajectory(
        t_start=t_start,
        duration=100.0,
        x_0=x0,
        os_0=os0,
        goals=goals,
        verbose=False
    )

    # Verify trajectory was created spanning both goals
    assert traj is not None
    assert traj.states.shape[1] > 50  # Should have timesteps after goal switch


# ==========================================
# MANUAL RUN
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("GOAL-BASED PLANNER TESTS")
    print("=" * 60)

    tests = [
        ("ECI goal +X", test_eci_goal_x_axis),
        ("ECI goal diagonal", test_eci_goal_diagonal),
        ("ECI goal -Z (180°)", test_eci_goal_negative_z),
        ("ECI goal small offset", test_eci_goal_small_offset),
        ("Coordinate goal nadir", test_coordinate_goal_nadir),
        ("Coordinate goal off-nadir", test_coordinate_goal_off_nadir),
        ("No_Goal rate damping", test_no_goal_rate_damping),
        ("No_Goal maintains attitude", test_no_goal_maintains_attitude),
        ("GoalList multiple goals", test_goallist_multiple_goals),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n{name}...", end=" ")
        try:
            test_func()
            print("PASSED")
            passed += 1
        except AssertionError as e:
            print(f"FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
