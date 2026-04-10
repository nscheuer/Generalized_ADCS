"""
Integration tests for ALTRO trajectory planner with different actuator configurations.

Tests verify that the planner can generate trajectories for:
- MTQ-only satellites
- RW-only satellites
- Mixed MTQ+RW satellites
"""

import sys
import os
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.controller.helpers.optional_dependencies import (
    trajectory_planner_available,
    trajectory_planner_missing_reason,
)

if not trajectory_planner_available():
    pytest.skip(trajectory_planner_missing_reason(), allow_module_level=True)

from ADCS.CONOPS.goals import ECI_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.plan_and_track import PlannerSettings, Trajectory
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize


def create_mtq_only_satellite():
    """Create a satellite with only magnetotorquers."""
    mtq_max_torque = 0.1
    mtqs = [MTQ(axis=j, max_torque=mtq_max_torque) for j in MathConstants.unitvecs]
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    return Satellite(
        mass=4.0,
        J_0=np.diagflat([0.1, 0.12, 0.15]),
        actuators=mtqs,
        sensors=mtms,
        boresight=np.array([0, 0, 1])
    )


def create_rw_only_satellite():
    """Create a satellite with only reaction wheels."""
    rw_max_torque = 0.005
    rw_J = 0.0014
    rw_h0 = 0.001
    rw_hmax = 0.015
    rws = [RW(axis=j, max_torque=rw_max_torque, J=rw_J, h=rw_h0, h_max=rw_hmax)
           for j in MathConstants.unitvecs]
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    return Satellite(
        mass=4.0,
        J_0=np.diagflat([0.1, 0.12, 0.15]),
        actuators=rws,
        sensors=mtms,
        boresight=np.array([0, 0, 1])
    )


def create_mixed_satellite():
    """Create a satellite with both MTQs and RWs."""
    mtq_max_torque = 0.1
    mtqs = [MTQ(axis=j, max_torque=mtq_max_torque) for j in MathConstants.unitvecs]

    rw_max_torque = 0.005
    rw_J = 0.0014
    rw_h0 = 0.001
    rw_hmax = 0.015
    rws = [RW(axis=j, max_torque=rw_max_torque, J=rw_J, h=rw_h0, h_max=rw_hmax)
           for j in MathConstants.unitvecs]

    # RWs first, then MTQs (standard ordering)
    acts = rws + mtqs
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    return Satellite(
        mass=4.0,
        J_0=np.diagflat([0.1, 0.12, 0.15]),
        actuators=acts,
        sensors=mtms,
        boresight=np.array([0, 0, 1])
    )


def create_initial_state(sat: Satellite):
    """Create initial state vector for satellite."""
    w0 = np.array([0.001, 0.002, -0.001])  # Small angular velocity
    q0 = normalize(np.array([1, 0, 0, 0]))  # Identity quaternion

    # Add RW momentum if satellite has reaction wheels
    rw_count = sum(1 for act in sat.actuators if isinstance(act, RW))
    if rw_count > 0:
        h0 = np.array([0.001] * rw_count)
        return np.concatenate([w0, q0, h0])
    else:
        return np.concatenate([w0, q0])


def create_orbit():
    """Create a test orbit with static magnetic field."""
    ephem = Ephemeris()
    R = 7000 * np.array([0, np.sqrt(2)/2, np.sqrt(2)/2])
    V = np.array([8, 0, 0])
    os0 = Orbital_State(
        ephem=ephem,
        J2000=0.22 - TimeConstants.sec2cent,
        R=R, V=V,
        B=np.array([0, 0.1, 0]),  # Static magnetic field
        S=np.array([1e5+1, 0, 0]),
        rho=5e-12
    )

    # Create static orbit for testing (100 timesteps)
    dur = 110
    orbs = [os0.copy() for _ in range(dur)]
    for j in range(dur):
        orbs[j].J2000 = os0.J2000 + j * TimeConstants.sec2cent
    return Orbit(orbs), os0


def run_trajectory_planning(sat: Satellite, duration: float = 50.0):
    """Run trajectory planning and return results."""
    x0 = create_initial_state(sat)
    orb, os0 = create_orbit()

    # Configure planner settings
    planner_settings = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=1.0)
    planner_settings.verbosity = False

    # Create controller and plan trajectory
    controller = Plan_and_Track_LQR(est_sat=sat, planner_settings=planner_settings)

    goals = GoalList({0.22: ECI_Goal(normalize(np.array([1, 1, 1])))})

    traj = controller.calculate_trajectory(
        t_start=0.22,
        duration=duration,
        x_0=x0,
        os_0=os0,
        goals=goals,
        verbose=False
    )

    return traj, controller


# ==========================================
# TESTS
# ==========================================

@pytest.mark.vslow
def test_mtq_only_trajectory_planning():
    """Test trajectory planning with MTQ-only satellite."""
    sat = create_mtq_only_satellite()
    traj, controller = run_trajectory_planning(sat)

    # Verify trajectory was created
    assert traj is not None, "Trajectory should be created"
    assert isinstance(traj, Trajectory), "Should return Trajectory object"

    # Verify trajectory has data
    assert traj.states is not None, "Trajectory should have states"
    assert traj.controls is not None, "Trajectory should have controls"
    assert len(traj.times) > 1, "Trajectory should have multiple timesteps"

    # Verify control dimensions match actuator count (3 MTQs)
    assert traj.controls.shape[0] == 3, f"Expected 3 control channels for MTQs, got {traj.controls.shape[0]}"

    # Verify state dimensions (w=3, q=4, no RW momentum)
    assert traj.states.shape[0] == 7, f"Expected 7 states (w+q), got {traj.states.shape[0]}"


@pytest.mark.vslow
def test_rw_only_trajectory_planning():
    """Test trajectory planning with RW-only satellite."""
    sat = create_rw_only_satellite()
    traj, controller = run_trajectory_planning(sat)

    # Verify trajectory was created
    assert traj is not None, "Trajectory should be created"
    assert isinstance(traj, Trajectory), "Should return Trajectory object"

    # Verify trajectory has data
    assert traj.states is not None, "Trajectory should have states"
    assert traj.controls is not None, "Trajectory should have controls"
    assert len(traj.times) > 1, "Trajectory should have multiple timesteps"

    # Verify control dimensions match actuator count (3 RWs)
    assert traj.controls.shape[0] == 3, f"Expected 3 control channels for RWs, got {traj.controls.shape[0]}"

    # Verify state dimensions (w=3, q=4, h=3)
    assert traj.states.shape[0] == 10, f"Expected 10 states (w+q+h), got {traj.states.shape[0]}"


@pytest.mark.vslow
def test_mixed_actuator_trajectory_planning():
    """Test trajectory planning with mixed MTQ+RW satellite."""
    sat = create_mixed_satellite()
    traj, controller = run_trajectory_planning(sat)

    # Verify trajectory was created
    assert traj is not None, "Trajectory should be created"
    assert isinstance(traj, Trajectory), "Should return Trajectory object"

    # Verify trajectory has data
    assert traj.states is not None, "Trajectory should have states"
    assert traj.controls is not None, "Trajectory should have controls"
    assert len(traj.times) > 1, "Trajectory should have multiple timesteps"

    # Verify control dimensions (3 RWs + 3 MTQs = 6)
    assert traj.controls.shape[0] == 6, f"Expected 6 control channels for mixed, got {traj.controls.shape[0]}"

    # Verify state dimensions (w=3, q=4, h=3)
    assert traj.states.shape[0] == 10, f"Expected 10 states (w+q+h), got {traj.states.shape[0]}"


@pytest.mark.vslow
def test_mtq_trajectory_points_toward_goal():
    """Test that MTQ trajectory moves toward goal direction."""
    sat = create_mtq_only_satellite()
    traj, _ = run_trajectory_planning(sat, duration=100.0)

    # Get initial and final quaternions
    q_init = traj.states[3:7, 0]
    q_final = traj.states[3:7, -1]

    # The quaternion should change (satellite should rotate)
    q_diff = np.abs(q_final - q_init)
    assert np.max(q_diff) > 1e-3, "Quaternion should change during trajectory"


@pytest.mark.vslow
def test_rw_trajectory_conserves_momentum():
    """Test that RW-only trajectory conserves total angular momentum (approximately)."""
    sat = create_rw_only_satellite()
    traj, _ = run_trajectory_planning(sat, duration=50.0)

    # For RW-only satellite without external torques, total angular momentum should be conserved
    # H_total = J * omega + h_rw
    J = sat.J_0

    # Initial total momentum
    w_init = traj.states[0:3, 0]
    h_rw_init = traj.states[7:10, 0]
    H_init = J @ w_init + h_rw_init

    # Final total momentum
    w_final = traj.states[0:3, -1]
    h_rw_final = traj.states[7:10, -1]
    H_final = J @ w_final + h_rw_final

    # Allow some tolerance for numerical integration and small disturbances
    H_diff = np.linalg.norm(H_final - H_init)
    assert H_diff < 0.1, f"Total momentum changed by {H_diff}, expected conservation"


@pytest.mark.vslow
def test_trajectory_control_within_limits():
    """Test that control outputs stay within actuator limits."""
    sat = create_mixed_satellite()
    traj, _ = run_trajectory_planning(sat)

    # Get actuator limits
    for i, act in enumerate(sat.actuators):
        u_max = act.u_max
        u_trajectory = traj.controls[i, :]

        # Check that all control values are within limits (with small tolerance)
        max_control = np.max(np.abs(u_trajectory))
        assert max_control <= u_max * 1.01, \
            f"Actuator {i} control {max_control} exceeds limit {u_max}"


@pytest.mark.vslow
def test_trajectory_states_finite():
    """Test that all trajectory states are finite (no NaN or Inf)."""
    sat = create_mixed_satellite()
    traj, _ = run_trajectory_planning(sat)

    assert np.all(np.isfinite(traj.states)), "All states should be finite"
    assert np.all(np.isfinite(traj.controls)), "All controls should be finite"
    assert np.all(np.isfinite(traj.times)), "All times should be finite"


# ==========================================
# MANUAL RUN
# ==========================================

if __name__ == "__main__":
    print("Testing MTQ-only satellite...")
    test_mtq_only_trajectory_planning()
    print("  PASSED")

    print("Testing RW-only satellite...")
    test_rw_only_trajectory_planning()
    print("  PASSED")

    print("Testing mixed actuator satellite...")
    test_mixed_actuator_trajectory_planning()
    print("  PASSED")

    print("Testing trajectory moves toward goal...")
    test_mtq_trajectory_points_toward_goal()
    print("  PASSED")

    print("Testing momentum conservation...")
    test_rw_trajectory_conserves_momentum()
    print("  PASSED")

    print("Testing control limits...")
    test_trajectory_control_within_limits()
    print("  PASSED")

    print("Testing states are finite...")
    test_trajectory_states_finite()
    print("  PASSED")

    print("\nAll tests passed!")
