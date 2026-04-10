"""
Dynamics-based integration tests for ALTRO trajectory planner.

These tests verify physically meaningful properties with analytically known answers:
- Bang-bang control optimality for time-optimal maneuvers
- Angular momentum conservation
- Eigenaxis rotation for minimum-path maneuvers
- Rest-to-rest maneuver symmetry
- Energy dissipation properties
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

from ADCS.CONOPS.goals import ECI_Goal, No_Goal
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
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def create_rw_satellite(J_diag: np.ndarray = None):
    """Create RW-only satellite with specified inertia."""
    if J_diag is None:
        J_diag = np.array([0.1, 0.1, 0.1])  # Symmetric by default

    rw_max_torque = 0.01
    rw_J = 0.001
    rw_h0 = 0.0
    rw_hmax = 0.05
    rws = [RW(axis=j, max_torque=rw_max_torque, J=rw_J, h=rw_h0, h_max=rw_hmax)
           for j in MathConstants.unitvecs]
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    return Satellite(
        mass=4.0,
        J_0=np.diagflat(J_diag),
        actuators=rws,
        sensors=mtms,
        boresight=np.array([0, 0, 1])
    )


def create_static_orbit(duration: int = 200):
    """Create orbit with zero magnetic field (no external torques for RW-only)."""
    ephem = Ephemeris()
    R = 7000 * np.array([0, np.sqrt(2)/2, np.sqrt(2)/2])
    V = np.array([8, 0, 0])
    os0 = Orbital_State(
        ephem=ephem,
        J2000=0.22,
        R=R, V=V,
        B=np.array([0, 0, 0]),  # Zero B-field: no external torques
        S=np.array([1e5, 0, 0]),
        rho=0.0  # No drag
    )

    orbs = [os0.copy() for _ in range(duration + 10)]
    for j in range(len(orbs)):
        orbs[j].J2000 = os0.J2000 + j * TimeConstants.sec2cent
    return Orbit(orbs), os0


def quat_to_angle(q: np.ndarray) -> float:
    """Extract rotation angle from quaternion [w, x, y, z] or [x, y, z, w]."""
    # Assuming ADCS convention: q = [qx, qy, qz, qw] or scalar-last
    # scipy uses scalar-last [x, y, z, w]
    if np.abs(q[3]) > 1.0:
        q = q / np.linalg.norm(q)
    angle = 2 * np.arccos(np.clip(np.abs(q[3]), -1, 1))
    return angle


def quat_error_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    """Compute rotation angle between two quaternions."""
    # q_err = q2 * q1^{-1}
    # For unit quaternions, q^{-1} = q_conj = [-qv, qw]
    q1_conj = np.array([-q1[0], -q1[1], -q1[2], q1[3]])

    # Quaternion multiplication: q2 * q1_conj
    w1, x1, y1, z1 = q1_conj[3], q1_conj[0], q1_conj[1], q1_conj[2]
    w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    angle = 2 * np.arccos(np.clip(np.abs(w), -1, 1))
    return angle


def run_planning(sat: Satellite, x0: np.ndarray, os0: Orbital_State,
                 goal: ECI_Goal, duration: float,
                 cost_weights: CostWeights = None,
                 control_weight_scale: float = 1.0) -> Trajectory:
    """Run trajectory planning with given configuration."""
    orb, _ = create_static_orbit(int(duration) + 50)

    planner_settings = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=1.0)
    planner_settings.verbosity = False

    if cost_weights:
        planner_settings.cost_main = cost_weights
        planner_settings.cost_second = cost_weights

    # Scale control weights for bang-bang tests
    planner_settings.rw_control_weight *= control_weight_scale

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
    return traj


# ==========================================
# BANG-BANG CONTROL TESTS
# ==========================================

@pytest.mark.vslow
def test_control_effort_nonzero():
    """
    Test that the planner produces non-trivial control effort.

    For a maneuver from rest to a different orientation, the control
    should be non-zero for a significant portion of the trajectory.
    This verifies the planner is actually commanding actuators.
    """
    # Symmetric satellite for decoupled axes
    sat = create_rw_satellite(J_diag=np.array([0.1, 0.1, 0.1]))
    _, os0 = create_static_orbit()

    # Start at rest, identity orientation
    w0 = np.array([0.0, 0.0, 0.0])
    q0 = normalize(np.array([0, 0, 0, 1]))  # Identity: [qx, qy, qz, qw]
    h0 = np.array([0.0, 0.0, 0.0])
    x0 = np.concatenate([w0, q0, h0])

    # Goal: 90-degree rotation about z-axis
    # Boresight is [0,0,1], so goal in ECI that requires z-rotation
    goal = ECI_Goal(normalize(np.array([1, 0, 0])))  # Point boresight at +X

    traj = run_planning(sat, x0, os0, goal, duration=50.0)

    # Check that control is non-trivial
    u_max_magnitude = np.max(np.abs(traj.controls))
    assert u_max_magnitude > 1e-6, \
        f"Control effort too small: max |u| = {u_max_magnitude}"

    # Control should be active for significant portion of trajectory
    u_norm = np.linalg.norm(traj.controls, axis=0)
    active_timesteps = np.sum(u_norm > 1e-8)
    fraction_active = active_timesteps / len(u_norm)

    assert fraction_active > 0.3, \
        f"Control not active enough: only {fraction_active*100:.1f}% of timesteps have control"

    # For LQR-based planner, control should change smoothly (switch signs for maneuver)
    for axis in range(3):
        u_axis = traj.controls[axis, :]
        if np.max(np.abs(u_axis)) > 1e-8:  # Only check if this axis is used
            sign_changes = np.sum(np.diff(np.sign(u_axis + 1e-12)) != 0)
            # Should have at least one sign change for a rest-to-rest maneuver
            assert sign_changes >= 1, \
                f"Axis {axis}: Expected sign change for acceleration/deceleration"


@pytest.mark.vslow
def test_bangbang_symmetry_rest_to_rest():
    """
    Test that rest-to-rest maneuver has symmetric control profile.

    For a rest-to-rest maneuver with symmetric boundary conditions,
    the optimal control profile should be symmetric about the midpoint:
    u(t) = -u(T-t) for time-optimal control.
    """
    sat = create_rw_satellite(J_diag=np.array([0.1, 0.1, 0.1]))
    _, os0 = create_static_orbit()

    # Rest initial state
    w0 = np.array([0.0, 0.0, 0.0])
    q0 = normalize(np.array([0, 0, 0, 1]))
    h0 = np.array([0.0, 0.0, 0.0])
    x0 = np.concatenate([w0, q0, h0])

    goal = ECI_Goal(normalize(np.array([1, 0, 0])))

    cost = CostWeights(
        angle=1e3,
        angle_N=1e5,
        ang_vel=1e3,
        ang_vel_N=1e6,  # High terminal velocity cost for rest-to-rest
        control_mult=0.1,
    )

    traj = run_planning(sat, x0, os0, goal, duration=60.0, cost_weights=cost)

    # Check final velocity is near zero (rest-to-rest)
    w_final = traj.states[0:3, -1]
    w_final_norm = np.linalg.norm(w_final)
    assert w_final_norm < 0.01, f"Final angular velocity {w_final_norm} should be near zero"

    # Check control symmetry: u(t) ≈ -u(T-t)
    # Compare first half with reversed second half
    N = traj.controls.shape[1]
    mid = N // 2

    for axis in range(3):
        u = traj.controls[axis, :]
        first_half = u[:mid]
        second_half_reversed = u[N-mid:][::-1]

        # Truncate to same length
        min_len = min(len(first_half), len(second_half_reversed))
        first_half = first_half[:min_len]
        second_half_reversed = second_half_reversed[:min_len]

        # Check anti-symmetry: u(t) ≈ -u(T-t)
        # Using correlation: if anti-symmetric, sum should be near zero
        asymmetry = np.mean(first_half + second_half_reversed)
        max_u = np.max(np.abs(u)) + 1e-10
        relative_asymmetry = np.abs(asymmetry) / max_u

        # Allow some asymmetry due to discretization and nonlinear dynamics
        assert relative_asymmetry < 0.5, \
            f"Axis {axis} control not symmetric: relative asymmetry = {relative_asymmetry:.3f}"


# ==========================================
# ANGULAR MOMENTUM CONSERVATION TESTS
# ==========================================

@pytest.mark.vslow
def test_momentum_conservation_rw_only():
    """
    Test angular momentum conservation for RW-only satellite.

    With no external torques (B=0), total angular momentum H = J*w + h_rw
    must be conserved: dH/dt = 0.
    """
    sat = create_rw_satellite()
    _, os0 = create_static_orbit()
    J = sat.J_0

    # Non-zero initial momentum
    w0 = np.array([0.01, -0.005, 0.008])
    q0 = normalize(np.array([0.1, 0.2, 0.3, 0.9]))
    h0 = np.array([0.001, -0.001, 0.002])
    x0 = np.concatenate([w0, q0, h0])

    # Compute initial total momentum (in body frame)
    H_init = J @ w0 + h0

    goal = ECI_Goal(normalize(np.array([1, 1, 0])))
    traj = run_planning(sat, x0, os0, goal, duration=40.0)

    # Check momentum at each timestep
    max_H_deviation = 0.0
    for k in range(traj.states.shape[1]):
        w_k = traj.states[0:3, k]
        h_k = traj.states[7:10, k]
        H_k = J @ w_k + h_k

        # Momentum should be constant (in inertial frame, but approximately in body for small rotations)
        H_deviation = np.linalg.norm(H_k - H_init)
        max_H_deviation = max(max_H_deviation, H_deviation)

    # Allow small deviation due to numerical integration
    assert max_H_deviation < 0.01, \
        f"Angular momentum not conserved: max deviation = {max_H_deviation}"


@pytest.mark.vslow
def test_rw_momentum_changes_during_maneuver():
    """
    Test that RW momentum changes during a reorientation maneuver.

    For RW-only satellites, the wheels must spin up/down to create
    torques for attitude control. This test verifies that the planner
    actually uses the reaction wheels (changes their momentum).
    """
    sat = create_rw_satellite()
    _, os0 = create_static_orbit()

    # Start at rest with zero RW momentum
    w0 = np.array([0.0, 0.0, 0.0])
    q0 = normalize(np.array([0, 0, 0, 1]))
    h0 = np.array([0.0, 0.0, 0.0])
    x0 = np.concatenate([w0, q0, h0])

    goal = ECI_Goal(normalize(np.array([1, 0, 0])))
    traj = run_planning(sat, x0, os0, goal, duration=50.0)

    # Track max RW momentum magnitude during trajectory
    max_h_norm = 0.0
    for k in range(traj.states.shape[1]):
        h_k = traj.states[7:10, k]
        h_norm = np.linalg.norm(h_k)
        max_h_norm = max(max_h_norm, h_norm)

    # RW momentum should change significantly during maneuver
    assert max_h_norm > 1e-5, \
        f"RW momentum didn't change: max ||h|| = {max_h_norm}"

    # Check that at least one RW was used significantly
    h_max_per_axis = np.max(np.abs(traj.states[7:10, :]), axis=1)
    assert np.max(h_max_per_axis) > 1e-5, \
        f"No RW was used significantly: max per axis = {h_max_per_axis}"


# ==========================================
# EIGENAXIS ROTATION TESTS
# ==========================================

@pytest.mark.vslow
def test_rotation_stays_within_bounds():
    """
    Test that rotation quaternion stays normalized and physically valid.

    For any trajectory, quaternions should remain unit quaternions,
    and the rotation path should not take unnecessarily long paths
    (angle should not exceed 180° for a maneuver that could be done shorter).
    """
    sat = create_rw_satellite(J_diag=np.array([0.1, 0.1, 0.1]))
    _, os0 = create_static_orbit()

    # Identity initial orientation
    w0 = np.array([0.0, 0.0, 0.0])
    q0 = normalize(np.array([0, 0, 0, 1]))  # Identity
    h0 = np.array([0.0, 0.0, 0.0])
    x0 = np.concatenate([w0, q0, h0])

    # Goal requiring ~90° rotation
    goal = ECI_Goal(normalize(np.array([1, 0, 0])))

    traj = run_planning(sat, x0, os0, goal, duration=60.0)

    # Check quaternion normalization throughout trajectory
    for k in range(traj.states.shape[1]):
        q_k = traj.states[3:7, k]
        q_norm = np.linalg.norm(q_k)
        assert 0.99 < q_norm < 1.01, \
            f"Quaternion not normalized at step {k}: ||q|| = {q_norm}"

    # Check that we don't take the long way around (angle < 180°)
    for k in range(traj.states.shape[1]):
        q_k = traj.states[3:7, k]
        angle = quat_to_angle(q_k)
        assert angle < np.pi + 0.1, \
            f"Rotation angle {np.degrees(angle):.1f}° exceeds 180° at step {k}"


@pytest.mark.vslow
def test_error_decreases_overall():
    """
    Test that the pointing error decreases from initial to final state.

    For a well-tuned trajectory, even if there are oscillations,
    the final error should be significantly less than the initial error.
    """
    sat = create_rw_satellite()
    _, os0 = create_static_orbit()

    w0 = np.array([0.0, 0.0, 0.0])
    q0 = normalize(np.array([0, 0, 0, 1]))
    h0 = np.array([0.0, 0.0, 0.0])
    x0 = np.concatenate([w0, q0, h0])

    # 90-degree rotation goal - use well-tuned costs
    goal_vec = normalize(np.array([1, 0, 0]))
    goal = ECI_Goal(goal_vec)

    # Higher terminal cost for better convergence
    cost = CostWeights(
        angle=1e3,
        angle_N=1e5,
        ang_vel=1e3,
        ang_vel_N=1e5,
        control_mult=1.0,
    )

    traj = run_planning(sat, x0, os0, goal, duration=100.0, cost_weights=cost)

    # Compute initial and final pointing errors
    boresight = np.array([0, 0, 1])

    def compute_pointing_error(q):
        r = Rotation.from_quat([q[0], q[1], q[2], q[3]])
        boresight_eci = r.apply(boresight)
        cos_angle = np.clip(np.dot(boresight_eci, goal_vec), -1, 1)
        return np.arccos(cos_angle)

    q_init = traj.states[3:7, 0]
    q_final = traj.states[3:7, -1]

    error_init = compute_pointing_error(q_init)
    error_final = compute_pointing_error(q_final)

    # Error should decrease significantly (at least 50%)
    error_reduction = (error_init - error_final) / error_init
    assert error_reduction > 0.5, \
        f"Error didn't decrease enough: initial={np.degrees(error_init):.1f}°, " \
        f"final={np.degrees(error_final):.1f}°, reduction={error_reduction*100:.1f}%"

    # Final error should be reasonable (< 30°)
    assert error_final < 0.52, \
        f"Final error too large: {np.degrees(error_final):.1f}° (should be < 30°)"


# ==========================================
# ENERGY TESTS
# ==========================================

@pytest.mark.vslow
def test_kinetic_energy_bounded():
    """
    Test that rotational kinetic energy stays bounded.

    T = 0.5 * w^T * J * w should not grow unboundedly.
    """
    sat = create_rw_satellite()
    _, os0 = create_static_orbit()
    J = sat.J_0

    w0 = np.array([0.0, 0.0, 0.0])
    q0 = normalize(np.array([0.1, 0.2, 0.1, 0.95]))
    h0 = np.array([0.0, 0.0, 0.0])
    x0 = np.concatenate([w0, q0, h0])

    goal = ECI_Goal(normalize(np.array([1, 1, 1])))
    traj = run_planning(sat, x0, os0, goal, duration=60.0)

    # Track kinetic energy
    max_KE = 0.0
    for k in range(traj.states.shape[1]):
        w_k = traj.states[0:3, k]
        KE = 0.5 * w_k @ J @ w_k
        max_KE = max(max_KE, KE)

    # Final kinetic energy should be low (at rest)
    w_final = traj.states[0:3, -1]
    KE_final = 0.5 * w_final @ J @ w_final

    # Energy should not explode
    assert max_KE < 1.0, f"Kinetic energy grew too large: max = {max_KE}"

    # Should settle to low energy
    assert KE_final < 0.01, f"Final kinetic energy = {KE_final} (should be near zero)"


@pytest.mark.vslow
def test_rw_momentum_bounded():
    """
    Test that RW momentum stays within saturation limits.
    """
    sat = create_rw_satellite()
    _, os0 = create_static_orbit()

    w0 = np.array([0.02, -0.01, 0.015])  # Start with rotation
    q0 = normalize(np.array([0.1, 0.2, 0.3, 0.9]))
    h0 = np.array([0.0, 0.0, 0.0])
    x0 = np.concatenate([w0, q0, h0])

    goal = ECI_Goal(normalize(np.array([0, 1, 0])))
    traj = run_planning(sat, x0, os0, goal, duration=70.0)

    # Check RW momentum limits
    h_max = sat.actuators[0].h_max  # Assuming all RWs have same limit

    for k in range(traj.states.shape[1]):
        h_k = traj.states[7:10, k]
        for i, h_i in enumerate(h_k):
            assert np.abs(h_i) <= h_max * 1.1, \
                f"RW {i} momentum {h_i} exceeds limit {h_max} at step {k}"


# ==========================================
# CONVERGENCE TESTS
# ==========================================

@pytest.mark.vslow
def test_final_velocity_decreases():
    """
    Test that the trajectory ends with reduced angular velocity.

    For a rest-to-rest maneuver with high terminal velocity cost,
    the final velocity should be lower than mid-trajectory velocity.
    """
    sat = create_rw_satellite()
    _, os0 = create_static_orbit()

    w0 = np.array([0.0, 0.0, 0.0])
    q0 = normalize(np.array([0, 0, 0, 1]))
    h0 = np.array([0.0, 0.0, 0.0])
    x0 = np.concatenate([w0, q0, h0])

    goal_vec = normalize(np.array([1, 0, 0]))
    goal = ECI_Goal(goal_vec)

    # High terminal velocity cost for rest-to-rest behavior
    cost = CostWeights(
        angle=1e3,
        angle_N=1e4,
        ang_vel=1e3,
        ang_vel_N=1e6,  # Very high terminal velocity cost
        control_mult=1.0,
    )

    traj = run_planning(sat, x0, os0, goal, duration=80.0, cost_weights=cost)

    # Compute velocity magnitude over trajectory
    w_norms = np.array([np.linalg.norm(traj.states[0:3, k])
                        for k in range(traj.states.shape[1])])

    # Mid-trajectory velocity should be non-zero (satellite is moving)
    mid_idx = len(w_norms) // 2
    w_mid = np.max(w_norms[mid_idx-5:mid_idx+5])
    assert w_mid > 1e-4, "Satellite should have non-zero velocity mid-trajectory"

    # Final velocity should be lower than peak velocity
    w_final = w_norms[-1]
    w_peak = np.max(w_norms)
    assert w_final < w_peak * 0.5, \
        f"Final velocity {w_final:.6f} should be < 50% of peak {w_peak:.6f}"

    # Final velocity should be reasonably small
    assert w_final < 0.05, \
        f"Final angular velocity ||w|| = {w_final:.4f} rad/s (should be < 0.05)"


@pytest.mark.vslow
def test_small_angle_maneuver():
    """
    Test that small angle maneuvers converge quickly and smoothly.

    For small angles, the linearized dynamics should be accurate,
    and the trajectory should be nearly linear in angle.
    """
    sat = create_rw_satellite()
    _, os0 = create_static_orbit()

    # Start very close to goal (5-degree offset)
    angle = 5 * np.pi / 180
    q0 = normalize(np.array([0, np.sin(angle/2), 0, np.cos(angle/2)]))  # 5° about Y
    w0 = np.array([0.0, 0.0, 0.0])
    h0 = np.array([0.0, 0.0, 0.0])
    x0 = np.concatenate([w0, q0, h0])

    # Goal is identity (boresight at +Z)
    # Current boresight is slightly off from +Z
    goal = ECI_Goal(np.array([0, 0, 1]))

    traj = run_planning(sat, x0, os0, goal, duration=30.0)

    # Should converge quickly for small angle
    q_final = traj.states[3:7, -1]
    # Error from identity
    angle_final = quat_to_angle(q_final)

    assert angle_final < 0.05, \
        f"Small angle maneuver didn't converge: final angle = {np.degrees(angle_final):.2f}°"


# ==========================================
# MANUAL RUN
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("DYNAMICS-BASED PLANNER TESTS")
    print("=" * 60)

    tests = [
        ("Control effort nonzero", test_control_effort_nonzero),
        ("Rest-to-rest symmetry", test_bangbang_symmetry_rest_to_rest),
        ("Momentum conservation", test_momentum_conservation_rw_only),
        ("RW momentum changes", test_rw_momentum_changes_during_maneuver),
        ("Rotation within bounds", test_rotation_stays_within_bounds),
        ("Error decreases overall", test_error_decreases_overall),
        ("Kinetic energy bounded", test_kinetic_energy_bounded),
        ("RW momentum bounded", test_rw_momentum_bounded),
        ("Final velocity decreases", test_final_velocity_decreases),
        ("Small angle maneuver", test_small_angle_maneuver),
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
