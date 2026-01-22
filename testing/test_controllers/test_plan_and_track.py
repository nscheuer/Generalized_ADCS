"""
Unit and integration tests for Plan_and_Track controllers.

Tests cover:
- Control clipping functionality
- TVLQR feedback gain (K) application
- State error computation
- Disturbance estimation and adaptation
- Trajectory tracking behavior
"""

import sys
import os
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.CONOPS.goals import ECI_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.plan_and_track_lqr_disturbed import Plan_and_Track_LQR_Disturbed
from ADCS.controller.plan_and_track_exact import Plan_and_Track_Exact
from ADCS.controller.helpers import PlannerSettings, Trajectory
from ADCS.controller.helpers.planner_subsettings import CostWeights
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import RW, MTQ
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize, quat_diff, quat_to_vec3


# ==========================================
# FIXTURES AND HELPER FUNCTIONS
# ==========================================

def create_test_satellite_rw_only():
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


def create_test_satellite_mtq_rw():
    """Create a satellite with MTQs and RWs for testing."""
    rw_max_torque = 0.005
    mtq_max_moment = 0.2

    rws = [RW(axis=np.array([0, 0, 1]), max_torque=rw_max_torque, J=0.001, h=0.0, h_max=0.05)]
    mtqs = [MTQ(axis=j, max_torque=mtq_max_moment) for j in MathConstants.unitvecs]
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]

    return Satellite(
        mass=4.0,
        J_0=np.diagflat([0.05, 0.05, 0.03]),
        actuators=mtqs + rws,  # MTQs first, then RW
        sensors=mtms,
        boresight=np.array([0, 0, 1])
    )


def create_test_orbit(duration: int = 150):
    """Create test orbit."""
    ephem = Ephemeris()
    R = 6778 * np.array([1, 0, 0])
    V = np.array([0, 7.67, 0])
    os0 = Orbital_State(
        ephem=ephem,
        J2000=0.22,
        R=R, V=V,
        B=np.array([2e-5, 1e-5, 3e-5]),  # Non-zero B-field for MTQ tests
        S=np.array([1e5, 0, 0]),
        rho=0.0
    )
    orbs = [os0.copy() for _ in range(duration + 10)]
    for j in range(len(orbs)):
        orbs[j].J2000 = os0.J2000 + j * TimeConstants.sec2cent
    return Orbit(orbs), os0


def create_initial_state(n_rw: int = 3):
    """Create standard initial state."""
    w0 = np.array([0.0, 0.0, 0.0])
    q0 = normalize(np.array([0, 0, 0, 1]))
    h0 = np.zeros(n_rw)
    return np.concatenate([w0, q0, h0])


def setup_controller_and_trajectory(controller_class, sat, duration: float = 60.0, **kwargs):
    """Setup a controller and compute a trajectory."""
    x0 = create_initial_state(n_rw=len(sat.rw_actuators))
    _, os0 = create_test_orbit(int(duration) + 50)

    planner_settings = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=10.0, dt_tvlqr=1.0)
    planner_settings.verbosity = False

    controller = controller_class(est_sat=sat, planner_settings=planner_settings, **kwargs)

    goal = ECI_Goal(np.array([1, 0, 0]))
    goals = GoalList({os0.J2000: goal})

    traj = controller.calculate_trajectory(
        t_start=os0.J2000,
        duration=duration,
        x_0=x0,
        os_0=os0,
        goals=goals,
        verbose=False
    )
    controller.set_active_trajectory(traj)

    return controller, traj, os0, x0


# ==========================================
# CLIPPING TESTS
# ==========================================

class TestControlClipping:
    """Tests for control clipping functionality."""

    def test_clip_control_within_limits(self):
        """Test that controls within limits are unchanged."""
        sat = create_test_satellite_rw_only()
        controller, traj, os0, x0 = setup_controller_and_trajectory(Plan_and_Track_LQR, sat)

        # Create a control vector within limits
        u_small = np.array([0.001, 0.001, 0.001])  # Well within 0.01 limit

        u_clipped = controller.clip_control(u_small, clip=True)
        np.testing.assert_array_almost_equal(u_clipped, u_small)

    def test_clip_control_exceeds_limits(self):
        """Test that controls exceeding limits are clipped."""
        sat = create_test_satellite_rw_only()
        controller, traj, os0, x0 = setup_controller_and_trajectory(Plan_and_Track_LQR, sat)

        # Create a control vector exceeding limits (RW u_max = 0.01)
        u_large = np.array([0.1, -0.05, 0.02])

        u_clipped = controller.clip_control(u_large, clip=True)

        # Check each actuator is clipped to its limit
        for i, act in enumerate(sat.actuators):
            assert abs(u_clipped[i]) <= act.u_max, \
                f"Control {i} exceeds limit: {u_clipped[i]} > {act.u_max}"

    def test_clip_control_disabled(self):
        """Test that clip=False returns unclipped control."""
        sat = create_test_satellite_rw_only()
        controller, traj, os0, x0 = setup_controller_and_trajectory(Plan_and_Track_LQR, sat)

        u_large = np.array([0.1, -0.05, 0.02])

        u_unclipped = controller.clip_control(u_large, clip=False)
        np.testing.assert_array_equal(u_unclipped, u_large)

    def test_clip_uses_hardware_limits_not_planner_limits(self):
        """Verify clipping uses actual hardware limits, not scaled planner limits."""
        sat = create_test_satellite_rw_only()
        controller, traj, os0, x0 = setup_controller_and_trajectory(Plan_and_Track_LQR, sat)

        # planner_settings.umax is scaled by control_limit_scale (0.75)
        # Hardware limit is 0.01, planner limit is 0.0075
        # A control of 0.008 should NOT be clipped (within hardware limit)
        u_between = np.array([0.008, 0.008, 0.008])

        u_clipped = controller.clip_control(u_between, clip=True)
        np.testing.assert_array_almost_equal(u_clipped, u_between)

    def test_find_u_clips_by_default(self):
        """Test that find_u clips control by default."""
        sat = create_test_satellite_rw_only()
        controller, traj, os0, x0 = setup_controller_and_trajectory(Plan_and_Track_LQR, sat)

        # Create a state with large error to generate large control
        x_large_error = x0.copy()
        x_large_error[0:3] = np.array([1.0, 1.0, 1.0])  # Large angular velocity error

        u = controller.find_u(x_hat=x_large_error, sens=np.array([]), est_sat=sat, os_hat=os0)

        # Verify all controls are within hardware limits
        for i, act in enumerate(sat.actuators):
            assert abs(u[i]) <= act.u_max + 1e-10, \
                f"Control {i} exceeds limit: {u[i]} > {act.u_max}"

    def test_find_u_no_clip_option(self):
        """Test that find_u can disable clipping."""
        sat = create_test_satellite_rw_only()
        controller, traj, os0, x0 = setup_controller_and_trajectory(Plan_and_Track_LQR, sat)

        # Create a state with large error
        x_large_error = x0.copy()
        x_large_error[0:3] = np.array([1.0, 1.0, 1.0])

        u_clipped = controller.find_u(x_hat=x_large_error, sens=np.array([]),
                                       est_sat=sat, os_hat=os0, clip=True)
        u_unclipped = controller.find_u(x_hat=x_large_error, sens=np.array([]),
                                         est_sat=sat, os_hat=os0, clip=False)

        # If error is large enough, unclipped should differ from clipped
        # (this depends on K gains being non-trivial)
        # At minimum, unclipped should have same sign as clipped
        assert np.all(np.sign(u_clipped) == np.sign(u_unclipped)) or \
               np.allclose(u_clipped, u_unclipped)


class TestClippingMixedActuators:
    """Test clipping with mixed actuator types (MTQ + RW)."""

    def test_clip_mixed_actuator_limits(self):
        """Test that each actuator type is clipped to its own limit."""
        sat = create_test_satellite_mtq_rw()
        controller, traj, os0, x0 = setup_controller_and_trajectory(Plan_and_Track_LQR, sat)

        # Create control exceeding all limits
        # 3 MTQs (u_max=0.2) + 1 RW (u_max=0.005)
        u_large = np.array([0.5, 0.5, 0.5, 0.01])

        u_clipped = controller.clip_control(u_large, clip=True)

        # MTQs should clip to 0.2
        np.testing.assert_almost_equal(u_clipped[0], 0.2)
        np.testing.assert_almost_equal(u_clipped[1], 0.2)
        np.testing.assert_almost_equal(u_clipped[2], 0.2)
        # RW should clip to 0.005
        np.testing.assert_almost_equal(u_clipped[3], 0.005)


# ==========================================
# K GAIN APPLICATION TESTS
# ==========================================

class TestKGainApplication:
    """Tests for TVLQR feedback gain application."""

    def test_zero_error_gives_reference_control(self):
        """When state matches reference, control should equal reference control."""
        sat = create_test_satellite_rw_only()
        controller, traj, os0, x0 = setup_controller_and_trajectory(Plan_and_Track_LQR, sat)

        # Get reference state at a mid-trajectory time
        t_mid = traj.start_time + (traj.end_time - traj.start_time) / 2
        x_ref = traj.get_state_at(t_mid)
        u_ref = traj.get_control_at(t_mid)

        # Compute control with zero error
        u = traj.compute_tracking_control(t_mid, x_ref)

        np.testing.assert_array_almost_equal(u, u_ref, decimal=10)

    def test_positive_error_produces_correction(self):
        """Positive state error should produce corrective control."""
        sat = create_test_satellite_rw_only()
        controller, traj, os0, x0 = setup_controller_and_trajectory(Plan_and_Track_LQR, sat)

        t_mid = traj.start_time + (traj.end_time - traj.start_time) / 2
        x_ref = traj.get_state_at(t_mid)
        u_ref = traj.get_control_at(t_mid)

        # Perturb angular velocity
        x_perturbed = x_ref.copy()
        x_perturbed[0:3] += np.array([0.01, 0.01, 0.01])

        u = traj.compute_tracking_control(t_mid, x_perturbed)

        # Control should differ from reference
        assert not np.allclose(u, u_ref), "Control should change with state error"

    def test_gain_shape_matches_error_state(self):
        """Verify gain matrix dimensions are correct."""
        sat = create_test_satellite_rw_only()
        controller, traj, os0, x0 = setup_controller_and_trajectory(Plan_and_Track_LQR, sat)

        t_mid = traj.start_time + (traj.end_time - traj.start_time) / 2
        K = traj.get_gain_at(t_mid)

        n_ctrl = len(sat.actuators)
        n_rw = len(sat.rw_actuators)
        n_error = 6 + n_rw  # 3 omega + 3 attitude (reduced from quaternion) + n_rw momentum

        assert K.shape == (n_ctrl, n_error), \
            f"K shape {K.shape} should be ({n_ctrl}, {n_error})"

    def test_state_diff_reduces_quaternion(self):
        """Verify _state_diff reduces 4D quaternion to 3D attitude error."""
        sat = create_test_satellite_rw_only()
        controller, traj, os0, x0 = setup_controller_and_trajectory(Plan_and_Track_LQR, sat)

        x_ref = traj.get_state_at(traj.start_time)
        x_curr = x_ref.copy()

        # Small quaternion perturbation
        x_curr[3:7] = normalize(x_curr[3:7] + np.array([0.01, 0.01, 0.01, 0]))

        dx = traj._state_diff(x_curr, x_ref)

        # Error state should be 6 + n_rw (not 7 + n_rw)
        n_rw = len(sat.rw_actuators)
        assert len(dx) == 6 + n_rw

    def test_angular_velocity_error_in_state_diff(self):
        """Test that angular velocity error is correctly computed."""
        sat = create_test_satellite_rw_only()
        controller, traj, os0, x0 = setup_controller_and_trajectory(Plan_and_Track_LQR, sat)

        x_ref = traj.get_state_at(traj.start_time)
        x_curr = x_ref.copy()

        # Add known angular velocity error
        w_error = np.array([0.05, -0.03, 0.02])
        x_curr[0:3] = x_ref[0:3] + w_error

        dx = traj._state_diff(x_curr, x_ref)

        np.testing.assert_array_almost_equal(dx[0:3], w_error)

    def test_rw_momentum_error_in_state_diff(self):
        """Test that RW momentum error is correctly computed."""
        sat = create_test_satellite_rw_only()
        controller, traj, os0, x0 = setup_controller_and_trajectory(Plan_and_Track_LQR, sat)

        x_ref = traj.get_state_at(traj.start_time)
        x_curr = x_ref.copy()

        n_rw = len(sat.rw_actuators)
        h_error = np.array([0.001, -0.002, 0.0015])[:n_rw]
        x_curr[7:7+n_rw] = x_ref[7:7+n_rw] + h_error

        dx = traj._state_diff(x_curr, x_ref)

        np.testing.assert_array_almost_equal(dx[6:6+n_rw], h_error)


class TestKGainEffect:
    """Test that K gains produce expected control behavior."""

    def test_larger_error_produces_larger_correction(self):
        """Larger state errors should produce larger control corrections."""
        sat = create_test_satellite_rw_only()
        controller, traj, os0, x0 = setup_controller_and_trajectory(Plan_and_Track_LQR, sat)

        t_mid = traj.start_time + (traj.end_time - traj.start_time) / 2
        x_ref = traj.get_state_at(t_mid)
        u_ref = traj.get_control_at(t_mid)

        # Small error
        x_small = x_ref.copy()
        x_small[0] += 0.01
        u_small = traj.compute_tracking_control(t_mid, x_small)

        # Large error (same direction)
        x_large = x_ref.copy()
        x_large[0] += 0.1
        u_large = traj.compute_tracking_control(t_mid, x_large)

        # Correction magnitude should be larger for larger error
        correction_small = np.linalg.norm(u_small - u_ref)
        correction_large = np.linalg.norm(u_large - u_ref)

        assert correction_large > correction_small, \
            f"Large error correction {correction_large} should exceed small error {correction_small}"

    def test_opposite_errors_produce_opposite_corrections(self):
        """Opposite state errors should produce opposite control corrections."""
        sat = create_test_satellite_rw_only()
        controller, traj, os0, x0 = setup_controller_and_trajectory(Plan_and_Track_LQR, sat)

        t_mid = traj.start_time + (traj.end_time - traj.start_time) / 2
        x_ref = traj.get_state_at(t_mid)
        u_ref = traj.get_control_at(t_mid)

        # Positive error
        x_pos = x_ref.copy()
        x_pos[0:3] += np.array([0.05, 0.05, 0.05])
        u_pos = traj.compute_tracking_control(t_mid, x_pos)

        # Negative error
        x_neg = x_ref.copy()
        x_neg[0:3] -= np.array([0.05, 0.05, 0.05])
        u_neg = traj.compute_tracking_control(t_mid, x_neg)

        # Corrections should be in opposite directions
        correction_pos = u_pos - u_ref
        correction_neg = u_neg - u_ref

        # Dot product should be negative (opposite directions)
        dot = np.dot(correction_pos, correction_neg)
        assert dot < 0, f"Corrections should be opposite: dot product = {dot}"


# ==========================================
# DISTURBANCE ESTIMATION TESTS
# ==========================================

def create_test_satellite_single_rw():
    """Create a satellite with a single RW for KwDist testing.

    Note: The KwDist C++ planner has dimension issues with 3 RWs,
    so we use a single RW configuration for disturbance tests.
    """
    rw = RW(axis=np.array([0, 0, 1]), max_torque=0.01, J=0.001, h=0.0, h_max=0.05)
    mtqs = [MTQ(axis=j, max_torque=0.1) for j in MathConstants.unitvecs]
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    return Satellite(
        mass=4.0,
        J_0=np.diagflat([0.1, 0.1, 0.1]),
        actuators=mtqs + [rw],  # 3 MTQs + 1 RW
        sensors=mtms,
        boresight=np.array([0, 0, 1])
    )


class TestDisturbanceEstimationUnit:
    """Unit tests for Plan_and_Track_LQR_Disturbed (no trajectory needed)."""

    def test_controller_initializes(self):
        """Controller should initialize without error."""
        sat = create_test_satellite_rw_only()
        planner_settings = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=10.0, dt_tvlqr=1.0)
        controller = Plan_and_Track_LQR_Disturbed(
            est_sat=sat, planner_settings=planner_settings
        )

        # Controller should have planner initialized
        assert controller.planner is not None
        assert controller.est_sat is sat


class TestDisturbanceCompensation:
    """Tests for disturbance compensation using satellite disturbance models.

    The LQR_Disturbed controller pulls disturbance torques from the
    EstimatedSatellite's disturbance models (e.g., SRP, drag, dipole).
    """

    def test_controller_uses_satellite_disturbance_models(self):
        """Controller should call est_sat.dist_torques() to get disturbance."""
        # This is a design/integration test - the controller uses the satellite's
        # disturbance models rather than computing disturbances internally
        sat = create_test_satellite_rw_only()

        # Verify satellite has dist_torques method
        assert hasattr(sat, 'dist_torques')

        # Create a simple state and orbital state
        x0 = create_initial_state(n_rw=len(sat.rw_actuators))
        _, os0 = create_test_orbit(50)

        # Compute disturbance torque - should work without error
        dist = sat.dist_torques(x=x0, os=os0)
        assert dist.shape == (3,)

    @pytest.mark.slow
    def test_kwdist_gain_has_disturbance_columns(self):
        """KwDist mode should have gains with extra columns for disturbance."""
        sat = create_test_satellite_single_rw()
        x0 = create_initial_state(n_rw=1)
        _, os0 = create_test_orbit(100)

        planner_settings = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=10.0, dt_tvlqr=1.0)
        controller = Plan_and_Track_LQR_Disturbed(
            est_sat=sat, planner_settings=planner_settings
        )

        goal = ECI_Goal(np.array([1, 0, 0]))
        goals = GoalList({os0.J2000: goal})

        traj = controller.calculate_trajectory(
            t_start=os0.J2000, duration=30.0, x_0=x0, os_0=os0, goals=goals, verbose=False
        )

        assert traj.use_disturbance_estimation == True

        # Get gain matrix - should have 3 extra columns for disturbance
        t_mid = traj.start_time + (traj.end_time - traj.start_time) / 2
        K = traj.get_gain_at(t_mid)

        n_ctrl = len(sat.actuators)
        n_rw = len(sat.rw_actuators)
        n_error_base = 6 + n_rw  # 3 omega + 3 attitude + n_rw momentum
        n_error_with_dist = n_error_base + 3

        assert K.shape == (n_ctrl, n_error_with_dist), \
            f"KwDist gain shape {K.shape} should be ({n_ctrl}, {n_error_with_dist})"

    @pytest.mark.slow
    def test_disturbance_fed_to_trajectory(self):
        """Disturbance torque should be passed to trajectory for control computation."""
        sat = create_test_satellite_single_rw()
        x0 = create_initial_state(n_rw=1)
        _, os0 = create_test_orbit(100)

        planner_settings = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=10.0, dt_tvlqr=1.0)
        controller = Plan_and_Track_LQR_Disturbed(
            est_sat=sat, planner_settings=planner_settings
        )

        goal = ECI_Goal(np.array([1, 0, 0]))
        goals = GoalList({os0.J2000: goal})

        traj = controller.calculate_trajectory(
            t_start=os0.J2000, duration=30.0, x_0=x0, os_0=os0, goals=goals, verbose=False
        )
        controller.set_active_trajectory(traj)

        # Call find_u - should not raise any errors
        u = controller.find_u(x_hat=x0, sens=np.array([]), est_sat=sat, os_hat=os0)

        # Control should be a valid vector
        assert u.shape == (len(sat.actuators),)
        assert np.all(np.isfinite(u))


# ==========================================
# PLAN_AND_TRACK_EXACT TESTS
# ==========================================

class TestPlanAndTrackExact:
    """Tests for Plan_and_Track_Exact controller."""

    def test_exact_returns_open_loop_control(self):
        """Exact controller should return trajectory control regardless of state."""
        sat = create_test_satellite_rw_only()
        controller, traj, os0, x0 = setup_controller_and_trajectory(Plan_and_Track_Exact, sat)

        t_mid = traj.start_time + (traj.end_time - traj.start_time) / 2
        u_ref = traj.get_control_at(t_mid)

        # Create orbital state at mid time
        os_mid = os0.copy()
        os_mid.J2000 = t_mid

        # Control should match reference regardless of current state
        x_random = np.random.randn(len(x0))
        x_random[3:7] = normalize(x_random[3:7])  # Normalize quaternion

        u = controller.find_u(x_hat=x_random, sens=np.array([]), est_sat=sat, os_hat=os_mid)

        # Should be close to reference (may differ slightly due to clipping)
        u_unclipped = controller.find_u(x_hat=x_random, sens=np.array([]),
                                         est_sat=sat, os_hat=os_mid, clip=False)
        np.testing.assert_array_almost_equal(u_unclipped, u_ref)

    def test_exact_still_clips(self):
        """Exact controller should still clip if enabled."""
        sat = create_test_satellite_rw_only()
        controller, traj, os0, x0 = setup_controller_and_trajectory(Plan_and_Track_Exact, sat)

        u = controller.find_u(x_hat=x0, sens=np.array([]), est_sat=sat, os_hat=os0)

        for i, act in enumerate(sat.actuators):
            assert abs(u[i]) <= act.u_max + 1e-10


# ==========================================
# INTEGRATION TESTS
# ==========================================

@pytest.mark.slow
class TestTrajectoryTracking:
    """Integration tests for trajectory tracking behavior."""

    def test_lqr_reduces_tracking_error(self):
        """LQR controller should reduce tracking error over time."""
        sat = create_test_satellite_rw_only()
        controller, traj, os0, x0 = setup_controller_and_trajectory(
            Plan_and_Track_LQR, sat, duration=30.0
        )

        # Start with error from reference
        x = x0.copy()
        x[0:3] = np.array([0.02, -0.01, 0.015])  # Initial velocity error

        errors = []
        dt = 1.0

        for i in range(10):
            t = traj.start_time + i * dt * TimeConstants.sec2cent
            if not traj.is_valid_time(t):
                break

            os_t = os0.copy()
            os_t.J2000 = t

            x_ref = traj.get_state_at(t)
            error = np.linalg.norm(x[0:3] - x_ref[0:3])
            errors.append(error)

            # Get control
            u = controller.find_u(x_hat=x, sens=np.array([]), est_sat=sat, os_hat=os_t)

            # Simple Euler integration for test (not accurate but shows trend)
            x[0:3] += dt * (np.linalg.inv(sat.J_0) @ (sat.actuators[0].axis * u[0] +
                                                       sat.actuators[1].axis * u[1] +
                                                       sat.actuators[2].axis * u[2]))

        # Error should generally decrease (may not be monotonic due to simple integration)
        assert errors[-1] < errors[0], \
            f"Error should decrease: initial={errors[0]:.4f}, final={errors[-1]:.4f}"

    def test_lqr_and_disturbed_have_different_trajectories(self):
        """LQR and LQR_Disturbed should produce trajectories with different gain structures."""
        # Use single-RW satellite for KwDist compatibility
        sat = create_test_satellite_single_rw()
        x0 = create_initial_state(n_rw=1)
        _, os0 = create_test_orbit(100)

        planner_settings = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=10.0, dt_tvlqr=1.0)

        # Setup LQR controller
        controller_lqr = Plan_and_Track_LQR(est_sat=sat, planner_settings=planner_settings)
        goal = ECI_Goal(np.array([1, 0, 0]))
        goals = GoalList({os0.J2000: goal})
        traj_lqr = controller_lqr.calculate_trajectory(
            t_start=os0.J2000, duration=30.0, x_0=x0, os_0=os0, goals=goals, verbose=False
        )

        # Setup LQR_Disturbed controller
        controller_dist = Plan_and_Track_LQR_Disturbed(
            est_sat=sat, planner_settings=planner_settings
        )
        traj_dist = controller_dist.calculate_trajectory(
            t_start=os0.J2000, duration=30.0, x_0=x0, os_0=os0, goals=goals, verbose=False
        )

        # LQR_Disturbed trajectory should have disturbance estimation enabled
        assert traj_lqr.use_disturbance_estimation == False
        assert traj_dist.use_disturbance_estimation == True

        # Gain matrices should have different shapes (KwDist has 3 extra columns)
        t_mid = traj_lqr.start_time + (traj_lqr.end_time - traj_lqr.start_time) / 2
        K_lqr = traj_lqr.get_gain_at(t_mid)
        K_dist = traj_dist.get_gain_at(t_mid)

        # LQR_Disturbed gains should have 3 extra columns for disturbance state
        assert K_dist.shape[1] == K_lqr.shape[1] + 3, \
            f"KwDist gain should have 3 extra columns: LQR={K_lqr.shape}, Dist={K_dist.shape}"


# ==========================================
# MANUAL RUN
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("PLAN AND TRACK CONTROLLER TESTS")
    print("=" * 60)

    # Run pytest with verbose output
    pytest.main([__file__, "-v", "--tb=short", "-x"])
