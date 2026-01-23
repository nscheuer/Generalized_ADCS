"""
Comprehensive tests for the Thruster actuator class.

Tests cover:
- Basic setup and parameter validation
- Torque computation (clean and with bias/noise)
- Force computation
- Mass flow and propellant tracking
- Jacobians and Hessians (numerical verification)
- Edge cases (zero moment arm, bidirectional, etc.)
- Integration with Satellite class
"""

import sys
import os
import numpy as np
import numdifftools as nd
import pytest
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.actuators import Actuator, RW, MTQ, Thruster, Noise, Bias
from ADCS.satellite_hardware.disturbances.disturbance_mode import DisturbanceMode
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.helpers.math_helpers import normalize, rot_mat, random_n_unit_vec
from ADCS.helpers.math_constants import MathConstants


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def simple_thruster():
    """Create a simple cold gas thruster for testing."""
    return Thruster(
        thrust_direction=np.array([1, 0, 0]),
        position=np.array([0.05, 0.05, 0.0]),
        max_thrust=0.1,
        isp=65,
        min_on_time=0.01
    )


@pytest.fixture
def biprop_thruster():
    """Create a bipropellant thruster."""
    return Thruster(
        thrust_direction=np.array([0, 1, 0]),
        position=np.array([1.0, 0, 0]),
        max_thrust=22.0,
        isp=290,
        bidirectional=True
    )


@pytest.fixture
def thruster_with_noise():
    """Create thruster with bias and noise models."""
    return Thruster(
        thrust_direction=np.array([0, 0, 1]),
        position=np.array([0.1, 0.1, 0]),
        max_thrust=0.5,
        isp=220,
        bias=Bias(bias=0.01, std_bias=0.001),
        noise=Noise(noise=0.0, std_noise=0.02)
    )


@pytest.fixture
def orbital_state():
    """Create a standard orbital state for testing."""
    ephem = Ephemeris()
    return Orbital_State(
        ephem=ephem,
        J2000=0.22,
        R=np.array([7000, 0, 0]),
        V=np.array([0, 7.5, 0]),
        B=np.array([2e-5, 1e-5, 3e-5])
    )


@pytest.fixture
def spacecraft_state():
    """Create a standard spacecraft state vector."""
    w0 = np.array([0.01, -0.02, 0.005])
    q0 = normalize(np.array([0.1, 0.2, 0.3, 0.9]))
    return np.hstack((w0, q0))


# =============================================================================
# BASIC SETUP TESTS
# =============================================================================

class TestThrusterSetup:
    """Tests for thruster initialization and parameter handling."""

    def test_basic_initialization(self):
        """Test basic thruster creation."""
        t = Thruster(
            thrust_direction=np.array([1, 0, 0]),
            position=np.array([0.1, 0.2, 0.3]),
            max_thrust=10.0,
            isp=200
        )
        
        assert np.allclose(t.thrust_direction, np.array([1, 0, 0]))
        assert np.allclose(t.position, np.array([0.1, 0.2, 0.3]))
        assert t.max_thrust == 10.0
        assert t.isp == 200
        assert t.min_on_time == 0.0
        assert t.bidirectional is False

    def test_direction_normalization(self):
        """Test that thrust direction is normalized."""
        t = Thruster(
            thrust_direction=np.array([3, 4, 0]),  # Not unit
            position=np.array([0.1, 0, 0]),
            max_thrust=1.0,
            isp=100
        )
        
        assert np.isclose(np.linalg.norm(t.thrust_direction), 1.0)
        assert np.allclose(t.thrust_direction, np.array([0.6, 0.8, 0]))

    def test_effective_torque_axis(self):
        """Test effective torque axis computation."""
        # Thrust in +X, position at (0, 0.1, 0) -> torque about -Z
        t = Thruster(
            thrust_direction=np.array([1, 0, 0]),
            position=np.array([0, 0.1, 0]),
            max_thrust=10.0,
            isp=100
        )
        
        # τ = r × F = [0, 0.1, 0] × [10, 0, 0] = [0, 0, -1]
        expected = np.array([0, 0, -1.0])
        assert np.allclose(t.effective_torque_axis, expected)

    def test_zero_moment_arm(self):
        """Test thruster with thrust through CoM (no torque)."""
        t = Thruster(
            thrust_direction=np.array([1, 0, 0]),
            position=np.array([0.1, 0, 0]),  # Along thrust direction
            max_thrust=10.0,
            isp=100
        )
        
        # r × n = [0.1, 0, 0] × [1, 0, 0] = [0, 0, 0]
        assert np.allclose(t.effective_torque_axis, np.zeros(3), atol=1e-12)

    def test_bidirectional_flag(self):
        """Test bidirectional thruster flag."""
        t_uni = Thruster(
            thrust_direction=np.array([1, 0, 0]),
            position=np.array([0, 0.1, 0]),
            max_thrust=1.0,
            isp=100,
            bidirectional=False
        )
        
        t_bi = Thruster(
            thrust_direction=np.array([1, 0, 0]),
            position=np.array([0, 0.1, 0]),
            max_thrust=1.0,
            isp=100,
            bidirectional=True
        )
        
        assert t_uni.bidirectional is False
        assert t_bi.bidirectional is True

    def test_u_max_is_normalized(self):
        """Test that u_max is always 1.0 for normalized command."""
        t = Thruster(
            thrust_direction=np.array([1, 0, 0]),
            position=np.array([0, 0.1, 0]),
            max_thrust=100.0,  # High thrust
            isp=100
        )
        
        assert t.u_max == 1.0


# =============================================================================
# TORQUE COMPUTATION TESTS
# =============================================================================

class TestThrusterTorque:
    """Tests for torque computation."""

    def test_torque_clean(self, simple_thruster, orbital_state, spacecraft_state):
        """Test torque computation without bias/noise."""
        dmode = DisturbanceMode(add_bias=False, add_noise=False, 
                                update_bias=False, update_noise=False)
        
        # Full thrust
        tau = simple_thruster.torque(u=1.0, x=spacecraft_state, os=orbital_state, dmode=dmode)
        expected = simple_thruster.effective_torque_axis * 1.0
        assert np.allclose(tau, expected)
        
        # Half thrust
        tau_half = simple_thruster.torque(u=0.5, x=spacecraft_state, os=orbital_state, dmode=dmode)
        assert np.allclose(tau_half, expected * 0.5)
        
        # Zero thrust
        tau_zero = simple_thruster.torque(u=0.0, x=spacecraft_state, os=orbital_state, dmode=dmode)
        assert np.allclose(tau_zero, np.zeros(3))

    def test_torque_linearity(self, simple_thruster, orbital_state, spacecraft_state):
        """Test that torque is linear in command."""
        dmode = DisturbanceMode(add_bias=False, add_noise=False,
                                update_bias=False, update_noise=False)
        
        u_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        torques = [simple_thruster.torque(u=u, x=spacecraft_state, os=orbital_state, dmode=dmode) 
                   for u in u_values]
        
        # Check linearity: τ(u) = τ(1) * u
        tau_full = torques[-1]
        for i, u in enumerate(u_values):
            assert np.allclose(torques[i], tau_full * u, atol=1e-12)

    def test_torque_bidirectional(self, biprop_thruster, orbital_state, spacecraft_state):
        """Test bidirectional thruster with negative commands."""
        dmode = DisturbanceMode(add_bias=False, add_noise=False,
                                update_bias=False, update_noise=False)
        
        tau_pos = biprop_thruster.torque(u=1.0, x=spacecraft_state, os=orbital_state, dmode=dmode)
        tau_neg = biprop_thruster.torque(u=-1.0, x=spacecraft_state, os=orbital_state, dmode=dmode)
        
        # Negative command should give opposite torque
        assert np.allclose(tau_neg, -tau_pos)

    def test_torque_unidirectional_clamp(self, simple_thruster, orbital_state, spacecraft_state):
        """Test that unidirectional thruster clamps negative commands."""
        dmode = DisturbanceMode(add_bias=False, add_noise=False,
                                update_bias=False, update_noise=False)
        
        with pytest.warns(UserWarning, match="negative command"):
            tau = simple_thruster.torque(u=-0.5, x=spacecraft_state, os=orbital_state, dmode=dmode)
        
        # Should be clamped to zero
        assert np.allclose(tau, np.zeros(3))

    def test_torque_exceeds_limit_warning(self, simple_thruster, orbital_state, spacecraft_state):
        """Test warning when command exceeds 1.0."""
        dmode = DisturbanceMode(add_bias=False, add_noise=False,
                                update_bias=False, update_noise=False)
        
        with pytest.warns(UserWarning, match="exceeds normalized limit"):
            simple_thruster.torque(u=1.5, x=spacecraft_state, os=orbital_state, dmode=dmode)

    def test_torque_state_independence(self, simple_thruster, orbital_state):
        """Test that torque doesn't depend on spacecraft state."""
        dmode = DisturbanceMode(add_bias=False, add_noise=False,
                                update_bias=False, update_noise=False)
        
        # Different states
        x1 = np.hstack((np.array([0.01, -0.02, 0.005]), normalize(np.array([0.1, 0.2, 0.3, 0.9]))))
        x2 = np.hstack((np.array([0.1, 0.1, 0.1]), normalize(np.array([0.5, 0.5, 0.5, 0.5]))))
        
        tau1 = simple_thruster.torque(u=0.7, x=x1, os=orbital_state, dmode=dmode)
        tau2 = simple_thruster.torque(u=0.7, x=x2, os=orbital_state, dmode=dmode)
        
        assert np.allclose(tau1, tau2)

    def test_torque_with_bias(self, thruster_with_noise, orbital_state, spacecraft_state):
        """Test torque computation with bias."""
        dmode_bias = DisturbanceMode(add_bias=True, add_noise=False,
                                     update_bias=False, update_noise=False)
        dmode_clean = DisturbanceMode(add_bias=False, add_noise=False,
                                      update_bias=False, update_noise=False)
        
        tau_bias = thruster_with_noise.torque(u=0.5, x=spacecraft_state, os=orbital_state, dmode=dmode_bias)
        tau_clean = thruster_with_noise.torque(u=0.5, x=spacecraft_state, os=orbital_state, dmode=dmode_clean)
        
        # With positive bias, should have slightly more torque
        bias_value = thruster_with_noise.bias.get_bias(orbital_state.J2000)
        expected_diff = thruster_with_noise.effective_torque_axis * bias_value
        
        assert np.allclose(tau_bias - tau_clean, expected_diff, atol=1e-10)


# =============================================================================
# FORCE COMPUTATION TESTS
# =============================================================================

class TestThrusterForce:
    """Tests for force computation."""

    def test_force_basic(self, simple_thruster, orbital_state, spacecraft_state):
        """Test basic force computation."""
        dmode = DisturbanceMode(add_bias=False, add_noise=False,
                                update_bias=False, update_noise=False)
        
        force = simple_thruster.force(u=1.0, x=spacecraft_state, os=orbital_state, dmode=dmode)
        
        expected = simple_thruster.thrust_direction * simple_thruster.max_thrust
        assert np.allclose(force, expected)

    def test_force_direction(self, biprop_thruster, orbital_state, spacecraft_state):
        """Test that force is along thrust direction."""
        dmode = DisturbanceMode(add_bias=False, add_noise=False,
                                update_bias=False, update_noise=False)
        
        force = biprop_thruster.force(u=0.5, x=spacecraft_state, os=orbital_state, dmode=dmode)
        
        # Force should be parallel to thrust direction
        force_dir = force / np.linalg.norm(force)
        assert np.allclose(np.abs(np.dot(force_dir, biprop_thruster.thrust_direction)), 1.0)

    def test_force_magnitude(self, simple_thruster, orbital_state, spacecraft_state):
        """Test force magnitude scales with command."""
        dmode = DisturbanceMode(add_bias=False, add_noise=False,
                                update_bias=False, update_noise=False)
        
        for u in [0.2, 0.5, 0.8, 1.0]:
            force = simple_thruster.force(u=u, x=spacecraft_state, os=orbital_state, dmode=dmode)
            assert np.isclose(np.linalg.norm(force), simple_thruster.max_thrust * u)


# =============================================================================
# PROPELLANT TRACKING TESTS
# =============================================================================

class TestPropellantTracking:
    """Tests for propellant consumption tracking."""

    def test_mass_flow_rate(self, simple_thruster):
        """Test mass flow rate calculation."""
        # mdot = F / (Isp * g0)
        expected = simple_thruster.max_thrust / (simple_thruster.isp * 9.80665)
        actual = simple_thruster.mass_flow_rate(1.0)
        
        assert np.isclose(actual, expected)

    def test_mass_flow_scales_with_thrust(self, simple_thruster):
        """Test mass flow scales linearly with command."""
        mdot_full = simple_thruster.mass_flow_rate(1.0)
        mdot_half = simple_thruster.mass_flow_rate(0.5)
        
        assert np.isclose(mdot_half, mdot_full * 0.5)

    def test_propellant_tracking_accumulation(self, simple_thruster):
        """Test propellant usage accumulation."""
        simple_thruster.reset_propellant_tracking()
        
        dt = 1.0  # 1 second
        mass1 = simple_thruster.update_propellant_usage(u=1.0, dt=dt)
        mass2 = simple_thruster.update_propellant_usage(u=0.5, dt=dt)
        
        total_mass = mass1 + mass2
        assert np.isclose(simple_thruster.total_mass_expended, total_mass)
        
        expected_impulse = simple_thruster.max_thrust * (1.0 + 0.5) * dt
        assert np.isclose(simple_thruster.total_impulse, expected_impulse)

    def test_minimum_impulse_bit(self, simple_thruster):
        """Test minimum impulse bit calculation."""
        mib = simple_thruster.minimum_impulse_bit()
        expected = simple_thruster.max_thrust * simple_thruster.min_on_time
        
        assert np.isclose(mib, expected)

    def test_reset_tracking(self, simple_thruster):
        """Test propellant tracking reset."""
        simple_thruster.update_propellant_usage(u=1.0, dt=10.0)
        assert simple_thruster.total_mass_expended > 0
        
        simple_thruster.reset_propellant_tracking()
        
        assert simple_thruster.total_mass_expended == 0.0
        assert simple_thruster.total_impulse == 0.0


# =============================================================================
# JACOBIAN TESTS (NUMERICAL VERIFICATION)
# =============================================================================

class TestThrusterJacobians:
    """Tests for first derivatives using numerical differentiation."""

    def test_dtorq_du_numerical(self, simple_thruster, orbital_state, spacecraft_state):
        """Verify dτ/du against numerical differentiation."""
        dmode = DisturbanceMode(add_bias=False, add_noise=False,
                                update_bias=False, update_noise=False)
        
        u0 = 0.5
        
        # Numerical Jacobian with smaller step size to stay in valid range [0, 1]
        # numdifftools explores approximately u0 ± 2*step_nom, so use step_nom=0.1
        def torque_func(u):
            return simple_thruster.torque(u=u, x=spacecraft_state, os=orbital_state, dmode=dmode)
        
        J_num = nd.Jacobian(torque_func, step=0.01)(u0).T
        J_ana = simple_thruster.dtorq__du(u=u0, x=spacecraft_state, os=orbital_state)
        
        assert np.allclose(J_num, J_ana, atol=1e-6)  # Slightly relaxed tolerance for small step

    def test_dtorq_dbasestate_numerical(self, simple_thruster, orbital_state, spacecraft_state):
        """Verify dτ/dx against numerical differentiation."""
        dmode = DisturbanceMode(add_bias=False, add_noise=False,
                                update_bias=False, update_noise=False)
        
        u0 = 0.5
        
        def torque_func(x):
            return simple_thruster.torque(u=u0, x=x, os=orbital_state, dmode=dmode)
        
        J_num = nd.Jacobian(torque_func)(spacecraft_state).T
        J_ana = simple_thruster.dtorq__dbasestate(u=u0, x=spacecraft_state, os=orbital_state)
        
        # Should be zero for thruster
        assert np.allclose(J_num, J_ana, atol=1e-8)
        assert np.allclose(J_ana, np.zeros((7, 3)))

    def test_dtorq_du_equals_effective_axis(self, simple_thruster, orbital_state, spacecraft_state):
        """Test that dτ/du equals the effective torque axis."""
        J = simple_thruster.dtorq__du(u=0.5, x=spacecraft_state, os=orbital_state)
        
        assert J.shape == (1, 3)
        assert np.allclose(J.flatten(), simple_thruster.effective_torque_axis)


# =============================================================================
# HESSIAN TESTS
# =============================================================================

class TestThrusterHessians:
    """Tests for second derivatives."""

    def test_hessians_are_zero(self, simple_thruster, orbital_state, spacecraft_state):
        """Test that all Hessians are zero (linear torque model)."""
        u0 = 0.5
        
        H_dudu = simple_thruster.ddtorq__dudu(u=u0, x=spacecraft_state, os=orbital_state)
        H_dudx = simple_thruster.ddtorq__dudbasestate(u=u0, x=spacecraft_state, os=orbital_state)
        H_dxdx = simple_thruster.ddtorq__dbasestatedbasestate(u=u0, x=spacecraft_state, os=orbital_state)
        
        assert np.allclose(H_dudu, 0)
        assert np.allclose(H_dudx, 0)
        assert np.allclose(H_dxdx, 0)

    def test_hessian_shapes(self, simple_thruster, orbital_state, spacecraft_state):
        """Test Hessian shapes are correct."""
        u0 = 0.5
        
        assert simple_thruster.ddtorq__dudu(u=u0, x=spacecraft_state, os=orbital_state).shape == (1, 1, 3)
        assert simple_thruster.ddtorq__dudbasestate(u=u0, x=spacecraft_state, os=orbital_state).shape == (1, 7, 3)
        assert simple_thruster.ddtorq__dbasestatedbasestate(u=u0, x=spacecraft_state, os=orbital_state).shape == (7, 7, 3)
        assert simple_thruster.ddtorq__dudh(u=u0, x=spacecraft_state, os=orbital_state).shape == (1, 0, 3)
        assert simple_thruster.ddtorq__dhdh(u=u0, x=spacecraft_state, os=orbital_state).shape == (0, 0, 3)


# =============================================================================
# STORAGE TORQUE TESTS
# =============================================================================

class TestStorageTorque:
    """Tests for momentum storage (should be empty for thrusters)."""

    def test_storage_torque_empty(self, simple_thruster, orbital_state, spacecraft_state):
        """Test that thrusters have no storage torque."""
        st = simple_thruster.storage_torque(u=1.0, x=spacecraft_state, os=orbital_state)
        
        assert st.shape == (0,)


# =============================================================================
# SATELLITE INTEGRATION TESTS
# =============================================================================

class TestThrusterSatelliteIntegration:
    """Tests for thruster integration with Satellite class."""

    def test_satellite_with_thrusters(self, orbital_state):
        """Test creating a satellite with thrusters."""
        thrusters = [
            Thruster(
                thrust_direction=np.array([0, 1, 0]),
                position=np.array([0.1, 0, 0]),
                max_thrust=0.1,
                isp=65
            ),
            Thruster(
                thrust_direction=np.array([0, -1, 0]),
                position=np.array([-0.1, 0, 0]),
                max_thrust=0.1,
                isp=65
            )
        ]
        
        sat = Satellite(
            mass=4.0,
            J_0=np.diagflat([0.05, 0.05, 0.03]),
            actuators=thrusters
        )
        
        assert len(sat.actuators) == 2
        assert all(isinstance(a, Thruster) for a in sat.actuators)

    def test_mixed_actuator_satellite(self, orbital_state):
        """Test satellite with mixed actuator types."""
        actuators = [
            RW(axis=np.array([1, 0, 0]), max_torque=0.01, J=0.001, h=0, h_max=0.05),
            MTQ(axis=np.array([0, 1, 0]), max_torque=0.5),
            Thruster(
                thrust_direction=np.array([0, 0, 1]),
                position=np.array([0.1, 0.1, 0]),
                max_thrust=0.5,
                isp=220
            )
        ]
        
        sat = Satellite(
            mass=4.0,
            J_0=np.diagflat([0.1, 0.1, 0.1]),
            actuators=actuators
        )
        
        assert len(sat.actuators) == 3
        assert isinstance(sat.actuators[0], RW)
        assert isinstance(sat.actuators[1], MTQ)
        assert isinstance(sat.actuators[2], Thruster)


# =============================================================================
# EDGE CASES AND SPECIAL CONFIGURATIONS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and special configurations."""

    def test_thruster_at_origin(self, orbital_state, spacecraft_state):
        """Test thruster positioned at CoM (zero torque)."""
        t = Thruster(
            thrust_direction=np.array([1, 0, 0]),
            position=np.array([0, 0, 0]),
            max_thrust=10.0,
            isp=100
        )
        
        dmode = DisturbanceMode(add_bias=False, add_noise=False,
                                update_bias=False, update_noise=False)
        tau = t.torque(u=1.0, x=spacecraft_state, os=orbital_state, dmode=dmode)
        
        assert np.allclose(tau, np.zeros(3))

    def test_very_small_thrust(self, orbital_state, spacecraft_state):
        """Test with micronewton thruster."""
        t = Thruster(
            thrust_direction=np.array([1, 0, 0]),
            position=np.array([0, 0.1, 0]),
            max_thrust=1e-6,  # 1 μN
            isp=3000  # High Isp ion thruster
        )
        
        dmode = DisturbanceMode(add_bias=False, add_noise=False,
                                update_bias=False, update_noise=False)
        tau = t.torque(u=1.0, x=spacecraft_state, os=orbital_state, dmode=dmode)
        
        # Should still compute correctly
        expected = np.array([0, 0, -1e-7])  # 0.1 * 1e-6 N·m
        assert np.allclose(tau, expected, atol=1e-15)

    def test_diagonal_thrust_direction(self, orbital_state, spacecraft_state):
        """Test thruster with diagonal thrust direction."""
        t = Thruster(
            thrust_direction=np.array([1, 1, 1]),  # Will be normalized
            position=np.array([0.1, 0, 0]),
            max_thrust=1.0,
            isp=100
        )
        
        # Verify normalization
        assert np.isclose(np.linalg.norm(t.thrust_direction), 1.0)
        
        # Check effective torque axis
        expected_dir = normalize(np.array([1, 1, 1]))
        expected_axis = np.cross(np.array([0.1, 0, 0]), expected_dir) * 1.0
        assert np.allclose(t.effective_torque_axis, expected_axis)

    def test_repr(self, simple_thruster):
        """Test string representation."""
        repr_str = repr(simple_thruster)
        
        assert "Thruster" in repr_str
        assert "F_max=0.1" in repr_str
        assert "Isp=65" in repr_str


# =============================================================================
# REALISTIC SCENARIO TESTS
# =============================================================================

class TestRealisticScenarios:
    """Tests with realistic thruster configurations."""

    def test_cubesat_cold_gas_system(self, orbital_state, spacecraft_state):
        """Test a realistic CubeSat cold gas RCS configuration."""
        # 8 thrusters in a typical CubeSat configuration
        # Positioned at corners, firing tangentially for pure torque
        
        positions = [
            np.array([0.05, 0.05, 0.05]),
            np.array([0.05, 0.05, -0.05]),
            np.array([0.05, -0.05, 0.05]),
            np.array([0.05, -0.05, -0.05]),
            np.array([-0.05, 0.05, 0.05]),
            np.array([-0.05, 0.05, -0.05]),
            np.array([-0.05, -0.05, 0.05]),
            np.array([-0.05, -0.05, -0.05]),
        ]
        
        # Thrusters firing tangentially (perpendicular to position)
        # This is more realistic for attitude control
        thrusters = []
        for pos in positions:
            # Create tangent direction by rotating the position vector
            direction = normalize(np.cross(pos, np.array([0, 0, 1])))
            if np.linalg.norm(direction) < 0.1:
                # Fallback for positions along Z axis
                direction = normalize(np.cross(pos, np.array([1, 0, 0])))
            
            t = Thruster(
                thrust_direction=direction,
                position=pos,
                max_thrust=0.01,  # 10 mN cold gas
                isp=65,
                min_on_time=0.005
            )
            thrusters.append(t)
        
        assert len(thrusters) == 8
        
        # All should have non-zero effective torque axes (tangent firing)
        for t in thrusters:
            assert np.linalg.norm(t.effective_torque_axis) > 0

    def test_geosat_biprop_system(self, orbital_state, spacecraft_state):
        """Test a realistic GEO satellite bipropellant thruster pair."""
        # Roll control thrusters
        t_pos = Thruster(
            thrust_direction=np.array([0, 1, 0]),
            position=np.array([2.0, 0, 0]),
            max_thrust=22.0,  # Typical 22N thruster
            isp=290,
            bidirectional=False
        )
        
        t_neg = Thruster(
            thrust_direction=np.array([0, -1, 0]),
            position=np.array([-2.0, 0, 0]),
            max_thrust=22.0,
            isp=290,
            bidirectional=False
        )
        
        dmode = DisturbanceMode(add_bias=False, add_noise=False,
                                update_bias=False, update_noise=False)
        
        # Firing both should create pure roll torque
        tau_pos = t_pos.torque(u=1.0, x=spacecraft_state, os=orbital_state, dmode=dmode)
        tau_neg = t_neg.torque(u=1.0, x=spacecraft_state, os=orbital_state, dmode=dmode)
        
        # Combined torque should be about Z axis
        tau_combined = tau_pos + tau_neg
        assert np.isclose(tau_combined[0], 0, atol=1e-10)
        assert np.isclose(tau_combined[1], 0, atol=1e-10)
        # Z component: 2 * 2.0m * 22N = 88 N·m
        assert np.isclose(abs(tau_combined[2]), 88.0, atol=1e-10)

    def test_propellant_budget_simulation(self):
        """Simulate propellant consumption over a mission scenario."""
        t = Thruster(
            thrust_direction=np.array([1, 0, 0]),
            position=np.array([0, 0.5, 0]),
            max_thrust=5.0,
            isp=220  # Hydrazine
        )
        
        t.reset_propellant_tracking()
        
        # Simulate 100 pulses of 0.1s at 50% thrust
        n_pulses = 100
        pulse_duration = 0.1  # seconds
        thrust_level = 0.5
        
        for _ in range(n_pulses):
            t.update_propellant_usage(u=thrust_level, dt=pulse_duration)
        
        # Expected impulse: 100 * 0.1s * 5N * 0.5 = 25 N·s
        assert np.isclose(t.total_impulse, 25.0)
        
        # Expected mass: I / (Isp * g0) = 25 / (220 * 9.80665) ≈ 0.0116 kg
        expected_mass = 25.0 / (220 * 9.80665)
        assert np.isclose(t.total_mass_expended, expected_mass, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
