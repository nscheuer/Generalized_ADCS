"""
TODO-DATA-1: Thruster Actuator Tests
====================================

Paper: Generalized Control Paper (Generalized_ACS_MASTER)
TODO ID: TODO-DATA-1
Description: Generate LTI/LTV rank tables for additional configurations (CMG, thruster)

This module tests the thruster actuator model, including:
- Minimum impulse bit (MIB) quantization behavior
- Propellant consumption tracking
- Integration with satellite dynamics
- Controller compatibility (experimental)

IMPORTANT: Thruster integration with control allocation is EXPERIMENTAL.
These tests validate the physics model, not closed-loop control.

Adjustable Parameters
---------------------
- MIB parameters (min_on_time, control_dt)
- Propellant parameters (Isp, max_thrust)
- Test tolerances
"""

import sys
import os
import numpy as np
import pytest
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ADCS.satellite_hardware.actuators import Thruster, MIBBehavior, reset_thruster_warnings
from ADCS.satellite_hardware.actuators import Bias, Noise
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.disturbances.disturbance_mode import DisturbanceMode
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.helpers.math_helpers import normalize
from ADCS.helpers.math_constants import MathConstants


# =============================================================================
# ADJUSTABLE PARAMETERS
# =============================================================================

# Thruster parameters
COLD_GAS_ISP = 65             # Cold gas specific impulse [s]
HYDRAZINE_ISP = 220           # Monopropellant Isp [s]
BIPROP_ISP = 290              # Bipropellant Isp [s]

CUBESAT_THRUST = 0.01         # CubeSat cold gas [N]
SMALLSAT_THRUST = 1.0         # Small sat thruster [N]
LARGE_SAT_THRUST = 22.0       # GEO sat thruster [N]

# MIB parameters
TYPICAL_MIN_ON_TIME = 0.01    # Typical min on-time [s]
CONTROL_DT = 0.1              # Control loop period [s]

# Test tolerances
TORQUE_TOLERANCE = 1e-10
MASS_FLOW_TOLERANCE = 1e-8


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def reset_warnings():
    """Reset thruster warnings before each test."""
    reset_thruster_warnings()
    yield


@pytest.fixture
def orbital_state():
    """Standard orbital state for testing."""
    ephem = Ephemeris()
    return Orbital_State(
        ephem=ephem, J2000=0.22,
        R=np.array([7000, 0, 0]),
        V=np.array([0, 7.5, 0]),
        B=np.array([2e-5, 1e-5, 3e-5])
    )


@pytest.fixture
def spacecraft_state():
    """Standard spacecraft state."""
    return np.hstack((np.zeros(3), normalize(np.array([0, 0, 0, 1]))))


@pytest.fixture
def cold_gas_thruster():
    """Typical CubeSat cold gas thruster."""
    return Thruster(
        thrust_direction=np.array([1, 0, 0]),
        position=np.array([0.05, 0.05, 0]),
        max_thrust=CUBESAT_THRUST,
        isp=COLD_GAS_ISP,
        min_on_time=TYPICAL_MIN_ON_TIME,
        mib_behavior=MIBBehavior.QUANTIZE_TO_ZERO,
        control_dt=CONTROL_DT
    )


@pytest.fixture
def hydrazine_thruster():
    """Monopropellant hydrazine thruster."""
    return Thruster(
        thrust_direction=np.array([0, 1, 0]),
        position=np.array([0.5, 0, 0]),
        max_thrust=SMALLSAT_THRUST,
        isp=HYDRAZINE_ISP,
        min_on_time=0.02,
        mib_behavior=MIBBehavior.QUANTIZE_TO_MIB,
        control_dt=CONTROL_DT
    )


# =============================================================================
# TODO-DATA-1: BASIC THRUSTER MODEL TESTS
# =============================================================================

class TestThrusterBasicModel:
    """
    TODO-DATA-1 Part 1: Basic thruster physics validation.
    """

    def test_torque_direction(self, cold_gas_thruster, orbital_state, spacecraft_state):
        """Torque should be perpendicular to position and thrust direction."""
        dmode = DisturbanceMode(add_bias=False, add_noise=False, 
                                update_bias=False, update_noise=False)
        
        tau = cold_gas_thruster.torque(u=1.0, x=spacecraft_state, os=orbital_state, dmode=dmode)
        
        # τ = r × F, so τ ⊥ r and τ ⊥ F
        r = cold_gas_thruster.position
        F_dir = cold_gas_thruster.thrust_direction
        
        assert abs(np.dot(tau, r)) < TORQUE_TOLERANCE, "Torque should be ⊥ to position"
        assert abs(np.dot(tau, F_dir)) < TORQUE_TOLERANCE, "Torque should be ⊥ to thrust"
        
        print(f"\n[TODO-DATA-1] Torque direction verified: τ={tau}")

    def test_torque_magnitude(self, cold_gas_thruster, orbital_state, spacecraft_state):
        """Torque magnitude should equal |r × F|."""
        dmode = DisturbanceMode(add_bias=False, add_noise=False,
                                update_bias=False, update_noise=False)
        
        tau = cold_gas_thruster.torque(u=1.0, x=spacecraft_state, os=orbital_state, dmode=dmode)
        
        r = cold_gas_thruster.position
        F = cold_gas_thruster.thrust_direction * cold_gas_thruster.max_thrust
        expected_tau = np.cross(r, F)
        
        assert np.allclose(tau, expected_tau, atol=TORQUE_TOLERANCE), \
            f"Torque {tau} != expected {expected_tau}"
        
        print(f"\n[TODO-DATA-1] Torque magnitude verified: |τ|={np.linalg.norm(tau):.6f} N·m")

    def test_mass_flow_rate(self, cold_gas_thruster):
        """Mass flow should follow ṁ = F / (Isp * g₀)."""
        mdot = cold_gas_thruster.mass_flow_rate(u=1.0)
        expected = cold_gas_thruster.max_thrust / (cold_gas_thruster.isp * 9.80665)
        
        assert np.isclose(mdot, expected, rtol=MASS_FLOW_TOLERANCE), \
            f"Mass flow {mdot} != expected {expected}"
        
        print(f"\n[TODO-DATA-1] Mass flow rate: {mdot*1000:.4f} g/s")


# =============================================================================
# TODO-DATA-1: MINIMUM IMPULSE BIT TESTS
# =============================================================================

class TestMIBQuantization:
    """
    TODO-DATA-1 Part 2: Minimum impulse bit behavior tests.
    
    Physical thrusters have minimum on-time due to valve dynamics.
    Tests validate three quantization behaviors:
    1. QUANTIZE_TO_ZERO: Small command → no fire
    2. QUANTIZE_TO_MIB: Small command → fire full MIB
    3. ACCUMULATE: Small commands accumulate until MIB
    """

    def test_mib_threshold_calculation(self, cold_gas_thruster):
        """Test MIB threshold is computed correctly."""
        # u_min = min_on_time / control_dt
        expected_u_min = TYPICAL_MIN_ON_TIME / CONTROL_DT
        assert np.isclose(cold_gas_thruster.u_min, expected_u_min), \
            f"u_min={cold_gas_thruster.u_min} != expected {expected_u_min}"
        
        print(f"\n[TODO-DATA-1] MIB threshold: u_min={cold_gas_thruster.u_min:.4f}")

    def test_quantize_to_zero_below_mib(self, orbital_state, spacecraft_state):
        """QUANTIZE_TO_ZERO: Commands below MIB produce zero torque."""
        thruster = Thruster(
            thrust_direction=np.array([1, 0, 0]),
            position=np.array([0, 0.1, 0]),
            max_thrust=0.1,
            isp=65,
            min_on_time=0.02,
            mib_behavior=MIBBehavior.QUANTIZE_TO_ZERO,
            control_dt=0.1
        )
        
        dmode = DisturbanceMode(add_bias=False, add_noise=False,
                                update_bias=False, update_noise=False)
        
        # u_min = 0.02 / 0.1 = 0.2, so u=0.1 is below threshold
        with pytest.warns(UserWarning, match="below MIB threshold"):
            tau = thruster.torque(u=0.1, x=spacecraft_state, os=orbital_state, dmode=dmode)
        
        # Should be quantized to zero
        assert np.allclose(tau, 0), f"Expected zero torque, got {tau}"
        assert thruster.firing_count == 0, "Should not have fired"
        
        print("\n[TODO-DATA-1] QUANTIZE_TO_ZERO: sub-MIB command → zero torque ✓")

    def test_quantize_to_mib_below_threshold(self, orbital_state, spacecraft_state):
        """QUANTIZE_TO_MIB: Commands below MIB fire full MIB pulse."""
        thruster = Thruster(
            thrust_direction=np.array([1, 0, 0]),
            position=np.array([0, 0.1, 0]),
            max_thrust=0.1,
            isp=65,
            min_on_time=0.02,
            mib_behavior=MIBBehavior.QUANTIZE_TO_MIB,
            control_dt=0.1
        )
        
        dmode = DisturbanceMode(add_bias=False, add_noise=False,
                                update_bias=False, update_noise=False)
        
        # u_min = 0.2, u=0.1 → fire at u=0.2
        with pytest.warns(UserWarning, match="Quantized to MIB"):
            tau = thruster.torque(u=0.1, x=spacecraft_state, os=orbital_state, dmode=dmode)
        
        # Should fire at MIB level
        expected_tau = thruster.effective_torque_axis * thruster.u_min
        assert np.allclose(tau, expected_tau, rtol=0.01), \
            f"Expected MIB torque {expected_tau}, got {tau}"
        assert thruster.firing_count == 1, "Should have fired once"
        
        print("\n[TODO-DATA-1] QUANTIZE_TO_MIB: sub-MIB command → MIB pulse ✓")

    def test_above_mib_threshold_fires_normally(self, orbital_state, spacecraft_state):
        """Commands above MIB fire at requested level."""
        thruster = Thruster(
            thrust_direction=np.array([1, 0, 0]),
            position=np.array([0, 0.1, 0]),
            max_thrust=0.1,
            isp=65,
            min_on_time=0.02,
            mib_behavior=MIBBehavior.QUANTIZE_TO_ZERO,
            control_dt=0.1
        )
        
        dmode = DisturbanceMode(add_bias=False, add_noise=False,
                                update_bias=False, update_noise=False)
        
        # u_min = 0.2, u=0.5 → fire normally
        # Should trigger integration warning but not MIB warning
        tau = thruster.torque(u=0.5, x=spacecraft_state, os=orbital_state, dmode=dmode)
        
        expected_tau = thruster.effective_torque_axis * 0.5
        assert np.allclose(tau, expected_tau), f"Expected {expected_tau}, got {tau}"
        
        print("\n[TODO-DATA-1] Above MIB: fires at requested level ✓")

    def test_accumulate_mode(self, orbital_state, spacecraft_state):
        """ACCUMULATE: Small commands sum until MIB reached."""
        thruster = Thruster(
            thrust_direction=np.array([1, 0, 0]),
            position=np.array([0, 0.1, 0]),
            max_thrust=0.1,
            isp=65,
            min_on_time=0.02,
            mib_behavior=MIBBehavior.ACCUMULATE,
            control_dt=0.1
        )
        
        dmode = DisturbanceMode(add_bias=False, add_noise=False,
                                update_bias=False, update_noise=False)
        
        # u_min = 0.2, send u=0.1 three times
        # First two: accumulate, third: fire
        
        tau1 = thruster.torque(u=0.1, x=spacecraft_state, os=orbital_state, dmode=dmode)
        assert np.allclose(tau1, 0), "First call should accumulate"
        assert thruster.accumulated_command == 0.1
        
        tau2 = thruster.torque(u=0.1, x=spacecraft_state, os=orbital_state, dmode=dmode)
        # 0.1 + 0.1 = 0.2 >= u_min, should fire
        assert not np.allclose(tau2, 0), "Should have fired after accumulation"
        assert thruster.accumulated_command == 0, "Should reset after firing"
        
        print("\n[TODO-DATA-1] ACCUMULATE mode: verified accumulation and firing ✓")


# =============================================================================
# TODO-DATA-1: PROPELLANT TRACKING TESTS
# =============================================================================

class TestPropellantTracking:
    """
    TODO-DATA-1 Part 3: Propellant consumption tracking tests.
    
    Supports mission planning and ΔV budgets.
    """

    def test_propellant_accumulation(self, cold_gas_thruster):
        """Test propellant usage accumulates correctly."""
        cold_gas_thruster.reset_propellant_tracking()
        
        # Fire for 10 seconds at 50% thrust
        dt = 1.0
        for _ in range(10):
            cold_gas_thruster.update_propellant_usage(u=0.5, dt=dt)
        
        # Expected impulse: 0.01 N * 0.5 * 10 s = 0.05 N·s
        expected_impulse = CUBESAT_THRUST * 0.5 * 10
        assert np.isclose(cold_gas_thruster.total_impulse, expected_impulse), \
            f"Impulse {cold_gas_thruster.total_impulse} != expected {expected_impulse}"
        
        # Expected mass: I / (Isp * g0)
        expected_mass = expected_impulse / (COLD_GAS_ISP * 9.80665)
        assert np.isclose(cold_gas_thruster.total_mass_expended, expected_mass, rtol=0.01), \
            f"Mass {cold_gas_thruster.total_mass_expended} != expected {expected_mass}"
        
        print(f"\n[TODO-DATA-1] Propellant tracking:")
        print(f"  Total impulse: {cold_gas_thruster.total_impulse:.4f} N·s")
        print(f"  Mass expended: {cold_gas_thruster.total_mass_expended*1000:.4f} g")

    def test_firing_count_tracking(self, cold_gas_thruster, orbital_state, spacecraft_state):
        """Test firing count is tracked."""
        cold_gas_thruster.reset_propellant_tracking()
        
        dmode = DisturbanceMode(add_bias=False, add_noise=False,
                                update_bias=False, update_noise=False)
        
        # Fire above MIB several times
        for _ in range(5):
            cold_gas_thruster.torque(u=0.5, x=spacecraft_state, os=orbital_state, dmode=dmode)
        
        assert cold_gas_thruster.firing_count == 5, \
            f"Firing count {cold_gas_thruster.firing_count} != 5"
        
        print(f"\n[TODO-DATA-1] Firing count: {cold_gas_thruster.firing_count}")

    def test_status_reporting(self, cold_gas_thruster, orbital_state, spacecraft_state):
        """Test status dict for telemetry."""
        cold_gas_thruster.reset_propellant_tracking()
        
        dmode = DisturbanceMode(add_bias=False, add_noise=False,
                                update_bias=False, update_noise=False)
        
        cold_gas_thruster.torque(u=0.5, x=spacecraft_state, os=orbital_state, dmode=dmode)
        cold_gas_thruster.update_propellant_usage(u=0.5, dt=1.0)
        
        status = cold_gas_thruster.get_status()
        
        assert 'total_impulse_Ns' in status
        assert 'total_mass_kg' in status
        assert 'firing_count' in status
        assert 'mib_behavior' in status
        
        print(f"\n[TODO-DATA-1] Thruster status: {status}")


# =============================================================================
# TODO-DATA-1: INTEGRATION WARNINGS TESTS
# =============================================================================

class TestIntegrationWarnings:
    """
    Test that appropriate warnings are issued for experimental features.
    """

    def test_integration_warning_issued_once(self, cold_gas_thruster, orbital_state, spacecraft_state):
        """Integration warning should appear once per session."""
        reset_thruster_warnings()
        
        dmode = DisturbanceMode(add_bias=False, add_noise=False,
                                update_bias=False, update_noise=False)
        
        # First call should warn
        with pytest.warns(UserWarning, match="THRUSTER INTEGRATION WARNING"):
            cold_gas_thruster.torque(u=0.5, x=spacecraft_state, os=orbital_state, dmode=dmode)
        
        # Second call should NOT warn (already shown)
        # Create new thruster to test global flag
        thruster2 = Thruster(
            thrust_direction=np.array([0, 1, 0]),
            position=np.array([0.1, 0, 0]),
            max_thrust=0.1,
            isp=65
        )
        
        # This should NOT raise warning (already shown in session)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            thruster2.torque(u=0.5, x=spacecraft_state, os=orbital_state, dmode=dmode)
            
            # Filter for integration warnings only
            integration_warnings = [x for x in w if "INTEGRATION" in str(x.message)]
            assert len(integration_warnings) == 0, \
                "Integration warning should only appear once per session"
        
        print("\n[TODO-DATA-1] Integration warning: appears once per session ✓")


# =============================================================================
# TODO-DATA-1: SATELLITE INTEGRATION TESTS
# =============================================================================

class TestSatelliteIntegration:
    """
    TODO-DATA-1 Part 4: Test thruster integration with Satellite class.
    
    NOTE: Control allocation (LP/QP) does NOT yet support thrusters.
    These tests verify the physics layer only.
    """

    def test_satellite_with_thrusters_created(self):
        """Test Satellite can be created with thrusters."""
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
            J_0=np.diagflat([0.1, 0.1, 0.1]),
            actuators=thrusters
        )
        
        assert len(sat.actuators) == 2
        
        print("\n[TODO-DATA-1] Satellite with thrusters: created ✓")

    def test_mixed_actuator_types(self):
        """Test Satellite with MTQ, RW, and Thruster."""
        from ADCS.satellite_hardware.actuators import MTQ, RW
        
        actuators = [
            MTQ(axis=np.array([1, 0, 0]), max_torque=0.5),
            RW(axis=np.array([0, 0, 1]), max_torque=0.01, J=0.001, h=0, h_max=0.05),
            Thruster(
                thrust_direction=np.array([0, 1, 0]),
                position=np.array([0.1, 0, 0]),
                max_thrust=0.1,
                isp=65
            )
        ]
        
        sat = Satellite(
            mass=4.0,
            J_0=np.diagflat([0.1, 0.1, 0.1]),
            actuators=actuators
        )
        
        assert len(sat.actuators) == 3
        
        # Count by type
        n_mtq = sum(1 for a in sat.actuators if isinstance(a, MTQ))
        n_rw = sum(1 for a in sat.actuators if isinstance(a, RW))
        n_thr = sum(1 for a in sat.actuators if isinstance(a, Thruster))
        
        assert n_mtq == 1 and n_rw == 1 and n_thr == 1
        
        print(f"\n[TODO-DATA-1] Mixed actuators: {n_mtq} MTQ, {n_rw} RW, {n_thr} Thruster ✓")


# =============================================================================
# TODO-DATA-1: ISP COMPARISON TESTS
# =============================================================================

class TestIspComparison:
    """
    TODO-DATA-1 Part 5: Compare different thruster types.
    
    Supports paper Table: Thruster Technology Comparison.
    """

    @pytest.mark.parametrize("isp,name", [
        (COLD_GAS_ISP, "Cold Gas"),
        (HYDRAZINE_ISP, "Hydrazine"),
        (BIPROP_ISP, "Bipropellant"),
    ])
    def test_mass_efficiency_by_isp(self, isp, name):
        """Higher Isp = lower mass consumption for same impulse."""
        thruster = Thruster(
            thrust_direction=np.array([1, 0, 0]),
            position=np.array([0, 0.1, 0]),
            max_thrust=1.0,  # Same thrust for comparison
            isp=isp
        )
        
        mdot = thruster.mass_flow_rate(u=1.0)
        
        # For 1 N thrust, mass flow = 1 / (Isp * 9.80665)
        expected = 1.0 / (isp * 9.80665)
        assert np.isclose(mdot, expected)
        
        print(f"\n[TODO-DATA-1] {name} (Isp={isp}s): ṁ={mdot*1000:.4f} g/s at 1N")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
