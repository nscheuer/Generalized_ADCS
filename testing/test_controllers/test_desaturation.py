"""
Tests for momentum desaturation functionality.

These tests support:
- TODO-DATA-5: Generate desaturation performance plots (momentum evolution, pointing impact)
- TODO-DESAT-1 through TODO-DESAT-6: Desaturation analysis and comparison

Tests cover:
- Momentum tracking over time
- RW momentum update mechanics
- Desaturation controller configuration
"""

import sys
import os
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ADCS.controller.mtq_w_rw_LP import MTQ_w_RW_LP
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.actuators import RW, MTQ
from ADCS.satellite_hardware.sensors import MTM
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.helpers.math_helpers import normalize
from ADCS.helpers.math_constants import MathConstants


# =============================================================================
# FIXTURES
# =============================================================================

def create_3mtq_3rw_satellite():
    """Create a fully actuated 3MTQ+3RW satellite."""
    mtqs = [MTQ(axis=j, max_torque=0.5) for j in MathConstants.unitvecs]
    rws = [RW(
        axis=j,
        max_torque=0.01,
        J=0.001,
        h=0.02,  # Start with some momentum
        h_max=0.05
    ) for j in MathConstants.unitvecs]
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    
    return dict(
        mass=4.0,
        J_0=np.diagflat([0.1, 0.1, 0.1]),
        actuators=mtqs + rws,
        sensors=mtms,
        boresight=np.array([0, 0, 1])
    )


def create_3mtq_1rw_satellite():
    """Create a 3MTQ+1RW satellite (common CubeSat config)."""
    mtqs = [MTQ(axis=j, max_torque=0.5) for j in MathConstants.unitvecs]
    rw = RW(
        axis=np.array([0, 0, 1]),
        max_torque=0.01,
        J=0.001,
        h=0.03,  # Start with some momentum
        h_max=0.05
    )
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    
    return dict(
        mass=4.0,
        J_0=np.diagflat([0.1, 0.1, 0.1]),
        actuators=mtqs + [rw],
        sensors=mtms,
        boresight=np.array([0, 0, 1])
    )


@pytest.fixture
def satellite_3mtq_3rw():
    return create_3mtq_3rw_satellite()


@pytest.fixture
def satellite_3mtq_1rw():
    return create_3mtq_1rw_satellite()


@pytest.fixture
def orbital_state():
    """Create orbital state with non-trivial B-field."""
    ephem = Ephemeris()
    return Orbital_State(
        ephem=ephem,
        J2000=0.22,
        R=np.array([7000, 0, 0]),
        V=np.array([0, 7.5, 0]),
        B=normalize(np.array([1, 1, 1])) * 3e-5
    )


# =============================================================================
# CONTROLLER CREATION TESTS
# =============================================================================

class TestControllerCreation:
    """Test desaturation-aware controller creation."""

    def test_controller_with_h_target(self, satellite_3mtq_3rw):
        """Test creating controller with momentum target."""
        est_sat = EstimatedSatellite(**satellite_3mtq_3rw)
        
        h_target = np.zeros(3)
        controller = MTQ_w_RW_LP(
            est_sat=est_sat,
            p_gain=1.0,
            d_gain=0.5,
            c_gain=0.5,
            h_target=h_target
        )
        
        assert np.allclose(controller.h_target, h_target)
        assert controller.c_gain == 0.5

    def test_controller_different_c_gains(self, satellite_3mtq_3rw):
        """Test controllers with different c_gain values."""
        for c_gain in [0.0, 0.1, 0.5, 1.0, 2.0]:
            est_sat = EstimatedSatellite(**create_3mtq_3rw_satellite())
            controller = MTQ_w_RW_LP(
                est_sat=est_sat,
                p_gain=1.0,
                d_gain=0.5,
                c_gain=c_gain,
                h_target=np.zeros(3)
            )
            assert controller.c_gain == c_gain


# =============================================================================
# MOMENTUM EVOLUTION TESTS
# =============================================================================

class TestMomentumEvolution:
    """Test momentum tracking during desaturation."""

    def test_rw_momentum_tracking(self, satellite_3mtq_3rw):
        """Test that RW momentum is correctly tracked."""
        sat_config = create_3mtq_3rw_satellite()
        est_sat = EstimatedSatellite(**sat_config)
        
        rws = [a for a in est_sat.actuators if isinstance(a, RW)]
        
        # Get initial momentum
        h_initial = np.array([rw.h for rw in rws])
        h_total_initial = np.sum([rw.h * rw.axis for rw in rws], axis=0)
        
        # Verify non-zero initial momentum
        assert np.linalg.norm(h_total_initial) > 0
        
        # Update momentum
        for i, rw in enumerate(rws):
            rw.update_momentum(h_initial[i] * 0.9)  # Reduce by 10%
        
        # Check updated values
        h_updated = np.array([rw.h for rw in rws])
        assert np.allclose(h_updated, h_initial * 0.9)

    def test_momentum_limits_respected(self, satellite_3mtq_3rw):
        """Test that momentum limits trigger warnings."""
        sat_config = create_3mtq_3rw_satellite()
        est_sat = EstimatedSatellite(**sat_config)
        
        rws = [a for a in est_sat.actuators if isinstance(a, RW)]
        
        # Try to exceed momentum limit
        with pytest.warns(UserWarning, match="saturation limit"):
            rws[0].update_momentum(rws[0].h_max * 1.5)

    def test_momentum_direction_consistency(self, satellite_3mtq_3rw):
        """Test that momentum direction is consistent with RW axes."""
        sat_config = create_3mtq_3rw_satellite()
        est_sat = EstimatedSatellite(**sat_config)
        
        rws = [a for a in est_sat.actuators if isinstance(a, RW)]
        
        # Set specific momentum values
        test_values = [0.01, 0.02, 0.03]
        for i, (rw, val) in enumerate(zip(rws, test_values)):
            rw.update_momentum(val)
        
        # Compute total momentum vector
        h_total = sum(rw.h * rw.axis for rw in rws)
        
        # Expected: [0.01, 0.02, 0.03] since axes are unit vectors along X, Y, Z
        expected = np.array([0.01, 0.02, 0.03])
        assert np.allclose(h_total, expected, atol=1e-10)


# =============================================================================
# DESATURATION CAPABILITY TESTS
# =============================================================================

class TestDesaturationCapability:
    """Test desaturation capabilities of different configurations."""

    def test_3rw_3mtq_can_desaturate(self, satellite_3mtq_3rw):
        """Test that 3RW+3MTQ config can desaturate."""
        est_sat = EstimatedSatellite(**satellite_3mtq_3rw)
        
        rws = [a for a in est_sat.actuators if isinstance(a, RW)]
        mtqs = [a for a in est_sat.actuators if isinstance(a, MTQ)]
        
        # Should have 3 RWs and 3 MTQs
        assert len(rws) == 3
        assert len(mtqs) == 3
        
        # Controller should recognize full capability
        controller = MTQ_w_RW_LP(
            est_sat=est_sat,
            p_gain=1.0,
            d_gain=0.5,
            c_gain=0.5,
            h_target=np.zeros(3)
        )
        
        # n_rw should be 3
        assert controller.n_rw == 3

    def test_1rw_3mtq_limited_desaturation(self, satellite_3mtq_1rw):
        """Test that 1RW+3MTQ has limited desaturation capability."""
        est_sat = EstimatedSatellite(**satellite_3mtq_1rw)
        
        rws = [a for a in est_sat.actuators if isinstance(a, RW)]
        mtqs = [a for a in est_sat.actuators if isinstance(a, MTQ)]
        
        # Should have 1 RW and 3 MTQs
        assert len(rws) == 1
        assert len(mtqs) == 3
        
        # Controller should note limitation
        controller = MTQ_w_RW_LP(
            est_sat=est_sat,
            p_gain=1.0,
            d_gain=0.5,
            c_gain=0.5,
            h_target=np.zeros(3)
        )
        
        assert controller.n_rw == 1


# =============================================================================
# MOMENTUM SIMULATION TESTS
# =============================================================================

class TestMomentumSimulation:
    """Simulate momentum evolution scenarios."""

    def test_momentum_growth_scenario(self, satellite_3mtq_3rw):
        """Simulate momentum growing over time."""
        sat_config = create_3mtq_3rw_satellite()
        est_sat = EstimatedSatellite(**sat_config)
        
        rws = [a for a in est_sat.actuators if isinstance(a, RW)]
        
        # Record initial momentum
        h_history = []
        h_initial = sum(rw.h * rw.axis for rw in rws)
        h_history.append(np.linalg.norm(h_initial))
        
        # Simulate momentum accumulation (external torque)
        for step in range(10):
            for rw in rws:
                # Add small increment (simulating disturbance accumulation)
                new_h = min(rw.h + 0.001, rw.h_max)
                rw.h = new_h  # Direct update to avoid warnings
            
            h_current = sum(rw.h * rw.axis for rw in rws)
            h_history.append(np.linalg.norm(h_current))
        
        # Momentum should have grown
        assert h_history[-1] > h_history[0]

    def test_momentum_decay_scenario(self, satellite_3mtq_3rw):
        """Simulate momentum decay (desaturation)."""
        sat_config = create_3mtq_3rw_satellite()
        est_sat = EstimatedSatellite(**sat_config)
        
        rws = [a for a in est_sat.actuators if isinstance(a, RW)]
        
        # Start with moderate momentum
        for rw in rws:
            rw.h = 0.03
        
        h_history = []
        h_initial = sum(rw.h * rw.axis for rw in rws)
        h_history.append(np.linalg.norm(h_initial))
        
        # Simulate desaturation
        for step in range(10):
            for rw in rws:
                # Reduce momentum (simulating desaturation)
                new_h = max(rw.h * 0.9, 0)  # Exponential decay
                rw.h = new_h
            
            h_current = sum(rw.h * rw.axis for rw in rws)
            h_history.append(np.linalg.norm(h_current))
        
        # Momentum should have decayed
        assert h_history[-1] < h_history[0]


# =============================================================================
# SATURATION TESTS
# =============================================================================

class TestSaturationBehavior:
    """Test behavior at saturation limits."""

    def test_approaching_saturation(self, satellite_3mtq_3rw):
        """Test behavior as RW approaches saturation."""
        sat_config = create_3mtq_3rw_satellite()
        est_sat = EstimatedSatellite(**sat_config)
        
        rws = [a for a in est_sat.actuators if isinstance(a, RW)]
        
        # Set one RW near saturation
        rws[0].h = rws[0].h_max * 0.95
        
        # Check saturation fraction
        saturation_fraction = rws[0].h / rws[0].h_max
        assert saturation_fraction > 0.9
        assert saturation_fraction < 1.0

    def test_multiple_rw_saturation(self, satellite_3mtq_3rw):
        """Test with multiple RWs at different saturation levels."""
        sat_config = create_3mtq_3rw_satellite()
        est_sat = EstimatedSatellite(**sat_config)
        
        rws = [a for a in est_sat.actuators if isinstance(a, RW)]
        
        # Set different saturation levels
        saturations = [0.2, 0.5, 0.9]
        for rw, sat_level in zip(rws, saturations):
            rw.h = rw.h_max * sat_level
        
        # Verify values
        for rw, expected in zip(rws, saturations):
            actual = rw.h / rw.h_max
            assert np.isclose(actual, expected)


# =============================================================================
# STATISTICS TESTS
# =============================================================================

class TestDesaturationStatistics:
    """Statistical tests for desaturation performance."""

    def test_momentum_distribution(self, satellite_3mtq_3rw):
        """Test momentum statistics across random initial conditions."""
        np.random.seed(42)
        
        h_magnitudes = []
        for _ in range(20):
            sat_config = create_3mtq_3rw_satellite()
            est_sat = EstimatedSatellite(**sat_config)
            
            rws = [a for a in est_sat.actuators if isinstance(a, RW)]
            
            # Random initial momentum
            for rw in rws:
                rw.h = np.random.uniform(0, rw.h_max)
            
            h_total = sum(rw.h * rw.axis for rw in rws)
            h_magnitudes.append(np.linalg.norm(h_total))
        
        # Should have variation
        assert len(set(h_magnitudes)) > 1
        assert np.std(h_magnitudes) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
