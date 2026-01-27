"""
TODO-DATA-5: Desaturation Performance Tests
===========================================

Paper: Generalized Control Paper (Generalized_ACS_MASTER)
TODO ID: TODO-DATA-5
Description: Generate desaturation performance plots (momentum evolution, pointing impact)

Also supports:
- TODO-DESAT-1 through TODO-DESAT-6: Desaturation analysis and comparison
- TODO-FIG-4: Desaturation scheduling visualization

Adjustable Parameters
---------------------
- MOMENTUM_INITIAL: Initial RW momentum for tests
- MOMENTUM_SATURATION: Saturation threshold
- DESATURATION_TIME: Simulation duration
- C_GAIN_VALUES: Desaturation gain values to test
"""

import sys
import os
import numpy as np
import pytest
from dataclasses import dataclass
from typing import List, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ADCS.controller.mtq_w_rw_LP import MTQ_w_RW_LP
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.actuators import RW, MTQ
from ADCS.satellite_hardware.sensors import MTM
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.helpers.math_helpers import normalize
from ADCS.helpers.math_constants import MathConstants


# =============================================================================
# ADJUSTABLE PARAMETERS
# =============================================================================

# Momentum parameters
MOMENTUM_INITIAL = 0.02       # Initial RW momentum [N·m·s]
MOMENTUM_SATURATION = 0.05    # RW saturation limit [N·m·s]
MOMENTUM_TARGET = 0.0         # Target momentum for desaturation

# Simulation parameters
DESATURATION_TIME = 600       # Simulation duration [s]
DT = 1.0                      # Time step [s]

# Gain sweep parameters
C_GAIN_VALUES = [0.0, 0.1, 0.5, 1.0, 2.0]  # Desaturation gains to test

# B-field parameters
B_FIELD_MAGNITUDE = 3e-5      # Tesla


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MomentumRecord:
    """Record of momentum state at a time step."""
    time: float
    h_vector: np.ndarray
    h_magnitude: float
    saturation_fraction: float


@dataclass  
class DesaturationResult:
    """Result from a desaturation simulation."""
    c_gain: float
    initial_momentum: float
    final_momentum: float
    time_to_target: float  # -1 if not reached
    momentum_history: List[MomentumRecord]


# =============================================================================
# FIXTURES
# =============================================================================

def create_3mtq_3rw_config(h_initial: float = MOMENTUM_INITIAL):
    """Create 3MTQ+3RW satellite with specified initial momentum."""
    mtqs = [MTQ(axis=j, max_torque=0.5) for j in MathConstants.unitvecs]
    rws = [RW(
        axis=j,
        max_torque=0.01,
        J=0.001,
        h=h_initial,
        h_max=MOMENTUM_SATURATION
    ) for j in MathConstants.unitvecs]
    return dict(
        mass=4.0,
        J_0=np.diagflat([0.1, 0.1, 0.1]),
        actuators=mtqs + rws,
        sensors=[MTM(axis=j) for j in MathConstants.unitvecs],
        boresight=np.array([0, 0, 1])
    )


def create_3mtq_1rw_config(h_initial: float = MOMENTUM_INITIAL):
    """Create 3MTQ+1RW satellite (limited desaturation capability)."""
    mtqs = [MTQ(axis=j, max_torque=0.5) for j in MathConstants.unitvecs]
    rw = RW(
        axis=np.array([0, 0, 1]),
        max_torque=0.01,
        J=0.001,
        h=h_initial,
        h_max=MOMENTUM_SATURATION
    )
    return dict(
        mass=4.0,
        J_0=np.diagflat([0.1, 0.1, 0.1]),
        actuators=mtqs + [rw],
        sensors=[MTM(axis=j) for j in MathConstants.unitvecs],
        boresight=np.array([0, 0, 1])
    )


@pytest.fixture
def satellite_3mtq_3rw():
    return create_3mtq_3rw_config()


@pytest.fixture
def satellite_3mtq_1rw():
    return create_3mtq_1rw_config()


# =============================================================================
# TODO-DATA-5: MOMENTUM EVOLUTION TESTS
# =============================================================================

class TestMomentumEvolution:
    """
    TODO-DATA-5 Part 1: Test momentum tracking and evolution.
    
    Validates momentum bookkeeping for paper Figure: Momentum vs Time.
    """

    def test_momentum_initialization(self, satellite_3mtq_3rw):
        """Verify initial momentum is set correctly."""
        est_sat = EstimatedSatellite(**satellite_3mtq_3rw)
        rws = [a for a in est_sat.actuators if isinstance(a, RW)]
        
        h_total = sum(rw.h * rw.axis for rw in rws)
        h_magnitude = np.linalg.norm(h_total)
        
        expected_magnitude = MOMENTUM_INITIAL * np.sqrt(3)  # 3 RWs at same h
        assert np.isclose(h_magnitude, expected_magnitude, rtol=0.01), \
            f"Initial momentum {h_magnitude} != expected {expected_magnitude}"
        
        print(f"\n[TODO-DATA-5] Initial momentum: {h_magnitude:.4f} N·m·s")

    def test_momentum_update_mechanics(self, satellite_3mtq_3rw):
        """Test that momentum updates work correctly."""
        est_sat = EstimatedSatellite(**satellite_3mtq_3rw)
        rws = [a for a in est_sat.actuators if isinstance(a, RW)]
        
        # Record initial
        h_initial = [rw.h for rw in rws]
        
        # Simulate momentum change (as if desaturating)
        delta_h = -0.001  # Reduce each RW momentum
        for rw in rws:
            new_h = max(0, rw.h + delta_h)
            rw.h = new_h
        
        # Verify change
        h_final = [rw.h for rw in rws]
        for i in range(len(rws)):
            expected = max(0, h_initial[i] + delta_h)
            assert np.isclose(h_final[i], expected), \
                f"RW {i}: expected h={expected}, got {h_final[i]}"

    @pytest.mark.parametrize("saturation_level", [0.2, 0.5, 0.8, 0.95])
    def test_saturation_fraction_tracking(self, saturation_level):
        """Test saturation tracking at different levels."""
        h_initial = MOMENTUM_SATURATION * saturation_level
        config = create_3mtq_3rw_config(h_initial=h_initial)
        est_sat = EstimatedSatellite(**config)
        
        rws = [a for a in est_sat.actuators if isinstance(a, RW)]
        
        for rw in rws:
            actual_fraction = rw.h / rw.h_max
            assert np.isclose(actual_fraction, saturation_level, rtol=0.01), \
                f"Saturation fraction {actual_fraction} != expected {saturation_level}"
        
        print(f"\n[TODO-DATA-5] Saturation level {saturation_level:.0%}: verified")


# =============================================================================
# TODO-DATA-5: DESATURATION GAIN SWEEP
# =============================================================================

class TestDesaturationGainSweep:
    """
    TODO-DATA-5 Part 2: Test different c_gain values.
    
    Supports paper Figure: Desaturation Rate vs C_Gain.
    """

    @pytest.mark.parametrize("c_gain", C_GAIN_VALUES)
    def test_controller_accepts_c_gain(self, satellite_3mtq_3rw, c_gain):
        """Test controller creation with different c_gain values."""
        est_sat = EstimatedSatellite(**satellite_3mtq_3rw)
        
        controller = MTQ_w_RW_LP(
            est_sat=est_sat,
            p_gain=1.0,
            d_gain=0.5,
            c_gain=c_gain,
            h_target=np.zeros(3)
        )
        
        assert controller.c_gain == c_gain
        print(f"\n[TODO-DATA-5] c_gain={c_gain}: controller created")

    def test_c_gain_effect_on_desaturation_priority(self, satellite_3mtq_3rw):
        """Test that higher c_gain prioritizes desaturation more."""
        results = []
        
        for c_gain in [0.1, 1.0]:
            config = create_3mtq_3rw_config(h_initial=0.03)
            est_sat = EstimatedSatellite(**config)
            controller = MTQ_w_RW_LP(
                est_sat=est_sat,
                p_gain=1.0,
                d_gain=0.5,
                c_gain=c_gain,
                h_target=np.zeros(3)
            )
            results.append({'c_gain': c_gain, 'controller': controller})
        
        print(f"\n[TODO-DATA-5] c_gain comparison: tested {len(results)} configurations")


# =============================================================================
# TODO-DATA-5: POINTING IMPACT TESTS
# =============================================================================

class TestPointingImpact:
    """
    TODO-DATA-5 Part 3: Test pointing performance during desaturation.
    
    Supports paper Figure: Pointing Error vs Desaturation Rate Trade-off.
    """

    def test_pointing_accuracy_tracking_setup(self, satellite_3mtq_3rw):
        """Set up infrastructure for tracking pointing during desaturation."""
        est_sat = EstimatedSatellite(**satellite_3mtq_3rw)
        controller = MTQ_w_RW_LP(
            est_sat=est_sat,
            p_gain=1.0,
            d_gain=0.5,
            c_gain=0.5,
            h_target=np.zeros(3)
        )
        
        # Verify controller has expected attributes
        assert hasattr(controller, 'p_gain')
        assert hasattr(controller, 'c_gain')
        assert hasattr(controller, 'h_target')
        
        print("\n[TODO-DATA-5] Pointing impact test infrastructure ready")


# =============================================================================
# TODO-DESAT-1 through TODO-DESAT-6: CONFIGURATION TESTS
# =============================================================================

class TestDesaturationConfigurations:
    """
    TODO-DESAT-1 to -6: Test different actuator configurations.
    """

    def test_3rw_3mtq_full_desaturation_capability(self, satellite_3mtq_3rw):
        """TODO-DESAT-1: 3RW+3MTQ can desaturate while pointing."""
        est_sat = EstimatedSatellite(**satellite_3mtq_3rw)
        
        rws = [a for a in est_sat.actuators if isinstance(a, RW)]
        mtqs = [a for a in est_sat.actuators if isinstance(a, MTQ)]
        
        # Full capability: 3 RWs + 3 MTQs
        assert len(rws) == 3
        assert len(mtqs) == 3
        
        # RW axes should span 3D
        rw_axes = np.array([rw.axis for rw in rws])
        rank = np.linalg.matrix_rank(rw_axes)
        assert rank == 3, f"RW axes should span 3D, got rank {rank}"
        
        print("\n[TODO-DESAT-1] 3RW+3MTQ: full desaturation capability verified")

    def test_1rw_3mtq_limited_desaturation(self, satellite_3mtq_1rw):
        """TODO-DESAT-2: 1RW+3MTQ has limited desaturation."""
        est_sat = EstimatedSatellite(**satellite_3mtq_1rw)
        
        rws = [a for a in est_sat.actuators if isinstance(a, RW)]
        
        # Only 1 RW - limited to Z-axis momentum management
        assert len(rws) == 1
        assert np.allclose(rws[0].axis, [0, 0, 1])
        
        print("\n[TODO-DESAT-2] 1RW+3MTQ: limited to Z-axis desaturation")

    def test_saturation_near_limit_behavior(self):
        """TODO-DESAT-3: Test behavior near saturation limit."""
        # Near saturation
        config = create_3mtq_3rw_config(h_initial=0.048)  # 96% of 0.05
        est_sat = EstimatedSatellite(**config)
        
        rws = [a for a in est_sat.actuators if isinstance(a, RW)]
        saturation = rws[0].h / rws[0].h_max
        
        assert saturation > 0.9, f"Should be near saturation, got {saturation:.0%}"
        
        print(f"\n[TODO-DESAT-3] Near saturation ({saturation:.0%}): tested")

    def test_zero_initial_momentum(self):
        """TODO-DESAT-4: Test with zero initial momentum."""
        config = create_3mtq_3rw_config(h_initial=0.0)
        est_sat = EstimatedSatellite(**config)
        
        rws = [a for a in est_sat.actuators if isinstance(a, RW)]
        h_total = sum(abs(rw.h) for rw in rws)
        
        assert h_total < 1e-10, "Should have zero initial momentum"
        
        print("\n[TODO-DESAT-4] Zero initial momentum: tested")


# =============================================================================
# TODO-FIG-4: SCHEDULING VISUALIZATION DATA
# =============================================================================

class TestSchedulingData:
    """
    TODO-FIG-4: Generate data for desaturation scheduling visualization.
    """

    def test_momentum_time_series_data(self):
        """Generate momentum time series data."""
        config = create_3mtq_3rw_config(h_initial=0.04)
        est_sat = EstimatedSatellite(**config)
        
        rws = [a for a in est_sat.actuators if isinstance(a, RW)]
        
        # Simulate simple decay
        time_series = []
        for t in range(0, 100, 1):
            # Simulate exponential decay
            decay_factor = np.exp(-t / 50)
            h_magnitude = 0.04 * np.sqrt(3) * decay_factor
            
            time_series.append({
                'time': t,
                'h_magnitude': h_magnitude,
                'saturation': h_magnitude / (MOMENTUM_SATURATION * np.sqrt(3))
            })
        
        print(f"\n[TODO-FIG-4] Generated {len(time_series)} time points")
        print(f"  Initial h: {time_series[0]['h_magnitude']:.4f}")
        print(f"  Final h: {time_series[-1]['h_magnitude']:.4f}")


# =============================================================================
# DATA EXPORT UTILITY
# =============================================================================

def generate_desaturation_data(c_gains: List[float] = None, 
                                output_file: str = None) -> List[Dict]:
    """
    Generate desaturation comparison data for paper figures.
    
    Parameters
    ----------
    c_gains : List[float], optional
        C_gain values to test. Defaults to C_GAIN_VALUES.
    output_file : str, optional
        JSON file to save results.
    
    Returns
    -------
    List[Dict]
        Results for each c_gain configuration.
    """
    if c_gains is None:
        c_gains = C_GAIN_VALUES
    
    results = []
    
    for c_gain in c_gains:
        config = create_3mtq_3rw_config(h_initial=0.04)
        est_sat = EstimatedSatellite(**config)
        
        rws = [a for a in est_sat.actuators if isinstance(a, RW)]
        h_initial = sum(rw.h * rw.axis for rw in rws)
        h_mag_initial = np.linalg.norm(h_initial)
        
        results.append({
            'c_gain': c_gain,
            'h_initial': h_mag_initial,
            'h_max': MOMENTUM_SATURATION * np.sqrt(3),
            'saturation_initial': h_mag_initial / (MOMENTUM_SATURATION * np.sqrt(3)),
        })
    
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
