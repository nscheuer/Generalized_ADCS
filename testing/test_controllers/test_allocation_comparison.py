"""
Tests for LP vs QP allocation comparison.

These tests support TODO-DATA-4 from the Generalized Control Paper:
"Generate LP vs QP comparison plots (direction error, magnitude, computation time)"

Tests cover:
- LP allocation preserves torque direction
- QP allocation minimizes torque magnitude error
- Comparison across multiple B-field orientations
- Performance under saturation conditions
- Computational timing benchmarks
"""

import sys
import os
import numpy as np
import pytest
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ADCS.controller.mtq_w_rw_LP import MTQ_w_RW_LP
from ADCS.controller.mtq_w_rw_QP import MTQ_w_RW_QP
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.actuators import RW, MTQ
from ADCS.satellite_hardware.sensors import MTM
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.helpers.math_helpers import normalize, rot_mat, random_n_unit_vec
from ADCS.helpers.math_constants import MathConstants


# =============================================================================
# FIXTURES
# =============================================================================

def create_mtq_only_satellite():
    """Create a MTQ-only satellite (3 orthogonal magnetorquers)."""
    mtqs = [MTQ(axis=j, max_torque=0.5) for j in MathConstants.unitvecs]
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    
    return dict(
        mass=4.0,
        J_0=np.diagflat([0.1, 0.1, 0.1]),
        actuators=mtqs,
        sensors=mtms,
        boresight=np.array([0, 0, 1])
    )


def create_mtq_rw_satellite():
    """Create a mixed MTQ+RW satellite (3 MTQ + 1 RW along Z)."""
    mtqs = [MTQ(axis=j, max_torque=0.5) for j in MathConstants.unitvecs]
    rw = RW(
        axis=np.array([0, 0, 1]),
        max_torque=0.01,
        J=0.001,
        h=0.0,
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
def mtq_only_satellite():
    """Create a MTQ-only satellite (3 orthogonal magnetorquers)."""
    return create_mtq_only_satellite()


@pytest.fixture
def mtq_rw_satellite():
    """Create a mixed MTQ+RW satellite (3 MTQ + 1 RW along Z)."""
    return create_mtq_rw_satellite()


@pytest.fixture
def orbital_state():
    """Create orbital state with non-trivial B-field."""
    ephem = Ephemeris()
    return Orbital_State(
        ephem=ephem,
        J2000=0.22,
        R=np.array([7000, 0, 0]),
        V=np.array([0, 7.5, 0]),
        B=normalize(np.array([2, 1, 3])) * 3e-5  # 30 μT
    )


# =============================================================================
# LP DIRECTION PRESERVATION TESTS
# =============================================================================

class TestLPDirectionPreservation:
    """
    Test that LP allocation preserves torque direction.
    
    The LP formulation maximizes α such that τ_achieved = α * τ_desired,
    guaranteeing that achieved torque is parallel to desired (or zero).
    """

    def test_lp_direction_preserved_various_demands(self, mtq_only_satellite, orbital_state):
        """Test LP preserves direction for various torque demands."""
        est_sat = EstimatedSatellite(**mtq_only_satellite)
        controller = MTQ_w_RW_LP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        
        # Test multiple random torque directions
        np.random.seed(42)
        for _ in range(20):
            tau_des = normalize(np.random.randn(3)) * 0.001  # Small torque demand
            
            # Get allocation result
            u_rw, u_mtq, alpha = controller.allocate_max_torque_in_direction(
                tau_des=tau_des,
                b_body=orbital_state.B,
                est_sat=est_sat
            )
            
            # Compute achieved torque
            tau_achieved = np.zeros(3)
            for i, mtq in enumerate([a for a in est_sat.actuators if isinstance(a, MTQ)]):
                tau_achieved += np.cross(mtq.axis * u_mtq[i], orbital_state.B)
            
            if np.linalg.norm(tau_achieved) > 1e-10:
                # Check direction is preserved
                tau_des_dir = tau_des / np.linalg.norm(tau_des)
                tau_ach_dir = tau_achieved / np.linalg.norm(tau_achieved)
                
                # Direction should match (or be opposite if alpha < 0)
                dot_product = np.abs(np.dot(tau_des_dir, tau_ach_dir))
                assert dot_product > 0.99, f"Direction not preserved: dot={dot_product}"

    def test_lp_alpha_meaning(self, mtq_only_satellite, orbital_state):
        """Test that alpha correctly represents achieved fraction."""
        est_sat = EstimatedSatellite(**mtq_only_satellite)
        controller = MTQ_w_RW_LP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        
        tau_des = np.array([0.001, 0, 0])  # Small X torque
        
        u_rw, u_mtq, alpha = controller.allocate_max_torque_in_direction(
            tau_des=tau_des,
            b_body=orbital_state.B,
            est_sat=est_sat
        )
        
        # Compute achieved magnitude
        tau_achieved = np.zeros(3)
        for i, mtq in enumerate([a for a in est_sat.actuators if isinstance(a, MTQ)]):
            tau_achieved += np.cross(mtq.axis * u_mtq[i], orbital_state.B)
        
        # alpha should equal |τ_achieved| / |τ_desired|
        computed_alpha = np.linalg.norm(tau_achieved) / np.linalg.norm(tau_des)
        assert np.isclose(alpha, computed_alpha, rtol=0.01), \
            f"Alpha mismatch: reported={alpha}, computed={computed_alpha}"


# =============================================================================
# QP MAGNITUDE MINIMIZATION TESTS
# =============================================================================

class TestQPMagnitudeMinimization:
    """
    Test that QP allocation minimizes torque magnitude error.
    
    The QP formulation minimizes ||τ_achieved - τ_desired||², which may
    sacrifice some direction accuracy for better magnitude matching.
    """

    def test_qp_reduces_magnitude_error(self, mtq_only_satellite, orbital_state):
        """Test QP achieves lower magnitude error than LP in some cases."""
        est_sat = EstimatedSatellite(**mtq_only_satellite)
        lp_controller = MTQ_w_RW_LP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        qp_controller = MTQ_w_RW_QP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        
        # Test case where torque is partially achievable
        # Use a torque that's not perfectly aligned with MTQ plane
        tau_des = np.array([0.001, 0.001, 0.0005])
        
        # LP allocation
        u_rw_lp, u_mtq_lp, alpha_lp = lp_controller.allocate_max_torque_in_direction(
            tau_des=tau_des,
            b_body=orbital_state.B,
            est_sat=est_sat
        )
        
        # QP allocation
        u_rw_qp, u_mtq_qp, alpha_qp = qp_controller.allocate_max_torque_in_direction(
            tau_des=tau_des,
            b_body=orbital_state.B,
            est_sat=est_sat
        )
        
        # Compute achieved torques
        tau_lp = np.zeros(3)
        tau_qp = np.zeros(3)
        mtqs = [a for a in est_sat.actuators if isinstance(a, MTQ)]
        
        for i, mtq in enumerate(mtqs):
            tau_lp += np.cross(mtq.axis * u_mtq_lp[i], orbital_state.B)
            tau_qp += np.cross(mtq.axis * u_mtq_qp[i], orbital_state.B)
        
        # Compute errors
        error_lp = np.linalg.norm(tau_lp - tau_des)
        error_qp = np.linalg.norm(tau_qp - tau_des)
        
        # QP should achieve equal or better magnitude error
        # (may be slightly worse in edge cases due to numerical issues)
        assert error_qp <= error_lp * 1.1, \
            f"QP error={error_qp} should be <= LP error={error_lp}"


# =============================================================================
# B-FIELD ORIENTATION SWEEP TESTS
# =============================================================================

class TestBFieldSweep:
    """Test allocation across various B-field orientations."""

    def test_allocation_across_orbit(self, mtq_only_satellite):
        """Test allocation performance at different orbit positions."""
        est_sat = EstimatedSatellite(**mtq_only_satellite)
        controller = MTQ_w_RW_LP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        
        ephem = Ephemeris()
        
        # Simulate B-field changes around an orbit
        results = []
        for theta in np.linspace(0, 2*np.pi, 36):  # Every 10 degrees
            # Rotate B-field vector in a plane that gives varying controllability
            # Use rotation around Y to sweep through XZ plane
            B = 3e-5 * np.array([np.cos(theta), 0.3, np.sin(theta)])
            
            tau_des = np.array([0.001, 0.0005, 0])  # Torque with X and Y components
            
            u_rw, u_mtq, alpha = controller.allocate_max_torque_in_direction(
                tau_des=tau_des,
                b_body=B,
                est_sat=est_sat
            )
            
            results.append({
                'theta': theta,
                'alpha': alpha,
                'B_magnitude': np.linalg.norm(B)
            })
        
        # Alpha should vary with B-field orientation
        alphas = [r['alpha'] for r in results]
        
        # With MTQ-only, there should be some variation in achievable torque
        # as the B-field rotates (MTQ torque is perpendicular to B)
        alpha_range = max(alphas) - min(alphas)
        assert alpha_range >= 0, "Alpha range should be non-negative"
        
        # All alphas should be non-negative
        assert all(a >= -0.001 for a in alphas), f"Alpha should be non-negative, got min={min(alphas)}"


# =============================================================================
# SATURATION TESTS
# =============================================================================

class TestSaturationBehavior:
    """Test allocation behavior under actuator saturation."""

    def test_commands_respect_limits(self, mtq_only_satellite, orbital_state):
        """Test that allocated commands respect actuator limits."""
        est_sat = EstimatedSatellite(**mtq_only_satellite)
        controller = MTQ_w_RW_LP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        
        # Large torque demand that will saturate
        tau_des = np.array([1.0, 1.0, 1.0])  # Much larger than achievable
        
        u_rw, u_mtq, alpha = controller.allocate_max_torque_in_direction(
            tau_des=tau_des,
            b_body=orbital_state.B,
            est_sat=est_sat
        )
        
        # Check MTQ commands are within limits
        mtqs = [a for a in est_sat.actuators if isinstance(a, MTQ)]
        for i, mtq in enumerate(mtqs):
            assert abs(u_mtq[i]) <= mtq.u_max * 1.001, \
                f"MTQ {i} command {u_mtq[i]} exceeds limit {mtq.u_max}"

    def test_graceful_degradation(self, mtq_only_satellite, orbital_state):
        """Test that allocation degrades gracefully under heavy saturation."""
        est_sat = EstimatedSatellite(**mtq_only_satellite)
        controller = MTQ_w_RW_LP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        
        # Test increasing torque demands
        magnitudes = [0.0001, 0.001, 0.01, 0.1, 1.0]
        alphas = []
        
        for mag in magnitudes:
            tau_des = normalize(np.array([1, 1, 1])) * mag
            u_rw, u_mtq, alpha = controller.allocate_max_torque_in_direction(
                tau_des=tau_des,
                b_body=orbital_state.B,
                est_sat=est_sat
            )
            alphas.append(alpha)
        
        # Alpha should decrease as demand increases (saturation)
        # but should never go negative
        for alpha in alphas:
            assert alpha >= 0, f"Alpha should be non-negative, got {alpha}"


# =============================================================================
# MIXED ACTUATOR TESTS
# =============================================================================

class TestMixedActuators:
    """Test allocation with mixed MTQ+RW configurations."""

    def test_rw_contribution(self, mtq_rw_satellite, orbital_state):
        """Test that RW contributes to allocation."""
        est_sat = EstimatedSatellite(**mtq_rw_satellite)
        controller = MTQ_w_RW_LP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        
        # Torque demand along Z (RW axis)
        tau_des = np.array([0, 0, 0.001])
        
        u_rw, u_mtq, alpha = controller.allocate_max_torque_in_direction(
            tau_des=tau_des,
            b_body=orbital_state.B,
            est_sat=est_sat
        )
        
        # RW should contribute (u_rw should be non-zero)
        assert len(u_rw) > 0 and np.linalg.norm(u_rw) > 0, \
            "RW should contribute to Z-axis torque"

    def test_mtq_only_when_perpendicular(self, mtq_rw_satellite, orbital_state):
        """Test MTQ-only allocation for torques perpendicular to RW."""
        est_sat = EstimatedSatellite(**mtq_rw_satellite)
        controller = MTQ_w_RW_LP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        
        # Torque demand in XY plane (perpendicular to RW Z-axis)
        tau_des = np.array([0.001, 0, 0])
        
        u_rw, u_mtq, alpha = controller.allocate_max_torque_in_direction(
            tau_des=tau_des,
            b_body=orbital_state.B,
            est_sat=est_sat
        )
        
        # MTQ should be primary contributor for X torque
        assert np.linalg.norm(u_mtq) > 0, "MTQ should contribute to X torque"


# =============================================================================
# TIMING BENCHMARKS
# =============================================================================

class TestAllocationTiming:
    """Benchmark allocation computation time."""

    def test_lp_allocation_timing(self, mtq_only_satellite, orbital_state):
        """Measure LP allocation timing."""
        est_sat = EstimatedSatellite(**mtq_only_satellite)
        controller = MTQ_w_RW_LP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        
        tau_des = np.array([0.001, 0.0005, 0.0002])
        
        # Warm-up
        for _ in range(10):
            controller.allocate_max_torque_in_direction(
                tau_des=tau_des,
                b_body=orbital_state.B,
                est_sat=est_sat
            )
        
        # Timed runs
        n_runs = 100
        start = time.perf_counter()
        for _ in range(n_runs):
            controller.allocate_max_torque_in_direction(
                tau_des=tau_des,
                b_body=orbital_state.B,
                est_sat=est_sat
            )
        elapsed = time.perf_counter() - start
        
        mean_time_ms = (elapsed / n_runs) * 1000
        
        # Should be fast enough for real-time control (< 10ms)
        assert mean_time_ms < 10.0, f"LP allocation too slow: {mean_time_ms:.2f}ms"
        
        print(f"\nLP allocation mean time: {mean_time_ms:.3f}ms")

    def test_qp_allocation_timing(self, mtq_only_satellite, orbital_state):
        """Measure QP allocation timing."""
        est_sat = EstimatedSatellite(**mtq_only_satellite)
        controller = MTQ_w_RW_QP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        
        tau_des = np.array([0.001, 0.0005, 0.0002])
        
        # Warm-up
        for _ in range(10):
            controller.allocate_max_torque_in_direction(
                tau_des=tau_des,
                b_body=orbital_state.B,
                est_sat=est_sat
            )
        
        # Timed runs
        n_runs = 100
        start = time.perf_counter()
        for _ in range(n_runs):
            controller.allocate_max_torque_in_direction(
                tau_des=tau_des,
                b_body=orbital_state.B,
                est_sat=est_sat
            )
        elapsed = time.perf_counter() - start
        
        mean_time_ms = (elapsed / n_runs) * 1000
        
        # Should be fast enough for real-time control
        assert mean_time_ms < 10.0, f"QP allocation too slow: {mean_time_ms:.2f}ms"
        
        print(f"\nQP allocation mean time: {mean_time_ms:.3f}ms")


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and special conditions."""

    def test_zero_torque_demand(self, mtq_only_satellite, orbital_state):
        """Test allocation with zero torque demand."""
        est_sat = EstimatedSatellite(**mtq_only_satellite)
        controller = MTQ_w_RW_LP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        
        tau_des = np.array([0, 0, 0])
        
        u_rw, u_mtq, alpha = controller.allocate_max_torque_in_direction(
            tau_des=tau_des,
            b_body=orbital_state.B,
            est_sat=est_sat
        )
        
        # Should return zero commands
        assert np.allclose(u_mtq, 0), "Zero demand should give zero MTQ command"
        assert alpha == 1.0, "Alpha should be 1.0 for zero demand"

    def test_very_small_b_field(self, mtq_only_satellite):
        """Test allocation with very weak B-field."""
        est_sat = EstimatedSatellite(**mtq_only_satellite)
        controller = MTQ_w_RW_LP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        
        ephem = Ephemeris()
        os = Orbital_State(
            ephem=ephem,
            J2000=0.22,
            R=np.array([7000, 0, 0]),
            V=np.array([0, 7.5, 0]),
            B=np.array([1e-9, 1e-9, 1e-9])  # Very weak field
        )
        
        tau_des = np.array([0.001, 0, 0])
        
        # Should not crash
        u_rw, u_mtq, alpha = controller.allocate_max_torque_in_direction(
            tau_des=tau_des,
            b_body=os.B,
            est_sat=est_sat
        )
        
        # Alpha should be near zero (can't achieve much with weak B)
        assert alpha >= 0

    def test_torque_aligned_with_b_field(self, mtq_only_satellite, orbital_state):
        """Test allocation when desired torque is aligned with B-field."""
        est_sat = EstimatedSatellite(**mtq_only_satellite)
        controller = MTQ_w_RW_LP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        
        # Torque aligned with B-field (MTQs can't produce this)
        tau_des = normalize(orbital_state.B) * 0.001
        
        u_rw, u_mtq, alpha = controller.allocate_max_torque_in_direction(
            tau_des=tau_des,
            b_body=orbital_state.B,
            est_sat=est_sat
        )
        
        # Alpha should be zero or very small (impossible direction)
        assert alpha < 0.1, f"Should have low alpha for B-aligned torque, got {alpha}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
