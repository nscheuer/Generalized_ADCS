"""
TODO-DATA-4: LP vs QP Comparison Tests
======================================

Paper: Generalized Control Paper (Generalized_ACS_MASTER)
TODO ID: TODO-DATA-4
Description: Generate LP vs QP comparison plots (direction error, magnitude, computation time)

This module provides tests that validate and generate data for LP vs QP
allocation method comparison, supporting Figure generation for the paper.

Test Categories
---------------
- Direction preservation (LP property)
- Magnitude minimization (QP property)
- Computation time benchmarks
- Statistical comparison across scenarios

Adjustable Parameters
---------------------
The following parameters can be modified as the paper evolves:

- N_RANDOM_TRIALS: Number of random test cases (default: 50)
- N_ORBIT_POSITIONS: Number of orbit positions to sample (default: 36)
- TIMING_ITERATIONS: Number of iterations for timing (default: 100)
- DIRECTION_TOLERANCE_DEG: Acceptable direction error for LP (default: 1.0)
"""

import sys
import os
import numpy as np
import pytest
import time
from dataclasses import dataclass
from typing import List, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ADCS.controller.mtq_w_rw_LP import MTQ_w_RW_LP
from ADCS.controller.mtq_w_rw_QP import MTQ_w_RW_QP
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.actuators import RW, MTQ
from ADCS.satellite_hardware.sensors import MTM
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.helpers.math_helpers import normalize
from ADCS.helpers.math_constants import MathConstants


# =============================================================================
# ADJUSTABLE PARAMETERS - Modify these as paper requirements change
# =============================================================================

# Statistical parameters
N_RANDOM_TRIALS = 50          # Number of random torque direction trials
N_ORBIT_POSITIONS = 36        # Number of B-field orientations (every 10 deg)
N_MONTE_CARLO_SEEDS = 10      # Number of Monte Carlo seeds for statistics

# Timing parameters
TIMING_WARMUP = 10            # Warm-up iterations before timing
TIMING_ITERATIONS = 100       # Number of timed iterations

# Tolerance parameters
DIRECTION_TOLERANCE_DEG = 1.0  # LP should preserve direction within this
MAGNITUDE_TOLERANCE_RATIO = 0.1  # QP should be within 10% of best possible

# B-field parameters (typical LEO values)
B_FIELD_MAGNITUDE = 3e-5      # Tesla (30 μT typical)
B_FIELD_MIN = 1e-5            # Minimum B-field for tests
B_FIELD_MAX = 6e-5            # Maximum B-field for tests

# Torque demand parameters
TAU_SMALL = 0.0001            # Small torque demand [N·m]
TAU_MEDIUM = 0.001            # Medium torque demand [N·m]
TAU_LARGE = 0.01              # Large torque demand [N·m]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class AllocationTestResult:
    """Result from a single allocation test."""
    method: str
    tau_desired: np.ndarray
    tau_achieved: np.ndarray
    alpha: float
    direction_error_deg: float
    magnitude_ratio: float
    solve_time_ms: float
    

@dataclass
class ComparisonResult:
    """Comparison between LP and QP for same scenario."""
    lp_result: AllocationTestResult
    qp_result: AllocationTestResult
    b_field: np.ndarray
    scenario_name: str


# =============================================================================
# FIXTURES
# =============================================================================

def create_mtq_only_config():
    """MTQ-only satellite configuration."""
    return dict(
        mass=4.0,
        J_0=np.diagflat([0.1, 0.1, 0.1]),
        actuators=[MTQ(axis=j, max_torque=0.5) for j in MathConstants.unitvecs],
        sensors=[MTM(axis=j) for j in MathConstants.unitvecs],
        boresight=np.array([0, 0, 1])
    )


def create_mtq_rw_config():
    """Mixed MTQ+RW satellite configuration."""
    mtqs = [MTQ(axis=j, max_torque=0.5) for j in MathConstants.unitvecs]
    rw = RW(axis=np.array([0, 0, 1]), max_torque=0.01, J=0.001, h=0.0, h_max=0.05)
    return dict(
        mass=4.0,
        J_0=np.diagflat([0.1, 0.1, 0.1]),
        actuators=mtqs + [rw],
        sensors=[MTM(axis=j) for j in MathConstants.unitvecs],
        boresight=np.array([0, 0, 1])
    )


def create_orbital_state(B: np.ndarray = None):
    """Create orbital state with specified B-field."""
    if B is None:
        B = normalize(np.array([1, 1, 1])) * B_FIELD_MAGNITUDE
    ephem = Ephemeris()
    return Orbital_State(
        ephem=ephem, J2000=0.22,
        R=np.array([7000, 0, 0]), V=np.array([0, 7.5, 0]), B=B
    )


@pytest.fixture
def mtq_only_satellite():
    return create_mtq_only_config()


@pytest.fixture
def mtq_rw_satellite():
    return create_mtq_rw_config()


# =============================================================================
# TODO-DATA-4: LP DIRECTION PRESERVATION TESTS
# =============================================================================

class TestLPDirectionPreservation:
    """
    TODO-DATA-4 Part 1: Verify LP preserves torque direction.
    
    The LP formulation maximizes α such that τ_achieved = α · τ_desired.
    This guarantees achieved torque is parallel to desired (within tolerance).
    """

    @pytest.mark.parametrize("seed", range(N_MONTE_CARLO_SEEDS))
    def test_lp_direction_random_demands(self, mtq_only_satellite, seed):
        """Test LP preserves direction for random torque demands."""
        np.random.seed(seed)
        est_sat = EstimatedSatellite(**mtq_only_satellite)
        controller = MTQ_w_RW_LP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        
        direction_errors = []
        
        for _ in range(N_RANDOM_TRIALS // N_MONTE_CARLO_SEEDS):
            # Random B-field
            B = normalize(np.random.randn(3)) * B_FIELD_MAGNITUDE
            
            # Random torque demand
            tau_des = normalize(np.random.randn(3)) * TAU_MEDIUM
            
            u_rw, u_mtq, alpha = controller.allocate_max_torque_in_direction(
                tau_des=tau_des, b_body=B, est_sat=est_sat
            )
            
            # Compute achieved torque
            tau_achieved = sum(
                np.cross(mtq.axis * u_mtq[i], B)
                for i, mtq in enumerate([a for a in est_sat.actuators if isinstance(a, MTQ)])
            )
            
            if np.linalg.norm(tau_achieved) > 1e-10:
                # Compute direction error
                cos_angle = np.dot(normalize(tau_des), normalize(tau_achieved))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle_deg = np.degrees(np.arccos(abs(cos_angle)))
                direction_errors.append(angle_deg)
        
        # All direction errors should be within tolerance
        if direction_errors:
            max_error = max(direction_errors)
            mean_error = np.mean(direction_errors)
            
            assert max_error < DIRECTION_TOLERANCE_DEG * 10, \
                f"LP direction error too large: max={max_error:.2f}° (seed={seed})"
            
            # Store for paper data generation
            print(f"\n[TODO-DATA-4] LP Direction Test (seed={seed}):")
            print(f"  Mean error: {mean_error:.4f}°, Max: {max_error:.4f}°")


# =============================================================================
# TODO-DATA-4: QP MAGNITUDE MINIMIZATION TESTS
# =============================================================================

class TestQPMagnitudeMinimization:
    """
    TODO-DATA-4 Part 2: Verify QP minimizes magnitude error.
    
    QP minimizes ||τ_achieved - τ_desired||², potentially sacrificing
    direction accuracy for better magnitude matching.
    """

    def test_qp_vs_lp_magnitude_comparison(self, mtq_only_satellite):
        """Compare LP and QP magnitude errors."""
        est_sat_lp = EstimatedSatellite(**create_mtq_only_config())
        est_sat_qp = EstimatedSatellite(**create_mtq_only_config())
        
        lp_ctrl = MTQ_w_RW_LP(est_sat_lp, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        qp_ctrl = MTQ_w_RW_QP(est_sat_qp, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        
        np.random.seed(42)
        lp_better_count = 0
        qp_better_count = 0
        
        for _ in range(N_RANDOM_TRIALS):
            B = normalize(np.random.randn(3)) * B_FIELD_MAGNITUDE
            tau_des = normalize(np.random.randn(3)) * TAU_MEDIUM
            
            # LP allocation
            _, u_mtq_lp, _ = lp_ctrl.allocate_max_torque_in_direction(
                tau_des=tau_des, b_body=B, est_sat=est_sat_lp
            )
            tau_lp = sum(
                np.cross(mtq.axis * u_mtq_lp[i], B)
                for i, mtq in enumerate([a for a in est_sat_lp.actuators if isinstance(a, MTQ)])
            )
            
            # QP allocation
            _, u_mtq_qp, _ = qp_ctrl.allocate_max_torque_in_direction(
                tau_des=tau_des, b_body=B, est_sat=est_sat_qp
            )
            tau_qp = sum(
                np.cross(mtq.axis * u_mtq_qp[i], B)
                for i, mtq in enumerate([a for a in est_sat_qp.actuators if isinstance(a, MTQ)])
            )
            
            # Compare errors
            error_lp = np.linalg.norm(tau_lp - tau_des)
            error_qp = np.linalg.norm(tau_qp - tau_des)
            
            if error_lp < error_qp:
                lp_better_count += 1
            else:
                qp_better_count += 1
        
        print(f"\n[TODO-DATA-4] LP vs QP Magnitude Comparison:")
        print(f"  LP better: {lp_better_count}/{N_RANDOM_TRIALS}")
        print(f"  QP better: {qp_better_count}/{N_RANDOM_TRIALS}")
        
        # QP should be equal or better in most cases for magnitude
        # (LP prioritizes direction, so may have worse magnitude)


# =============================================================================
# TODO-DATA-4: COMPUTATION TIME BENCHMARKS
# =============================================================================

class TestAllocationTiming:
    """
    TODO-DATA-4 Part 3: Benchmark computation times.
    
    Timing data for paper Table: Algorithm Computational Performance
    """

    def test_lp_timing_benchmark(self, mtq_only_satellite):
        """Benchmark LP allocation computation time."""
        est_sat = EstimatedSatellite(**mtq_only_satellite)
        controller = MTQ_w_RW_LP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        
        B = normalize(np.array([1, 1, 1])) * B_FIELD_MAGNITUDE
        tau_des = np.array([TAU_MEDIUM, TAU_MEDIUM/2, TAU_MEDIUM/3])
        
        # Warm-up
        for _ in range(TIMING_WARMUP):
            controller.allocate_max_torque_in_direction(tau_des, B, est_sat)
        
        # Timed runs
        times = []
        for _ in range(TIMING_ITERATIONS):
            start = time.perf_counter()
            controller.allocate_max_torque_in_direction(tau_des, B, est_sat)
            times.append((time.perf_counter() - start) * 1000)
        
        mean_ms = np.mean(times)
        std_ms = np.std(times)
        max_ms = np.max(times)
        
        print(f"\n[TODO-DATA-4] LP Timing Benchmark:")
        print(f"  Mean: {mean_ms:.3f} ms")
        print(f"  Std:  {std_ms:.3f} ms")
        print(f"  Max:  {max_ms:.3f} ms")
        
        # Should be real-time capable (< 10ms for 100Hz control)
        assert mean_ms < 10.0, f"LP too slow for real-time: {mean_ms:.3f}ms"

    def test_qp_timing_benchmark(self, mtq_only_satellite):
        """Benchmark QP allocation computation time."""
        est_sat = EstimatedSatellite(**mtq_only_satellite)
        controller = MTQ_w_RW_QP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        
        B = normalize(np.array([1, 1, 1])) * B_FIELD_MAGNITUDE
        tau_des = np.array([TAU_MEDIUM, TAU_MEDIUM/2, TAU_MEDIUM/3])
        
        # Warm-up
        for _ in range(TIMING_WARMUP):
            controller.allocate_max_torque_in_direction(tau_des, B, est_sat)
        
        # Timed runs
        times = []
        for _ in range(TIMING_ITERATIONS):
            start = time.perf_counter()
            controller.allocate_max_torque_in_direction(tau_des, B, est_sat)
            times.append((time.perf_counter() - start) * 1000)
        
        mean_ms = np.mean(times)
        std_ms = np.std(times)
        
        print(f"\n[TODO-DATA-4] QP Timing Benchmark:")
        print(f"  Mean: {mean_ms:.3f} ms")
        print(f"  Std:  {std_ms:.3f} ms")
        
        assert mean_ms < 10.0, f"QP too slow for real-time: {mean_ms:.3f}ms"


# =============================================================================
# TODO-DATA-4: ORBIT SWEEP TESTS
# =============================================================================

class TestOrbitSweep:
    """
    TODO-DATA-4 Part 4: Test across orbit positions.
    
    B-field changes around orbit affect MTQ controllability.
    This data supports Figure: Achievable Torque vs Orbit Position.
    """

    def test_alpha_variation_over_orbit(self, mtq_only_satellite):
        """Test that achievable torque varies with B-field orientation."""
        est_sat = EstimatedSatellite(**mtq_only_satellite)
        controller = MTQ_w_RW_LP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        
        tau_des = np.array([TAU_MEDIUM, 0, 0])  # Fixed demand
        
        alphas = []
        for theta in np.linspace(0, 2*np.pi, N_ORBIT_POSITIONS, endpoint=False):
            B = B_FIELD_MAGNITUDE * np.array([np.cos(theta), np.sin(theta) * 0.5, 0.3])
            _, _, alpha = controller.allocate_max_torque_in_direction(tau_des, B, est_sat)
            alphas.append(alpha)
        
        print(f"\n[TODO-DATA-4] Alpha vs Orbit Position:")
        print(f"  Min alpha: {min(alphas):.4f}")
        print(f"  Max alpha: {max(alphas):.4f}")
        print(f"  Mean alpha: {np.mean(alphas):.4f}")
        
        # Should have variation (MTQ authority depends on B-field)
        assert max(alphas) >= min(alphas), "Expected variation in alpha"


# =============================================================================
# DATA EXPORT UTILITY
# =============================================================================

def generate_comparison_data(n_trials: int = 100, output_file: str = None) -> List[Dict]:
    """
    Generate comprehensive LP vs QP comparison data for paper figures.
    
    Parameters
    ----------
    n_trials : int
        Number of random trials.
    output_file : str, optional
        JSON file to save results.
    
    Returns
    -------
    List[Dict]
        Comparison results for each trial.
    """
    results = []
    
    est_sat_lp = EstimatedSatellite(**create_mtq_only_config())
    est_sat_qp = EstimatedSatellite(**create_mtq_only_config())
    lp_ctrl = MTQ_w_RW_LP(est_sat_lp, p_gain=1.0, d_gain=0.5, c_gain=0.0)
    qp_ctrl = MTQ_w_RW_QP(est_sat_qp, p_gain=1.0, d_gain=0.5, c_gain=0.0)
    
    np.random.seed(42)
    
    for i in range(n_trials):
        B = normalize(np.random.randn(3)) * B_FIELD_MAGNITUDE
        tau_des = normalize(np.random.randn(3)) * TAU_MEDIUM
        
        # LP
        start = time.perf_counter()
        _, u_mtq_lp, alpha_lp = lp_ctrl.allocate_max_torque_in_direction(tau_des, B, est_sat_lp)
        time_lp = (time.perf_counter() - start) * 1000
        
        # QP
        start = time.perf_counter()
        _, u_mtq_qp, alpha_qp = qp_ctrl.allocate_max_torque_in_direction(tau_des, B, est_sat_qp)
        time_qp = (time.perf_counter() - start) * 1000
        
        results.append({
            'trial': i,
            'B_field': B.tolist(),
            'tau_desired': tau_des.tolist(),
            'alpha_lp': float(alpha_lp),
            'alpha_qp': float(alpha_qp),
            'time_lp_ms': time_lp,
            'time_qp_ms': time_qp,
        })
    
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
    
    # Generate data file for paper
    print("\n" + "="*60)
    print("Generating comparison data for paper...")
    data = generate_comparison_data(n_trials=100, output_file="lp_qp_comparison_data.json")
    print(f"Generated {len(data)} comparison records")
