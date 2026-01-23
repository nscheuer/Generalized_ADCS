"""
TODO-DATA-5, TODO-DATA-9: Sensitivity Analysis Tests
====================================================

Papers: Generalized Control Paper, Package Paper, Planner Paper
TODO IDs:
  - TODO-DATA-5: Run sensitivity analysis sweeps (100-trial MC for each error condition)
  - TODO-DATA-9: Generate sensitivity analysis plots (inertia error, B-field error, alignment error)
  - TODO-JGCD-4: Add robustness analysis (parameter uncertainty, estimation error)

This module tests controller robustness to parameter uncertainties.

Adjustable Parameters
---------------------
- INERTIA_ERROR_RANGE: Range of inertia errors to test [%]
- BFIELD_ERROR_RANGE: Range of B-field errors to test [%]
- ALIGNMENT_ERROR_RANGE: Range of alignment errors [deg]
"""

import sys
import os
import numpy as np
import pytest
from dataclasses import dataclass
from typing import List, Dict, Tuple

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

# Error ranges for sensitivity sweeps
INERTIA_ERROR_RANGE = [-20, -10, -5, 0, 5, 10, 20]    # Percent
BFIELD_ERROR_RANGE = [-30, -15, 0, 15, 30]             # Percent  
ALIGNMENT_ERROR_RANGE = [-3, -1, 0, 1, 3]              # Degrees

# Number of trials per condition
N_TRIALS_PER_CONDITION = 10

# Pretty output
PRETTY_OUTPUT = True


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SensitivityResult:
    """Result from sensitivity analysis."""
    parameter_name: str
    parameter_value: float
    parameter_unit: str
    mean_performance: float
    std_performance: float
    worst_case: float
    n_trials: int


# =============================================================================
# PRETTY OUTPUT
# =============================================================================

class PrettyOutput:
    HEADER = '\033[95m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    
    @staticmethod
    def header(text: str) -> None:
        if PRETTY_OUTPUT:
            print(f"\n{PrettyOutput.BOLD}{PrettyOutput.HEADER}{'='*70}")
            print(f"  {text}")
            print(f"{'='*70}{PrettyOutput.ENDC}\n")
    
    @staticmethod
    def sensitivity_table(results: List[SensitivityResult], title: str) -> None:
        """Print sensitivity analysis results as table."""
        print(f"\n{PrettyOutput.BOLD}  {title}{PrettyOutput.ENDC}")
        print("  " + "─" * 65)
        
        headers = ["Parameter", "Mean Perf", "Std", "Worst Case"]
        widths = [20, 15, 12, 15]
        
        header_row = "  │ " + " │ ".join(f"{h:^{w}}" for h, w in zip(headers, widths)) + " │"
        print(header_row)
        print("  ├─" + "─┼─".join("─" * w for w in widths) + "─┤")
        
        for r in results:
            param_str = f"{r.parameter_value:+.1f}{r.parameter_unit}"
            cols = [
                param_str,
                f"{r.mean_performance:.4f}",
                f"{r.std_performance:.4f}",
                f"{r.worst_case:.4f}",
            ]
            row = "  │ " + " │ ".join(f"{c:^{w}}" for c, w in zip(cols, widths)) + " │"
            print(row)
        
        print("  └─" + "─┴─".join("─" * w for w in widths) + "─┘")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_satellite_with_inertia_error(error_pct: float) -> dict:
    """Create satellite with perturbed inertia."""
    J_base = np.diagflat([0.1, 0.1, 0.1])
    J_perturbed = J_base * (1 + error_pct / 100)
    
    mtqs = [MTQ(axis=j, max_torque=0.5) for j in MathConstants.unitvecs]
    rw = RW(axis=np.array([0, 0, 1]), max_torque=0.01, J=0.001, h=0.0, h_max=0.05)
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    
    return dict(
        mass=4.0,
        J_0=J_perturbed,
        actuators=mtqs + [rw],
        sensors=mtms,
        boresight=np.array([0, 0, 1])
    )


def create_bfield_with_error(error_pct: float) -> np.ndarray:
    """Create B-field with magnitude error."""
    B_nominal = normalize(np.array([1, 1, 1])) * 3e-5
    return B_nominal * (1 + error_pct / 100)


def evaluate_controller_performance(config: dict, B: np.ndarray, seed: int) -> float:
    """Evaluate controller performance (simplified metric)."""
    np.random.seed(seed)
    
    est_sat = EstimatedSatellite(**config)
    controller = MTQ_w_RW_LP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
    
    # Test allocation on random torque demand
    tau_des = normalize(np.random.randn(3)) * 0.001
    
    _, u_mtq, alpha = controller.allocate_max_torque_in_direction(
        tau_des=tau_des,
        b_body=B,
        est_sat=est_sat
    )
    
    # Performance metric: achieved fraction (higher is better)
    return max(0, alpha)


# =============================================================================
# TODO-DATA-5: INERTIA SENSITIVITY TESTS
# =============================================================================

class TestInertiaSensitivity:
    """
    TODO-DATA-5, TODO-DATA-9: Inertia uncertainty sensitivity analysis.
    """

    def test_inertia_error_sweep(self):
        """Sweep through inertia errors and measure performance."""
        PrettyOutput.header("TODO-DATA-9: Inertia Sensitivity Analysis")
        
        B_nominal = normalize(np.array([1, 1, 1])) * 3e-5
        results = []
        
        for error_pct in INERTIA_ERROR_RANGE:
            performances = []
            
            for trial in range(N_TRIALS_PER_CONDITION):
                config = create_satellite_with_inertia_error(error_pct)
                perf = evaluate_controller_performance(config, B_nominal, trial)
                performances.append(perf)
            
            results.append(SensitivityResult(
                parameter_name="Inertia Error",
                parameter_value=error_pct,
                parameter_unit="%",
                mean_performance=np.mean(performances),
                std_performance=np.std(performances),
                worst_case=np.min(performances),
                n_trials=len(performances),
            ))
        
        PrettyOutput.sensitivity_table(results, "Inertia Error Sensitivity")
        
        # Verify degradation is graceful
        nominal_perf = [r.mean_performance for r in results if r.parameter_value == 0][0]
        for r in results:
            # Performance shouldn't degrade more than 50% even at ±20%
            if abs(r.parameter_value) <= 20:
                assert r.mean_performance >= nominal_perf * 0.5, \
                    f"Too much degradation at {r.parameter_value}%"


# =============================================================================
# TODO-DATA-9: B-FIELD SENSITIVITY TESTS
# =============================================================================

class TestBFieldSensitivity:
    """
    TODO-DATA-9: B-field estimation error sensitivity analysis.
    """

    def test_bfield_error_sweep(self):
        """Sweep through B-field errors and measure performance."""
        PrettyOutput.header("TODO-DATA-9: B-Field Sensitivity Analysis")
        
        results = []
        
        for error_pct in BFIELD_ERROR_RANGE:
            performances = []
            B = create_bfield_with_error(error_pct)
            
            for trial in range(N_TRIALS_PER_CONDITION):
                config = create_satellite_with_inertia_error(0)  # Nominal inertia
                perf = evaluate_controller_performance(config, B, trial)
                performances.append(perf)
            
            results.append(SensitivityResult(
                parameter_name="B-Field Error",
                parameter_value=error_pct,
                parameter_unit="%",
                mean_performance=np.mean(performances),
                std_performance=np.std(performances),
                worst_case=np.min(performances),
                n_trials=len(performances),
            ))
        
        PrettyOutput.sensitivity_table(results, "B-Field Error Sensitivity")


# =============================================================================
# TODO-JGCD-4: COMBINED ROBUSTNESS
# =============================================================================

class TestCombinedRobustness:
    """
    TODO-JGCD-4: Combined parameter uncertainty analysis.
    """

    def test_combined_uncertainty(self):
        """Test with multiple simultaneous uncertainties."""
        PrettyOutput.header("TODO-JGCD-4: Combined Robustness Analysis")
        
        # Test conditions: (inertia_error%, bfield_error%)
        conditions = [
            (0, 0, "Nominal"),
            (10, 15, "Mild"),
            (20, 30, "Severe"),
        ]
        
        results = []
        
        for inertia_err, bfield_err, label in conditions:
            performances = []
            
            for trial in range(N_TRIALS_PER_CONDITION):
                config = create_satellite_with_inertia_error(inertia_err)
                B = create_bfield_with_error(bfield_err)
                perf = evaluate_controller_performance(config, B, trial)
                performances.append(perf)
            
            print(f"  {label} (I:{inertia_err:+d}%, B:{bfield_err:+d}%):")
            print(f"    Mean: {np.mean(performances):.4f}")
            print(f"    Std:  {np.std(performances):.4f}")
            print(f"    Min:  {np.min(performances):.4f}")
            print()
        
        print("  Combined robustness analysis complete ✓")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
