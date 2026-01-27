"""
TODO-DATA-8, TODO-DAA-3, TODO-DAA-4, TODO-COMP-1, TODO-SMALLSAT-2: Data Generation Tests
=========================================================================================

Papers: Generalized Control Paper, Package Paper
TODO IDs:
  - TODO-DATA-8: Comparison table vs prior methods (Lovera, Wisniewski)
  - TODO-DAA-3: Numerical examples for CubeSat configurations
  - TODO-DAA-4: WCDTA comparison table across configurations
  - TODO-COMP-1: Disturbance compensation examples
  - TODO-SMALLSAT-1: 5-minute demo capability
  - TODO-SMALLSAT-2: Practitioner metrics (power, mass, cost proxies)
  - TODO-SMALLSAT-3: Failure modes documentation data

These tests verify data generation capabilities for paper tables and examples.

Adjustable Parameters
---------------------
- CUBESAT_CONFIGS: CubeSat configuration definitions
- N_COMPARISON_TRIALS: Number of trials for statistical comparison
"""

import sys
import os
import numpy as np
import pytest
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ADCS.controller.mtq_w_rw_LP import MTQ_w_RW_LP
from ADCS.controller.mtq_w_rw_QP import MTQ_w_RW_QP
from ADCS.controller.bdot import BDot
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.actuators import RW, MTQ
from ADCS.satellite_hardware.sensors import MTM
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.helpers.math_helpers import normalize, skewsym
from ADCS.helpers.math_constants import MathConstants


# =============================================================================
# ADJUSTABLE PARAMETERS
# =============================================================================

# CubeSat configurations for TODO-DAA-3
CUBESAT_CONFIGS = {
    "1U_Basic": {
        "mass": 1.33,  # kg
        "J": np.diagflat([0.002, 0.002, 0.002]),  # kg⋅m²
        "mtq_torque": 0.2,  # Am² (typical 1U)
        "n_rw": 0,
        "rw_torque": 0.0,
    },
    "3U_Standard": {
        "mass": 4.0,
        "J": np.diagflat([0.035, 0.035, 0.007]),
        "mtq_torque": 0.5,
        "n_rw": 3,
        "rw_torque": 0.004,  # 4 mNm typical
    },
    "6U_HighPerf": {
        "mass": 12.0,
        "J": np.diagflat([0.08, 0.08, 0.02]),
        "mtq_torque": 1.0,
        "n_rw": 4,
        "rw_torque": 0.01,
    },
    "12U_Advanced": {
        "mass": 24.0,
        "J": np.diagflat([0.2, 0.2, 0.05]),
        "mtq_torque": 2.0,
        "n_rw": 4,
        "rw_torque": 0.02,
    },
}

# Comparison parameters
N_COMPARISON_TRIALS = 20

# Disturbance levels
DISTURBANCE_TORQUE_NM = 1e-6  # Typical gravity gradient for CubeSat

# Pretty output
PRETTY_OUTPUT = True


# =============================================================================
# PRETTY OUTPUT
# =============================================================================

class PrettyOutput:
    HEADER = '\033[95m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    
    @staticmethod
    def header(text: str) -> None:
        if PRETTY_OUTPUT:
            print(f"\n{PrettyOutput.BOLD}{PrettyOutput.HEADER}{'='*70}")
            print(f"  {text}")
            print(f"{'='*70}{PrettyOutput.ENDC}\n")
    
    @staticmethod
    def subheader(text: str) -> None:
        if PRETTY_OUTPUT:
            print(f"\n{PrettyOutput.BOLD}{PrettyOutput.CYAN}  ── {text} ──{PrettyOutput.ENDC}")
    
    @staticmethod
    def table_row(cols: List[str], widths: List[int]) -> str:
        return "  │ " + " │ ".join(f"{c:^{w}}" for c, w in zip(cols, widths)) + " │"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_cubesat_config(name: str) -> dict:
    """Create satellite config from CubeSat parameters."""
    params = CUBESAT_CONFIGS[name]
    
    mtqs = [MTQ(axis=j, max_torque=params["mtq_torque"]) for j in MathConstants.unitvecs]
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    
    rws = []
    if params["n_rw"] == 3:
        rws = [RW(axis=j, max_torque=params["rw_torque"], J=0.001, h=0.0, h_max=0.05) 
               for j in MathConstants.unitvecs]
    elif params["n_rw"] == 4:
        # Pyramid configuration
        axes = [
            normalize(np.array([1, 1, 1])),
            normalize(np.array([1, -1, -1])),
            normalize(np.array([-1, 1, -1])),
            normalize(np.array([-1, -1, 1])),
        ]
        rws = [RW(axis=ax, max_torque=params["rw_torque"], J=0.001, h=0.0, h_max=0.05) 
               for ax in axes]
    
    return dict(
        mass=params["mass"],
        J_0=params["J"],
        actuators=mtqs + rws,
        sensors=mtms,
        boresight=np.array([0, 0, 1])
    )


def create_orbital_state() -> Orbital_State:
    """Create standard orbital state."""
    ephem = Ephemeris()
    return Orbital_State(
        ephem=ephem, J2000=0.22,
        R=np.array([7000, 0, 0]), V=np.array([0, 7.5, 0]),
        B=normalize(np.array([1, 1, 1])) * 3e-5
    )


# =============================================================================
# TODO-DATA-8: COMPARISON WITH PRIOR METHODS
# =============================================================================

class TestPriorMethodsComparison:
    """
    TODO-DATA-8: Generate comparison table vs prior methods.
    """

    def test_controller_comparison_table(self):
        """Generate comparison data for LP, QP, B-dot controllers."""
        PrettyOutput.header("TODO-DATA-8: Prior Methods Comparison Table")
        
        config = create_cubesat_config("3U_Standard")
        os = create_orbital_state()
        
        # Create controllers
        est_sat = EstimatedSatellite(**config)
        
        controllers = {
            "LP (Proposed)": MTQ_w_RW_LP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0),
            "QP (Proposed)": MTQ_w_RW_QP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0),
            "B-dot": BDot(est_sat=est_sat, gain=1e4),
        }
        
        comparison_data = []
        
        for name, ctrl in controllers.items():
            # Test allocation performance
            alphas = []
            times_ms = []
            
            for trial in range(N_COMPARISON_TRIALS):
                np.random.seed(trial)
                tau_des = normalize(np.random.randn(3)) * 0.005
                
                start = time.perf_counter()
                
                if hasattr(ctrl, 'allocate_max_torque_in_direction'):
                    _, _, alpha = ctrl.allocate_max_torque_in_direction(
                        tau_des=tau_des, b_body=os.B, est_sat=est_sat
                    )
                    alphas.append(alpha)
                else:
                    # B-dot doesn't have allocation
                    alphas.append(0.0)
                
                times_ms.append((time.perf_counter() - start) * 1000)
            
            comparison_data.append({
                'controller': name,
                'mean_alpha': np.mean(alphas) if alphas[0] > 0 else "N/A",
                'mean_time_ms': np.mean(times_ms),
                'supports_rw': hasattr(ctrl, 'allocate_max_torque_in_direction'),
                'supports_desat': hasattr(ctrl, 'c_gain'),
            })
        
        PrettyOutput.subheader("Controller Comparison")
        
        headers = ["Controller", "Mean α", "Time (ms)", "RW Support", "Desat"]
        widths = [15, 10, 10, 12, 8]
        
        print("  ┌─" + "─┬─".join("─" * w for w in widths) + "─┐")
        print(PrettyOutput.table_row(headers, widths))
        print("  ├─" + "─┼─".join("─" * w for w in widths) + "─┤")
        
        for d in comparison_data:
            alpha_str = f"{d['mean_alpha']:.4f}" if isinstance(d['mean_alpha'], float) else d['mean_alpha']
            cols = [
                d['controller'],
                alpha_str,
                f"{d['mean_time_ms']:.3f}",
                "✓" if d['supports_rw'] else "✗",
                "✓" if d['supports_desat'] else "✗",
            ]
            print(PrettyOutput.table_row(cols, widths))
        
        print("  └─" + "─┴─".join("─" * w for w in widths) + "─┘")
        
        assert len(comparison_data) == 3


# =============================================================================
# TODO-DAA-3: CUBESAT NUMERICAL EXAMPLES
# =============================================================================

class TestCubeSatNumericalExamples:
    """
    TODO-DAA-3: Generate numerical examples for CubeSat configurations.
    """

    def test_cubesat_configurations(self):
        """Generate data for CubeSat configuration examples."""
        PrettyOutput.header("TODO-DAA-3: CubeSat Configuration Examples")
        
        os = create_orbital_state()
        results = []
        
        for name in CUBESAT_CONFIGS.keys():
            config = create_cubesat_config(name)
            params = CUBESAT_CONFIGS[name]
            
            est_sat = EstimatedSatellite(**config)
            controller = MTQ_w_RW_LP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
            
            # Compute achievable torque in multiple directions
            directions = [normalize(np.random.randn(3)) for _ in range(10)]
            alphas = []
            
            for d in directions:
                tau_des = d * 0.01
                _, _, alpha = controller.allocate_max_torque_in_direction(
                    tau_des=tau_des, b_body=os.B, est_sat=est_sat
                )
                alphas.append(alpha)
            
            # Compute WCDTA
            wcdta = np.min(alphas)
            
            results.append({
                'config': name,
                'mass': params['mass'],
                'n_rw': params['n_rw'],
                'mtq_torque': params['mtq_torque'],
                'wcdta': wcdta,
                'mean_alpha': np.mean(alphas),
            })
        
        PrettyOutput.subheader("CubeSat Configuration Summary")
        
        headers = ["Config", "Mass (kg)", "RWs", "MTQ (Am²)", "WCDTA", "Mean α"]
        widths = [12, 10, 6, 10, 8, 8]
        
        print("  ┌─" + "─┬─".join("─" * w for w in widths) + "─┐")
        print(PrettyOutput.table_row(headers, widths))
        print("  ├─" + "─┼─".join("─" * w for w in widths) + "─┤")
        
        for r in results:
            cols = [
                r['config'][:10],
                f"{r['mass']:.1f}",
                str(r['n_rw']),
                f"{r['mtq_torque']:.1f}",
                f"{r['wcdta']:.4f}",
                f"{r['mean_alpha']:.4f}",
            ]
            print(PrettyOutput.table_row(cols, widths))
        
        print("  └─" + "─┴─".join("─" * w for w in widths) + "─┘")
        
        assert len(results) == len(CUBESAT_CONFIGS)


# =============================================================================
# TODO-DAA-4: WCDTA COMPARISON TABLE
# =============================================================================

class TestWCDTAComparisonTable:
    """
    TODO-DAA-4: Generate WCDTA comparison table across configurations.
    """

    def test_wcdta_vs_b_field(self):
        """Compare WCDTA across B-field orientations."""
        PrettyOutput.header("TODO-DAA-4: WCDTA vs B-Field Comparison")
        
        B_orientations = {
            "Along X": normalize(np.array([1, 0, 0])) * 3e-5,
            "Along Y": normalize(np.array([0, 1, 0])) * 3e-5,
            "Along Z": normalize(np.array([0, 0, 1])) * 3e-5,
            "Diagonal": normalize(np.array([1, 1, 1])) * 3e-5,
        }
        
        configs_to_test = ["1U_Basic", "3U_Standard", "6U_HighPerf"]
        
        results = []
        
        for config_name in configs_to_test:
            config = create_cubesat_config(config_name)
            est_sat = EstimatedSatellite(**config)
            controller = MTQ_w_RW_LP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
            
            row = {'config': config_name}
            
            for b_name, B in B_orientations.items():
                # Sample directions
                directions = [normalize(np.random.randn(3)) for _ in range(20)]
                alphas = []
                
                for d in directions:
                    tau_des = d * 0.01
                    _, _, alpha = controller.allocate_max_torque_in_direction(
                        tau_des=tau_des, b_body=B, est_sat=est_sat
                    )
                    alphas.append(alpha)
                
                row[b_name] = np.min(alphas)
            
            results.append(row)
        
        PrettyOutput.subheader("WCDTA vs B-Field Orientation")
        
        headers = ["Config"] + list(B_orientations.keys())
        widths = [12] + [10] * len(B_orientations)
        
        print("  ┌─" + "─┬─".join("─" * w for w in widths) + "─┐")
        print(PrettyOutput.table_row(headers, widths))
        print("  ├─" + "─┼─".join("─" * w for w in widths) + "─┤")
        
        for r in results:
            cols = [r['config'][:10]] + [f"{r[k]:.4f}" for k in B_orientations.keys()]
            print(PrettyOutput.table_row(cols, widths))
        
        print("  └─" + "─┴─".join("─" * w for w in widths) + "─┘")
        
        assert len(results) == len(configs_to_test)


# =============================================================================
# TODO-COMP-1: DISTURBANCE COMPENSATION
# =============================================================================

class TestDisturbanceCompensation:
    """
    TODO-COMP-1: Generate disturbance compensation example data.
    """

    def test_disturbance_rejection_capability(self):
        """Test ability to reject typical disturbance torques."""
        PrettyOutput.header("TODO-COMP-1: Disturbance Compensation Examples")
        
        disturbances = {
            "Gravity Gradient": np.array([1e-6, 0.5e-6, 0.2e-6]),
            "Aero Drag": np.array([0.5e-6, 0.5e-6, 0]),
            "SRP": np.array([0.1e-6, 0.1e-6, 0.1e-6]),
            "Residual Dipole": np.array([0.3e-6, 0.3e-6, 0.3e-6]),
            "Combined Worst": np.array([2e-6, 1.5e-6, 1e-6]),
        }
        
        config = create_cubesat_config("3U_Standard")
        est_sat = EstimatedSatellite(**config)
        controller = MTQ_w_RW_LP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        os = create_orbital_state()
        
        results = []
        
        for name, tau_dist in disturbances.items():
            # Can we allocate enough torque to reject this?
            _, _, alpha = controller.allocate_max_torque_in_direction(
                tau_des=tau_dist,
                b_body=os.B,
                est_sat=est_sat
            )
            
            results.append({
                'disturbance': name,
                'magnitude_nNm': np.linalg.norm(tau_dist) * 1e9,
                'alpha': alpha,
                'can_reject': alpha >= 0.99,
            })
        
        PrettyOutput.subheader("Disturbance Rejection Capability")
        
        headers = ["Disturbance", "|τ| (nNm)", "α", "Rejectable"]
        widths = [18, 12, 8, 12]
        
        print("  ┌─" + "─┬─".join("─" * w for w in widths) + "─┐")
        print(PrettyOutput.table_row(headers, widths))
        print("  ├─" + "─┼─".join("─" * w for w in widths) + "─┤")
        
        for r in results:
            status = "✓ Yes" if r['can_reject'] else "✗ No"
            cols = [
                r['disturbance'],
                f"{r['magnitude_nNm']:.1f}",
                f"{r['alpha']:.4f}",
                status,
            ]
            print(PrettyOutput.table_row(cols, widths))
        
        print("  └─" + "─┴─".join("─" * w for w in widths) + "─┘")
        
        # All typical disturbances should be rejectable
        assert all(r['can_reject'] for r in results[:-1])  # Except worst-case


# =============================================================================
# TODO-SMALLSAT-1: 5-MINUTE DEMO
# =============================================================================

class TestFiveMinuteDemo:
    """
    TODO-SMALLSAT-1: Verify 5-minute demo capability.
    """

    def test_quick_setup_demo(self):
        """Demonstrate quick setup and simulation."""
        PrettyOutput.header("TODO-SMALLSAT-1: 5-Minute Demo")
        
        start_time = time.time()
        
        # Step 1: Create satellite (< 1 min)
        print("  Step 1: Create satellite configuration...")
        mtqs = [MTQ(axis=j, max_torque=0.5) for j in MathConstants.unitvecs]
        rws = [RW(axis=j, max_torque=0.01, J=0.001, h=0.0, h_max=0.05) 
               for j in MathConstants.unitvecs]
        mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
        
        sat = Satellite(mass=4.0, J_0=np.diagflat([0.1, 0.1, 0.1]), actuators=mtqs + rws)
        est_sat = EstimatedSatellite(
            mass=4.0, J_0=np.diagflat([0.1, 0.1, 0.1]),
            actuators=mtqs + rws, sensors=mtms, boresight=np.array([0, 0, 1])
        )
        print(f"    ✓ Satellite created ({time.time() - start_time:.2f}s)")
        
        # Step 2: Create controller (< 1 min)
        print("  Step 2: Create controller...")
        controller = MTQ_w_RW_LP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        print(f"    ✓ Controller created ({time.time() - start_time:.2f}s)")
        
        # Step 3: Create orbit (< 1 min)
        print("  Step 3: Set up orbit...")
        ephem = Ephemeris()
        os = Orbital_State(
            ephem=ephem, J2000=0.22,
            R=np.array([7000, 0, 0]), V=np.array([0, 7.5, 0]),
            B=np.array([2e-5, 1e-5, 3e-5])
        )
        print(f"    ✓ Orbit created ({time.time() - start_time:.2f}s)")
        
        # Step 4: Run quick simulation (< 2 min)
        print("  Step 4: Run simulation...")
        x = np.hstack([np.zeros(3), np.array([0, 0, 0, 1]), np.zeros(3)])
        for i in range(10):
            tau_des = np.array([0.001, 0.0005, 0])
            u_rw, u_mtq, alpha = controller.allocate_max_torque_in_direction(
                tau_des=tau_des, b_body=os.B, est_sat=est_sat
            )
            u = np.concatenate([u_mtq, u_rw])
            xdot = sat.dynamics_core(x, u, os)
            x = x + xdot * 0.5
            x[3:7] = normalize(x[3:7])
        print(f"    ✓ Simulation complete ({time.time() - start_time:.2f}s)")
        
        total_time = time.time() - start_time
        print(f"\n  Total setup + run time: {total_time:.2f}s")
        
        success = total_time < 60  # Should complete in under 1 minute
        PrettyOutput.subheader("Demo Result")
        print(f"  {'✓ PASS' if success else '✗ FAIL'}: Demo completed in {total_time:.1f}s (target: <60s)")
        
        assert success


# =============================================================================
# TODO-SMALLSAT-2: PRACTITIONER METRICS
# =============================================================================

class TestPractitionerMetrics:
    """
    TODO-SMALLSAT-2: Generate practitioner-relevant metrics.
    """

    def test_power_mass_metrics(self):
        """Generate power and mass proxy metrics."""
        PrettyOutput.header("TODO-SMALLSAT-2: Practitioner Metrics")
        
        # Rough estimates for actuator power/mass
        actuator_specs = {
            "MTQ": {"power_w": 0.5, "mass_kg": 0.05},
            "RW_small": {"power_w": 1.0, "mass_kg": 0.1},
            "RW_large": {"power_w": 3.0, "mass_kg": 0.3},
        }
        
        configs = {
            "MTQ-only": {"mtq": 3, "rw_small": 0, "rw_large": 0},
            "3U Standard": {"mtq": 3, "rw_small": 3, "rw_large": 0},
            "6U High-Perf": {"mtq": 3, "rw_small": 0, "rw_large": 4},
        }
        
        results = []
        
        for name, cfg in configs.items():
            total_power = (
                cfg["mtq"] * actuator_specs["MTQ"]["power_w"] +
                cfg["rw_small"] * actuator_specs["RW_small"]["power_w"] +
                cfg["rw_large"] * actuator_specs["RW_large"]["power_w"]
            )
            total_mass = (
                cfg["mtq"] * actuator_specs["MTQ"]["mass_kg"] +
                cfg["rw_small"] * actuator_specs["RW_small"]["mass_kg"] +
                cfg["rw_large"] * actuator_specs["RW_large"]["mass_kg"]
            )
            
            results.append({
                'config': name,
                'n_actuators': cfg["mtq"] + cfg["rw_small"] + cfg["rw_large"],
                'power_w': total_power,
                'mass_kg': total_mass,
            })
        
        PrettyOutput.subheader("Actuator System Metrics")
        
        headers = ["Configuration", "Actuators", "Power (W)", "Mass (kg)"]
        widths = [15, 10, 12, 12]
        
        print("  ┌─" + "─┬─".join("─" * w for w in widths) + "─┐")
        print(PrettyOutput.table_row(headers, widths))
        print("  ├─" + "─┼─".join("─" * w for w in widths) + "─┤")
        
        for r in results:
            cols = [
                r['config'],
                str(r['n_actuators']),
                f"{r['power_w']:.1f}",
                f"{r['mass_kg']:.2f}",
            ]
            print(PrettyOutput.table_row(cols, widths))
        
        print("  └─" + "─┴─".join("─" * w for w in widths) + "─┘")
        
        assert len(results) == len(configs)


# =============================================================================
# TODO-SMALLSAT-3: FAILURE MODES
# =============================================================================

class TestFailureModes:
    """
    TODO-SMALLSAT-3: Document failure modes data.
    """

    def test_failure_mode_enumeration(self):
        """Enumerate and characterize failure modes."""
        PrettyOutput.header("TODO-SMALLSAT-3: Failure Modes Documentation")
        
        failure_modes = [
            {
                'mode': "Single MTQ failure",
                'cause': "Coil open/short",
                'impact': "Reduced torque authority in one direction",
                'mitigation': "Redundant MTQ axis",
                'severity': "Low",
            },
            {
                'mode': "Single RW failure",
                'cause': "Motor/bearing failure",
                'impact': "Loss of momentum storage in one axis",
                'mitigation': "4th RW in pyramid config",
                'severity': "Medium",
            },
            {
                'mode': "MTM failure",
                'cause': "Sensor drift/failure",
                'impact': "Degraded B-field knowledge",
                'mitigation': "Redundant MTM, IGRF fallback",
                'severity': "Medium",
            },
            {
                'mode': "All RW saturation",
                'cause': "Persistent disturbance",
                'impact': "Loss of pointing, need desaturation",
                'mitigation': "Desaturation scheduling",
                'severity': "High",
            },
            {
                'mode': "Power loss",
                'cause': "Battery/EPS failure",
                'impact': "Total ADCS loss",
                'mitigation': "Safe mode, tumble recovery",
                'severity': "Critical",
            },
        ]
        
        PrettyOutput.subheader("Failure Mode Summary")
        
        headers = ["Failure Mode", "Severity", "Mitigation"]
        widths = [20, 10, 25]
        
        print("  ┌─" + "─┬─".join("─" * w for w in widths) + "─┐")
        print(PrettyOutput.table_row(headers, widths))
        print("  ├─" + "─┼─".join("─" * w for w in widths) + "─┤")
        
        for fm in failure_modes:
            cols = [
                fm['mode'][:18],
                fm['severity'],
                fm['mitigation'][:23],
            ]
            print(PrettyOutput.table_row(cols, widths))
        
        print("  └─" + "─┴─".join("─" * w for w in widths) + "─┘")
        
        print("\n  Detailed Failure Mode Analysis:")
        for i, fm in enumerate(failure_modes, 1):
            print(f"\n  {i}. {fm['mode']}")
            print(f"     Cause: {fm['cause']}")
            print(f"     Impact: {fm['impact']}")
        
        assert len(failure_modes) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
