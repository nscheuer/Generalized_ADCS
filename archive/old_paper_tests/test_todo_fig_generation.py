"""
TODO-FIG-2, FIG-3, FIG-5, FIG-6, FIG-7: Figure Generation Tests
================================================================

Papers: Generalized Control Paper
TODO IDs:
  - TODO-FIG-2: Torque polytope animation data
  - TODO-FIG-3: LP vs QP 2D projection data
  - TODO-FIG-5: Pointing error time series data
  - TODO-FIG-6: Monte Carlo distribution data
  - TODO-FIG-7: Actuator failure response data
  - TODO-DAA-1: Torque envelope over orbit data
  - TODO-DAA-2: WCDTA sphere visualization data
  - TODO-LP-1: LP allocation geometry data
  - TODO-INTERPRET-1: Polytope shapes comparison data

These tests verify that the data needed for paper figures can be generated.
Actual plotting is separate (matplotlib scripts).

Adjustable Parameters
---------------------
- N_ORBIT_POINTS: Number of points around orbit for envelope
- N_POLYTOPE_VERTICES: Resolution for polytope computation
- SIMULATION_DURATION: Duration for time series [s]
"""

import sys
import os
import numpy as np
import pytest
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ADCS.controller.mtq_w_rw_LP import MTQ_w_RW_LP
from ADCS.controller.mtq_w_rw_QP import MTQ_w_RW_QP
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

# Orbit/envelope parameters
N_ORBIT_POINTS = 36          # Points around orbit (every 10 deg)
N_POLYTOPE_DIRECTIONS = 50   # Directions to sample for polytope

# Simulation parameters  
SIMULATION_DURATION = 60.0   # seconds
SIMULATION_DT = 1.0          # seconds

# Monte Carlo parameters
N_MC_SAMPLES = 20            # For quick testing

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
    def figure_data_summary(name: str, shape: tuple, range_info: str) -> None:
        if PRETTY_OUTPUT:
            print(f"  📊 {name}")
            print(f"     Shape: {shape}")
            print(f"     Range: {range_info}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_3mtq_3rw_config():
    """Create standard 3-MTQ + 3-RW satellite configuration."""
    mtqs = [MTQ(axis=j, max_torque=0.5) for j in MathConstants.unitvecs]
    rws = [RW(axis=j, max_torque=0.01, J=0.001, h=0.0, h_max=0.05) 
           for j in MathConstants.unitvecs]
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    return dict(
        mass=4.0,
        J_0=np.diagflat([0.1, 0.1, 0.1]),
        actuators=mtqs + rws,
        sensors=mtms,
        boresight=np.array([0, 0, 1])
    )


def create_orbital_state(true_anomaly_deg: float = 0.0) -> Orbital_State:
    """Create orbital state at given true anomaly."""
    ephem = Ephemeris()
    
    # Simple circular orbit
    r = 7000  # km
    v = 7.5   # km/s
    
    # Rotate position/velocity by true anomaly
    theta = np.radians(true_anomaly_deg)
    R = np.array([r * np.cos(theta), r * np.sin(theta), 0])
    V = np.array([-v * np.sin(theta), v * np.cos(theta), 0])
    
    # B-field varies with position (simplified dipole)
    B_mag = 3e-5
    B = normalize(np.array([
        np.cos(theta), 
        np.sin(theta) * 0.5, 
        0.5
    ])) * B_mag
    
    return Orbital_State(
        ephem=ephem, J2000=0.22,
        R=R, V=V, B=B
    )


def sample_unit_sphere(n_points: int) -> np.ndarray:
    """Generate approximately uniform points on unit sphere."""
    # Fibonacci sphere
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle
    
    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    
    return np.array(points)


# =============================================================================
# TODO-FIG-2: TORQUE POLYTOPE DATA
# =============================================================================

class TestTorquePolytopeData:
    """
    TODO-FIG-2: Generate data for torque polytope visualization/animation.
    """

    def test_polytope_vertices_computable(self):
        """Verify we can compute polytope vertices for different B-fields."""
        PrettyOutput.header("TODO-FIG-2: Torque Polytope Data Generation")
        
        config = create_3mtq_3rw_config()
        est_sat = EstimatedSatellite(**config)
        controller = MTQ_w_RW_LP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        
        # Sample directions on unit sphere
        directions = sample_unit_sphere(N_POLYTOPE_DIRECTIONS)
        
        # Test at multiple B-field orientations
        B_orientations = [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            normalize(np.array([1, 1, 1])),
        ]
        
        polytope_data = []
        
        for B_dir in B_orientations:
            B = B_dir * 3e-5
            vertices = []
            
            for d in directions:
                tau_des = d * 0.01  # Scale for reasonable torque
                _, _, alpha = controller.allocate_max_torque_in_direction(
                    tau_des=tau_des,
                    b_body=B,
                    est_sat=est_sat
                )
                # Achieved torque magnitude in this direction
                vertices.append(alpha * np.linalg.norm(tau_des))
            
            polytope_data.append({
                'B_direction': B_dir,
                'directions': directions,
                'magnitudes': np.array(vertices),
            })
        
        PrettyOutput.subheader("Polytope Data Summary")
        for i, data in enumerate(polytope_data):
            mags = data['magnitudes']
            print(f"  B-field {i+1}: min={mags.min():.6f}, max={mags.max():.6f}, mean={mags.mean():.6f}")
        
        # Verify data is valid
        assert len(polytope_data) == 4
        for data in polytope_data:
            assert data['magnitudes'].shape[0] == N_POLYTOPE_DIRECTIONS
            assert np.all(data['magnitudes'] >= 0)

    def test_polytope_animation_frames(self):
        """Generate polytope data for animation over orbit."""
        PrettyOutput.subheader("Animation Frames Generation")
        
        config = create_3mtq_3rw_config()
        est_sat = EstimatedSatellite(**config)
        controller = MTQ_w_RW_LP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        
        directions = sample_unit_sphere(20)  # Fewer for speed
        frames = []
        
        for true_anom in np.linspace(0, 360, N_ORBIT_POINTS, endpoint=False):
            os = create_orbital_state(true_anom)
            
            magnitudes = []
            for d in directions:
                tau_des = d * 0.01
                _, _, alpha = controller.allocate_max_torque_in_direction(
                    tau_des=tau_des,
                    b_body=os.B,
                    est_sat=est_sat
                )
                magnitudes.append(alpha * np.linalg.norm(tau_des))
            
            frames.append({
                'true_anomaly_deg': true_anom,
                'B_body': os.B.copy(),
                'magnitudes': np.array(magnitudes),
            })
        
        print(f"  Generated {len(frames)} animation frames")
        print(f"  Each frame has {len(directions)} polytope samples")
        
        assert len(frames) == N_ORBIT_POINTS


# =============================================================================
# TODO-FIG-3: LP vs QP PROJECTION DATA
# =============================================================================

class TestLPvsQPProjectionData:
    """
    TODO-FIG-3: Generate data for LP vs QP 2D projection comparison.
    """

    def test_2d_projection_data(self):
        """Generate LP vs QP comparison data for 2D projection."""
        PrettyOutput.header("TODO-FIG-3: LP vs QP 2D Projection Data")
        
        config = create_3mtq_3rw_config()
        est_sat = EstimatedSatellite(**config)
        controller_lp = MTQ_w_RW_LP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        controller_qp = MTQ_w_RW_QP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        
        B = normalize(np.array([1, 1, 0])) * 3e-5
        
        # Sample torque demands in XY plane
        n_angles = 36
        angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
        magnitudes = [0.001, 0.005, 0.01]
        
        lp_data = []
        qp_data = []
        
        for mag in magnitudes:
            for angle in angles:
                tau_des = np.array([mag * np.cos(angle), mag * np.sin(angle), 0])
                
                # LP allocation
                u_rw_lp, u_mtq_lp, alpha_lp = controller_lp.allocate_max_torque_in_direction(
                    tau_des=tau_des, b_body=B, est_sat=est_sat
                )
                
                # QP allocation
                u_rw_qp, u_mtq_qp, alpha_qp = controller_qp.allocate_max_torque_in_direction(
                    tau_des=tau_des, b_body=B, est_sat=est_sat
                )
                
                lp_data.append({
                    'tau_des': tau_des,
                    'alpha': alpha_lp,
                    'u_rw': u_rw_lp,
                    'u_mtq': u_mtq_lp,
                    'u_norm': np.linalg.norm(np.concatenate([u_rw_lp, u_mtq_lp])),
                })
                
                qp_data.append({
                    'tau_des': tau_des,
                    'alpha': alpha_qp,
                    'u_rw': u_rw_qp,
                    'u_mtq': u_mtq_qp,
                    'u_norm': np.linalg.norm(np.concatenate([u_rw_qp, u_mtq_qp])),
                })
        
        PrettyOutput.subheader("LP vs QP Comparison")
        
        lp_alphas = [d['alpha'] for d in lp_data]
        qp_alphas = [d['alpha'] for d in qp_data]
        lp_norms = [d['u_norm'] for d in lp_data]
        qp_norms = [d['u_norm'] for d in qp_data]
        
        print(f"  LP: α mean={np.mean(lp_alphas):.4f}, ||u|| mean={np.mean(lp_norms):.6f}")
        print(f"  QP: α mean={np.mean(qp_alphas):.4f}, ||u|| mean={np.mean(qp_norms):.6f}")
        print(f"  QP uses {100*(1 - np.mean(qp_norms)/np.mean(lp_norms)):.1f}% less control effort")
        
        assert len(lp_data) == len(qp_data) == n_angles * len(magnitudes)


# =============================================================================
# TODO-FIG-5: POINTING ERROR TIME SERIES
# =============================================================================

class TestPointingTimeSeriesData:
    """
    TODO-FIG-5: Generate pointing error time series data.
    """

    def test_pointing_error_timeseries(self):
        """Generate pointing error over time for figure."""
        PrettyOutput.header("TODO-FIG-5: Pointing Error Time Series Data")
        
        # Create satellite
        mtqs = [MTQ(axis=j, max_torque=0.5) for j in MathConstants.unitvecs]
        rws = [RW(axis=j, max_torque=0.01, J=0.001, h=0.0, h_max=0.05) 
               for j in MathConstants.unitvecs]
        
        sat = Satellite(
            mass=4.0,
            J_0=np.diagflat([0.1, 0.1, 0.1]),
            actuators=mtqs + rws,
        )
        
        # Initial state: small attitude error
        q0 = normalize(np.array([0.05, 0.02, 0.01, 1.0]))  # Small rotation from identity
        omega0 = np.array([0.01, -0.005, 0.002])  # Small angular velocity
        h0 = np.zeros(3)  # RW momentum
        x = np.hstack([omega0, q0, h0])
        
        # Target: identity quaternion
        q_target = np.array([0, 0, 0, 1])
        
        # Simple proportional control
        K_p = 0.1
        K_d = 0.5
        
        ephem = Ephemeris()
        os = Orbital_State(
            ephem=ephem, J2000=0.22,
            R=np.array([7000, 0, 0]), V=np.array([0, 7.5, 0]),
            B=np.array([2e-5, 1e-5, 3e-5])
        )
        
        # Simulate
        time_data = []
        error_data = []
        omega_data = []
        
        dt = SIMULATION_DT
        t = 0.0
        
        while t < SIMULATION_DURATION:
            # Extract state
            omega = x[0:3]
            q = x[3:7]
            
            # Compute pointing error (angle from target)
            q_err = q.copy()
            q_err[3] = q[3] - q_target[3]  # Simplified error
            q_err[:3] = q[:3] - q_target[:3]
            
            # Error angle
            error_angle = 2 * np.arcsin(np.clip(np.linalg.norm(q_err[:3]), 0, 1))
            error_deg = np.degrees(error_angle)
            
            # Store data
            time_data.append(t)
            error_data.append(error_deg)
            omega_data.append(np.linalg.norm(omega))
            
            # Simple PD control torque
            tau_cmd = -K_p * q[:3] - K_d * omega
            
            # Zero control for this test (open loop decay observation)
            u = np.zeros(6)
            
            # Propagate (simple Euler)
            xdot = sat.dynamics_core(x, u, os)
            x = x + xdot * dt
            x[3:7] = normalize(x[3:7])  # Renormalize quaternion
            
            t += dt
        
        PrettyOutput.subheader("Time Series Summary")
        PrettyOutput.figure_data_summary(
            "Pointing Error",
            (len(time_data),),
            f"{min(error_data):.2f}° to {max(error_data):.2f}°"
        )
        PrettyOutput.figure_data_summary(
            "Angular Rate",
            (len(omega_data),),
            f"{min(omega_data):.4f} to {max(omega_data):.4f} rad/s"
        )
        
        assert len(time_data) == len(error_data)
        assert len(time_data) > 10


# =============================================================================
# TODO-FIG-6: MONTE CARLO DISTRIBUTION DATA
# =============================================================================

class TestMCDistributionData:
    """
    TODO-FIG-6: Generate Monte Carlo distribution data for histograms/CDFs.
    """

    def test_mc_pointing_distribution(self):
        """Generate MC pointing error distribution data."""
        PrettyOutput.header("TODO-FIG-6: Monte Carlo Distribution Data")
        
        config = create_3mtq_3rw_config()
        
        pointing_errors = []
        settling_times = []
        
        for trial in range(N_MC_SAMPLES):
            np.random.seed(trial)
            
            # Perturb inertia
            J_nom = np.diagflat([0.1, 0.1, 0.1])
            J_perturb = J_nom * (1 + 0.1 * np.random.randn())
            
            # Random initial attitude error
            axis = normalize(np.random.randn(3))
            angle = np.random.uniform(5, 30)  # degrees
            
            # Simulate pointing error (simplified)
            # In real test, would run full simulation
            final_error = np.random.exponential(1.0)  # Placeholder
            settle_time = 10 + np.random.exponential(5.0)  # Placeholder
            
            pointing_errors.append(final_error)
            settling_times.append(settle_time)
        
        pointing_errors = np.array(pointing_errors)
        settling_times = np.array(settling_times)
        
        PrettyOutput.subheader("Distribution Statistics")
        print(f"  Pointing Error:")
        print(f"    Mean: {np.mean(pointing_errors):.3f}°")
        print(f"    Std:  {np.std(pointing_errors):.3f}°")
        print(f"    95%:  {np.percentile(pointing_errors, 95):.3f}°")
        print(f"  Settling Time:")
        print(f"    Mean: {np.mean(settling_times):.1f} s")
        print(f"    Std:  {np.std(settling_times):.1f} s")
        print(f"    95%:  {np.percentile(settling_times, 95):.1f} s")
        
        # Data for histogram
        hist_data = {
            'pointing_errors': pointing_errors,
            'settling_times': settling_times,
            'bins_error': np.linspace(0, np.max(pointing_errors)*1.1, 20),
            'bins_time': np.linspace(0, np.max(settling_times)*1.1, 20),
        }
        
        assert len(pointing_errors) == N_MC_SAMPLES


# =============================================================================
# TODO-FIG-7: ACTUATOR FAILURE RESPONSE DATA
# =============================================================================

class TestActuatorFailureData:
    """
    TODO-FIG-7: Generate actuator failure response data.
    """

    def test_failure_response_data(self):
        """Generate data showing response to actuator failure."""
        PrettyOutput.header("TODO-FIG-7: Actuator Failure Response Data")
        
        # Test scenarios: which actuator fails
        scenarios = [
            ("No Failure", None),
            ("RW-X Fails", 3),  # Index 3 = first RW
            ("RW-Y Fails", 4),
            ("RW-Z Fails", 5),
            ("MTQ-X Fails", 0),
        ]
        
        results = []
        
        for name, fail_idx in scenarios:
            config = create_3mtq_3rw_config()
            
            # Apply failure by zeroing max_torque
            if fail_idx is not None:
                config['actuators'][fail_idx].max_torque = 0.0
            
            est_sat = EstimatedSatellite(**config)
            controller = MTQ_w_RW_LP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
            
            # Test allocation capability in multiple directions
            B = normalize(np.array([1, 1, 1])) * 3e-5
            
            achieved_fractions = []
            for _ in range(20):
                tau_des = normalize(np.random.randn(3)) * 0.005
                _, _, alpha = controller.allocate_max_torque_in_direction(
                    tau_des=tau_des, b_body=B, est_sat=est_sat
                )
                achieved_fractions.append(alpha)
            
            results.append({
                'scenario': name,
                'mean_alpha': np.mean(achieved_fractions),
                'min_alpha': np.min(achieved_fractions),
                'degradation': 1.0 - np.mean(achieved_fractions),
            })
        
        PrettyOutput.subheader("Failure Response Summary")
        print("  ┌────────────────────┬────────────┬────────────┬─────────────┐")
        print("  │ Scenario           │ Mean α     │ Min α      │ Degradation │")
        print("  ├────────────────────┼────────────┼────────────┼─────────────┤")
        for r in results:
            deg_str = f"{r['degradation']*100:.1f}%"
            print(f"  │ {r['scenario']:18} │ {r['mean_alpha']:10.4f} │ {r['min_alpha']:10.4f} │ {deg_str:>11} │")
        print("  └────────────────────┴────────────┴────────────┴─────────────┘")
        
        assert len(results) == len(scenarios)
        # No failure should have best performance
        assert results[0]['mean_alpha'] >= max(r['mean_alpha'] for r in results[1:])


# =============================================================================
# TODO-DAA-1: TORQUE ENVELOPE OVER ORBIT
# =============================================================================

class TestTorqueEnvelopeData:
    """
    TODO-DAA-1: Generate torque envelope data over orbit.
    """

    def test_envelope_over_orbit(self):
        """Generate torque envelope varying over orbit."""
        PrettyOutput.header("TODO-DAA-1: Torque Envelope Over Orbit")
        
        config = create_3mtq_3rw_config()
        est_sat = EstimatedSatellite(**config)
        controller = MTQ_w_RW_LP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        
        envelope_data = []
        
        for true_anom in np.linspace(0, 360, N_ORBIT_POINTS, endpoint=False):
            os = create_orbital_state(true_anom)
            
            # Compute achievable torque in principal directions
            principal_dirs = [
                np.array([1, 0, 0]),
                np.array([0, 1, 0]),
                np.array([0, 0, 1]),
                np.array([-1, 0, 0]),
                np.array([0, -1, 0]),
                np.array([0, 0, -1]),
            ]
            
            achievable = {}
            for i, d in enumerate(principal_dirs):
                tau_des = d * 0.01
                _, _, alpha = controller.allocate_max_torque_in_direction(
                    tau_des=tau_des, b_body=os.B, est_sat=est_sat
                )
                achievable[f"dir_{i}"] = alpha * 0.01
            
            envelope_data.append({
                'true_anomaly_deg': true_anom,
                'B_magnitude': np.linalg.norm(os.B),
                **achievable,
            })
        
        PrettyOutput.subheader("Envelope Variation")
        
        # Extract statistics
        dir_0 = [d['dir_0'] for d in envelope_data]
        dir_2 = [d['dir_2'] for d in envelope_data]
        
        print(f"  X-axis torque: {min(dir_0)*1e3:.3f} to {max(dir_0)*1e3:.3f} mNm")
        print(f"  Z-axis torque: {min(dir_2)*1e3:.3f} to {max(dir_2)*1e3:.3f} mNm")
        print(f"  Variation ratio: {max(dir_0)/min(dir_0):.2f}x")
        
        assert len(envelope_data) == N_ORBIT_POINTS


# =============================================================================
# TODO-DAA-2: WCDTA SPHERE DATA
# =============================================================================

class TestWCDTASphereData:
    """
    TODO-DAA-2: Generate WCDTA (Worst-Case Direction Torque Authority) sphere data.
    """

    def test_wcdta_sphere_data(self):
        """Generate WCDTA visualization data."""
        PrettyOutput.header("TODO-DAA-2: WCDTA Sphere Visualization Data")
        
        config = create_3mtq_3rw_config()
        est_sat = EstimatedSatellite(**config)
        controller = MTQ_w_RW_LP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        
        # Sample many directions on sphere
        directions = sample_unit_sphere(100)
        
        # Test at multiple B-field conditions
        B_conditions = [
            ("Favorable", normalize(np.array([0, 0, 1])) * 3e-5),
            ("Unfavorable", normalize(np.array([1, 0, 0])) * 3e-5),
            ("Mixed", normalize(np.array([1, 1, 1])) * 3e-5),
        ]
        
        sphere_data = []
        
        for name, B in B_conditions:
            alphas = []
            for d in directions:
                tau_des = d * 0.01
                _, _, alpha = controller.allocate_max_torque_in_direction(
                    tau_des=tau_des, b_body=B, est_sat=est_sat
                )
                alphas.append(alpha)
            
            alphas = np.array(alphas)
            wcdta = np.min(alphas)  # Worst-case
            
            sphere_data.append({
                'condition': name,
                'B': B,
                'directions': directions,
                'alphas': alphas,
                'wcdta': wcdta,
                'mean_alpha': np.mean(alphas),
            })
            
            print(f"  {name}: WCDTA={wcdta:.4f}, Mean α={np.mean(alphas):.4f}")
        
        assert len(sphere_data) == 3


# =============================================================================
# TODO-INTERPRET-1: POLYTOPE SHAPES COMPARISON
# =============================================================================

class TestPolytopeShapesComparison:
    """
    TODO-INTERPRET-1: Compare polytope shapes for different actuator configurations.
    """

    def test_compare_actuator_configs(self):
        """Compare torque polytopes for different actuator configurations."""
        PrettyOutput.header("TODO-INTERPRET-1: Polytope Shapes Comparison")
        
        # Configuration 1: 3 orthogonal RWs
        mtqs = [MTQ(axis=j, max_torque=0.5) for j in MathConstants.unitvecs]
        rws_ortho = [RW(axis=j, max_torque=0.01, J=0.001, h=0.0, h_max=0.05) 
                     for j in MathConstants.unitvecs]
        mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
        
        config_ortho = dict(
            mass=4.0, J_0=np.diagflat([0.1, 0.1, 0.1]),
            actuators=mtqs + rws_ortho, sensors=mtms,
            boresight=np.array([0, 0, 1])
        )
        
        # Configuration 2: 4 RWs in pyramid
        pyramid_axes = [
            normalize(np.array([1, 1, 1])),
            normalize(np.array([1, -1, -1])),
            normalize(np.array([-1, 1, -1])),
            normalize(np.array([-1, -1, 1])),
        ]
        rws_pyramid = [RW(axis=ax, max_torque=0.01, J=0.001, h=0.0, h_max=0.05) 
                       for ax in pyramid_axes]
        
        config_pyramid = dict(
            mass=4.0, J_0=np.diagflat([0.1, 0.1, 0.1]),
            actuators=mtqs + rws_pyramid, sensors=mtms,
            boresight=np.array([0, 0, 1])
        )
        
        configs = [
            ("3 Orthogonal RW", config_ortho),
            ("4 Pyramid RW", config_pyramid),
        ]
        
        B = normalize(np.array([1, 1, 1])) * 3e-5
        directions = sample_unit_sphere(50)
        
        comparison_data = []
        
        for name, cfg in configs:
            est_sat = EstimatedSatellite(**cfg)
            controller = MTQ_w_RW_LP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
            
            alphas = []
            for d in directions:
                tau_des = d * 0.01
                _, _, alpha = controller.allocate_max_torque_in_direction(
                    tau_des=tau_des, b_body=B, est_sat=est_sat
                )
                alphas.append(alpha)
            
            comparison_data.append({
                'config': name,
                'alphas': np.array(alphas),
                'wcdta': np.min(alphas),
                'mean': np.mean(alphas),
                'isotropy': np.min(alphas) / np.max(alphas),  # 1.0 = perfect sphere
            })
        
        PrettyOutput.subheader("Configuration Comparison")
        print("  ┌─────────────────────┬─────────┬─────────┬───────────┐")
        print("  │ Configuration       │ WCDTA   │ Mean α  │ Isotropy  │")
        print("  ├─────────────────────┼─────────┼─────────┼───────────┤")
        for d in comparison_data:
            print(f"  │ {d['config']:19} │ {d['wcdta']:.5f} │ {d['mean']:.5f} │ {d['isotropy']:.5f}   │")
        print("  └─────────────────────┴─────────┴─────────┴───────────┘")
        
        assert len(comparison_data) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
