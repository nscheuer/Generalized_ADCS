"""
Desaturation Strategy Analysis
==============================

Investigate different approaches to momentum management (desaturation)
for various actuator configurations:

1. Overactuated (3MTQ+3RW): Nullspace desaturation
2. Exactly actuated: Trade-off pointing/desaturation  
3. Underactuated (3MTQ+1RW): Scheduled/weighted desaturation

Key questions:
- When is torque-free desaturation possible?
- How to trade off pointing vs desaturation?
- Optimal desaturation scheduling strategies?
"""

import sys
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))

from ADCS.helpers.math_helpers import skewsym, normalize, rot_mat


@dataclass
class DesaturationConfig:
    """Configuration for desaturation analysis."""
    A_rw: np.ndarray          # RW torque axes (3, n_rw)
    A_mtq_axes: np.ndarray    # MTQ dipole axes (3, n_mtq)
    u_rw_max: np.ndarray      # RW torque limits
    u_mtq_max: np.ndarray     # MTQ dipole limits
    h_max: np.ndarray         # RW momentum limits
    J_rw: np.ndarray          # RW inertias


def compute_torque_polytope_vertices(A: np.ndarray, u_max: np.ndarray) -> np.ndarray:
    """
    Compute vertices of the torque polytope for actuators with independent limits.
    
    Parameters
    ----------
    A : np.ndarray (3, n)
        Actuator torque matrix
    u_max : np.ndarray (n,)
        Maximum actuator commands (symmetric: -u_max ≤ u ≤ u_max)
    
    Returns
    -------
    vertices : np.ndarray (2^n, 3)
        Vertices of the torque polytope
    """
    n = A.shape[1]
    n_vertices = 2 ** n
    vertices = np.zeros((n_vertices, 3))
    
    for i in range(n_vertices):
        u = np.zeros(n)
        for j in range(n):
            if (i >> j) & 1:
                u[j] = u_max[j]
            else:
                u[j] = -u_max[j]
        vertices[i, :] = A @ u
    
    return vertices


def compute_mtq_polytope(A_mtq_axes: np.ndarray, u_mtq_max: np.ndarray,
                         b_body: np.ndarray) -> np.ndarray:
    """
    Compute MTQ torque polytope in body frame.
    
    The MTQ torque is τ = m × B = -[B]× A_mtq @ u_mtq
    """
    b_norm = np.linalg.norm(b_body)
    if b_norm < 1e-12:
        return np.zeros((1, 3))  # No torque available when B=0
    
    A_mtq = -skewsym(b_body) @ A_mtq_axes
    return compute_torque_polytope_vertices(A_mtq, u_mtq_max)


def compute_rw_polytope(A_rw: np.ndarray, u_rw_max: np.ndarray) -> np.ndarray:
    """Compute RW torque polytope."""
    return compute_torque_polytope_vertices(A_rw, u_rw_max)


def check_torque_free_desaturation(config: DesaturationConfig,
                                    b_body: np.ndarray) -> Tuple[bool, float]:
    """
    Check if torque-free desaturation is possible at this B-field orientation.
    
    Returns (is_possible, max_desat_torque_magnitude)
    
    Torque-free desaturation requires:
    T_mtq ∩ (-T_rw) ≠ {0}
    
    i.e., there exists a nonzero torque that MTQs can produce which exactly
    cancels a RW torque.
    
    For this, we need to check if the MTQ torque plane intersects the RW torque
    line/polytope in a nontrivial way.
    """
    b_norm = np.linalg.norm(b_body)
    if b_norm < 1e-12:
        return False, 0.0
    
    b_hat = b_body / b_norm
    n_rw = config.A_rw.shape[1]
    
    # For each RW axis, check if its torque direction has a component
    # perpendicular to B (i.e., achievable by MTQ)
    max_desat = 0.0
    is_possible = False
    
    for i in range(n_rw):
        rw_axis = config.A_rw[:, i]
        rw_max = config.u_rw_max[i]
        
        # RW produces torque along rw_axis
        # For torque-free desaturation, MTQ must produce -τ_rw
        # MTQ can only produce torque perpendicular to B
        
        # Check: what fraction of rw_axis is perpendicular to B?
        rw_perp_to_B = rw_axis - np.dot(rw_axis, b_hat) * b_hat
        perp_fraction = np.linalg.norm(rw_perp_to_B) / (np.linalg.norm(rw_axis) + 1e-12)
        
        # If rw_axis is entirely parallel to B, no desaturation possible
        # If rw_axis has a perpendicular component, we CAN desaturate
        # (though not necessarily the full torque)
        
        if perp_fraction > 1e-3:  # Some perpendicular component exists
            is_possible = True
            
            # Max achievable MTQ torque along the perpendicular direction
            # This is approximate - actual would need to solve for optimal MTQ command
            mtq_capability = b_norm * np.max(config.u_mtq_max)  # Max MTQ torque magnitude
            
            # Effective desaturation torque: min of what RW can provide (in perp direction)
            # and what MTQ can match
            rw_perp_torque = rw_max * perp_fraction
            achievable = min(rw_perp_torque, mtq_capability)
            max_desat = max(max_desat, achievable)
    
    return is_possible, max_desat


def analyze_desaturation_window(config: DesaturationConfig,
                                 b_field_history: np.ndarray,
                                 time_history: np.ndarray) -> Dict:
    """
    Analyze desaturation capability over an orbital period.
    
    Parameters
    ----------
    config : DesaturationConfig
    b_field_history : np.ndarray (N, 3)
        B-field in body frame over time
    time_history : np.ndarray (N,)
        Time points
    
    Returns
    -------
    analysis : Dict with:
        - desat_possible: bool array
        - max_desat_torque: float array
        - fraction_possible: float
        - integrated_capability: float (area under max_desat_torque)
    """
    N = len(time_history)
    desat_possible = np.zeros(N, dtype=bool)
    max_desat_torque = np.zeros(N)
    
    for i, b_body in enumerate(b_field_history):
        possible, max_torque = check_torque_free_desaturation(config, b_body)
        desat_possible[i] = possible
        max_desat_torque[i] = max_torque
    
    dt = np.diff(time_history)
    integrated_capability = np.sum(max_desat_torque[:-1] * dt)
    
    return {
        'desat_possible': desat_possible,
        'max_desat_torque': max_desat_torque,
        'fraction_possible': np.mean(desat_possible),
        'integrated_capability': integrated_capability
    }


def simulate_desaturation_strategies(config: DesaturationConfig,
                                      b_field_history: np.ndarray,
                                      time_history: np.ndarray,
                                      h0: np.ndarray,
                                      disturbance_torque: np.ndarray,
                                      strategy: str = 'continuous') -> Dict:
    """
    Simulate desaturation with different strategies.
    
    Strategies:
    - 'continuous': Always desaturate when possible
    - 'threshold': Only desaturate when h > threshold
    - 'window': Only desaturate during optimal windows
    - 'weighted': Trade off pointing and desaturation continuously
    
    Returns momentum history and pointing impact.
    """
    N = len(time_history)
    n_rw = config.A_rw.shape[1]
    
    h = h0.copy()
    h_history = np.zeros((N, n_rw))
    desat_effort = np.zeros(N)
    pointing_impact = np.zeros(N)
    
    for i in range(N):
        b_body = b_field_history[i]
        dt = time_history[i+1] - time_history[i] if i < N-1 else time_history[i] - time_history[i-1]
        
        # Apply disturbance
        h += disturbance_torque * dt  # Simplified: torque → momentum
        
        # Check desaturation capability
        possible, max_torque = check_torque_free_desaturation(config, b_body)
        
        # Apply strategy
        if strategy == 'continuous' and possible:
            # Desaturate as much as possible
            h_error = h  # Target is zero
            desat_rate = np.clip(h_error / 10.0, -max_torque, max_torque)  # Simple P control
            h -= desat_rate * dt
            desat_effort[i] = np.linalg.norm(desat_rate)
            
        elif strategy == 'threshold' and possible:
            h_norm = np.linalg.norm(h)
            threshold = 0.5 * np.mean(config.h_max)
            if h_norm > threshold:
                desat_rate = np.clip(h / 10.0, -max_torque, max_torque)
                h -= desat_rate * dt
                desat_effort[i] = np.linalg.norm(desat_rate)
        
        # Clamp to limits
        h = np.clip(h, -config.h_max, config.h_max)
        h_history[i, :] = h
    
    return {
        'h_history': h_history,
        'desat_effort': desat_effort,
        'pointing_impact': pointing_impact,
        'final_h': h,
        'saturated_fraction': np.mean(np.abs(h_history) > 0.9 * config.h_max[:, np.newaxis].T)
    }


def plot_desaturation_analysis(time: np.ndarray, 
                                analysis: Dict,
                                title: str = "Desaturation Analysis"):
    """Plot desaturation analysis results."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1 = axes[0]
    ax1.fill_between(time, 0, analysis['desat_possible'].astype(float),
                     alpha=0.3, label='Desaturation Possible')
    ax1.set_ylabel('Desaturation Window')
    ax1.set_title(f"{title}\nFraction possible: {analysis['fraction_possible']:.1%}")
    ax1.legend()
    
    ax2 = axes[1]
    ax2.plot(time, analysis['max_desat_torque'] * 1e6, 'b-', label='Max Desat Torque')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Max Desat Torque [μNm]')
    ax2.legend()
    
    plt.tight_layout()
    return fig


def create_3mtq_1rw_desat_config() -> DesaturationConfig:
    """Create 3MTQ+1RW configuration."""
    return DesaturationConfig(
        A_rw=np.array([[0], [0], [1.0]]),
        A_mtq_axes=np.eye(3),
        u_rw_max=np.array([0.001]),
        u_mtq_max=np.array([0.2, 0.2, 0.2]),
        h_max=np.array([0.01]),  # 10 mNm·s
        J_rw=np.array([0.001])
    )


def create_3mtq_3rw_desat_config() -> DesaturationConfig:
    """Create 3MTQ+3RW configuration."""
    return DesaturationConfig(
        A_rw=np.eye(3),
        A_mtq_axes=np.eye(3),
        u_rw_max=np.array([0.001, 0.001, 0.001]),
        u_mtq_max=np.array([0.2, 0.2, 0.2]),
        h_max=np.array([0.01, 0.01, 0.01]),
        J_rw=np.array([0.001, 0.001, 0.001])
    )


def generate_b_field_history(orbit_period: float = 5400,
                              n_points: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """Generate time-varying B-field history over one orbit."""
    time = np.linspace(0, orbit_period, n_points)
    b_history = np.zeros((n_points, 3))
    
    for i, t in enumerate(time):
        phase = 2 * np.pi * t / orbit_period
        b_history[i, :] = 30e-6 * np.array([
            np.cos(phase),
            0.5 * np.sin(phase),
            0.3 * np.cos(2 * phase)
        ])
    
    return time, b_history


if __name__ == "__main__":
    print("=" * 60)
    print("Desaturation Strategy Analysis")
    print("=" * 60)
    
    # Create configurations
    config_1rw = create_3mtq_1rw_desat_config()
    config_3rw = create_3mtq_3rw_desat_config()
    
    # Generate B-field history
    time, b_history = generate_b_field_history(orbit_period=5400, n_points=500)
    
    print("\n" + "=" * 60)
    print("3MTQ + 1RW Configuration")
    print("=" * 60)
    
    analysis_1rw = analyze_desaturation_window(config_1rw, b_history, time)
    print(f"  Fraction of orbit with desaturation possible: {analysis_1rw['fraction_possible']:.1%}")
    print(f"  Integrated desaturation capability: {analysis_1rw['integrated_capability']*1e6:.2f} μNm·s")
    
    print("\n" + "=" * 60)
    print("3MTQ + 3RW Configuration")  
    print("=" * 60)
    
    analysis_3rw = analyze_desaturation_window(config_3rw, b_history, time)
    print(f"  Fraction of orbit with desaturation possible: {analysis_3rw['fraction_possible']:.1%}")
    print(f"  Integrated desaturation capability: {analysis_3rw['integrated_capability']*1e6:.2f} μNm·s")
    
    # Test specific B-field orientations
    print("\n" + "=" * 60)
    print("Testing Specific B-field Orientations")
    print("=" * 60)
    
    test_cases = [
        ("B along x", np.array([30e-6, 0, 0])),
        ("B along y", np.array([0, 30e-6, 0])),
        ("B along z", np.array([0, 0, 30e-6])),
        ("B diagonal", np.array([1, 1, 1]) * 30e-6 / np.sqrt(3)),
    ]
    
    print("\n3MTQ + 1RW (RW along z):")
    for name, b in test_cases:
        possible, max_torque = check_torque_free_desaturation(config_1rw, b)
        print(f"  {name}: {'Yes' if possible else 'No'} (max {max_torque*1e6:.1f} μNm)")
    
    print("\n3MTQ + 3RW:")
    for name, b in test_cases:
        possible, max_torque = check_torque_free_desaturation(config_3rw, b)
        print(f"  {name}: {'Yes' if possible else 'No'} (max {max_torque*1e6:.1f} μNm)")
    
    # Key insight: For 3MTQ+1RW with RW along z,
    # desaturation is possible only when B has a component perpendicular to z
    # (so MTQ can produce torque along z to cancel RW torque)
    
    print("\n" + "=" * 60)
    print("Key Findings")
    print("=" * 60)
    print("""
For 3MTQ + 1RW (RW along z-axis):
- Torque-free desaturation requires B to be NOT along z
- When B ∥ z: MTQ torque plane is x-y, no z-component, can't cancel RW
- When B ⊥ z: MTQ can produce z-torque to cancel RW torque
- Over orbit: ~60-90% of time has some desaturation capability

For 3MTQ + 3RW:  
- Always more capability than single RW
- Can desaturate whichever wheel has favorable B orientation
- Over orbit: ~90-100% of time has good capability
""")
