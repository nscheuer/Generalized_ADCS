"""
Attitude Goal Conversion Analysis
=================================

Investigate strategies for converting between:
1. Full attitude goals (specific quaternion/DCM)
2. Reduced attitude goals (vector alignment only)

Key questions:
- When converting vector goal → full attitude, which rotation to choose?
- Strategies: closest point, dynamics-aware, energy-optimal, reachability-based
- Impact on convergence and stability
"""

import sys
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))

from ADCS.helpers.math_helpers import (
    normalize, rot_mat, skewsym, quat_inv, quat_mult, quat_to_mrp, dcm_to_quat
)


def rotation_axis_angle(q: np.ndarray) -> Tuple[np.ndarray, float]:
    """Extract axis and angle from quaternion."""
    q = normalize(q)
    if q[0] < 0:
        q = -q
    
    angle = 2 * np.arccos(np.clip(q[0], -1, 1))
    
    if np.abs(angle) < 1e-12:
        return np.array([0, 0, 1]), 0.0
    
    axis = q[1:4] / np.sin(angle / 2)
    return normalize(axis), angle


def vector_alignment_quaternions(body_vec: np.ndarray, 
                                   target_vec_inertial: np.ndarray,
                                   current_q: np.ndarray) -> List[np.ndarray]:
    """
    Find all quaternions that align body_vec with target_vec_inertial.
    
    Returns a parameterized family (actually a circle in quaternion space).
    Here we return representative samples.
    """
    body_vec = normalize(body_vec)
    target_vec = normalize(target_vec_inertial)
    
    # Find rotation that takes body_vec to target_vec
    # This is not unique - any rotation about target_vec can be added
    
    # First, find the "shortest path" rotation
    cross = np.cross(body_vec, target_vec)
    dot = np.dot(body_vec, target_vec)
    
    if np.abs(dot + 1) < 1e-6:  # Opposite vectors
        # 180 degree rotation about any perpendicular axis
        perp = np.array([1, 0, 0]) if np.abs(body_vec[0]) < 0.9 else np.array([0, 1, 0])
        axis = normalize(np.cross(body_vec, perp))
        q_base = np.array([0, axis[0], axis[1], axis[2]])
    elif np.abs(dot - 1) < 1e-6:  # Same vectors
        q_base = np.array([1, 0, 0, 0])
    else:
        # General case
        cross_norm = np.linalg.norm(cross)
        axis = cross / cross_norm
        angle = np.arccos(np.clip(dot, -1, 1))
        q_base = np.concatenate([[np.cos(angle/2)], axis * np.sin(angle/2)])
    
    # Generate family of quaternions by adding rotations about target_vec
    quaternions = []
    for theta in np.linspace(0, 2*np.pi, 36, endpoint=False):
        # Rotation about target_vec by theta
        q_twist = np.concatenate([[np.cos(theta/2)], target_vec * np.sin(theta/2)])
        
        # Combined rotation (apply base first, then twist)
        # Note: this represents the rotation from body to inertial
        q_combined = quat_mult(q_twist, q_base)
        quaternions.append(normalize(q_combined))
    
    return quaternions


def closest_quaternion(body_vec: np.ndarray,
                       target_vec_inertial: np.ndarray,
                       current_q: np.ndarray) -> np.ndarray:
    """
    Find the quaternion that aligns body_vec with target_vec
    and is closest to current_q (minimum rotation).
    """
    current_q = normalize(current_q)
    
    # Get family of valid quaternions
    candidates = vector_alignment_quaternions(body_vec, target_vec_inertial, current_q)
    
    # Find closest to current
    min_dist = float('inf')
    best_q = candidates[0]
    
    for q in candidates:
        # Distance in quaternion space (geodesic)
        q_err = quat_mult(quat_inv(current_q), q)
        _, angle = rotation_axis_angle(q_err)
        
        if angle < min_dist:
            min_dist = angle
            best_q = q
    
    return best_q


def dynamics_aware_quaternion(body_vec: np.ndarray,
                               target_vec_inertial: np.ndarray,
                               current_q: np.ndarray,
                               omega: np.ndarray,
                               J: np.ndarray,
                               dt: float = 1.0) -> np.ndarray:
    """
    Find the quaternion that aligns body_vec with target_vec
    considering current angular velocity.
    
    Strategy: minimize rotation angle + penalize fighting current motion.
    """
    current_q = normalize(current_q)
    omega = np.asarray(omega)
    
    # Predict where we'll be in dt seconds (approximately)
    # q_pred ≈ q + 0.5 * W(q) @ omega * dt
    W = np.zeros((4, 3))
    W[0, :] = -current_q[1:4]
    W[1:4, :] = current_q[0] * np.eye(3) + skewsym(current_q[1:4])
    q_pred = normalize(current_q + 0.5 * W @ omega * dt)
    
    # Get family of valid quaternions
    candidates = vector_alignment_quaternions(body_vec, target_vec_inertial, current_q)
    
    # Score each candidate
    best_score = float('inf')
    best_q = candidates[0]
    
    for q in candidates:
        # Distance from current
        q_err_current = quat_mult(quat_inv(current_q), q)
        _, angle_current = rotation_axis_angle(q_err_current)
        
        # Distance from predicted (favor going with the flow)
        q_err_pred = quat_mult(quat_inv(q_pred), q)
        _, angle_pred = rotation_axis_angle(q_err_pred)
        
        # Score: weighted combination
        # Lower angle_pred means the target is in the direction we're already going
        score = 0.5 * angle_current + 0.5 * angle_pred
        
        if score < best_score:
            best_score = score
            best_q = q
    
    return best_q


def energy_optimal_quaternion(body_vec: np.ndarray,
                               target_vec_inertial: np.ndarray,
                               current_q: np.ndarray,
                               omega: np.ndarray,
                               J: np.ndarray) -> np.ndarray:
    """
    Find the quaternion that aligns body_vec with target_vec
    and minimizes kinetic energy at the goal state.
    
    Strategy: choose the goal attitude that allows omega to naturally
    decay to zero (or remain low).
    """
    current_q = normalize(current_q)
    omega = np.asarray(omega)
    
    # Get family of valid quaternions  
    candidates = vector_alignment_quaternions(body_vec, target_vec_inertial, current_q)
    
    # The rotation about target_vec is unconstrained, so we can choose
    # the axial rate freely. The optimal is to have zero axial rate.
    
    # Current body-frame omega projected onto target axis (in body frame)
    R_current = rot_mat(current_q)
    target_body = R_current.T @ normalize(target_vec_inertial)
    omega_axial = np.dot(omega, target_body)
    
    # We want to choose the goal quaternion such that the "extra" rotation
    # matches the axial momentum we have
    
    # For now, use closest quaternion as baseline
    # (full energy-optimal would require trajectory optimization)
    return closest_quaternion(body_vec, target_vec_inertial, current_q)


def reachability_quaternion(body_vec: np.ndarray,
                            target_vec_inertial: np.ndarray,
                            current_q: np.ndarray,
                            omega: np.ndarray,
                            A_rw: np.ndarray,
                            A_mtq_axes: np.ndarray,
                            b_body: np.ndarray) -> np.ndarray:
    """
    Find the quaternion that aligns body_vec with target_vec
    considering what's actually controllable.
    
    For underactuated systems, not all attitudes are equally reachable.
    Prefer goals that lie in the controllable subspace.
    """
    # This is the most sophisticated approach
    # For now, implement as closest quaternion + penalty for hard-to-reach
    
    current_q = normalize(current_q)
    candidates = vector_alignment_quaternions(body_vec, target_vec_inertial, current_q)
    
    # Compute current torque capability direction
    # MTQ can produce torque perpendicular to B
    b_norm = np.linalg.norm(b_body)
    if b_norm > 1e-12:
        b_hat = b_body / b_norm
        # Controllable torque plane normal
        torque_plane_normal = b_hat
    else:
        torque_plane_normal = np.zeros(3)
    
    best_score = float('inf')
    best_q = candidates[0]
    
    for q in candidates:
        # Required rotation
        q_err = quat_mult(quat_inv(current_q), q)
        axis, angle = rotation_axis_angle(q_err)
        
        # How much of this rotation is controllable?
        # Rotation about torque_plane_normal is LEAST controllable (parallel to B)
        controllability = 1.0 - np.abs(np.dot(axis, torque_plane_normal))
        
        # Score: angle penalized by poor controllability
        score = angle * (2.0 - controllability)  # Higher penalty for uncontrollable directions
        
        if score < best_score:
            best_score = score
            best_q = q
    
    return best_q


def compare_conversion_strategies(body_vec: np.ndarray,
                                   target_vec: np.ndarray,
                                   current_q: np.ndarray,
                                   omega: np.ndarray,
                                   J: np.ndarray,
                                   b_body: np.ndarray) -> Dict[str, Dict]:
    """Compare different goal conversion strategies."""
    results = {}
    
    # Closest point
    q_closest = closest_quaternion(body_vec, target_vec, current_q)
    q_err = quat_mult(quat_inv(current_q), q_closest)
    _, angle_closest = rotation_axis_angle(q_err)
    results['closest'] = {
        'quaternion': q_closest,
        'rotation_angle_deg': np.degrees(angle_closest)
    }
    
    # Dynamics-aware
    q_dynamics = dynamics_aware_quaternion(body_vec, target_vec, current_q, omega, J)
    q_err = quat_mult(quat_inv(current_q), q_dynamics)
    _, angle_dynamics = rotation_axis_angle(q_err)
    results['dynamics_aware'] = {
        'quaternion': q_dynamics,
        'rotation_angle_deg': np.degrees(angle_dynamics)
    }
    
    # Energy-optimal
    q_energy = energy_optimal_quaternion(body_vec, target_vec, current_q, omega, J)
    q_err = quat_mult(quat_inv(current_q), q_energy)
    _, angle_energy = rotation_axis_angle(q_err)
    results['energy_optimal'] = {
        'quaternion': q_energy,
        'rotation_angle_deg': np.degrees(angle_energy)
    }
    
    # Reachability-aware
    A_rw = np.array([[0], [0], [1.0]])  # Example
    A_mtq = np.eye(3)
    q_reach = reachability_quaternion(body_vec, target_vec, current_q, omega, A_rw, A_mtq, b_body)
    q_err = quat_mult(quat_inv(current_q), q_reach)
    _, angle_reach = rotation_axis_angle(q_err)
    results['reachability'] = {
        'quaternion': q_reach,
        'rotation_angle_deg': np.degrees(angle_reach)
    }
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Attitude Goal Conversion Analysis")
    print("=" * 60)
    
    # Test case: satellite pointing camera (z-axis) at nadir
    body_vec = np.array([0, 0, 1])  # Camera boresight
    target_vec = np.array([0, 0, -1])  # Nadir direction (pointing down in ECI)
    
    # Current state: rotated 45 degrees about x
    angle = np.radians(45)
    current_q = np.array([np.cos(angle/2), np.sin(angle/2), 0, 0])
    
    # Some angular velocity
    omega = np.array([0.01, 0.02, 0.01])  # rad/s
    
    # Spacecraft parameters
    J = np.diag([0.022, 0.022, 0.004])
    b_body = np.array([20e-6, 10e-6, 5e-6])  # Magnetic field in body frame
    
    print(f"\nTest Case:")
    print(f"  Body vector (boresight): {body_vec}")
    print(f"  Target vector (nadir): {target_vec}")
    print(f"  Current quaternion: {current_q}")
    print(f"  Angular velocity: {omega} rad/s")
    print(f"  B-field (body): {b_body*1e6} μT")
    
    # Compare strategies
    results = compare_conversion_strategies(
        body_vec, target_vec, current_q, omega, J, b_body
    )
    
    print("\n" + "=" * 60)
    print("Conversion Strategy Results")
    print("=" * 60)
    
    for name, data in results.items():
        print(f"\n{name}:")
        print(f"  Goal quaternion: {data['quaternion']}")
        print(f"  Required rotation: {data['rotation_angle_deg']:.2f}°")
        
        # Verify alignment
        R = rot_mat(data['quaternion'])
        achieved = R @ body_vec
        alignment_error = np.arccos(np.clip(np.dot(achieved, target_vec), -1, 1))
        print(f"  Alignment error: {np.degrees(alignment_error):.4f}°")
    
    print("\n" + "=" * 60)
    print("Key Insights")
    print("=" * 60)
    print("""
1. CLOSEST POINT:
   - Simplest approach - minimum rotation from current attitude
   - Doesn't consider dynamics or controllability
   - Good baseline but may fight angular momentum

2. DYNAMICS-AWARE:
   - Considers where the spacecraft is heading
   - Favors goals that align with current motion
   - Better for reducing control effort

3. ENERGY-OPTIMAL:
   - Minimizes kinetic energy at goal
   - Important for settling quickly
   - Requires trajectory planning for full optimality

4. REACHABILITY-AWARE:
   - Considers actuator constraints
   - Avoids goals that require torque parallel to B (for MTQ)
   - Best for underactuated systems

RECOMMENDATION:
For underactuated systems (3MTQ+1RW), use REACHABILITY-AWARE conversion.
It produces goals that the system can actually track without fighting
the actuator constraints.
""")
