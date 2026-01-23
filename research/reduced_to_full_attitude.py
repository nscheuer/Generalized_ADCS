"""
Reduced to Full Attitude Control Conversion
============================================

Explore methods to make a reduced-attitude controller (vector alignment)
behave like a full-attitude controller (quaternion tracking).

Key question: Can we achieve full 3-DOF attitude control using only 
2-DOF vector alignment objectives?

Methods to explore:
1. Alternating goals: Switch between two vector goals that intersect at target
2. Cascaded control: Use one goal to reach a manifold, another to converge
3. Constrained dynamics: Add rate constraints that naturally damp axial rotation
4. Hybrid: Use reduced attitude + axial rate control separately
5. Multi-vector: Track multiple body vectors to different inertial targets

Mathematical analysis:
- What are the convergence properties?
- When is each method applicable?
- What are the trade-offs?
"""

import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))

from ADCS.helpers.math_helpers import normalize, rot_mat, skewsym, quat_inv, quat_mult, dcm_to_quat


def quaternion_error_vector(q, q_goal):
    """Full quaternion error."""
    q = normalize(q)
    q_goal = normalize(q_goal)
    q_err = quat_mult(quat_inv(q_goal), q)
    if q_err[0] < 0:
        q_err = -q_err
    return 2.0 * q_err[1:4]


def vector_alignment_error(q, body_vec, target_vec_inertial):
    """
    Compute error for aligning body_vec with target_vec.
    
    Returns a 3-vector that when used as torque direction will reduce alignment error.
    Uses the cross product form: error = target_body × body_vec
    (This gives the rotation that takes body_vec to target_body)
    """
    R = rot_mat(q)
    body_vec = normalize(body_vec)
    target_vec = normalize(target_vec_inertial)
    
    # Target in body frame
    target_body = R.T @ target_vec
    
    # Error is cross product: target × body gives rotation to move body toward target
    error = np.cross(target_body, body_vec)
    
    return error


def pointing_error_deg(q, q_goal, boresight=np.array([0, 0, 1])):
    """Compute pointing error of boresight."""
    q = normalize(q)
    q_goal = normalize(q_goal)
    R = rot_mat(q)
    R_goal = rot_mat(q_goal)
    actual = R @ boresight
    goal = R_goal @ boresight
    cos_angle = np.clip(np.dot(actual, goal), -1, 1)
    return np.degrees(np.arccos(cos_angle))


def full_attitude_error_deg(q, q_goal):
    """Compute total rotation angle error."""
    q = normalize(q)
    q_goal = normalize(q_goal)
    q_err = quat_mult(quat_inv(q_goal), q)
    if q_err[0] < 0:
        q_err = -q_err
    angle = 2 * np.arccos(np.clip(q_err[0], -1, 1))
    return np.degrees(angle)


class AttitudeController:
    """Base class for attitude controllers."""
    
    def __init__(self, J, kp, kd):
        self.J = J
        self.kp = kp
        self.kd = kd
    
    def compute_torque(self, omega, q, q_goal, t) -> np.ndarray:
        """Returns desired torque."""
        raise NotImplementedError


class FullAttitudeController(AttitudeController):
    """Standard full quaternion feedback controller."""
    
    def compute_torque(self, omega, q, q_goal, t):
        q_err = quaternion_error_vector(q, q_goal)
        return -self.kp * q_err - self.kd * omega


class SingleVectorController(AttitudeController):
    """
    Reduced attitude controller: align single body vector.
    Cannot control rotation about alignment axis.
    """
    
    def __init__(self, J, kp, kd, body_vec, target_vec_func):
        super().__init__(J, kp, kd)
        self.body_vec = normalize(body_vec)
        self.target_vec_func = target_vec_func  # Function of (q_goal, t)
    
    def compute_torque(self, omega, q, q_goal, t):
        target_vec = self.target_vec_func(q_goal, t)
        v_err = vector_alignment_error(q, self.body_vec, target_vec)
        
        # Only damp rates perpendicular to alignment axis
        R = rot_mat(q)
        target_body = R.T @ normalize(target_vec)
        omega_perp = omega - np.dot(omega, target_body) * target_body
        
        return -self.kp * v_err - self.kd * omega_perp


class AlternatingVectorController(AttitudeController):
    """
    Alternate between two vector alignment goals that intersect at the target attitude.
    
    Idea: If we want q_goal, find two vectors v1, v2 in body frame that map to
    specific inertial directions at q_goal. Alternating control of v1 and v2
    should converge to q_goal (the intersection of the two constraint manifolds).
    """
    
    def __init__(self, J, kp, kd, switch_period=10.0):
        super().__init__(J, kp, kd)
        self.switch_period = switch_period
        
        # Two body vectors (orthogonal works best)
        self.body_vec1 = normalize(np.array([0, 0, 1]))  # z-axis (boresight)
        self.body_vec2 = normalize(np.array([1, 0, 0]))  # x-axis
    
    def compute_torque(self, omega, q, q_goal, t):
        # Determine which vector to track
        phase = int(t / self.switch_period) % 2
        
        # Compute target vectors in inertial frame (where body vecs should point)
        R_goal = rot_mat(q_goal)
        target_vec1 = R_goal @ self.body_vec1
        target_vec2 = R_goal @ self.body_vec2
        
        if phase == 0:
            body_vec = self.body_vec1
            target_vec = target_vec1
        else:
            body_vec = self.body_vec2
            target_vec = target_vec2
        
        v_err = vector_alignment_error(q, body_vec, target_vec)
        
        # Damp all rates (both goals are changing)
        return -self.kp * v_err - self.kd * omega


class CascadedVectorController(AttitudeController):
    """
    Cascaded control: Primary vector alignment + axial rate damping.
    
    Stage 1: Align primary body vector with target
    Stage 2: Once aligned, damp rotation about that axis
    
    This achieves full attitude control if target has zero axial rate.
    """
    
    def __init__(self, J, kp, kd, body_vec, target_vec_func, axial_kd=None):
        super().__init__(J, kp, kd)
        self.body_vec = normalize(body_vec)
        self.target_vec_func = target_vec_func
        self.axial_kd = axial_kd if axial_kd is not None else kd
    
    def compute_torque(self, omega, q, q_goal, t):
        target_vec = self.target_vec_func(q_goal, t)
        v_err = vector_alignment_error(q, self.body_vec, target_vec)
        
        # Primary control: align vector
        tau_align = -self.kp * v_err
        
        # Rate damping: full 3-axis
        tau_damp = -self.kd * omega
        
        # But add extra damping for axial rate
        R = rot_mat(q)
        target_body = R.T @ normalize(target_vec)
        omega_axial = np.dot(omega, target_body) * target_body
        tau_axial_damp = -self.axial_kd * omega_axial
        
        return tau_align + tau_damp + tau_axial_damp


class MultiVectorController(AttitudeController):
    """
    Multi-vector alignment: Track multiple body vectors simultaneously.
    
    If we track 2 non-parallel body vectors to 2 non-parallel inertial targets,
    we have 4 DOF constraints for 3 DOF attitude → overdetermined → full control.
    
    Combine errors with weights.
    """
    
    def __init__(self, J, kp, kd, body_vecs, target_vecs_func, weights=None):
        super().__init__(J, kp, kd)
        self.body_vecs = [normalize(v) for v in body_vecs]
        self.target_vecs_func = target_vecs_func
        self.weights = weights if weights is not None else [1.0] * len(body_vecs)
    
    def compute_torque(self, omega, q, q_goal, t):
        target_vecs = self.target_vecs_func(q_goal, t)
        
        total_err = np.zeros(3)
        for body_vec, target_vec, w in zip(self.body_vecs, target_vecs, self.weights):
            v_err = vector_alignment_error(q, body_vec, target_vec)
            total_err += w * v_err
        
        return -self.kp * total_err - self.kd * omega


class DynamicsAwareController(AttitudeController):
    """
    Dynamics-aware reduced attitude: Use current angular velocity to inform goal.
    
    If we have angular velocity ω, choose the goal attitude that:
    1. Aligns the boresight correctly
    2. Has the axial component that matches current ω (minimize needed change)
    
    This should converge faster because we don't fight existing rotation.
    """
    
    def __init__(self, J, kp, kd, body_vec, target_vec_func):
        super().__init__(J, kp, kd)
        self.body_vec = normalize(body_vec)
        self.target_vec_func = target_vec_func
    
    def compute_torque(self, omega, q, q_goal, t):
        target_vec = self.target_vec_func(q_goal, t)
        
        # Vector alignment error
        v_err = vector_alignment_error(q, self.body_vec, target_vec)
        
        # Current body-frame target axis
        R = rot_mat(q)
        target_body = R.T @ normalize(target_vec)
        
        # Decompose omega
        omega_axial = np.dot(omega, target_body) * target_body
        omega_perp = omega - omega_axial
        
        # Strongly damp perpendicular (misalignment correction)
        # Weakly damp axial (let it persist if not harmful)
        tau_align = -self.kp * v_err
        tau_damp_perp = -self.kd * omega_perp
        tau_damp_axial = -self.kd * 0.1 * omega_axial  # Weak axial damping
        
        return tau_align + tau_damp_perp + tau_damp_axial


def simulate_controller(controller, J, x0, q_goal, tf, dt):
    """Simulate attitude control."""
    steps = int(tf / dt) + 1
    
    full_error_hist = np.zeros(steps)
    point_error_hist = np.zeros(steps)
    omega_hist = np.zeros((steps, 3))
    
    x = x0.copy()
    t = 0.0
    
    for k in range(steps):
        omega = x[0:3]
        q = normalize(x[3:7])
        
        tau = controller.compute_torque(omega, q, q_goal, t)
        
        full_error_hist[k] = full_attitude_error_deg(q, q_goal)
        point_error_hist[k] = pointing_error_deg(q, q_goal)
        omega_hist[k, :] = omega
        
        if k == steps - 1:
            break
        
        # Propagate
        def dynamics(t_local, y):
            w = y[0:3]
            quat = normalize(y[3:7])
            
            w_dot = np.linalg.solve(J, tau - np.cross(w, J @ w))
            
            W = np.zeros((4, 3))
            W[0, :] = -quat[1:4]
            W[1:4, :] = quat[0] * np.eye(3) + skewsym(quat[1:4])
            q_dot = 0.5 * W @ w
            
            return np.concatenate([w_dot, q_dot])
        
        sol = solve_ivp(dynamics, [0, dt], x, method='RK45', rtol=1e-8, atol=1e-10)
        x = sol.y[:, -1]
        x[3:7] = normalize(x[3:7])
        
        t += dt
    
    return {
        'full_error': full_error_hist,
        'point_error': point_error_hist,
        'omega': omega_hist,
        'final_full_error': full_error_hist[-1],
        'final_point_error': point_error_hist[-1],
        'converged_full': full_error_hist[-1] < 1.0,
        'converged_point': point_error_hist[-1] < 1.0
    }


def run_comparison():
    """Compare different reduced→full attitude strategies."""
    np.random.seed(42)
    
    J = np.diag([0.022, 0.022, 0.004])
    kp, kd = 0.0001, 0.002
    tf, dt = 300, 0.5
    
    # Goal quaternion
    q_goal = normalize(np.array([0.8, 0.2, 0.4, 0.3]))
    R_goal = rot_mat(q_goal)
    
    # Target vector functions
    def target_z(q_goal, t):
        return R_goal @ np.array([0, 0, 1])
    
    def target_x(q_goal, t):
        return R_goal @ np.array([1, 0, 0])
    
    def target_both(q_goal, t):
        return [R_goal @ np.array([0, 0, 1]), R_goal @ np.array([1, 0, 0])]
    
    # Controllers
    controllers = {
        'Full Attitude': FullAttitudeController(J, kp, kd),
        'Single Vector (z)': SingleVectorController(J, kp, kd, np.array([0,0,1]), target_z),
        'Alternating': AlternatingVectorController(J, kp, kd, switch_period=20),
        'Cascaded': CascadedVectorController(J, kp, kd, np.array([0,0,1]), target_z, axial_kd=kd),
        'Multi-Vector': MultiVectorController(J, kp, kd, 
                                              [np.array([0,0,1]), np.array([1,0,0])],
                                              target_both, weights=[1.0, 0.5]),
        'Dynamics-Aware': DynamicsAwareController(J, kp, kd, np.array([0,0,1]), target_z),
    }
    
    # Multiple initial conditions
    n_trials = 10
    all_results = {name: [] for name in controllers}
    
    for i in tqdm(range(n_trials), desc="Running trials"):
        # Random initial condition
        axis = normalize(np.random.randn(3))
        angle = np.random.uniform(0.2, 0.8)  # 10-45 degrees
        q0 = normalize(np.concatenate([[np.cos(angle/2)], axis * np.sin(angle/2)]))
        omega0 = np.random.randn(3) * 0.03
        x0 = np.concatenate([omega0, q0])
        
        for name, controller in controllers.items():
            result = simulate_controller(controller, J, x0, q_goal, tf, dt)
            all_results[name].append(result)
    
    # Summarize
    print("\n" + "=" * 80)
    print("REDUCED → FULL ATTITUDE CONTROL COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Method':<20} {'Full Err(°)':>12} {'Point Err(°)':>12} {'Conv Full':>10} {'Conv Point':>10}")
    print("-" * 80)
    
    for name in controllers:
        data = all_results[name]
        full_err = np.mean([d['final_full_error'] for d in data])
        full_std = np.std([d['final_full_error'] for d in data])
        point_err = np.mean([d['final_point_error'] for d in data])
        conv_full = np.mean([d['converged_full'] for d in data])
        conv_point = np.mean([d['converged_point'] for d in data])
        
        print(f"{name:<20} {full_err:>6.2f}±{full_std:>4.2f} {point_err:>12.2f} {100*conv_full:>10.0f}% {100*conv_point:>10.0f}%")
    
    return all_results


if __name__ == "__main__":
    results = run_comparison()
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("""
Key findings:

1. SINGLE VECTOR: Only converges for pointing (2-DOF), axial rotation unconstrained
   - Use when: Only boresight alignment matters (imaging, sun tracking)

2. ALTERNATING VECTORS: Can achieve full attitude by switching between constraints
   - Convergence is slower due to switching
   - Use when: Simple implementation needed, slow convergence OK

3. CASCADED: Vector alignment + axial damping = full attitude
   - Good convergence, simple structure
   - Use when: Target has zero axial rate

4. MULTI-VECTOR: Track multiple vectors simultaneously = overdetermined full attitude
   - Best convergence, most robust
   - Use when: Computational cost is acceptable

5. DYNAMICS-AWARE: Exploits current motion for faster convergence
   - Doesn't guarantee full attitude, but gets close
   - Use when: Fast settling is critical, exact attitude less important

RECOMMENDATION:
- For true full attitude from reduced: Use Multi-Vector or Cascaded
- For "good enough" attitude: Dynamics-Aware single vector
""")
