#!/usr/bin/env python
"""
Test SLERP interpolation + control recomputation for dynamics consistency.

This script:
1. Creates a simple trajectory (coarse)
2. SLERP interpolates to fine grid  
3. Recomputes controls from SLERP states
4. Forward simulates with those controls
5. Compares forward-simmed states to SLERP states
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
from ADCS.helpers.math_helpers import rot_mat


def solve_controls_from_trajectory(Xset_interp, B_eci, dt, J, rw_axes, 
                                    m_max=None, rw_torq_max=None):
    """Solve for MTQ + RW controls given interpolated states."""
    N = Xset_interp.shape[1]
    n_rw = rw_axes.shape[0] if rw_axes is not None and len(rw_axes) > 0 else 0
    n_mtq = 3
    n_u = n_mtq + n_rw
    
    Uset = np.zeros((n_u, N))
    
    for k in range(N-1):
        w_curr = Xset_interp[0:3, k]
        w_next = Xset_interp[0:3, k+1]
        
        q = Xset_interp[3:7, k]
        R = rot_mat(q)
        
        w_dot = (w_next - w_curr) / dt
        tau_needed = J @ w_dot + np.cross(w_curr, J @ w_curr)
        
        # RW contribution
        tau_rw = np.zeros(3)
        if n_rw > 0:
            for i in range(n_rw):
                h_curr = Xset_interp[7+i, k]
                h_next = Xset_interp[7+i, k+1]
                h_dot = (h_next - h_curr) / dt
                rw_torque = -h_dot
                if rw_torq_max is not None:
                    rw_torque = np.clip(rw_torque, -rw_torq_max, rw_torq_max)
                Uset[n_mtq + i, k] = rw_torque
                tau_rw += rw_torque * rw_axes[i]
        
        tau_mtq_needed = tau_needed - tau_rw
        
        B_body = R.T @ B_eci[:, k]
        B_sq = np.dot(B_body, B_body)
        
        if B_sq > 1e-20:
            m = np.cross(B_body, tau_mtq_needed) / B_sq
            if m_max is not None:
                m = np.clip(m, -m_max, m_max)
            Uset[0:3, k] = m
    
    Uset[:, -1] = Uset[:, -2] if N > 1 else 0
    return Uset


def forward_simulate(x0, Uset, B_eci, dt, J, rw_axes, rw_J):
    """Simple forward simulation with MTQ + RW."""
    N = Uset.shape[1]
    n_rw = len(rw_axes) if rw_axes is not None else 0
    n_x = 7 + n_rw
    
    Xset = np.zeros((n_x, N))
    Xset[:, 0] = x0
    
    for k in range(N-1):
        w = Xset[0:3, k]
        q = Xset[3:7, k]
        R = rot_mat(q)
        
        # Controls
        m = Uset[0:3, k]
        
        # B in body frame
        B_body = R.T @ B_eci[:, k]
        
        # MTQ torque
        tau_mtq = np.cross(m, B_body)
        
        # RW torque
        tau_rw = np.zeros(3)
        if n_rw > 0:
            for i in range(n_rw):
                rw_torque = Uset[3+i, k]
                tau_rw += -rw_torque * rw_axes[i]  # Reaction on spacecraft
        
        # Total external torque
        tau = tau_mtq + tau_rw
        
        # Euler equation: J @ w_dot = tau - w × (J @ w)
        w_dot = np.linalg.solve(J, tau - np.cross(w, J @ w))
        
        # Quaternion kinematics: q_dot = 0.5 * Omega(w) @ q
        w_quat = np.array([0, w[0], w[1], w[2]])
        q_dot = 0.5 * quat_mult(w_quat, q)
        
        # Euler integration
        w_new = w + w_dot * dt
        q_new = q + q_dot * dt
        q_new = q_new / np.linalg.norm(q_new)  # Normalize
        
        Xset[0:3, k+1] = w_new
        Xset[3:7, k+1] = q_new
        
        # RW momentum update
        if n_rw > 0:
            for i in range(n_rw):
                h = Xset[7+i, k]
                h_dot = Uset[3+i, k] * rw_J[i]  # torque = J * alpha, h_dot = J * alpha
                # Actually h_dot = torque directly for RW
                h_dot = Uset[3+i, k]
                Xset[7+i, k+1] = h + h_dot * dt
    
    return Xset


def quat_mult(q1, q2):
    """Quaternion multiplication [w,x,y,z] convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def slerp_interpolate(Xset_coarse, N_fine):
    """SLERP interpolate quaternions, cubic for rest."""
    N_coarse = Xset_coarse.shape[1]
    n_states = Xset_coarse.shape[0]
    
    t_coarse = np.linspace(0, 1, N_coarse)
    t_fine = np.linspace(0, 1, N_fine)
    
    Xset_fine = np.zeros((n_states, N_fine))
    
    # Ensure quaternion continuity
    quats_coarse = Xset_coarse[3:7, :].T.copy()
    for k in range(1, N_coarse):
        if np.dot(quats_coarse[k], quats_coarse[k-1]) < 0:
            quats_coarse[k] *= -1
    
    # SLERP for quaternions
    quats_scipy = quats_coarse[:, [1, 2, 3, 0]]  # [w,x,y,z] -> [x,y,z,w]
    rotations = Rotation.from_quat(quats_scipy)
    slerp_interp = Slerp(t_coarse, rotations)
    rotations_fine = slerp_interp(t_fine)
    quats_fine_scipy = rotations_fine.as_quat()
    
    Xset_fine[3, :] = quats_fine_scipy[:, 3]  # w
    Xset_fine[4, :] = quats_fine_scipy[:, 0]  # x
    Xset_fine[5, :] = quats_fine_scipy[:, 1]  # y
    Xset_fine[6, :] = quats_fine_scipy[:, 2]  # z
    
    # Cubic for angular velocity and RW momentum
    for i in list(range(0, 3)) + list(range(7, n_states)):
        Xset_fine[i, :] = interp1d(t_coarse, Xset_coarse[i, :], 
                                   kind='cubic', fill_value='extrapolate')(t_fine)
    
    return Xset_fine


def main():
    print("=" * 60)
    print("SLERP + Control Recomputation Consistency Test")
    print("=" * 60)
    
    # Parameters
    dt_coarse = 10.0  # seconds
    dt_fine = 2.0
    tf = 100.0  # total time
    
    N_coarse = int(tf / dt_coarse) + 1
    N_fine = int(tf / dt_fine) + 1
    
    print(f"\nN_coarse = {N_coarse}, N_fine = {N_fine}")
    
    # Satellite parameters
    J = np.diag([0.01, 0.012, 0.008])
    rw_axes = np.array([[0, 0, 1]])  # Single RW along z
    rw_J = np.array([1e-5])
    m_max = 0.1
    rw_torq_max = 1e-3
    
    # Create coarse trajectory - a simple slew
    n_rw = 1
    n_x = 7 + n_rw
    
    Xset_coarse = np.zeros((n_x, N_coarse))
    
    # Initial state: small angular velocity, identity quaternion
    Xset_coarse[0:3, 0] = [0.01, 0.005, -0.008]  # rad/s
    Xset_coarse[3:7, 0] = [1, 0, 0, 0]  # identity
    Xset_coarse[7, 0] = 0.001  # small RW momentum
    
    # Create a smooth trajectory (exponential decay of angular velocity, rotation)
    for k in range(1, N_coarse):
        t = k * dt_coarse
        decay = np.exp(-t / 50.0)
        
        Xset_coarse[0:3, k] = Xset_coarse[0:3, 0] * decay
        
        # Quaternion: rotate about z
        angle = 0.5 * (1 - decay)  # radians
        Xset_coarse[3, k] = np.cos(angle/2)
        Xset_coarse[4, k] = 0
        Xset_coarse[5, k] = 0
        Xset_coarse[6, k] = np.sin(angle/2)
        
        Xset_coarse[7, k] = Xset_coarse[7, 0] * decay
    
    print(f"\nCoarse trajectory:")
    print(f"  ω range: [{Xset_coarse[0:3,:].min():.6f}, {Xset_coarse[0:3,:].max():.6f}]")
    print(f"  q[0] (w): [{Xset_coarse[3,:].min():.4f}, {Xset_coarse[3,:].max():.4f}]")
    
    # SLERP interpolate
    print(f"\n1. SLERP interpolating...")
    Xset_slerp = slerp_interpolate(Xset_coarse, N_fine)
    print(f"  ω range: [{Xset_slerp[0:3,:].min():.6f}, {Xset_slerp[0:3,:].max():.6f}]")
    
    # Create B-field (varying in ECI)
    B_eci = np.zeros((3, N_fine))
    for k in range(N_fine):
        t = k * dt_fine
        B_eci[0, k] = 3e-5 * np.cos(t / 100)
        B_eci[1, k] = 2e-5 * np.sin(t / 100)
        B_eci[2, k] = 4e-5
    
    # Recompute controls
    print(f"\n2. Recomputing controls from SLERP states...")
    Uset = solve_controls_from_trajectory(
        Xset_slerp, B_eci, dt_fine, J, rw_axes,
        m_max=m_max, rw_torq_max=rw_torq_max
    )
    print(f"  MTQ range: [{Uset[0:3,:].min():.6f}, {Uset[0:3,:].max():.6f}]")
    print(f"  RW range: [{Uset[3:,:].min():.6f}, {Uset[3:,:].max():.6f}]")
    
    # Forward simulate
    print(f"\n3. Forward simulating with recomputed controls...")
    Xset_fwd = forward_simulate(
        Xset_slerp[:, 0], Uset, B_eci, dt_fine, J, rw_axes, rw_J
    )
    
    # Compare
    print(f"\n4. Comparing SLERP vs Forward-simulated:")
    diff_w = np.abs(Xset_slerp[0:3, :] - Xset_fwd[0:3, :])
    diff_q = np.abs(Xset_slerp[3:7, :] - Xset_fwd[3:7, :])
    diff_h = np.abs(Xset_slerp[7:, :] - Xset_fwd[7:, :])
    
    print(f"  Max ω diff: {diff_w.max():.6f} rad/s")
    print(f"  Mean ω diff: {diff_w.mean():.6f} rad/s")
    print(f"  Max q diff: {diff_q.max():.6f}")
    print(f"  Mean q diff: {diff_q.mean():.6f}")
    print(f"  Max h diff: {diff_h.max():.6f}")
    
    # Check if acceptable
    if diff_w.max() < 0.01 and diff_q.max() < 0.01:
        print(f"\n✓ PASS: Dynamics are consistent!")
    else:
        print(f"\n✗ FAIL: Dynamics mismatch too large")
        
        # Debug: show where the differences are largest
        worst_k = np.argmax(diff_w.max(axis=0))
        print(f"\n  Worst timestep: k={worst_k}")
        print(f"    SLERP ω: {Xset_slerp[0:3, worst_k]}")
        print(f"    Fwd ω:   {Xset_fwd[0:3, worst_k]}")
        print(f"    Diff:    {diff_w[:, worst_k]}")


if __name__ == "__main__":
    main()
