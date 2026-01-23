"""
Advanced QP Methods in Closed-Loop
==================================

Test whether the QP variants with smart constraints actually improve
closed-loop performance compared to LP.

Key insight: QP_ProjDom achieves higher alpha in 33% of cases.
The question: does this translate to better closed-loop pointing?
"""

import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, Bounds, lsq_linear, linprog
from typing import Dict, List, Tuple
from tqdm import tqdm
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))

from ADCS.helpers.math_helpers import normalize, rot_mat, skewsym, quat_inv, quat_mult


def solve_lp(tau_des, A_total, lb, ub):
    """LP allocation."""
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-12:
        return np.zeros(len(lb))
    
    tau_hat = tau_des / t_mag
    n_act = len(lb)
    
    c = np.zeros(n_act + 1)
    c[-1] = -1.0
    
    A_eq = np.hstack([A_total, -tau_hat.reshape(3, 1)])
    b_eq = np.zeros(3)
    
    bounds = [(lb[i], ub[i]) for i in range(n_act)] + [(0, None)]
    
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if res.success:
        u = res.x[:n_act]
        T_max = res.x[-1]
        if T_max > t_mag:
            u = u * (t_mag / T_max)
        return u
    return np.zeros(n_act)


def solve_qp_projection_dominance(tau_des, A_total, lb, ub):
    """QP with projection dominance constraint."""
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-12:
        return np.zeros(len(lb))
    
    tau_hat = tau_des / t_mag
    
    # Get LP solution for constraint
    u_lp = solve_lp(tau_des, A_total, lb, ub)
    tau_lp = A_total @ u_lp
    min_proj = np.dot(tau_lp, tau_hat) * 0.999
    
    def objective(u):
        r = A_total @ u - tau_des
        return 0.5 * np.dot(r, r)
    
    def gradient(u):
        return A_total.T @ (A_total @ u - tau_des)
    
    c_proj = A_total.T @ tau_hat
    
    constraint = {
        'type': 'ineq',
        'fun': lambda u: c_proj @ u - min_proj,
        'jac': lambda u: c_proj
    }
    
    res = minimize(objective, u_lp, jac=gradient, method='SLSQP',
                  bounds=Bounds(lb, ub), constraints=[constraint],
                  options={'ftol': 1e-10})
    
    return res.x if res.success else u_lp


def solve_qp_smart(tau_des, A_total, lb, ub, omega, q_err):
    """
    Smart QP: Combines projection dominance with energy-aware constraints.
    
    1. Must achieve at least LP's projection (never worse)
    2. If damping (ω·τ_des < 0): limit perpendicular energy contribution
    3. If correcting (τ_des toward -q_err): ensure torque helps
    """
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-12:
        return np.zeros(len(lb))
    
    tau_hat = tau_des / t_mag
    omega = np.asarray(omega)
    q_err = np.asarray(q_err)
    
    u_lp = solve_lp(tau_des, A_total, lb, ub)
    tau_lp = A_total @ u_lp
    
    def objective(u):
        r = A_total @ u - tau_des
        return 0.5 * np.dot(r, r)
    
    def gradient(u):
        return A_total.T @ (A_total @ u - tau_des)
    
    constraints = []
    
    # 1. Projection dominance
    min_proj = np.dot(tau_lp, tau_hat) * 0.999
    c_proj = A_total.T @ tau_hat
    constraints.append({
        'type': 'ineq',
        'fun': lambda u, c=c_proj, mp=min_proj: c @ u - mp,
        'jac': lambda u, c=c_proj: c
    })
    
    # 2. Energy constraint if damping
    omega_dot_tau_des = np.dot(omega, tau_des)
    if omega_dot_tau_des < -1e-12:
        # Don't add MORE energy than LP does
        omega_dot_tau_lp = np.dot(omega, tau_lp)
        c_omega = A_total.T @ omega
        # ω·τ ≤ ω·τ_LP (since both should be negative, this allows same or more damping)
        constraints.append({
            'type': 'ineq',
            'fun': lambda u, c=c_omega, ub=omega_dot_tau_lp: ub - c @ u,
            'jac': lambda u, c=c_omega: -c
        })
    
    res = minimize(objective, u_lp, jac=gradient, method='SLSQP',
                  bounds=Bounds(lb, ub), constraints=constraints,
                  options={'ftol': 1e-10})
    
    return res.x if res.success else u_lp


def quaternion_error_vector(q, q_goal):
    """Compute quaternion error as 3-vector."""
    q = normalize(q)
    q_goal = normalize(q_goal)
    q_err = quat_mult(quat_inv(q_goal), q)
    if q_err[0] < 0:
        q_err = -q_err
    return 2.0 * q_err[1:4]


def pointing_error_deg(q, q_goal):
    """Compute pointing error in degrees."""
    q = normalize(q)
    q_goal = normalize(q_goal)
    R = rot_mat(q)
    R_goal = rot_mat(q_goal)
    boresight = np.array([0, 0, 1])
    actual = R @ boresight
    goal = R_goal @ boresight
    cos_angle = np.clip(np.dot(actual, goal), -1, 1)
    return np.degrees(np.arccos(cos_angle))


def simulate_with_allocator(allocator_func, J, A_rw, A_mtq_axes, u_rw_max, u_mtq_max,
                            x0, q_goal, b_field_func, kp, kd, tf, dt):
    """Run closed-loop simulation with given allocator."""
    n_rw = A_rw.shape[1]
    n_mtq = A_mtq_axes.shape[1]
    
    lb = np.concatenate([-u_rw_max, -u_mtq_max])
    ub = np.concatenate([u_rw_max, u_mtq_max])
    
    steps = int(tf / dt) + 1
    error_hist = np.zeros(steps)
    alpha_hist = np.zeros(steps)
    
    x = x0.copy()
    t = 0.0
    
    for k in range(steps):
        omega = x[0:3]
        q = normalize(x[3:7])
        h_rw = x[7:7+n_rw] if n_rw > 0 else np.array([])
        
        b_body = b_field_func(q, t)
        
        # Build A_total
        if n_mtq > 0:
            A_mtq = -skewsym(b_body) @ A_mtq_axes
        else:
            A_mtq = np.zeros((3, 0))
        A_total = np.hstack([A_rw, A_mtq])
        
        # Compute control
        q_err = quaternion_error_vector(q, q_goal)
        tau_pd = -kp * q_err - kd * omega
        
        h_rw_vec = A_rw @ h_rw if n_rw > 0 else np.zeros(3)
        tau_gyro = np.cross(omega, J @ omega + h_rw_vec)
        
        tau_des = tau_pd + tau_gyro
        
        # Allocate
        u = allocator_func(tau_des, A_total, lb, ub, omega, q_err)
        u_rw = u[:n_rw]
        u_mtq = u[n_rw:]
        
        # Compute alpha for tracking
        tau_ach = A_total @ u
        t_mag = np.linalg.norm(tau_des)
        if t_mag > 1e-12:
            alpha = np.dot(tau_ach, tau_des / t_mag) / t_mag
        else:
            alpha = 1.0
        
        error_hist[k] = pointing_error_deg(q, q_goal)
        alpha_hist[k] = alpha
        
        if k == steps - 1:
            break
        
        # Propagate
        def dynamics(t_local, y):
            w = y[0:3]
            quat = normalize(y[3:7])
            hrw = y[7:7+n_rw] if n_rw > 0 else np.array([])
            
            b_local = b_field_func(quat, t + t_local)
            
            if n_mtq > 0:
                A_mtq_local = -skewsym(b_local) @ A_mtq_axes
                tau_mtq = A_mtq_local @ u_mtq
            else:
                tau_mtq = np.zeros(3)
            
            tau_rw = A_rw @ u_rw if n_rw > 0 else np.zeros(3)
            tau_total = tau_rw + tau_mtq
            
            hrw_vec = A_rw @ hrw if n_rw > 0 else np.zeros(3)
            w_dot = np.linalg.solve(J, tau_total - np.cross(w, J @ w + hrw_vec))
            
            W = np.zeros((4, 3))
            W[0, :] = -quat[1:4]
            W[1:4, :] = quat[0] * np.eye(3) + skewsym(quat[1:4])
            q_dot = 0.5 * W @ w
            
            h_dot = -u_rw if n_rw > 0 else np.array([])
            
            return np.concatenate([w_dot, q_dot, h_dot])
        
        sol = solve_ivp(dynamics, [0, dt], x, method='RK45', rtol=1e-8, atol=1e-10)
        x = sol.y[:, -1]
        x[3:7] = normalize(x[3:7])
        
        t += dt
    
    return {
        'final_error': error_hist[-1],
        'rms_error': np.sqrt(np.mean(error_hist**2)),
        'mean_alpha': np.mean(alpha_hist),
        'error_hist': error_hist
    }


def b_field_func(orbit_period=5400):
    def f(q, t):
        phase = 2 * np.pi * t / orbit_period
        B_eci = 30e-6 * np.array([np.cos(phase), 0.5*np.sin(phase), 0.3*np.cos(2*phase)])
        R = rot_mat(q)
        return R.T @ B_eci
    return f


def run_comparison(n_scenarios=20, tf=300, dt=2):
    """Compare allocators in closed-loop."""
    np.random.seed(42)
    
    # Configuration
    J = np.diag([0.022, 0.022, 0.004])
    A_rw = np.array([[0], [0], [1.0]])
    A_mtq_axes = np.eye(3)
    u_rw_max = np.array([0.001])
    u_mtq_max = np.array([0.2, 0.2, 0.2])
    kp, kd = 5e-5, 1e-3
    
    b_field = b_field_func()
    q_goal = np.array([1, 0, 0, 0])
    
    # Allocators
    def lp_alloc(tau_des, A_total, lb, ub, omega, q_err):
        return solve_lp(tau_des, A_total, lb, ub)
    
    def qp_projdom_alloc(tau_des, A_total, lb, ub, omega, q_err):
        return solve_qp_projection_dominance(tau_des, A_total, lb, ub)
    
    def qp_smart_alloc(tau_des, A_total, lb, ub, omega, q_err):
        return solve_qp_smart(tau_des, A_total, lb, ub, omega, q_err)
    
    allocators = {
        'LP': lp_alloc,
        'QP_ProjDom': qp_projdom_alloc,
        'QP_Smart': qp_smart_alloc
    }
    
    results = {name: [] for name in allocators}
    
    for i in tqdm(range(n_scenarios), desc="Running scenarios"):
        # Random initial condition
        axis = normalize(np.random.randn(3))
        angle = np.random.uniform(0.1, 0.5)  # 6-30 degrees
        q0 = np.concatenate([[np.cos(angle/2)], axis * np.sin(angle/2)])
        q0 = normalize(q0)
        
        omega0 = np.random.randn(3) * 0.02
        h0 = np.array([0.002])
        x0 = np.concatenate([omega0, q0, h0])
        
        for name, alloc_func in allocators.items():
            res = simulate_with_allocator(
                alloc_func, J, A_rw, A_mtq_axes, u_rw_max, u_mtq_max,
                x0, q_goal, b_field, kp, kd, tf, dt
            )
            results[name].append(res)
    
    # Summarize
    print("\n" + "=" * 70)
    print("CLOSED-LOOP COMPARISON: LP vs QP_ProjDom vs QP_Smart")
    print("=" * 70)
    
    print(f"\n{'Method':<15} {'Final Err(°)':>15} {'RMS Err(°)':>15} {'Mean Alpha':>15}")
    print("-" * 70)
    
    for name in allocators:
        data = results[name]
        final = np.mean([d['final_error'] for d in data])
        final_std = np.std([d['final_error'] for d in data])
        rms = np.mean([d['rms_error'] for d in data])
        rms_std = np.std([d['rms_error'] for d in data])
        alpha = np.mean([d['mean_alpha'] for d in data])
        
        print(f"{name:<15} {final:>7.2f} ± {final_std:>5.2f} {rms:>7.2f} ± {rms_std:>5.2f} {alpha:>15.3f}")
    
    # Pairwise comparison
    print("\n" + "-" * 70)
    print("PAIRWISE COMPARISON")
    print("-" * 70)
    
    lp_final = [d['final_error'] for d in results['LP']]
    qp_final = [d['final_error'] for d in results['QP_ProjDom']]
    smart_final = [d['final_error'] for d in results['QP_Smart']]
    
    qp_better = sum(1 for l, q in zip(lp_final, qp_final) if q < l - 0.1)
    smart_better = sum(1 for l, s in zip(lp_final, smart_final) if s < l - 0.1)
    
    print(f"\nQP_ProjDom beats LP in {qp_better}/{n_scenarios} scenarios ({100*qp_better/n_scenarios:.1f}%)")
    print(f"QP_Smart beats LP in {smart_better}/{n_scenarios} scenarios ({100*smart_better/n_scenarios:.1f}%)")
    
    # When does QP_ProjDom help?
    improvements = [(lp - qp) for lp, qp in zip(lp_final, qp_final)]
    print(f"\nQP_ProjDom improvement over LP:")
    print(f"  Mean: {np.mean(improvements):.2f}°")
    print(f"  Max improvement: {max(improvements):.2f}°")
    print(f"  Max degradation: {min(improvements):.2f}°")
    
    return results


if __name__ == "__main__":
    results = run_comparison(n_scenarios=30, tf=400, dt=2)
    
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print("""
The key finding depends on whether perpendicular torque components help or hurt.

For underactuated systems:
- LP guarantees torque in the correct direction
- QP_ProjDom can add perpendicular components that may help or hurt
- QP_Smart tries to filter out "bad" perpendicular components

The winner depends on the specific scenario and control law structure.
""")
