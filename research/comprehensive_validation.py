"""
Comprehensive Validation Suite
==============================

Address critical gaps in our research:
1. Single QP closed-loop validation
2. Continuous desaturation robustness
3. Full-orbit simulations
4. Realistic disturbances
5. Actuator failure cases
"""

import sys
import os
import numpy as np
from scipy.optimize import minimize, Bounds, linprog, lsq_linear
from scipy.integrate import solve_ivp
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import time

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from ADCS.helpers.math_helpers import skewsym, normalize, rot_mat, quat_inv, quat_mult


# ============== HELPER FUNCTIONS ==============

def quaternion_error_vector(q, q_goal):
    q = normalize(q)
    q_goal = normalize(q_goal)
    q_err = quat_mult(quat_inv(q_goal), q)
    if q_err[0] < 0:
        q_err = -q_err
    return 2.0 * q_err[1:4]


def full_attitude_error_deg(q, q_goal):
    q = normalize(q)
    q_goal = normalize(q_goal)
    q_err = quat_mult(quat_inv(q_goal), q)
    if q_err[0] < 0:
        q_err = -q_err
    return np.degrees(2 * np.arccos(np.clip(q_err[0], -1, 1)))


# ============== ALLOCATORS ==============

def allocate_lp(tau_des, A_total, lb, ub):
    """Standard LP allocation."""
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-12:
        return np.zeros(len(lb))
    
    tau_hat = tau_des / t_mag
    n = len(lb)
    
    c = np.zeros(n + 1)
    c[-1] = -1.0
    A_eq = np.hstack([A_total, -tau_hat.reshape(3, 1)])
    b_eq = np.zeros(3)
    bounds = [(lb[i], ub[i]) for i in range(n)] + [(0, None)]
    
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if res.success:
        u = res.x[:n]
        T_max = res.x[-1]
        if T_max > t_mag:
            u = u * (t_mag / T_max)
        return u
    return np.zeros(n)


def allocate_single_qp(tau_des, A_total, lb, ub, w=1000.0):
    """Single QP with weighted projection priority."""
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-12:
        return np.zeros(len(lb))
    
    tau_hat = tau_des / t_mag
    n = len(lb)
    
    # Objective: -w*(A@u)·τ̂ + 0.5*||A@u - τ_des||²
    c_proj = A_total.T @ tau_hat
    H = A_total.T @ A_total
    
    def objective(u):
        r = A_total @ u - tau_des
        proj = c_proj @ u
        return 0.5 * np.dot(r, r) - w * proj
    
    def gradient(u):
        return H @ u - A_total.T @ tau_des - w * c_proj
    
    x0 = np.zeros(n)
    res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                  bounds=[(lb[i], ub[i]) for i in range(n)],
                  options={'ftol': 1e-12, 'maxiter': 100})
    
    return res.x if res.success else np.zeros(n)


def allocate_lp_then_qp(tau_des, A_total, lb, ub, omega=None):
    """Two-step LP then constrained QP."""
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-12:
        return np.zeros(len(lb))
    
    tau_hat = tau_des / t_mag
    n = len(lb)
    
    # Step 1: LP
    u_lp = allocate_lp(tau_des, A_total, lb, ub)
    proj_lp = np.dot(A_total @ u_lp, tau_hat)
    
    # Step 2: QP with projection constraint
    min_proj = proj_lp * 0.999
    
    def objective(u):
        r = A_total @ u - tau_des
        return 0.5 * np.dot(r, r)
    
    def gradient(u):
        return A_total.T @ (A_total @ u - tau_des)
    
    c_proj = A_total.T @ tau_hat
    constraints = [{
        'type': 'ineq',
        'fun': lambda u: c_proj @ u - min_proj,
        'jac': lambda u: c_proj
    }]
    
    # Add energy constraint if damping
    if omega is not None:
        omega_dot_tau = np.dot(omega, tau_des)
        if omega_dot_tau < -1e-12:
            energy_lp = np.dot(omega, A_total @ u_lp)
            c_omega = A_total.T @ omega
            constraints.append({
                'type': 'ineq',
                'fun': lambda u: energy_lp - c_omega @ u,
                'jac': lambda u: -c_omega
            })
    
    res = minimize(objective, u_lp, jac=gradient, method='SLSQP',
                  bounds=Bounds(lb, ub), constraints=constraints,
                  options={'ftol': 1e-12})
    
    return res.x if res.success else u_lp


# ============== DISTURBANCE MODELS ==============

def gravity_gradient_torque(q, J, r_orbit_eci, mu=3.986e14):
    """Compute gravity gradient torque."""
    R = rot_mat(q)
    r_body = R.T @ normalize(r_orbit_eci)
    r_mag = np.linalg.norm(r_orbit_eci)
    return 3 * mu / r_mag**3 * np.cross(r_body, J @ r_body)


def magnetic_disturbance_torque(q, m_residual, B_eci):
    """Residual magnetic dipole disturbance."""
    R = rot_mat(q)
    B_body = R.T @ B_eci
    return np.cross(m_residual, B_body)


def aerodynamic_torque(q, v_eci, rho, Cd, A_ref, cp_offset):
    """Simplified aerodynamic torque."""
    R = rot_mat(q)
    v_body = R.T @ v_eci
    v_mag = np.linalg.norm(v_body)
    if v_mag < 1e-6:
        return np.zeros(3)
    v_hat = v_body / v_mag
    F_aero = -0.5 * rho * v_mag**2 * Cd * A_ref * v_hat
    return np.cross(cp_offset, F_aero)


# ============== SIMULATION ENGINE ==============

@dataclass
class SimConfig:
    """Simulation configuration."""
    J: np.ndarray
    A_rw: np.ndarray
    A_mtq_axes: np.ndarray
    u_rw_max: np.ndarray
    u_mtq_max: np.ndarray
    kp: float
    kd: float
    orbit_period: float = 5400.0
    orbit_altitude: float = 400e3
    include_gg: bool = False
    include_mag_dist: bool = False
    include_aero: bool = False
    m_residual: np.ndarray = None
    failed_actuators: List[int] = None


def simulate(config: SimConfig, allocator_func, x0: np.ndarray, 
             q_goal: np.ndarray, tf: float, dt: float,
             continuous_desat: bool = False, desat_params: dict = None) -> Dict:
    """
    Run simulation with given configuration and allocator.
    """
    J = config.J
    A_rw = config.A_rw
    A_mtq_axes = config.A_mtq_axes
    u_rw_max = config.u_rw_max.copy()
    u_mtq_max = config.u_mtq_max.copy()
    kp, kd = config.kp, config.kd
    
    # Apply actuator failures
    if config.failed_actuators:
        for idx in config.failed_actuators:
            n_rw = len(u_rw_max)
            if idx < n_rw:
                u_rw_max[idx] = 0
            else:
                u_mtq_max[idx - n_rw] = 0
    
    n_rw = A_rw.shape[1]
    n_mtq = A_mtq_axes.shape[1]
    
    lb = np.concatenate([-u_rw_max, -u_mtq_max])
    ub = np.concatenate([u_rw_max, u_mtq_max])
    
    # Orbit parameters
    Re = 6.378e6
    r_orbit = Re + config.orbit_altitude
    v_orbit = np.sqrt(3.986e14 / r_orbit)
    
    def b_field_func(q, t):
        phase = 2 * np.pi * t / config.orbit_period
        # Simplified dipole model
        B_eci = 30e-6 * np.array([
            np.cos(phase) * 0.8,
            np.sin(phase) * 0.6,
            0.3 * np.cos(2*phase)
        ])
        R = rot_mat(q)
        return R.T @ B_eci, B_eci
    
    def orbit_state(t):
        phase = 2 * np.pi * t / config.orbit_period
        r_eci = r_orbit * np.array([np.cos(phase), np.sin(phase), 0])
        v_eci = v_orbit * np.array([-np.sin(phase), np.cos(phase), 0])
        return r_eci, v_eci
    
    # Continuous desaturation weight function
    def desat_weight(h_rw):
        if not continuous_desat or desat_params is None:
            return 0.0
        h_mag = np.linalg.norm(h_rw)
        h_low = desat_params.get('h_low', 0.003)
        h_high = desat_params.get('h_high', 0.008)
        w_max = desat_params.get('w_max', 10.0)
        
        if h_mag <= h_low:
            return 0.0
        if h_mag >= h_high:
            return w_max
        t = (h_mag - h_low) / (h_high - h_low)
        return w_max * t * t * (3 - 2*t)
    
    steps = int(tf / dt) + 1
    error_hist = np.zeros(steps)
    h_hist = np.zeros((steps, n_rw)) if n_rw > 0 else None
    
    x = x0.copy()
    t = 0.0
    
    for k in range(steps):
        omega = x[0:3]
        q = normalize(x[3:7])
        h_rw = x[7:7+n_rw] if n_rw > 0 else np.array([])
        
        b_body, B_eci = b_field_func(q, t)
        r_eci, v_eci = orbit_state(t)
        
        if n_mtq > 0:
            A_mtq = -skewsym(b_body) @ A_mtq_axes
            A_total = np.hstack([A_rw, A_mtq]) if n_rw > 0 else A_mtq
        else:
            A_total = A_rw
        
        # Compute disturbances
        tau_dist = np.zeros(3)
        if config.include_gg:
            tau_dist += gravity_gradient_torque(q, J, r_eci)
        if config.include_mag_dist and config.m_residual is not None:
            tau_dist += magnetic_disturbance_torque(q, config.m_residual, B_eci)
        if config.include_aero:
            rho = 1e-12  # Approximate density at 400km
            tau_dist += aerodynamic_torque(q, v_eci, rho, 2.2, 0.01, np.array([0.01, 0, 0]))
        
        # Control
        q_err = quaternion_error_vector(q, q_goal)
        h_rw_vec = A_rw @ h_rw if n_rw > 0 else np.zeros(3)
        tau_gyro = np.cross(omega, J @ omega + h_rw_vec)
        tau_des = -kp * q_err - kd * omega + tau_gyro
        
        # Allocate (with optional continuous desaturation)
        if continuous_desat and n_rw > 0:
            w = desat_weight(h_rw)
            if w > 0.01:
                # Modified allocation with desaturation
                u_desat = 0.1 * h_rw
                A_aug = np.vstack([A_total, np.sqrt(w) * np.hstack([np.eye(n_rw), np.zeros((n_rw, n_mtq))])])
                b_aug = np.concatenate([tau_des, np.sqrt(w) * u_desat])
                res = lsq_linear(A_aug, b_aug, bounds=(lb, ub), method='bvls')
                u = res.x if res.success else allocator_func(tau_des, A_total, lb, ub)
            else:
                u = allocator_func(tau_des, A_total, lb, ub)
        else:
            u = allocator_func(tau_des, A_total, lb, ub)
        
        u_rw = u[:n_rw] if n_rw > 0 else np.array([])
        u_mtq = u[n_rw:] if n_mtq > 0 else np.array([])
        
        error_hist[k] = full_attitude_error_deg(q, q_goal)
        if h_hist is not None:
            h_hist[k, :] = h_rw
        
        if k == steps - 1:
            break
        
        # Propagate
        def dynamics(t_local, y):
            w = y[0:3]
            quat = normalize(y[3:7])
            hrw = y[7:7+n_rw] if n_rw > 0 else np.array([])
            
            b_local, _ = b_field_func(quat, t + t_local)
            
            if n_mtq > 0:
                A_mtq_local = -skewsym(b_local) @ A_mtq_axes
                tau_mtq = A_mtq_local @ u_mtq
            else:
                tau_mtq = np.zeros(3)
            
            tau_rw = A_rw @ u_rw if n_rw > 0 else np.zeros(3)
            tau_total = tau_rw + tau_mtq + tau_dist
            
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
        'error_hist': error_hist,
        'h_hist': h_hist,
        'final_error': error_hist[-1],
        'mean_error': np.mean(error_hist),
        'max_error': np.max(error_hist),
        'converged': error_hist[-1] < 5.0,
        'final_h': np.linalg.norm(h_hist[-1]) if h_hist is not None else 0
    }


# ============== TEST SUITES ==============

def test_single_qp_closed_loop():
    """Test 1: Validate single QP in closed-loop."""
    print("\n" + "=" * 70)
    print("TEST 1: Single QP Closed-Loop Validation")
    print("=" * 70)
    
    np.random.seed(42)
    
    config = SimConfig(
        J=np.diag([0.022, 0.022, 0.004]),
        A_rw=np.array([[0], [0], [1.0]]),
        A_mtq_axes=np.eye(3),
        u_rw_max=np.array([0.001]),
        u_mtq_max=np.array([0.2, 0.2, 0.2]),
        kp=5e-5, kd=1e-3
    )
    
    q_goal = np.array([1, 0, 0, 0])
    tf, dt = 400, 2
    n_tests = 20
    
    allocators = {
        'LP': allocate_lp,
        'LP+QP': lambda t, A, l, u: allocate_lp_then_qp(t, A, l, u),
        'SingleQP_w100': lambda t, A, l, u: allocate_single_qp(t, A, l, u, w=100),
        'SingleQP_w1000': lambda t, A, l, u: allocate_single_qp(t, A, l, u, w=1000),
        'SingleQP_w10000': lambda t, A, l, u: allocate_single_qp(t, A, l, u, w=10000),
    }
    
    results = {name: [] for name in allocators}
    
    for i in tqdm(range(n_tests), desc="Testing allocators"):
        axis = normalize(np.random.randn(3))
        angle = np.random.uniform(0.2, 0.6)
        q0 = normalize(np.concatenate([[np.cos(angle/2)], axis*np.sin(angle/2)]))
        omega0 = np.random.randn(3) * 0.02
        h0 = np.array([0.002])
        x0 = np.concatenate([omega0, q0, h0])
        
        for name, alloc in allocators.items():
            res = simulate(config, alloc, x0, q_goal, tf, dt)
            results[name].append(res)
    
    print(f"\n{'Allocator':<20} {'Final Err':>12} {'Mean Err':>12} {'Conv Rate':>12}")
    print("-" * 60)
    
    for name in allocators:
        data = results[name]
        final = np.mean([d['final_error'] for d in data])
        mean = np.mean([d['mean_error'] for d in data])
        conv = np.mean([d['converged'] for d in data])
        print(f"{name:<20} {final:>12.2f}° {mean:>12.2f}° {100*conv:>11.0f}%")
    
    return results


def test_continuous_desat_robustness():
    """Test 2: Continuous desaturation robustness."""
    print("\n" + "=" * 70)
    print("TEST 2: Continuous Desaturation Robustness")
    print("=" * 70)
    
    np.random.seed(42)
    
    config = SimConfig(
        J=np.diag([0.022, 0.022, 0.004]),
        A_rw=np.array([[0], [0], [1.0]]),
        A_mtq_axes=np.eye(3),
        u_rw_max=np.array([0.001]),
        u_mtq_max=np.array([0.2, 0.2, 0.2]),
        kp=5e-5, kd=1e-3
    )
    
    q_goal = np.array([1, 0, 0, 0])
    tf, dt = 600, 2
    n_tests = 15
    
    desat_configs = [
        ('No desat', False, None),
        ('Continuous w=1', True, {'h_low': 0.003, 'h_high': 0.008, 'w_max': 1}),
        ('Continuous w=10', True, {'h_low': 0.003, 'h_high': 0.008, 'w_max': 10}),
        ('Continuous w=50', True, {'h_low': 0.003, 'h_high': 0.008, 'w_max': 50}),
    ]
    
    results = {name: [] for name, _, _ in desat_configs}
    
    for i in tqdm(range(n_tests), desc="Testing desaturation"):
        axis = normalize(np.random.randn(3))
        angle = np.random.uniform(0.2, 0.6)
        q0 = normalize(np.concatenate([[np.cos(angle/2)], axis*np.sin(angle/2)]))
        omega0 = np.random.randn(3) * 0.02
        h0 = np.array([np.random.uniform(0.004, 0.007)])  # Start with momentum
        x0 = np.concatenate([omega0, q0, h0])
        
        for name, use_desat, params in desat_configs:
            res = simulate(config, allocate_lp, x0, q_goal, tf, dt,
                          continuous_desat=use_desat, desat_params=params)
            results[name].append(res)
    
    print(f"\n{'Config':<20} {'Final h(mNm·s)':>15} {'Final Err':>12} {'Mean Err':>12}")
    print("-" * 70)
    
    for name, _, _ in desat_configs:
        data = results[name]
        h = np.mean([d['final_h'] for d in data]) * 1000
        h_std = np.std([d['final_h'] for d in data]) * 1000
        final = np.mean([d['final_error'] for d in data])
        mean = np.mean([d['mean_error'] for d in data])
        print(f"{name:<20} {h:>7.2f}±{h_std:>5.2f} {final:>12.2f}° {mean:>12.2f}°")
    
    return results


def test_full_orbit():
    """Test 3: Full orbit simulation."""
    print("\n" + "=" * 70)
    print("TEST 3: Full Orbit Simulation (5400s)")
    print("=" * 70)
    
    np.random.seed(42)
    
    config = SimConfig(
        J=np.diag([0.022, 0.022, 0.004]),
        A_rw=np.array([[0], [0], [1.0]]),
        A_mtq_axes=np.eye(3),
        u_rw_max=np.array([0.001]),
        u_mtq_max=np.array([0.2, 0.2, 0.2]),
        kp=5e-5, kd=1e-3,
        include_gg=True  # Include gravity gradient
    )
    
    q_goal = np.array([1, 0, 0, 0])
    tf, dt = 5400, 5  # Full orbit
    n_tests = 5
    
    allocators = {
        'LP': allocate_lp,
        'LP+QP': lambda t, A, l, u: allocate_lp_then_qp(t, A, l, u),
    }
    
    results = {name: [] for name in allocators}
    
    for i in tqdm(range(n_tests), desc="Full orbit tests"):
        axis = normalize(np.random.randn(3))
        angle = np.random.uniform(0.3, 0.5)
        q0 = normalize(np.concatenate([[np.cos(angle/2)], axis*np.sin(angle/2)]))
        omega0 = np.random.randn(3) * 0.01
        h0 = np.array([0.001])
        x0 = np.concatenate([omega0, q0, h0])
        
        for name, alloc in allocators.items():
            res = simulate(config, alloc, x0, q_goal, tf, dt)
            results[name].append(res)
    
    print(f"\n{'Allocator':<15} {'Final Err':>12} {'Mean Err':>12} {'Max Err':>12}")
    print("-" * 55)
    
    for name in allocators:
        data = results[name]
        final = np.mean([d['final_error'] for d in data])
        mean = np.mean([d['mean_error'] for d in data])
        max_e = np.mean([d['max_error'] for d in data])
        print(f"{name:<15} {final:>12.2f}° {mean:>12.2f}° {max_e:>12.2f}°")
    
    return results


def test_actuator_failures():
    """Test 4: Actuator failure robustness."""
    print("\n" + "=" * 70)
    print("TEST 4: Actuator Failure Robustness")
    print("=" * 70)
    
    np.random.seed(42)
    
    base_config = {
        'J': np.diag([0.022, 0.022, 0.004]),
        'A_rw': np.eye(3),  # 3 RWs
        'A_mtq_axes': np.eye(3),
        'u_rw_max': np.array([0.001, 0.001, 0.001]),
        'u_mtq_max': np.array([0.2, 0.2, 0.2]),
        'kp': 5e-5, 'kd': 1e-3
    }
    
    failure_scenarios = [
        ('No failures', None),
        ('RW 0 failed', [0]),
        ('RW 1 failed', [1]),
        ('MTQ 0 failed', [3]),
        ('RW 0 + MTQ 1 failed', [0, 4]),
    ]
    
    q_goal = np.array([1, 0, 0, 0])
    tf, dt = 300, 2
    n_tests = 10
    
    results = {name: [] for name, _ in failure_scenarios}
    
    for i in tqdm(range(n_tests), desc="Testing failures"):
        axis = normalize(np.random.randn(3))
        angle = np.random.uniform(0.2, 0.5)
        q0 = normalize(np.concatenate([[np.cos(angle/2)], axis*np.sin(angle/2)]))
        omega0 = np.random.randn(3) * 0.02
        h0 = np.zeros(3)
        x0 = np.concatenate([omega0, q0, h0])
        
        for name, failures in failure_scenarios:
            config = SimConfig(**base_config, failed_actuators=failures)
            res = simulate(config, allocate_lp, x0, q_goal, tf, dt)
            results[name].append(res)
    
    print(f"\n{'Scenario':<25} {'Final Err':>12} {'Conv Rate':>12}")
    print("-" * 55)
    
    for name, _ in failure_scenarios:
        data = results[name]
        final = np.mean([d['final_error'] for d in data])
        conv = np.mean([d['converged'] for d in data])
        print(f"{name:<25} {final:>12.2f}° {100*conv:>11.0f}%")
    
    return results


def run_all_tests():
    """Run all validation tests."""
    print("=" * 70)
    print("COMPREHENSIVE VALIDATION SUITE")
    print("=" * 70)
    
    results = {}
    
    results['single_qp'] = test_single_qp_closed_loop()
    results['desat'] = test_continuous_desat_robustness()
    results['full_orbit'] = test_full_orbit()
    results['failures'] = test_actuator_failures()
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
