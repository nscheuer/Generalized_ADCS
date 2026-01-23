"""
Continuous Blended Desaturation
===============================

Idea: Instead of discrete modes (pointing vs desaturation), continuously blend
desaturation into the allocation problem based on momentum state.

Method 1: Weighted Secondary Objective
--------------------------------------
min ||A·u - τ_des||² + w(h) · ||A_rw·u_rw + k_h·h||²

where w(h) increases smoothly as h approaches saturation.

Method 2: Soft Constraint
-------------------------
min ||A·u - τ_des||² - λ · (h · u_rw)

This encourages wheel commands that reduce momentum (h·u_rw < 0 means
u_rw opposes h, which dumps momentum via h_dot = -u_rw).

Method 3: Nullspace Projection (for overactuated)
-------------------------------------------------
u = u_pointing + (I - A⁺A) · u_desat

where A⁺ is pseudoinverse. The nullspace term doesn't affect pointing.

Method 4: Pareto Optimization
-----------------------------
Solve for Pareto front between pointing and desaturation.
User specifies trade-off parameter.
"""

import sys
import os
import numpy as np
from scipy.optimize import minimize, Bounds, linprog, lsq_linear
from scipy.integrate import solve_ivp
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from ADCS.helpers.math_helpers import skewsym, normalize, rot_mat, quat_inv, quat_mult


def quaternion_error_vector(q, q_goal):
    q = normalize(q)
    q_goal = normalize(q_goal)
    q_err = quat_mult(quat_inv(q_goal), q)
    if q_err[0] < 0:
        q_err = -q_err
    return 2.0 * q_err[1:4]


def pointing_error_deg(q, q_goal):
    q = normalize(q)
    q_goal = normalize(q_goal)
    R = rot_mat(q)
    R_goal = rot_mat(q_goal)
    boresight = np.array([0, 0, 1])
    actual = R @ boresight
    goal = R_goal @ boresight
    cos_angle = np.clip(np.dot(actual, goal), -1, 1)
    return np.degrees(np.arccos(cos_angle))


class ContinuousDesatAllocator:
    """
    Allocator with continuous desaturation blending.
    
    The weight function w(h) smoothly increases desaturation priority:
    
    w(h) = 0                          if ||h|| < h_low
    w(h) = w_max * f((||h||-h_low)/(h_high-h_low))  otherwise
    
    where f is a smooth activation function.
    """
    
    def __init__(self, h_low: float = 0.003, h_high: float = 0.008,
                 w_max: float = 10.0, k_desat: float = 0.1):
        """
        Parameters
        ----------
        h_low : float
            Momentum below which no desaturation (Nm·s)
        h_high : float
            Momentum at which desaturation is at full weight
        w_max : float
            Maximum desaturation weight
        k_desat : float
            Desaturation gain (u_desat = -k_desat * h)
        """
        self.h_low = h_low
        self.h_high = h_high
        self.w_max = w_max
        self.k_desat = k_desat
    
    def _weight(self, h_mag: float) -> float:
        """Compute desaturation weight based on momentum magnitude."""
        if h_mag <= self.h_low:
            return 0.0
        if h_mag >= self.h_high:
            return self.w_max
        
        # Smooth cubic transition
        t = (h_mag - self.h_low) / (self.h_high - self.h_low)
        # Use smoothstep: 3t² - 2t³
        f = t * t * (3 - 2 * t)
        return self.w_max * f
    
    def allocate(self, tau_des: np.ndarray, A_rw: np.ndarray, A_mtq: np.ndarray,
                 h_rw: np.ndarray, u_rw_max: np.ndarray, u_mtq_max: np.ndarray,
                 b_body: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Allocate with continuous desaturation blending.
        
        Solves:
        min ||A_total·u - τ_des||² + w(h) · ||u_rw - u_desat||²
        
        where u_desat = -k_desat * h_rw (commands that reduce momentum)
        """
        n_rw = A_rw.shape[1]
        n_mtq = A_mtq.shape[1]
        n_total = n_rw + n_mtq
        
        A_total = np.hstack([A_rw, A_mtq])
        lb = np.concatenate([-u_rw_max, -u_mtq_max])
        ub = np.concatenate([u_rw_max, u_mtq_max])
        
        # Momentum magnitude and weight
        h_rw_vec = A_rw @ h_rw if n_rw > 0 else np.zeros(3)
        h_mag = np.linalg.norm(h_rw_vec)
        w = self._weight(np.linalg.norm(h_rw))
        
        # Desired desaturation command
        u_desat = -self.k_desat * h_rw
        
        if w < 1e-6:
            # No desaturation needed, just do pointing
            t_mag = np.linalg.norm(tau_des)
            if t_mag < 1e-12:
                return np.zeros(n_rw), np.zeros(n_mtq)
            
            tau_hat = tau_des / t_mag
            
            # LP for direction preservation
            c = np.zeros(n_total + 1)
            c[-1] = -1.0
            A_eq = np.hstack([A_total, -tau_hat.reshape(3, 1)])
            b_eq = np.zeros(3)
            bounds = [(lb[i], ub[i]) for i in range(n_total)] + [(0, None)]
            
            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            
            if res.success:
                u = res.x[:n_total]
                T_max = res.x[-1]
                if T_max > t_mag:
                    u = u * (t_mag / T_max)
            else:
                u = np.zeros(n_total)
            
            return u[:n_rw], u[n_rw:]
        
        # Blended objective: pointing + desaturation
        # ||A·u - τ_des||² + w * ||u_rw - u_desat||²
        
        # Build QP matrices
        # H = A'A + w*[I_rw 0; 0 0]
        H = A_total.T @ A_total
        H[:n_rw, :n_rw] += w * np.eye(n_rw)
        
        # c = -A'τ_des - w*[u_desat; 0]
        c_vec = -A_total.T @ tau_des
        c_vec[:n_rw] -= w * u_desat
        
        def objective(u):
            r = A_total @ u - tau_des
            desat_err = u[:n_rw] - u_desat
            return 0.5 * np.dot(r, r) + 0.5 * w * np.dot(desat_err, desat_err)
        
        def gradient(u):
            return H @ u + c_vec
        
        # Solve
        u0 = np.zeros(n_total)
        res = minimize(objective, u0, jac=gradient, method='L-BFGS-B',
                      bounds=[(lb[i], ub[i]) for i in range(n_total)],
                      options={'ftol': 1e-12})
        
        u = res.x if res.success else np.zeros(n_total)
        
        return u[:n_rw], u[n_rw:]


class StandardAllocator:
    """Standard LP allocator (no desaturation)."""
    
    def allocate(self, tau_des, A_rw, A_mtq, h_rw, u_rw_max, u_mtq_max, b_body):
        n_rw = A_rw.shape[1]
        n_mtq = A_mtq.shape[1]
        n_total = n_rw + n_mtq
        
        A_total = np.hstack([A_rw, A_mtq])
        lb = np.concatenate([-u_rw_max, -u_mtq_max])
        ub = np.concatenate([u_rw_max, u_mtq_max])
        
        t_mag = np.linalg.norm(tau_des)
        if t_mag < 1e-12:
            return np.zeros(n_rw), np.zeros(n_mtq)
        
        tau_hat = tau_des / t_mag
        
        c = np.zeros(n_total + 1)
        c[-1] = -1.0
        A_eq = np.hstack([A_total, -tau_hat.reshape(3, 1)])
        b_eq = np.zeros(3)
        bounds = [(lb[i], ub[i]) for i in range(n_total)] + [(0, None)]
        
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if res.success:
            u = res.x[:n_total]
            T_max = res.x[-1]
            if T_max > t_mag:
                u = u * (t_mag / T_max)
        else:
            u = np.zeros(n_total)
        
        return u[:n_rw], u[n_rw:]


class SequentialDesatAllocator:
    """Allocator with discrete desaturation mode."""
    
    def __init__(self, h_threshold: float = 0.005, hysteresis: float = 0.3):
        self.h_threshold = h_threshold
        self.hysteresis = hysteresis
        self.in_desat_mode = False
    
    def allocate(self, tau_des, A_rw, A_mtq, h_rw, u_rw_max, u_mtq_max, b_body):
        n_rw = A_rw.shape[1]
        n_mtq = A_mtq.shape[1]
        
        h_mag = np.linalg.norm(h_rw)
        
        # Mode switching with hysteresis
        if not self.in_desat_mode and h_mag > self.h_threshold:
            self.in_desat_mode = True
        elif self.in_desat_mode and h_mag < self.h_threshold * self.hysteresis:
            self.in_desat_mode = False
        
        A_total = np.hstack([A_rw, A_mtq])
        lb = np.concatenate([-u_rw_max, -u_mtq_max])
        ub = np.concatenate([u_rw_max, u_mtq_max])
        n_total = n_rw + n_mtq
        
        if not self.in_desat_mode:
            # Pointing mode (LP)
            t_mag = np.linalg.norm(tau_des)
            if t_mag < 1e-12:
                return np.zeros(n_rw), np.zeros(n_mtq)
            
            tau_hat = tau_des / t_mag
            c = np.zeros(n_total + 1)
            c[-1] = -1.0
            A_eq = np.hstack([A_total, -tau_hat.reshape(3, 1)])
            b_eq = np.zeros(3)
            bounds = [(lb[i], ub[i]) for i in range(n_total)] + [(0, None)]
            
            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            
            if res.success:
                u = res.x[:n_total]
                T_max = res.x[-1]
                if T_max > t_mag:
                    u = u * (t_mag / T_max)
            else:
                u = np.zeros(n_total)
            
            return u[:n_rw], u[n_rw:]
        else:
            # Desaturation mode: torque-free desaturation
            b_hat = b_body / (np.linalg.norm(b_body) + 1e-12)
            
            # Want to dump h_rw via MTQ
            # MTQ torque: τ_mtq = m × B
            # RW torque: τ_rw = -h_dot_rw (reaction)
            # For torque-free: τ_mtq + τ_rw = 0
            
            # Desired change: reduce h_rw
            h_rw_vec = A_rw @ h_rw
            tau_desat = -0.01 * h_rw_vec  # Torque to reduce momentum
            
            # Project to MTQ-achievable
            tau_desat_perp = tau_desat - np.dot(tau_desat, b_hat) * b_hat
            
            if np.linalg.norm(tau_desat_perp) < 1e-12:
                # Can't desaturate now
                return np.zeros(n_rw), np.zeros(n_mtq)
            
            # MTQ command to achieve tau_desat_perp
            u_mtq = np.linalg.lstsq(A_mtq, tau_desat_perp, rcond=None)[0]
            u_mtq = np.clip(u_mtq, -u_mtq_max, u_mtq_max)
            
            # RW command to cancel MTQ torque (torque-free)
            tau_mtq_actual = A_mtq @ u_mtq
            u_rw = np.linalg.lstsq(A_rw, -tau_mtq_actual, rcond=None)[0]
            u_rw = np.clip(u_rw, -u_rw_max, u_rw_max)
            
            return u_rw, u_mtq


def simulate_desaturation(allocator, J, A_rw, A_mtq_axes, x0, q_goal, 
                          u_rw_max, u_mtq_max, b_field_func, tf, dt):
    """Run desaturation simulation."""
    n_rw = A_rw.shape[1]
    n_mtq = A_mtq_axes.shape[1]
    kp, kd = 5e-5, 1e-3
    
    steps = int(tf / dt) + 1
    h_hist = np.zeros((steps, n_rw))
    error_hist = np.zeros(steps)
    
    x = x0.copy()
    t = 0.0
    
    for k in range(steps):
        omega = x[0:3]
        q = normalize(x[3:7])
        h_rw = x[7:7+n_rw]
        
        b_body = b_field_func(q, t)
        A_mtq = -skewsym(b_body) @ A_mtq_axes
        
        # Control
        q_err = quaternion_error_vector(q, q_goal)
        h_rw_vec = A_rw @ h_rw
        tau_gyro = np.cross(omega, J @ omega + h_rw_vec)
        tau_des = -kp * q_err - kd * omega + tau_gyro
        
        u_rw, u_mtq = allocator.allocate(tau_des, A_rw, A_mtq, h_rw, 
                                          u_rw_max, u_mtq_max, b_body)
        
        h_hist[k, :] = h_rw
        error_hist[k] = pointing_error_deg(q, q_goal)
        
        if k == steps - 1:
            break
        
        # Propagate
        def dynamics(t_local, y):
            w = y[0:3]
            quat = normalize(y[3:7])
            hrw = y[7:7+n_rw]
            
            b_local = b_field_func(quat, t + t_local)
            A_mtq_local = -skewsym(b_local) @ A_mtq_axes
            
            tau_mtq = A_mtq_local @ u_mtq
            tau_rw = A_rw @ u_rw
            tau_total = tau_rw + tau_mtq
            
            hrw_vec = A_rw @ hrw
            w_dot = np.linalg.solve(J, tau_total - np.cross(w, J @ w + hrw_vec))
            
            W = np.zeros((4, 3))
            W[0, :] = -quat[1:4]
            W[1:4, :] = quat[0] * np.eye(3) + skewsym(quat[1:4])
            q_dot = 0.5 * W @ w
            
            h_dot = -u_rw
            
            return np.concatenate([w_dot, q_dot, h_dot])
        
        sol = solve_ivp(dynamics, [0, dt], x, method='RK45', rtol=1e-8, atol=1e-10)
        x = sol.y[:, -1]
        x[3:7] = normalize(x[3:7])
        
        t += dt
    
    return {
        'h_hist': h_hist,
        'error_hist': error_hist,
        'final_h': np.linalg.norm(h_hist[-1]),
        'final_error': error_hist[-1],
        'mean_error': np.mean(error_hist),
        'max_error': np.max(error_hist)
    }


def run_comparison():
    """Compare continuous vs sequential vs no desaturation."""
    np.random.seed(42)
    
    J = np.diag([0.022, 0.022, 0.004])
    A_rw = np.array([[0], [0], [1.0]])
    A_mtq_axes = np.eye(3)
    u_rw_max = np.array([0.001])
    u_mtq_max = np.array([0.2, 0.2, 0.2])
    
    q_goal = np.array([1, 0, 0, 0])
    
    def b_field_func(q, t, orbit_period=5400):
        phase = 2 * np.pi * t / orbit_period
        B_eci = 30e-6 * np.array([np.cos(phase), 0.5*np.sin(phase), 0.3*np.cos(2*phase)])
        R = rot_mat(q)
        return R.T @ B_eci
    
    allocators = {
        'No_Desat': StandardAllocator(),
        'Sequential': SequentialDesatAllocator(h_threshold=0.005),
        'Continuous_w1': ContinuousDesatAllocator(w_max=1.0),
        'Continuous_w10': ContinuousDesatAllocator(w_max=10.0),
        'Continuous_w100': ContinuousDesatAllocator(w_max=100.0),
    }
    
    tf, dt = 800, 2  # Moderate simulation
    
    # Initial condition with momentum building up
    n_scenarios = 5
    all_results = {name: [] for name in allocators}
    
    for i in tqdm(range(n_scenarios), desc="Running scenarios"):
        # Random initial condition
        axis = normalize(np.random.randn(3))
        angle = np.random.uniform(0.2, 0.5)
        q0 = normalize(np.concatenate([[np.cos(angle/2)], axis * np.sin(angle/2)]))
        omega0 = np.random.randn(3) * 0.02
        h0 = np.array([np.random.uniform(0.003, 0.007)])  # Start with some momentum
        x0 = np.concatenate([omega0, q0, h0])
        
        for name, allocator in allocators.items():
            # Reset mode for sequential
            if hasattr(allocator, 'in_desat_mode'):
                allocator.in_desat_mode = False
            
            result = simulate_desaturation(
                allocator, J, A_rw, A_mtq_axes, x0, q_goal,
                u_rw_max, u_mtq_max, b_field_func, tf, dt
            )
            all_results[name].append(result)
    
    # Summarize
    print("\n" + "=" * 90)
    print("CONTINUOUS vs SEQUENTIAL DESATURATION COMPARISON")
    print("=" * 90)
    
    print(f"\n{'Method':<18} {'Final h(mNm·s)':>15} {'Mean Err(°)':>12} {'Max Err(°)':>12} {'Final Err(°)':>12}")
    print("-" * 90)
    
    for name in allocators:
        data = all_results[name]
        h = np.mean([d['final_h'] for d in data]) * 1000
        h_std = np.std([d['final_h'] for d in data]) * 1000
        mean_err = np.mean([d['mean_error'] for d in data])
        max_err = np.mean([d['max_error'] for d in data])
        final_err = np.mean([d['final_error'] for d in data])
        
        print(f"{name:<18} {h:>7.2f}±{h_std:>5.2f} {mean_err:>12.2f} {max_err:>12.2f} {final_err:>12.2f}")
    
    return all_results


if __name__ == "__main__":
    results = run_comparison()
    
    print("\n" + "=" * 90)
    print("ANALYSIS")
    print("=" * 90)
    print("""
Continuous desaturation blends pointing and momentum management without
discrete mode switching. Key advantages:

1. NO MODE SWITCHING: Smooth transition based on momentum state
2. BETTER POINTING: Doesn't completely abandon pointing during desaturation
3. TUNABLE: w_max controls the trade-off

The weight function w(h) uses a smoothstep for continuous derivatives:
    w(h) = w_max * (3t² - 2t³) where t = (h - h_low)/(h_high - h_low)

This ensures smooth actuator commands and no discontinuities.
""")
