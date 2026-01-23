"""
Reachability-Aware Control Strategies
=====================================

Explore methods that use knowledge of:
1. Current angular velocity (momentum) 
2. Actuator capabilities
3. B-field variations over time

to choose better control actions and goals.

Key questions:
1. Can we exploit existing angular velocity to reach goals faster?
2. Should we plan maneuvers around favorable B-field windows?
3. Can adaptive goal selection improve convergence?
"""

import sys
import os
import numpy as np
from scipy.optimize import minimize, Bounds, linprog
from scipy.integrate import solve_ivp
from typing import Tuple, Dict, List, Optional, Callable
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


def full_attitude_error_rad(q, q_goal):
    q = normalize(q)
    q_goal = normalize(q_goal)
    q_err = quat_mult(quat_inv(q_goal), q)
    if q_err[0] < 0:
        q_err = -q_err
    return 2 * np.arccos(np.clip(q_err[0], -1, 1))


def quaternion_from_axis_angle(axis, angle):
    """Create quaternion from axis-angle."""
    axis = normalize(axis)
    return np.concatenate([[np.cos(angle/2)], axis * np.sin(angle/2)])


def quat_rotate_vector(q, v):
    """Rotate vector v by quaternion q."""
    R = rot_mat(q)
    return R @ v


class ReachabilityAnalyzer:
    """
    Analyze reachability and controllability given current state and actuators.
    """
    
    def __init__(self, J: np.ndarray, A_rw: np.ndarray, A_mtq_axes: np.ndarray,
                 u_rw_max: np.ndarray, u_mtq_max: np.ndarray):
        self.J = J
        self.A_rw = A_rw
        self.A_mtq_axes = A_mtq_axes
        self.u_rw_max = u_rw_max
        self.u_mtq_max = u_mtq_max
    
    def torque_capability(self, b_body: np.ndarray, direction: np.ndarray) -> float:
        """
        Compute maximum achievable torque in given direction.
        
        Returns the maximum alpha such that tau = alpha * ||direction|| * direction_hat
        is achievable.
        """
        b_body = np.asarray(b_body)
        direction = np.asarray(direction)
        
        d_mag = np.linalg.norm(direction)
        if d_mag < 1e-12:
            return 1.0
        
        d_hat = direction / d_mag
        
        # Build total torque matrix
        A_mtq = -skewsym(b_body) @ self.A_mtq_axes
        A_total = np.hstack([self.A_rw, A_mtq])
        
        lb = np.concatenate([-self.u_rw_max, -self.u_mtq_max])
        ub = np.concatenate([self.u_rw_max, self.u_mtq_max])
        
        # LP to maximize torque in direction
        n_act = len(lb)
        c = np.zeros(n_act + 1)
        c[-1] = -1.0
        
        A_eq = np.hstack([A_total, -d_hat.reshape(3, 1)])
        b_eq = np.zeros(3)
        
        bounds = [(lb[i], ub[i]) for i in range(n_act)] + [(0, None)]
        
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if res.success:
            return res.x[-1] / d_mag if d_mag > 0 else 1.0
        return 0.0
    
    def angular_acceleration_capability(self, b_body: np.ndarray, 
                                         direction: np.ndarray) -> float:
        """Maximum angular acceleration in given direction."""
        max_torque = self.torque_capability(b_body, direction)
        J_eff = direction @ self.J @ direction / (np.linalg.norm(direction)**2 + 1e-12)
        return max_torque / J_eff if J_eff > 1e-12 else 0.0
    
    def estimate_maneuver_time(self, q_current: np.ndarray, q_goal: np.ndarray,
                               omega_current: np.ndarray, b_body: np.ndarray) -> float:
        """
        Estimate time to reach q_goal from current state.
        
        Simple estimate based on rotation angle and acceleration capability.
        """
        q_err_vec = quaternion_error_vector(q_current, q_goal)
        angle = np.linalg.norm(q_err_vec)  # Approximate for small angles
        
        if angle < 1e-6:
            return 0.0
        
        axis = q_err_vec / angle
        
        # Acceleration capability in rotation direction
        alpha_max = self.angular_acceleration_capability(b_body, axis)
        
        # Current velocity projection
        omega_proj = np.dot(omega_current, axis)
        
        if alpha_max < 1e-12:
            # Can't accelerate in this direction
            if abs(omega_proj) > 1e-6:
                # Will we coast to the goal?
                coast_time = angle / abs(omega_proj)
                return coast_time
            return float('inf')
        
        # Bang-bang maneuver estimate
        # Time = 2 * sqrt(angle / alpha_max) for symmetric accel/decel
        bang_bang_time = 2 * np.sqrt(angle / alpha_max)
        
        # Adjust for initial velocity
        if omega_proj > 0:
            # Already moving toward goal
            bang_bang_time *= 0.7
        elif omega_proj < 0:
            # Moving away - need to reverse
            bang_bang_time *= 1.5
        
        return bang_bang_time


class ReachabilityAwareController:
    """
    Controller that uses reachability analysis to:
    1. Choose better intermediate goals
    2. Exploit existing angular velocity
    3. Time maneuvers around favorable B-field conditions
    """
    
    def __init__(self, J: np.ndarray, A_rw: np.ndarray, A_mtq_axes: np.ndarray,
                 u_rw_max: np.ndarray, u_mtq_max: np.ndarray,
                 kp: float, kd: float):
        self.J = J
        self.A_rw = A_rw
        self.A_mtq_axes = A_mtq_axes
        self.u_rw_max = u_rw_max
        self.u_mtq_max = u_mtq_max
        self.kp = kp
        self.kd = kd
        
        self.analyzer = ReachabilityAnalyzer(J, A_rw, A_mtq_axes, u_rw_max, u_mtq_max)
    
    def compute_optimal_intermediate_goal(self, q_current: np.ndarray, 
                                           q_final: np.ndarray,
                                           omega_current: np.ndarray,
                                           b_body: np.ndarray,
                                           n_candidates: int = 8) -> np.ndarray:
        """
        Find intermediate goal that exploits current momentum and B-field.
        
        Instead of going directly to q_final, find a nearby goal that:
        1. Is easier to reach given current omega
        2. Has better torque authority given current B
        3. Eventually leads to q_final
        """
        q_err_vec = quaternion_error_vector(q_current, q_final)
        err_mag = np.linalg.norm(q_err_vec)
        
        if err_mag < 0.05:  # Close enough, go direct
            return q_final
        
        # Generate candidate intermediate goals
        # These are rotations that partially move toward q_final
        candidates = []
        
        # Direct path
        candidates.append(q_final)
        
        # Paths that align with current omega
        if np.linalg.norm(omega_current) > 1e-6:
            omega_hat = omega_current / np.linalg.norm(omega_current)
            
            # Rotate in omega direction by various amounts
            for frac in [0.25, 0.5, 0.75]:
                angle = frac * err_mag
                q_via_omega = quat_mult(
                    quaternion_from_axis_angle(omega_hat, angle),
                    q_current
                )
                candidates.append(q_via_omega)
        
        # Paths that align with controllable direction (perp to B)
        b_hat = b_body / (np.linalg.norm(b_body) + 1e-12)
        
        for axis in [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]:
            # Project to perpendicular of B
            axis_perp = axis - np.dot(axis, b_hat) * b_hat
            if np.linalg.norm(axis_perp) > 0.1:
                axis_perp = normalize(axis_perp)
                for frac in [0.3, 0.6]:
                    angle = frac * err_mag
                    q_via_axis = quat_mult(
                        quaternion_from_axis_angle(axis_perp, angle),
                        q_current
                    )
                    candidates.append(q_via_axis)
        
        # Score candidates
        best_score = float('inf')
        best_goal = q_final
        
        for q_cand in candidates:
            # Score = estimated time to candidate + estimated time from candidate to final
            time_to_cand = self.analyzer.estimate_maneuver_time(
                q_current, q_cand, omega_current, b_body)
            
            # Rough estimate for remaining
            cand_to_final = full_attitude_error_rad(q_cand, q_final)
            
            score = time_to_cand + cand_to_final * 50  # Weight final distance
            
            if score < best_score and not np.isnan(score) and not np.isinf(score):
                best_score = score
                best_goal = q_cand
        
        return best_goal
    
    def compute_torque(self, omega: np.ndarray, q: np.ndarray, 
                       q_goal: np.ndarray, h_rw: np.ndarray,
                       b_body: np.ndarray) -> np.ndarray:
        """
        Compute control torque using reachability-aware strategy.
        """
        # Find optimal intermediate goal
        q_target = self.compute_optimal_intermediate_goal(q, q_goal, omega, b_body)
        
        # Standard PD control to intermediate goal
        q_err = quaternion_error_vector(q, q_target)
        h_rw_vec = self.A_rw @ h_rw
        tau_gyro = np.cross(omega, self.J @ omega + h_rw_vec)
        
        tau_des = -self.kp * q_err - self.kd * omega + tau_gyro
        
        return tau_des, q_target


class StandardController:
    """Standard PD controller for comparison."""
    
    def __init__(self, J, kp, kd, A_rw):
        self.J = J
        self.kp = kp
        self.kd = kd
        self.A_rw = A_rw
    
    def compute_torque(self, omega, q, q_goal, h_rw, b_body):
        q_err = quaternion_error_vector(q, q_goal)
        h_rw_vec = self.A_rw @ h_rw
        tau_gyro = np.cross(omega, self.J @ omega + h_rw_vec)
        tau_des = -self.kp * q_err - self.kd * omega + tau_gyro
        return tau_des, q_goal


def simulate_controller(controller, allocator, J, A_rw, A_mtq_axes, 
                        u_rw_max, u_mtq_max, x0, q_goal, b_field_func,
                        tf, dt):
    """Run simulation with given controller and allocator."""
    lb = np.concatenate([-u_rw_max, -u_mtq_max])
    ub = np.concatenate([u_rw_max, u_mtq_max])
    
    steps = int(tf / dt) + 1
    error_hist = np.zeros(steps)
    
    x = x0.copy()
    t = 0.0
    
    for k in range(steps):
        omega = x[0:3]
        q = normalize(x[3:7])
        h_rw = x[7:8]
        
        b_body = b_field_func(q, t)
        A_mtq = -skewsym(b_body) @ A_mtq_axes
        A_total = np.hstack([A_rw, A_mtq])
        
        tau_des, q_target = controller.compute_torque(omega, q, q_goal, h_rw, b_body)
        
        # Allocate
        t_mag = np.linalg.norm(tau_des)
        if t_mag > 1e-12:
            tau_hat = tau_des / t_mag
            n_act = 4
            c = np.zeros(n_act + 1)
            c[-1] = -1.0
            A_eq = np.hstack([A_total, -tau_hat.reshape(3, 1)])
            b_eq = np.zeros(3)
            bounds_lp = [(lb[i], ub[i]) for i in range(n_act)] + [(0, None)]
            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds_lp, method='highs')
            if res.success:
                u = res.x[:n_act]
                T_max = res.x[-1]
                if T_max > t_mag:
                    u = u * (t_mag / T_max)
            else:
                u = np.zeros(4)
        else:
            u = np.zeros(4)
        
        u_rw = u[:1]
        u_mtq = u[1:]
        
        error_hist[k] = np.degrees(full_attitude_error_rad(q, q_goal))
        
        if k == steps - 1:
            break
        
        # Propagate
        def dynamics(t_local, y):
            w = y[0:3]
            quat = normalize(y[3:7])
            hrw = y[7:8]
            
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
        'final_error': error_hist[-1],
        'rms_error': np.sqrt(np.mean(error_hist**2)),
        'min_error': np.min(error_hist),
        'error_hist': error_hist
    }


def run_comparison():
    """Compare standard vs reachability-aware control."""
    np.random.seed(42)
    
    J = np.diag([0.022, 0.022, 0.004])
    A_rw = np.array([[0], [0], [1.0]])
    A_mtq_axes = np.eye(3)
    u_rw_max = np.array([0.001])
    u_mtq_max = np.array([0.2, 0.2, 0.2])
    kp, kd = 5e-5, 1e-3
    
    tf, dt = 400, 2
    
    q_goal = np.array([1, 0, 0, 0])
    
    def b_field_func(q, t, orbit_period=5400):
        phase = 2 * np.pi * t / orbit_period
        B_eci = 30e-6 * np.array([np.cos(phase), 0.5*np.sin(phase), 0.3*np.cos(2*phase)])
        R = rot_mat(q)
        return R.T @ B_eci
    
    standard_ctrl = StandardController(J, kp, kd, A_rw)
    reachability_ctrl = ReachabilityAwareController(J, A_rw, A_mtq_axes, 
                                                     u_rw_max, u_mtq_max, kp, kd)
    
    n_scenarios = 30
    results_standard = []
    results_reachability = []
    
    for i in tqdm(range(n_scenarios), desc="Running scenarios"):
        # Random initial condition with significant angular velocity
        axis = normalize(np.random.randn(3))
        angle = np.random.uniform(0.2, 0.7)
        q0 = normalize(np.concatenate([[np.cos(angle/2)], axis * np.sin(angle/2)]))
        omega0 = np.random.randn(3) * 0.03  # Significant initial rate
        h0 = np.array([0.002])
        x0 = np.concatenate([omega0, q0, h0])
        
        res_std = simulate_controller(standard_ctrl, 'LP', J, A_rw, A_mtq_axes,
                                       u_rw_max, u_mtq_max, x0, q_goal, 
                                       b_field_func, tf, dt)
        results_standard.append(res_std)
        
        res_reach = simulate_controller(reachability_ctrl, 'LP', J, A_rw, A_mtq_axes,
                                         u_rw_max, u_mtq_max, x0, q_goal,
                                         b_field_func, tf, dt)
        results_reachability.append(res_reach)
    
    # Summarize
    print("\n" + "=" * 70)
    print("REACHABILITY-AWARE CONTROL COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Method':<20} {'Final Err (°)':>15} {'RMS Err (°)':>15} {'Min Err (°)':>15}")
    print("-" * 70)
    
    for name, data in [('Standard', results_standard), ('Reachability', results_reachability)]:
        final = np.mean([d['final_error'] for d in data])
        final_std = np.std([d['final_error'] for d in data])
        rms = np.mean([d['rms_error'] for d in data])
        min_err = np.mean([d['min_error'] for d in data])
        print(f"{name:<20} {final:>7.2f} ± {final_std:>5.2f} {rms:>15.2f} {min_err:>15.2f}")
    
    # Pairwise
    std_finals = [d['final_error'] for d in results_standard]
    reach_finals = [d['final_error'] for d in results_reachability]
    
    better = sum(1 for s, r in zip(std_finals, reach_finals) if r < s - 0.5)
    worse = sum(1 for s, r in zip(std_finals, reach_finals) if r > s + 0.5)
    
    print(f"\nReachability vs Standard: Better {better}, Worse {worse}, Same {n_scenarios - better - worse}")
    print(f"Average improvement: {np.mean([s-r for s,r in zip(std_finals, reach_finals)]):.2f}°")
    
    return results_standard, results_reachability


if __name__ == "__main__":
    results = run_comparison()
    
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print("""
The reachability-aware controller attempts to:
1. Find intermediate goals that exploit current angular velocity
2. Choose paths that have better torque authority given B-field
3. Plan maneuvers that work WITH the system dynamics

Benefits are most visible when:
- Initial angular velocity is significant
- Direct path has poor controllability
- B-field makes certain directions hard to control

Limitations:
- Added computational cost for goal selection
- May not help when already well-controlled
- Heuristic time estimation may not be accurate
""")
