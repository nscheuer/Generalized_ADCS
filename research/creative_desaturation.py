"""
Creative Desaturation Strategies
================================

Explore unconventional approaches to momentum management:

1. Sequential control: Point with MTQs, slew with RW, point back with MTQs
2. Oscillating strategies: Let spacecraft rock while desaturating
3. Relaxed-attitude desaturation: Accept temporary pointing degradation
4. Hybrid goal tracking: Track different goals for different DOFs

Mathematical analysis:
- When can "weird" strategies still guarantee convergence?
- What's the trade-off between pointing accuracy and desaturation rate?
"""

import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))

from ADCS.helpers.math_helpers import normalize, rot_mat, skewsym, quat_inv, quat_mult


@dataclass
class DesatResult:
    """Result from desaturation simulation."""
    time: np.ndarray
    h_hist: np.ndarray  # RW momentum history
    pointing_error_hist: np.ndarray  # degrees
    final_momentum: float
    final_pointing_error: float
    momentum_change_rate: float  # Average |dh/dt|
    method: str


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


class DesaturationController:
    """Base class for desaturation controllers."""
    
    def __init__(self, J, A_rw, A_mtq_axes, u_rw_max, u_mtq_max, kp, kd, kh):
        self.J = J
        self.A_rw = A_rw
        self.A_mtq_axes = A_mtq_axes
        self.u_rw_max = u_rw_max
        self.u_mtq_max = u_mtq_max
        self.kp = kp
        self.kd = kd
        self.kh = kh  # Desaturation gain
    
    def compute_command(self, omega, q, h_rw, q_goal, b_body, t) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (u_rw, u_mtq)."""
        raise NotImplementedError


class StandardDesatController(DesaturationController):
    """
    Standard approach: Try to point and desaturate simultaneously.
    Uses torque-free desaturation when possible, accepts degradation otherwise.
    """
    
    def compute_command(self, omega, q, h_rw, q_goal, b_body, t):
        n_rw = self.A_rw.shape[1]
        n_mtq = self.A_mtq_axes.shape[1]
        
        b_norm = np.linalg.norm(b_body)
        if b_norm < 1e-12:
            return np.zeros(n_rw), np.zeros(n_mtq)
        
        b_hat = b_body / b_norm
        
        # Build MTQ torque matrix
        A_mtq = -skewsym(b_body) @ self.A_mtq_axes
        
        # Pointing torque
        q_err = quaternion_error_vector(q, q_goal)
        tau_pointing = -self.kp * q_err - self.kd * omega
        
        # Gyroscopic compensation
        h_rw_vec = self.A_rw @ h_rw
        tau_gyro = np.cross(omega, self.J @ omega + h_rw_vec)
        
        # Total desired torque for pointing
        tau_des = tau_pointing + tau_gyro
        
        # Desaturation torque (want to reduce h_rw)
        h_target = np.zeros(n_rw)
        h_error = h_rw - h_target
        tau_desat = -self.kh * self.A_rw @ h_error  # Torque on body to reduce h
        
        # Project desaturation torque to MTQ-achievable plane
        tau_desat_perp = tau_desat - np.dot(tau_desat, b_hat) * b_hat
        
        # Try torque-free desaturation: MTQ provides -tau_desat, RW provides +tau_desat
        # Net body torque = 0
        
        # Can MTQ achieve tau_desat_perp?
        u_mtq_desat = np.linalg.lstsq(A_mtq, tau_desat_perp, rcond=None)[0]
        
        # Check saturation
        desat_scale = 1.0
        if np.any(np.abs(u_mtq_desat) > self.u_mtq_max):
            desat_scale = np.min(self.u_mtq_max / (np.abs(u_mtq_desat) + 1e-12))
            u_mtq_desat = u_mtq_desat * desat_scale
        
        # RW torque to cancel MTQ desaturation torque
        tau_mtq_actual = A_mtq @ u_mtq_desat
        u_rw_desat = np.linalg.lstsq(self.A_rw, -tau_mtq_actual, rcond=None)[0]
        u_rw_desat = np.clip(u_rw_desat, -self.u_rw_max, self.u_rw_max)
        
        # Also apply pointing torque via MTQ (use remaining capacity)
        tau_pointing_perp = tau_des - np.dot(tau_des, b_hat) * b_hat
        u_mtq_point = np.linalg.lstsq(A_mtq, tau_pointing_perp, rcond=None)[0]
        
        # Combine (with saturation check)
        u_mtq = u_mtq_desat + u_mtq_point
        u_mtq = np.clip(u_mtq, -self.u_mtq_max, self.u_mtq_max)
        
        # RW handles what MTQ can't
        tau_mtq_actual = A_mtq @ u_mtq
        tau_remaining = tau_des - tau_mtq_actual
        u_rw_point = np.linalg.lstsq(self.A_rw, tau_remaining, rcond=None)[0]
        
        u_rw = u_rw_desat + u_rw_point
        u_rw = np.clip(u_rw, -self.u_rw_max, self.u_rw_max)
        
        return u_rw, u_mtq


class SequentialDesatController(DesaturationController):
    """
    Sequential strategy: Alternate between pointing and desaturation phases.
    
    Phase 1: Point using MTQs only (RW torque = 0, let h accumulate if needed)
    Phase 2: Desaturate using RW (apply RW torque, MTQ cancels for torque-free)
    
    Switch phases based on h level and pointing error.
    """
    
    def __init__(self, *args, h_threshold=0.005, error_threshold=5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.h_threshold = h_threshold
        self.error_threshold = error_threshold
        self.phase = 'pointing'
    
    def compute_command(self, omega, q, h_rw, q_goal, b_body, t):
        n_rw = self.A_rw.shape[1]
        n_mtq = self.A_mtq_axes.shape[1]
        
        b_norm = np.linalg.norm(b_body)
        if b_norm < 1e-12:
            return np.zeros(n_rw), np.zeros(n_mtq)
        
        b_hat = b_body / b_norm
        A_mtq = -skewsym(b_body) @ self.A_mtq_axes
        
        h_rw_vec = self.A_rw @ h_rw
        h_mag = np.linalg.norm(h_rw_vec)
        error = pointing_error_deg(q, q_goal)
        
        # Phase switching logic
        if self.phase == 'pointing' and h_mag > self.h_threshold:
            self.phase = 'desaturating'
        elif self.phase == 'desaturating' and h_mag < self.h_threshold * 0.5:
            self.phase = 'pointing'
        
        if self.phase == 'pointing':
            # Pure pointing with MTQs, RW only for what MTQ can't do
            q_err = quaternion_error_vector(q, q_goal)
            tau_des = -self.kp * q_err - self.kd * omega
            tau_gyro = np.cross(omega, self.J @ omega + h_rw_vec)
            tau_total = tau_des + tau_gyro
            
            # Project to MTQ plane
            tau_perp = tau_total - np.dot(tau_total, b_hat) * b_hat
            u_mtq = np.linalg.lstsq(A_mtq, tau_perp, rcond=None)[0]
            u_mtq = np.clip(u_mtq, -self.u_mtq_max, self.u_mtq_max)
            
            # RW only for parallel component (can't be achieved by MTQ)
            tau_parallel = np.dot(tau_total, b_hat) * b_hat
            u_rw = np.linalg.lstsq(self.A_rw, tau_parallel, rcond=None)[0]
            u_rw = np.clip(u_rw, -self.u_rw_max, self.u_rw_max)
            
        else:  # desaturating
            # Pure desaturation: RW dumps, MTQ cancels
            tau_desat = -self.kh * h_rw_vec
            
            # Can only dump perpendicular to B
            tau_desat_perp = tau_desat - np.dot(tau_desat, b_hat) * b_hat
            
            if np.linalg.norm(tau_desat_perp) < 1e-12:
                # Can't desaturate right now, just do light pointing
                u_rw = np.zeros(n_rw)
                q_err = quaternion_error_vector(q, q_goal)
                tau_light = -self.kp * 0.1 * q_err  # Reduced gain
                tau_perp = tau_light - np.dot(tau_light, b_hat) * b_hat
                u_mtq = np.linalg.lstsq(A_mtq, tau_perp, rcond=None)[0]
                u_mtq = np.clip(u_mtq, -self.u_mtq_max, self.u_mtq_max)
            else:
                # Torque-free desaturation
                u_mtq = np.linalg.lstsq(A_mtq, tau_desat_perp, rcond=None)[0]
                u_mtq = np.clip(u_mtq, -self.u_mtq_max, self.u_mtq_max)
                
                tau_mtq_actual = A_mtq @ u_mtq
                u_rw = np.linalg.lstsq(self.A_rw, -tau_mtq_actual, rcond=None)[0]
                u_rw = np.clip(u_rw, -self.u_rw_max, self.u_rw_max)
        
        return u_rw, u_mtq


class OscillatingDesatController(DesaturationController):
    """
    Oscillating strategy: Intentionally let attitude oscillate to enable desaturation.
    
    Idea: If we're stuck (B parallel to needed torque), rock the spacecraft
    to change the B-field in body frame, enabling desaturation windows.
    
    This trades pointing accuracy for desaturation capability.
    """
    
    def __init__(self, *args, oscillation_amplitude=0.1, oscillation_period=60, **kwargs):
        super().__init__(*args, **kwargs)
        self.oscillation_amplitude = oscillation_amplitude  # radians
        self.oscillation_period = oscillation_period  # seconds
    
    def compute_command(self, omega, q, h_rw, q_goal, b_body, t):
        n_rw = self.A_rw.shape[1]
        n_mtq = self.A_mtq_axes.shape[1]
        
        b_norm = np.linalg.norm(b_body)
        if b_norm < 1e-12:
            return np.zeros(n_rw), np.zeros(n_mtq)
        
        b_hat = b_body / b_norm
        A_mtq = -skewsym(b_body) @ self.A_mtq_axes
        
        h_rw_vec = self.A_rw @ h_rw
        h_mag = np.linalg.norm(h_rw_vec)
        
        # Check if we can desaturate with current B
        tau_desat_ideal = -self.kh * h_rw_vec
        tau_desat_perp = tau_desat_ideal - np.dot(tau_desat_ideal, b_hat) * b_hat
        desat_capability = np.linalg.norm(tau_desat_perp) / (np.linalg.norm(tau_desat_ideal) + 1e-12)
        
        # If good capability, do standard control
        if desat_capability > 0.5:
            q_err = quaternion_error_vector(q, q_goal)
            tau_des = -self.kp * q_err - self.kd * omega
            tau_gyro = np.cross(omega, self.J @ omega + h_rw_vec)
            tau_total = tau_des + tau_gyro
            
            # Add desaturation
            tau_with_desat = tau_total + tau_desat_perp
            tau_perp = tau_with_desat - np.dot(tau_with_desat, b_hat) * b_hat
            
            u_mtq = np.linalg.lstsq(A_mtq, tau_perp, rcond=None)[0]
            u_mtq = np.clip(u_mtq, -self.u_mtq_max, self.u_mtq_max)
            
            tau_remaining = tau_total - A_mtq @ u_mtq
            u_rw = np.linalg.lstsq(self.A_rw, tau_remaining, rcond=None)[0]
            u_rw = np.clip(u_rw, -self.u_rw_max, self.u_rw_max)
        
        else:
            # Poor capability - add oscillation to change B in body frame
            oscillation_phase = 2 * np.pi * t / self.oscillation_period
            
            # Oscillate about an axis perpendicular to both B and RW axis
            rw_axis = self.A_rw[:, 0] / np.linalg.norm(self.A_rw[:, 0])
            oscillation_axis = np.cross(b_hat, rw_axis)
            if np.linalg.norm(oscillation_axis) < 0.1:
                oscillation_axis = np.cross(b_hat, np.array([1, 0, 0]))
            oscillation_axis = oscillation_axis / (np.linalg.norm(oscillation_axis) + 1e-12)
            
            # Desired oscillation velocity
            omega_oscillation = self.oscillation_amplitude * np.cos(oscillation_phase) * oscillation_axis
            
            # Track oscillation + pointing
            q_err = quaternion_error_vector(q, q_goal)
            omega_error = omega - omega_oscillation
            
            tau_des = -self.kp * q_err - self.kd * omega_error
            tau_gyro = np.cross(omega, self.J @ omega + h_rw_vec)
            tau_total = tau_des + tau_gyro
            
            tau_perp = tau_total - np.dot(tau_total, b_hat) * b_hat
            u_mtq = np.linalg.lstsq(A_mtq, tau_perp, rcond=None)[0]
            u_mtq = np.clip(u_mtq, -self.u_mtq_max, self.u_mtq_max)
            
            tau_remaining = tau_total - A_mtq @ u_mtq
            u_rw = np.linalg.lstsq(self.A_rw, tau_remaining, rcond=None)[0]
            u_rw = np.clip(u_rw, -self.u_rw_max, self.u_rw_max)
        
        return u_rw, u_mtq


def simulate_desaturation(controller, J, A_rw, A_mtq_axes, x0, q_goal, b_field_func, tf, dt):
    """Run desaturation simulation."""
    n_rw = A_rw.shape[1]
    n_mtq = A_mtq_axes.shape[1]
    
    steps = int(tf / dt) + 1
    time_hist = np.zeros(steps)
    h_hist = np.zeros((steps, n_rw))
    error_hist = np.zeros(steps)
    
    x = x0.copy()
    t = 0.0
    
    for k in range(steps):
        omega = x[0:3]
        q = normalize(x[3:7])
        h_rw = x[7:7+n_rw]
        
        b_body = b_field_func(q, t)
        
        u_rw, u_mtq = controller.compute_command(omega, q, h_rw, q_goal, b_body, t)
        
        time_hist[k] = t
        h_hist[k, :] = h_rw
        error_hist[k] = pointing_error_deg(q, q_goal)
        
        if k == steps - 1:
            break
        
        # Propagate
        A_mtq = -skewsym(b_body) @ A_mtq_axes
        
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
    
    h_change_rate = np.mean(np.abs(np.diff(np.linalg.norm(h_hist, axis=1)))) / dt
    
    return DesatResult(
        time=time_hist,
        h_hist=h_hist,
        pointing_error_hist=error_hist,
        final_momentum=np.linalg.norm(h_hist[-1]),
        final_pointing_error=error_hist[-1],
        momentum_change_rate=h_change_rate,
        method=controller.__class__.__name__
    )


def b_field_func(orbit_period=5400):
    def f(q, t):
        phase = 2 * np.pi * t / orbit_period
        B_eci = 30e-6 * np.array([np.cos(phase), 0.5*np.sin(phase), 0.3*np.cos(2*phase)])
        R = rot_mat(q)
        return R.T @ B_eci
    return f


def run_desaturation_comparison():
    """Compare desaturation strategies."""
    np.random.seed(42)
    
    # Configuration
    J = np.diag([0.022, 0.022, 0.004])
    A_rw = np.array([[0], [0], [1.0]])
    A_mtq_axes = np.eye(3)
    u_rw_max = np.array([0.001])
    u_mtq_max = np.array([0.2, 0.2, 0.2])
    kp, kd, kh = 5e-5, 1e-3, 0.01
    
    b_field = b_field_func()
    q_goal = np.array([1, 0, 0, 0])
    
    # Initial condition with high momentum
    omega0 = np.array([0.01, 0.005, 0.002])
    q0 = normalize(np.array([0.9, 0.1, 0.2, 0.3]))
    h0 = np.array([0.008])  # Near saturation
    x0 = np.concatenate([omega0, q0, h0])
    
    tf, dt = 1000, 2  # Long simulation to see desaturation
    
    # Create controllers
    controllers = {
        'Standard': StandardDesatController(J, A_rw, A_mtq_axes, u_rw_max, u_mtq_max, kp, kd, kh),
        'Sequential': SequentialDesatController(J, A_rw, A_mtq_axes, u_rw_max, u_mtq_max, kp, kd, kh),
        'Oscillating': OscillatingDesatController(J, A_rw, A_mtq_axes, u_rw_max, u_mtq_max, kp, kd, kh),
    }
    
    results = {}
    
    for name, controller in tqdm(controllers.items(), desc="Testing controllers"):
        result = simulate_desaturation(controller, J, A_rw, A_mtq_axes, x0, q_goal, b_field, tf, dt)
        results[name] = result
    
    # Print results
    print("\n" + "=" * 70)
    print("DESATURATION STRATEGY COMPARISON")
    print("=" * 70)
    
    print(f"\nInitial momentum: {np.linalg.norm(h0)*1000:.2f} mNm·s")
    print(f"Simulation time: {tf}s")
    
    print(f"\n{'Method':<15} {'Final h (mNm·s)':>15} {'Final Err (°)':>15} {'Δh Rate':>15}")
    print("-" * 70)
    
    for name, result in results.items():
        print(f"{name:<15} {result.final_momentum*1000:>15.3f} {result.final_pointing_error:>15.2f} {result.momentum_change_rate*1e6:>15.3f}")
    
    return results


if __name__ == "__main__":
    results = run_desaturation_comparison()
    
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print("""
Key findings:
1. Standard approach works when B-field allows torque-free desaturation
2. Sequential approach can be more aggressive about desaturation
3. Oscillating approach trades pointing accuracy for more desaturation windows

The optimal strategy depends on mission requirements:
- If pointing is critical: Standard or Sequential
- If desaturation is critical: Oscillating or Sequential with low threshold
""")
