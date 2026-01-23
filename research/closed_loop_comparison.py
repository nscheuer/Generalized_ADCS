"""
Closed-Loop Simulation Comparison
=================================

Compare LP, QP, and QPC allocation methods in full closed-loop simulations
to understand how torque allocation affects actual pointing performance.

The key question: Does preserving torque direction (LP) lead to better
pointing than achieving more total torque (QP)?
"""

import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from dataclasses import dataclass
import json
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))

from ADCS.helpers.math_helpers import normalize, rot_mat, skewsym, quat_inv, quat_mult

from research.allocation_comparison import (
    LPAllocator, QPAllocator, QPCAllocator, MinNormAllocator, TorqueAllocator
)


@dataclass
class ClosedLoopConfig:
    """Configuration for closed-loop simulation."""
    # Spacecraft parameters
    J: np.ndarray  # Inertia matrix
    
    # Actuator parameters
    A_rw: np.ndarray
    A_mtq_axes: np.ndarray
    u_rw_max: np.ndarray
    u_mtq_max: np.ndarray
    J_rw: np.ndarray  # RW inertias
    
    # Control gains
    kp: float  # Proportional gain
    kd: float  # Derivative gain
    
    # Simulation parameters
    tf: float  # Final time
    dt: float  # Time step


@dataclass 
class ClosedLoopResult:
    """Results from a closed-loop simulation."""
    time: np.ndarray
    state: np.ndarray  # [omega(3), q(4), h_rw(n_rw)]
    u: np.ndarray      # Actuator commands
    tau_des: np.ndarray
    tau_ach: np.ndarray
    pointing_error: np.ndarray  # degrees
    
    final_error: float
    convergence_time: float  # Time to reach 1 degree
    rms_error: float
    mean_alpha: float
    method: str


def quaternion_error_vector(q: np.ndarray, q_goal: np.ndarray) -> np.ndarray:
    """
    Compute attitude error as a vector proportional to rotation angle.
    Returns 2 * q_error_vector (MRP-like error).
    """
    q = normalize(q)
    q_goal = normalize(q_goal)
    
    # Error quaternion: q_err = q_goal^{-1} * q
    q_err = quat_mult(quat_inv(q_goal), q)
    
    # Ensure short rotation path
    if q_err[0] < 0:
        q_err = -q_err
    
    # Return 2 * vector part (small angle approximation gives rotation vector)
    return 2.0 * q_err[1:4]


def pointing_error_deg(q: np.ndarray, q_goal: np.ndarray,
                       boresight: np.ndarray = np.array([0, 0, 1])) -> float:
    """Compute pointing error in degrees between boresight and goal direction."""
    q = normalize(q)
    q_goal = normalize(q_goal)
    
    R = rot_mat(q)
    R_goal = rot_mat(q_goal)
    
    boresight_actual = R @ boresight
    boresight_goal = R_goal @ boresight
    
    cos_angle = np.clip(np.dot(boresight_actual, boresight_goal), -1, 1)
    return np.degrees(np.arccos(cos_angle))


def simulate_closed_loop(config: ClosedLoopConfig,
                         allocator: TorqueAllocator,
                         x0: np.ndarray,
                         q_goal: np.ndarray,
                         b_field_func,
                         verbose: bool = False) -> ClosedLoopResult:
    """
    Run closed-loop simulation with given allocator.
    
    Parameters
    ----------
    config : ClosedLoopConfig
        Simulation configuration
    allocator : TorqueAllocator
        Torque allocation method to use
    x0 : np.ndarray
        Initial state [omega(3), q(4), h_rw(n_rw)]
    q_goal : np.ndarray
        Goal quaternion
    b_field_func : callable
        Function that returns B-field in body frame given (q, t)
    verbose : bool
        Print progress
    
    Returns
    -------
    ClosedLoopResult
        Simulation results
    """
    n_rw = config.A_rw.shape[1]
    n_mtq = config.A_mtq_axes.shape[1]
    n_states = 7 + n_rw
    n_acts = n_rw + n_mtq
    
    steps = int(config.tf / config.dt) + 1
    
    # Storage
    time_hist = np.zeros(steps)
    state_hist = np.zeros((steps, n_states))
    u_hist = np.zeros((steps, n_acts))
    tau_des_hist = np.zeros((steps, 3))
    tau_ach_hist = np.zeros((steps, 3))
    error_hist = np.zeros(steps)
    alpha_hist = np.zeros(steps)
    
    x = x0.copy()
    t = 0.0
    
    for k in range(steps):
        omega = x[0:3]
        q = normalize(x[3:7])
        h_rw = x[7:7+n_rw] if n_rw > 0 else np.array([])
        
        # Get B-field in body frame
        b_body = b_field_func(q, t)
        
        # Compute control torque (PD control)
        q_err = quaternion_error_vector(q, q_goal)
        tau_pd = -config.kp * q_err - config.kd * omega
        
        # Gyroscopic compensation
        h_rw_vec = config.A_rw @ h_rw if n_rw > 0 else np.zeros(3)
        tau_gyro = np.cross(omega, config.J @ omega + h_rw_vec)
        
        tau_des = tau_pd + tau_gyro
        
        # Allocate torque
        result = allocator.allocate(
            tau_des=tau_des,
            b_body=b_body,
            A_rw=config.A_rw,
            A_mtq_axes=config.A_mtq_axes,
            u_rw_max=config.u_rw_max,
            u_mtq_max=config.u_mtq_max,
            omega=omega,
            h_rw=h_rw,
            J_rw=config.J_rw
        )
        
        u_rw = result.u_rw
        u_mtq = result.u_mtq
        
        # Compute actual torque achieved
        if n_mtq > 0:
            A_mtq = -skewsym(b_body) @ config.A_mtq_axes
        else:
            A_mtq = np.zeros((3, 0))
        
        tau_rw = config.A_rw @ u_rw if n_rw > 0 else np.zeros(3)
        tau_mtq = A_mtq @ u_mtq if n_mtq > 0 else np.zeros(3)
        tau_ach = tau_rw + tau_mtq
        
        # Store
        time_hist[k] = t
        state_hist[k, :] = x
        u_hist[k, :n_rw] = u_rw
        u_hist[k, n_rw:] = u_mtq
        tau_des_hist[k, :] = tau_des
        tau_ach_hist[k, :] = tau_ach
        error_hist[k] = pointing_error_deg(q, q_goal)
        alpha_hist[k] = result.alpha
        
        if k == steps - 1:
            break
        
        # Propagate dynamics
        def dynamics(t_local, y):
            w = y[0:3]
            quat = normalize(y[3:7])
            hrw = y[7:7+n_rw] if n_rw > 0 else np.array([])
            
            # Recompute B-field at current attitude
            b_local = b_field_func(quat, t + t_local)
            
            if n_mtq > 0:
                A_mtq_local = -skewsym(b_local) @ config.A_mtq_axes
                tau_mtq_local = A_mtq_local @ u_mtq
            else:
                tau_mtq_local = np.zeros(3)
            
            tau_rw_local = config.A_rw @ u_rw if n_rw > 0 else np.zeros(3)
            tau_total = tau_rw_local + tau_mtq_local
            
            hrw_vec = config.A_rw @ hrw if n_rw > 0 else np.zeros(3)
            
            # Angular acceleration
            w_dot = np.linalg.solve(config.J, 
                tau_total - np.cross(w, config.J @ w + hrw_vec))
            
            # Quaternion kinematics
            W = np.zeros((4, 3))
            W[0, :] = -quat[1:4]
            W[1:4, :] = quat[0] * np.eye(3) + skewsym(quat[1:4])
            q_dot = 0.5 * W @ w
            
            # RW momentum change
            h_dot = -u_rw if n_rw > 0 else np.array([])
            
            return np.concatenate([w_dot, q_dot, h_dot])
        
        sol = solve_ivp(dynamics, [0, config.dt], x, method='RK45',
                       rtol=1e-8, atol=1e-10)
        x = sol.y[:, -1]
        x[3:7] = normalize(x[3:7])
        
        t += config.dt
    
    # Compute summary statistics
    final_error = error_hist[-1]
    rms_error = np.sqrt(np.mean(error_hist**2))
    mean_alpha = np.mean(alpha_hist)
    
    # Find convergence time (first time error < 1 degree and stays there)
    threshold = 1.0  # degrees
    convergence_time = config.tf  # default if never converges
    for k in range(len(error_hist)):
        if error_hist[k] < threshold:
            # Check if it stays below
            if np.all(error_hist[k:] < threshold * 2):  # Allow some margin
                convergence_time = time_hist[k]
                break
    
    return ClosedLoopResult(
        time=time_hist,
        state=state_hist,
        u=u_hist,
        tau_des=tau_des_hist,
        tau_ach=tau_ach_hist,
        pointing_error=error_hist,
        final_error=final_error,
        convergence_time=convergence_time,
        rms_error=rms_error,
        mean_alpha=mean_alpha,
        method=allocator.name
    )


def create_3mtq_1rw_config(kp: float = 5e-5, kd: float = 1e-3,
                            tf: float = 500, dt: float = 2) -> ClosedLoopConfig:
    """Create configuration for 3MTQ + 1RW CubeSat."""
    return ClosedLoopConfig(
        J=np.diag([0.022, 0.022, 0.004]),  # 3U CubeSat
        A_rw=np.array([[0], [0], [1.0]]),  # Single RW along z
        A_mtq_axes=np.eye(3),  # Orthogonal MTQs
        u_rw_max=np.array([0.001]),  # 1 mNm max torque
        u_mtq_max=np.array([0.2, 0.2, 0.2]),  # 0.2 Am² max dipole
        J_rw=np.array([0.001]),  # RW inertia
        kp=kp,
        kd=kd,
        tf=tf,
        dt=dt
    )


def create_3mtq_3rw_config(kp: float = 5e-5, kd: float = 1e-3,
                            tf: float = 500, dt: float = 2) -> ClosedLoopConfig:
    """Create configuration for 3MTQ + 3RW (fully actuated)."""
    return ClosedLoopConfig(
        J=np.diag([0.022, 0.022, 0.004]),
        A_rw=np.eye(3),  # Three orthogonal RWs
        A_mtq_axes=np.eye(3),
        u_rw_max=np.array([0.001, 0.001, 0.001]),
        u_mtq_max=np.array([0.2, 0.2, 0.2]),
        J_rw=np.array([0.001, 0.001, 0.001]),
        kp=kp,
        kd=kd,
        tf=tf,
        dt=dt
    )


def time_varying_b_field(orbit_period: float = 5400):
    """
    Create a function that returns time-varying B-field in body frame.
    
    Simulates rotation of spacecraft in orbit.
    """
    def b_field(q, t):
        # B-field in inertial frame (simplified model)
        phase = 2 * np.pi * t / orbit_period
        B_eci = 30e-6 * np.array([
            np.cos(phase),
            0.5 * np.sin(phase),
            0.3 * np.cos(2 * phase)
        ])
        
        # Transform to body frame
        R = rot_mat(q)
        return R.T @ B_eci
    
    return b_field


def run_comparison(scenarios: List[Dict], 
                   allocators: List[TorqueAllocator],
                   config: ClosedLoopConfig,
                   b_field_func) -> Dict[str, List[ClosedLoopResult]]:
    """
    Run closed-loop comparison across multiple scenarios.
    """
    results = {alloc.name: [] for alloc in allocators}
    
    for scenario in tqdm(scenarios, desc="Running scenarios"):
        x0 = scenario['x0']
        q_goal = scenario['q_goal']
        
        for alloc in allocators:
            result = simulate_closed_loop(
                config=config,
                allocator=alloc,
                x0=x0,
                q_goal=q_goal,
                b_field_func=b_field_func
            )
            results[alloc.name].append(result)
    
    return results


def summarize_closed_loop_results(results: Dict[str, List[ClosedLoopResult]]) -> Dict:
    """Compute summary statistics for closed-loop results."""
    summary = {}
    
    for method, result_list in results.items():
        final_errors = [r.final_error for r in result_list]
        convergence_times = [r.convergence_time for r in result_list]
        rms_errors = [r.rms_error for r in result_list]
        mean_alphas = [r.mean_alpha for r in result_list]
        
        summary[method] = {
            'final_error_deg': {
                'mean': np.mean(final_errors),
                'std': np.std(final_errors),
                'max': np.max(final_errors),
                'median': np.median(final_errors)
            },
            'convergence_time_s': {
                'mean': np.mean(convergence_times),
                'std': np.std(convergence_times),
                'max': np.max(convergence_times),
                'median': np.median(convergence_times)
            },
            'rms_error_deg': {
                'mean': np.mean(rms_errors),
                'std': np.std(rms_errors)
            },
            'mean_alpha': {
                'mean': np.mean(mean_alphas),
                'std': np.std(mean_alphas)
            }
        }
    
    return summary


def generate_pointing_scenarios(n_scenarios: int = 20, seed: int = 42) -> List[Dict]:
    """Generate random pointing scenarios."""
    np.random.seed(seed)
    scenarios = []
    
    for i in range(n_scenarios):
        # Random initial orientation (near identity)
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        angle = np.random.uniform(0.1, 0.5)  # 6-30 degrees
        
        q0 = np.concatenate([[np.cos(angle/2)], axis * np.sin(angle/2)])
        q0 = normalize(q0)
        
        # Random initial angular velocity (small)
        omega0 = np.random.randn(3) * 0.02  # ~1 deg/s
        
        # Initial RW momentum
        h0 = np.array([0.002])  # Some initial momentum
        
        x0 = np.concatenate([omega0, q0, h0])
        
        # Goal is identity quaternion
        q_goal = np.array([1, 0, 0, 0])
        
        scenarios.append({
            'x0': x0,
            'q_goal': q_goal,
            'name': f'scenario_{i}'
        })
    
    return scenarios


if __name__ == "__main__":
    print("=" * 60)
    print("Closed-Loop Allocation Comparison")
    print("=" * 60)
    
    # Configuration
    config = create_3mtq_1rw_config(
        kp=5e-5,
        kd=1e-3,
        tf=300,  # 5 minutes
        dt=2     # 2 second steps
    )
    
    print(f"\nConfiguration:")
    print(f"  Spacecraft: 3U CubeSat")
    print(f"  Actuators: 3MTQ + 1RW")
    print(f"  Simulation: {config.tf}s at {config.dt}s steps")
    print(f"  Control: kp={config.kp}, kd={config.kd}")
    
    # Generate scenarios
    scenarios = generate_pointing_scenarios(n_scenarios=10, seed=42)
    print(f"\nGenerated {len(scenarios)} pointing scenarios")
    
    # Allocators to compare
    allocators = [
        LPAllocator(),
        QPAllocator(),
        QPCAllocator("A"),
        QPCAllocator("B"),
    ]
    print(f"\nComparing {len(allocators)} allocation methods: {[a.name for a in allocators]}")
    
    # B-field function
    b_field = time_varying_b_field(orbit_period=5400)
    
    # Run comparison
    print("\nRunning closed-loop simulations...")
    results = run_comparison(scenarios, allocators, config, b_field)
    
    # Summarize
    summary = summarize_closed_loop_results(results)
    
    print("\n" + "=" * 60)
    print("CLOSED-LOOP RESULTS SUMMARY")
    print("=" * 60)
    
    for method, stats in summary.items():
        print(f"\n{method}:")
        print(f"  Final Error: {stats['final_error_deg']['mean']:.3f}° ± {stats['final_error_deg']['std']:.3f}° (max {stats['final_error_deg']['max']:.3f}°)")
        print(f"  Convergence Time: {stats['convergence_time_s']['mean']:.1f}s ± {stats['convergence_time_s']['std']:.1f}s")
        print(f"  RMS Error: {stats['rms_error_deg']['mean']:.3f}° ± {stats['rms_error_deg']['std']:.3f}°")
        print(f"  Mean Alpha: {stats['mean_alpha']['mean']:.3f} ± {stats['mean_alpha']['std']:.3f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"/home/pmckeen/Generalized_ADCS/research/closed_loop_{timestamp}.json"
    
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        return obj
    
    with open(results_file, 'w') as f:
        json.dump(convert_for_json(summary), f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
