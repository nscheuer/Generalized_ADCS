"""
Full Simulation Test of Allocation Methods
==========================================

Uses the actual codebase simulation infrastructure to test allocation methods.
Compares:
1. Unbounded PD (pseudo-inverse - baseline)
2. LP allocation
3. QP allocation variants (unconstrained, 1a, 3b)
4. Sliding mode control variants

Based on NSSR_3+1 branch debug/generate infrastructure.
"""

import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linprog
import cvxpy as cp
from typing import Dict, List, Tuple, Optional, Callable
from tqdm import tqdm
from dataclasses import dataclass
import warnings

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))

from ADCS.CONOPS.goals import Goal, ECI_Goal, No_Goal
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize, rot_mat, skewsym

SCALE = 1e6


# =============================================================================
# SATELLITE CONFIGURATIONS
# =============================================================================

def create_3mtq_1rw_satellite() -> Tuple[Satellite, np.ndarray]:
    """Create 3MTQ + 1RW satellite (standard test config)."""
    mtq_max_torque = 0.4
    mtqs = [MTQ(axis=j, max_torque=mtq_max_torque) for j in MathConstants.unitvecs]
    
    rw_max_torque = 7 * 0.001
    rw_J = 0.001
    rw_h0 = 0.0
    rw_hmax = 16.2 * 0.001
    
    # Single RW on x-axis
    rws = [RW(axis=MathConstants.unitvecs[0], max_torque=rw_max_torque, J=rw_J, h=rw_h0, h_max=rw_hmax)]
    
    acts = mtqs + rws
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    
    sat = Satellite(
        mass=1.2,
        J_0=np.diagflat([0.022, 0.022, 0.004]),
        actuators=acts,
        sensors=mtms,
        boresight=np.array([0, 0, 1]),
    )
    
    # Initial state: zero rates, identity quaternion, zero RW momentum
    w0 = np.array([0.0, 0.0, 0.0])
    q0 = normalize(np.array([1.0, 0.0, 0.0, 0.0]))
    h0 = np.array([rw_h0])
    x0 = np.concatenate([w0, q0, h0])
    
    return sat, x0


def create_4rw_satellite() -> Tuple[Satellite, np.ndarray]:
    """Create 4RW pyramid satellite (fully actuated)."""
    rw_max_torque = 7 * 0.001
    rw_J = 0.001
    rw_h0 = 0.0
    rw_hmax = 16.2 * 0.001
    
    # Pyramid configuration
    beta = np.radians(54.74)
    rw_axes = [
        np.array([np.sin(beta), 0, np.cos(beta)]),
        np.array([0, np.sin(beta), np.cos(beta)]),
        np.array([-np.sin(beta), 0, np.cos(beta)]),
        np.array([0, -np.sin(beta), np.cos(beta)]),
    ]
    
    rws = [RW(axis=ax, max_torque=rw_max_torque, J=rw_J, h=rw_h0, h_max=rw_hmax) for ax in rw_axes]
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    
    sat = Satellite(
        mass=1.2,
        J_0=np.diagflat([0.022, 0.022, 0.004]),
        actuators=rws,
        sensors=mtms,
        boresight=np.array([0, 0, 1]),
    )
    
    w0 = np.array([0.0, 0.0, 0.0])
    q0 = normalize(np.array([1.0, 0.0, 0.0, 0.0]))
    h0 = np.zeros(4)
    x0 = np.concatenate([w0, q0, h0])
    
    return sat, x0


# =============================================================================
# ORBIT GENERATION
# =============================================================================

def create_test_orbit(tf: float, dt: float) -> Orbit:
    """Create test orbit."""
    ephem = Ephemeris()
    start_time = 0.22 - 1 * TimeConstants.sec2cent
    end_time = 0.22 + tf * TimeConstants.sec2cent
    
    R = 7000 * np.array([0.0, np.sqrt(2) / 2, np.sqrt(2) / 2])
    V = np.array([8.0, 0.0, 0.0])
    
    os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
    return Orbit(os0=os0, end_time=end_time, dt=dt, use_J2=True, fast=False)


def create_fake_orbit(tf: float, dt: float, B_body: np.ndarray = np.array([0.0, 30e-6, 10e-6])) -> Orbit:
    """Create fake orbit with constant B-field (for faster testing)."""
    ephem = Ephemeris()
    R = 7000 * np.array([0.0, np.sqrt(2) / 2, np.sqrt(2) / 2])
    V = np.array([8.0, 0.0, 0.0])
    
    os0 = Orbital_State(
        ephem=ephem,
        J2000=0.22 - 1 * TimeConstants.sec2cent,
        R=R, V=V,
        B=B_body,
        S=np.array([1e5 + 1.0, 0.0, 0.0]),
        rho=5e-12,
    )
    
    dur = int(tf / dt) + 10
    orbs = [os0] * (dur + 10)
    for j in range(dur):
        orbs[j] = os0.copy()
        orbs[j].J2000 = os0.J2000 + j * dt * TimeConstants.sec2cent
    return Orbit(orbs)


# =============================================================================
# CONTROL LAWS
# =============================================================================

def pd_control_law(q_err: np.ndarray, w_err: np.ndarray, K_p: float, K_d: float) -> np.ndarray:
    """Standard PD control law."""
    return -K_p * q_err - K_d * w_err


def sliding_mode_control_law(q_err: np.ndarray, w_err: np.ndarray, K_p: float, K_d: float,
                              lambda_smc: float = 0.5, eta: float = 0.0005) -> np.ndarray:
    """Sliding mode control with saturation."""
    s = w_err + lambda_smc * q_err
    tau_eq = -K_p * q_err - K_d * w_err
    tau_sw = -eta * np.clip(s / 0.01, -1, 1)  # Saturation boundary layer
    return tau_eq + tau_sw


# =============================================================================
# ALLOCATORS
# =============================================================================

def build_allocation_matrix(sat: Satellite, B_body: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build combined allocation matrix A, lb, ub."""
    rws = [a for a in sat.actuators if isinstance(a, RW)]
    mtqs = [a for a in sat.actuators if isinstance(a, MTQ)]
    
    # RW contribution: direct torque
    if rws:
        A_rw = np.column_stack([np.asarray(rw.axis, float).reshape(3,) for rw in rws])
        lb_rw = np.array([-rw.u_max for rw in rws])
        ub_rw = np.array([rw.u_max for rw in rws])
    else:
        A_rw = np.zeros((3, 0))
        lb_rw = np.array([])
        ub_rw = np.array([])
    
    # MTQ contribution: τ = m × B = -B × m
    if mtqs and np.linalg.norm(B_body) > 1e-9:
        B_skew = -skewsym(B_body)
        A_mtq_axes = np.column_stack([np.asarray(m.axis, float).reshape(3,) for m in mtqs])
        A_mtq = B_skew @ A_mtq_axes
        lb_mtq = np.array([-m.u_max for m in mtqs])
        ub_mtq = np.array([m.u_max for m in mtqs])
    else:
        A_mtq = np.zeros((3, 0))
        lb_mtq = np.array([])
        ub_mtq = np.array([])
    
    A = np.hstack([A_rw, A_mtq])
    lb = np.concatenate([lb_rw, lb_mtq])
    ub = np.concatenate([ub_rw, ub_mtq])
    
    return A, lb, ub


def allocate_unbounded(tau_des: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray, **kw) -> np.ndarray:
    """Pseudo-inverse allocation (ignores bounds)."""
    if A.shape[1] == 0:
        return np.zeros(0)
    return np.linalg.pinv(A) @ tau_des


def allocate_lp(tau_des: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray, **kw) -> np.ndarray:
    """LP allocation (direction preserving)."""
    n = len(lb)
    if n == 0:
        return np.zeros(0)
    
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-15:
        return np.zeros(n)
    
    tau_hat = tau_des / t_mag
    c = np.zeros(n + 1)
    c[-1] = -1
    A_eq = np.hstack([A, -tau_hat.reshape(3, 1)])
    bounds = [(lb[i], ub[i]) for i in range(n)] + [(0, None)]
    
    res = linprog(c, A_eq=A_eq, b_eq=np.zeros(3), bounds=bounds, method='highs')
    
    if res.success:
        u = res.x[:n]
        alpha = res.x[-1]
        if alpha > t_mag:
            u = u * (t_mag / alpha)
        return u
    return np.zeros(n)


def allocate_qp(tau_des: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray, **kw) -> np.ndarray:
    """QP allocation (L2 optimal with scaling fix)."""
    n = len(lb)
    if n == 0:
        return np.zeros(0)
    
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(SCALE * (tau - tau_des)))
    constraints = [u >= lb, u <= ub]
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
        if u.value is not None:
            return u.value
    except:
        pass
    return np.zeros(n)


def allocate_qp_1a(tau_des: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray, omega: np.ndarray, **kw) -> np.ndarray:
    """QP with power-brake constraint (1a)."""
    n = len(lb)
    if n == 0:
        return np.zeros(0)
    
    P_des = np.dot(omega, tau_des)
    
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(SCALE * (tau - tau_des)))
    constraints = [u >= lb, u <= ub]
    
    if P_des < -1e-12:
        constraints.append(omega @ tau <= 1e-12)
    
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
        if u.value is not None:
            return u.value
    except:
        pass
    return np.zeros(n)


def allocate_qp_3b(tau_des: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray, omega: np.ndarray, **kw) -> np.ndarray:
    """QP with sign-critical constraint (3b)."""
    n = len(lb)
    if n == 0:
        return np.zeros(0)
    
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(SCALE * (tau - tau_des)))
    constraints = [u >= lb, u <= ub]
    
    tau_threshold = 0.1 * np.linalg.norm(tau_des) + 1e-12
    
    for i in range(3):
        if abs(tau_des[i]) < tau_threshold:
            continue
        if omega[i] > 1e-6 and tau_des[i] < -1e-12:
            constraints.append(tau[i] <= 0)
        elif omega[i] < -1e-6 and tau_des[i] > 1e-12:
            constraints.append(tau[i] >= 0)
    
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
        if u.value is not None:
            return u.value
    except:
        pass
    return np.zeros(n)


# =============================================================================
# POINTING ERROR COMPUTATION
# =============================================================================

def compute_pointing_error_deg(q: np.ndarray, goal_eci: np.ndarray, 
                                body_boresight: np.ndarray = np.array([0, 0, 1])) -> float:
    """Compute pointing error in degrees."""
    R = rot_mat(q)
    boresight_eci = R @ body_boresight
    
    boresight_eci = boresight_eci / (np.linalg.norm(boresight_eci) + 1e-16)
    goal_eci = goal_eci / (np.linalg.norm(goal_eci) + 1e-16)
    
    c = float(np.clip(np.dot(boresight_eci, goal_eci), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def compute_quaternion_error(q: np.ndarray, q_goal: np.ndarray) -> np.ndarray:
    """Compute attitude error vector (MRP-like)."""
    q = normalize(q)
    q_goal = normalize(q_goal)
    
    # Error quaternion
    q_goal_inv = np.array([q_goal[0], -q_goal[1], -q_goal[2], -q_goal[3]])
    
    # q_err = q_goal^{-1} * q
    q_err = np.array([
        q_goal_inv[0]*q[0] - q_goal_inv[1]*q[1] - q_goal_inv[2]*q[2] - q_goal_inv[3]*q[3],
        q_goal_inv[0]*q[1] + q_goal_inv[1]*q[0] + q_goal_inv[2]*q[3] - q_goal_inv[3]*q[2],
        q_goal_inv[0]*q[2] - q_goal_inv[1]*q[3] + q_goal_inv[2]*q[0] + q_goal_inv[3]*q[1],
        q_goal_inv[0]*q[3] + q_goal_inv[1]*q[2] - q_goal_inv[2]*q[1] + q_goal_inv[3]*q[0],
    ])
    
    if q_err[0] < 0:
        q_err = -q_err
    
    return 2.0 * q_err[1:4]


# =============================================================================
# SIMULATION
# =============================================================================

@dataclass
class SimResult:
    """Simulation result."""
    name: str
    control_law: str
    allocator: str
    final_error_deg: float
    converged: bool
    error_history: np.ndarray


def run_simulation(
    sat: Satellite,
    x0: np.ndarray,
    orb: Orbit,
    goal_eci: np.ndarray,
    control_law: Callable,
    allocator: Callable,
    K_p: float = 0.00005,
    K_d: float = 0.001,
    tf: float = 500,
    dt: float = 2,
    name: str = "",
) -> SimResult:
    """Run full closed-loop simulation."""
    
    t0 = 0
    N = int((tf - t0) / dt)
    x = x0.copy()
    
    error_history = np.zeros(N)
    
    for i in range(N):
        t = t0 + i * dt
        J2000 = 0.22 + t * TimeConstants.sec2cent
        os = orb.get_os(J2000=J2000)
        
        # Get B-field
        B_body = os.B
        
        # Current state
        w = x[0:3]
        q = x[3:7]
        
        # Compute errors
        q_err = compute_quaternion_error(q, normalize(np.array([1, 0, 0, 0])))  # Assuming identity goal
        
        # For ECI goal, compute error differently
        R = rot_mat(q)
        boresight_eci = R @ sat.boresight
        goal_eci_norm = goal_eci / np.linalg.norm(goal_eci)
        
        # Cross product gives rotation axis, magnitude gives sin(angle)
        cross = np.cross(boresight_eci, goal_eci_norm)
        q_err = cross  # Simplified error
        
        w_err = w  # Assuming zero reference rate
        
        # Control law
        tau_des = control_law(q_err, w_err, K_p, K_d)
        
        # Add gyroscopic compensation
        n_rw = len([a for a in sat.actuators if isinstance(a, RW)])
        if n_rw > 0 and len(x) > 7:
            h_rw = x[7:7+n_rw]
            A_rw = np.column_stack([np.asarray(rw.axis, float).reshape(3,) for rw in sat.actuators if isinstance(rw, RW)])
            h_rw_body = A_rw @ h_rw
        else:
            h_rw_body = np.zeros(3)
        
        J = sat.J_0
        tau_gyro = np.cross(w, J @ w + h_rw_body)
        tau_des = tau_des + tau_gyro
        
        # Build allocation matrix
        A, lb, ub = build_allocation_matrix(sat, B_body)
        
        # Allocate
        if A.shape[1] > 0:
            u = allocator(tau_des, A, lb, ub, omega=w)
        else:
            u = np.zeros(0)
        
        # Store error
        error_history[i] = compute_pointing_error_deg(q, goal_eci, sat.boresight)
        
        # Propagate dynamics
        prev_os = os.copy()
        os_next = orb.get_os(0.22 + (t + dt) * TimeConstants.sec2cent)
        
        # Pad u to match actuator count
        u_full = np.zeros(len(sat.actuators))
        u_full[:len(u)] = u
        
        out = solve_ivp(
            fun=sat.dynamics_for_solver,
            t_span=(0, dt),
            y0=x,
            method="RK45",
            args=(u_full, prev_os, os_next),
            rtol=1e-7,
            atol=1e-7,
        )
        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])
    
    final_error = error_history[-1]
    converged = final_error < 5.0  # 5 degree threshold
    
    return SimResult(
        name=name,
        control_law=control_law.__name__,
        allocator=allocator.__name__,
        final_error_deg=final_error,
        converged=converged,
        error_history=error_history,
    )


# =============================================================================
# MAIN TEST
# =============================================================================

def run_all_tests():
    """Run comprehensive tests."""
    print("=" * 100)
    print("FULL SIMULATION TEST: ALLOCATION METHODS WITH REAL DYNAMICS")
    print("=" * 100)
    
    # Create satellite and orbit
    sat, x0 = create_3mtq_1rw_satellite()
    
    # Use fake orbit for speed (constant B-field)
    tf = 100
    dt = 2
    
    B_fields = [
        (np.array([0.0, 30e-6, 10e-6]), "B_yz"),
        (np.array([10e-6, 10e-6, 30e-6]), "B_strong_z"),
    ]
    
    goals = [
        (np.array([1.0, 0.0, 0.0]), "Goal_X"),
        (np.array([1.0, 1.0, 1.0]), "Goal_diag"),
    ]
    
    control_laws = [
        (pd_control_law, "PD"),
        (sliding_mode_control_law, "SMC"),
    ]
    
    allocators = [
        (allocate_unbounded, "Unbounded"),
        (allocate_lp, "LP"),
        (allocate_qp, "QP"),
        (allocate_qp_1a, "QP_1a"),
        (allocate_qp_3b, "QP_3b"),
    ]
    
    all_results = []
    
    for B_body, B_name in B_fields:
        orb = create_fake_orbit(tf, dt, B_body)
        
        for goal_eci, goal_name in goals:
            print(f"\n--- {B_name} | {goal_name} ---")
            print(f"{'Control':<8} {'Allocator':<12} {'Final Err':>12} {'Converged':>10}")
            print("-" * 50)
            
            for ctrl_func, ctrl_name in control_laws:
                for alloc_func, alloc_name in allocators:
                    # Reset initial state with random quaternion
                    x0_test = x0.copy()
                    x0_test[3:7] = normalize(np.random.randn(4))
                    
                    result = run_simulation(
                        sat=sat,
                        x0=x0_test,
                        orb=orb,
                        goal_eci=goal_eci,
                        control_law=ctrl_func,
                        allocator=alloc_func,
                        K_p=0.00005,
                        K_d=0.001,
                        tf=tf,
                        dt=dt,
                        name=f"{B_name}|{goal_name}",
                    )
                    all_results.append(result)
                    
                    conv_str = "✓" if result.converged else "✗"
                    print(f"{ctrl_name:<8} {alloc_name:<12} {result.final_error_deg:>11.2f}° {conv_str:>10}")
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY BY ALLOCATOR (across all scenarios)")
    print("=" * 100)
    
    for alloc_func, alloc_name in allocators:
        alloc_results = [r for r in all_results if r.allocator == alloc_func.__name__]
        if alloc_results:
            errors = [r.final_error_deg for r in alloc_results]
            n_conv = sum(1 for r in alloc_results if r.converged)
            print(f"{alloc_name:<12}: Mean={np.mean(errors):>7.2f}°, Max={np.max(errors):>7.1f}°, Conv={n_conv}/{len(alloc_results)}")
    
    return all_results


if __name__ == "__main__":
    np.random.seed(42)
    warnings.filterwarnings('ignore')
    
    results = run_all_tests()
