"""
Allocation Method Comparison Framework
======================================

This module provides a systematic framework for comparing different torque
allocation methods for spacecraft attitude control:

1. LP (Linear Program) - maximize torque along desired direction
2. QP (Quadratic Program) - minimize ||τ_achieved - τ_desired||²  
3. QPC (QP with Constraint) - QP with Lyapunov stability constraints
4. Various QPC variants with different constraint formulations

The framework supports:
- Single-torque allocation comparisons (geometry analysis)
- Full simulation comparisons (closed-loop performance)
- Monte Carlo analysis with varied initial conditions
- Multiple actuator configurations

Author: Research Framework for Generalized ADCS
Date: January 2026
"""

import sys
import os
import numpy as np
from scipy.optimize import linprog, lsq_linear, minimize, Bounds
from scipy.integrate import solve_ivp
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from tqdm import tqdm
import json
from datetime import datetime

# Add project path
sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))

from ADCS.helpers.math_helpers import skewsym, normalize, rot_mat, quat_mult, quat_inv


@dataclass
class AllocationResult:
    """Result from a single allocation problem solve."""
    u_rw: np.ndarray      # Reaction wheel commands
    u_mtq: np.ndarray     # Magnetorquer commands
    tau_achieved: np.ndarray  # Achieved torque
    tau_desired: np.ndarray   # Desired torque
    alpha: float          # Scalar effectiveness (τ_achieved · τ̂_desired / ||τ_desired||)
    direction_error_deg: float  # Angle between τ_achieved and τ_desired
    magnitude_ratio: float      # ||τ_achieved|| / ||τ_desired||
    solve_time_us: float        # Solve time in microseconds
    solver_success: bool        # Did the solver converge?
    method: str                 # Which method was used


@dataclass
class SimulationResult:
    """Result from a full closed-loop simulation."""
    time_hist: np.ndarray
    state_hist: np.ndarray  # [ω, q, h_rw]
    u_hist: np.ndarray
    tau_des_hist: np.ndarray
    tau_ach_hist: np.ndarray
    alpha_hist: np.ndarray
    pointing_error_hist: np.ndarray  # degrees
    final_pointing_error: float
    convergence_time: float  # time to reach within threshold
    rms_pointing_error: float
    method: str


class TorqueAllocator:
    """
    Base class for torque allocation methods.
    
    All allocators take the same inputs:
    - tau_des: Desired torque (3,)
    - b_body: Magnetic field in body frame (3,)
    - A_rw: RW torque mapping matrix (3, n_rw)
    - A_mtq_axes: MTQ dipole axes matrix (3, n_mtq)
    - u_rw_max: RW torque limits (n_rw,)
    - u_mtq_max: MTQ dipole limits (n_mtq,)
    - omega: Current angular velocity (optional, for QPC variants)
    - h_rw: Current RW momentum (optional, for some variants)
    """
    
    def __init__(self, name: str):
        self.name = name
    
    def allocate(self, tau_des: np.ndarray, b_body: np.ndarray,
                 A_rw: np.ndarray, A_mtq_axes: np.ndarray,
                 u_rw_max: np.ndarray, u_mtq_max: np.ndarray,
                 omega: Optional[np.ndarray] = None,
                 h_rw: Optional[np.ndarray] = None,
                 J_rw: Optional[np.ndarray] = None) -> AllocationResult:
        raise NotImplementedError("Subclasses must implement allocate()")


class LPAllocator(TorqueAllocator):
    """
    Linear Program allocator: maximize torque along desired direction.
    
    Solves:
        max α
        s.t. A_total @ u = α * τ̂_des
             -u_max ≤ u ≤ u_max
             α ≥ 0
    
    Properties:
    - Direction preservation: τ_achieved is always parallel to τ_desired (or zero)
    - May "waste" actuator authority perpendicular to desired direction
    """
    
    def __init__(self):
        super().__init__("LP")
    
    def allocate(self, tau_des: np.ndarray, b_body: np.ndarray,
                 A_rw: np.ndarray, A_mtq_axes: np.ndarray,
                 u_rw_max: np.ndarray, u_mtq_max: np.ndarray,
                 omega: Optional[np.ndarray] = None,
                 h_rw: Optional[np.ndarray] = None,
                 J_rw: Optional[np.ndarray] = None) -> AllocationResult:
        import time
        start = time.perf_counter()
        
        tau_des = np.asarray(tau_des, float).reshape(3,)
        t_mag = np.linalg.norm(tau_des)
        
        n_rw = A_rw.shape[1] if A_rw.size > 0 else 0
        n_mtq = A_mtq_axes.shape[1] if A_mtq_axes.size > 0 else 0
        
        if t_mag < 1e-12:
            return AllocationResult(
                u_rw=np.zeros(n_rw), u_mtq=np.zeros(n_mtq),
                tau_achieved=np.zeros(3), tau_desired=tau_des,
                alpha=1.0, direction_error_deg=0.0, magnitude_ratio=1.0,
                solve_time_us=0.0, solver_success=True, method=self.name
            )
        
        # Build A_mtq from B-field cross product
        if n_mtq > 0:
            A_mtq = -skewsym(b_body) @ A_mtq_axes
        else:
            A_mtq = np.zeros((3, 0))
        
        # Combined matrix
        A_total = np.hstack([A_rw, A_mtq])
        n_act = n_rw + n_mtq
        
        if n_act == 0:
            return AllocationResult(
                u_rw=np.zeros(0), u_mtq=np.zeros(0),
                tau_achieved=np.zeros(3), tau_desired=tau_des,
                alpha=0.0, direction_error_deg=90.0, magnitude_ratio=0.0,
                solve_time_us=0.0, solver_success=False, method=self.name
            )
        
        tau_hat = tau_des / t_mag
        
        # LP: maximize T_available
        # Variables: [u_1, ..., u_n, T_available]
        c = np.zeros(n_act + 1)
        c[-1] = -1.0  # minimize -T_available = maximize T_available
        
        # Equality constraint: A_total @ u = T_available * tau_hat
        A_eq = np.hstack([A_total, -tau_hat.reshape(3, 1)])
        b_eq = np.zeros(3)
        
        # Bounds
        lb = np.concatenate([-u_rw_max, -u_mtq_max, [0.0]])
        ub = np.concatenate([u_rw_max, u_mtq_max, [None]])
        bounds = [(lb[i], ub[i] if ub[i] is not None else None) for i in range(n_act + 1)]
        
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        solve_time = (time.perf_counter() - start) * 1e6  # microseconds
        
        if res.success:
            u_sol = res.x[:n_act]
            T_max = res.x[-1]
            
            # Scale if we have more capacity than needed
            if T_max > t_mag:
                scale = t_mag / T_max
                u_sol = u_sol * scale
                alpha = 1.0
            else:
                alpha = T_max / t_mag if t_mag > 0 else 0.0
            
            u_rw = u_sol[:n_rw]
            u_mtq = u_sol[n_rw:]
            tau_achieved = A_total @ u_sol
        else:
            u_rw = np.zeros(n_rw)
            u_mtq = np.zeros(n_mtq)
            tau_achieved = np.zeros(3)
            alpha = 0.0
        
        # Compute metrics
        tau_ach_mag = np.linalg.norm(tau_achieved)
        if tau_ach_mag > 1e-12 and t_mag > 1e-12:
            cos_angle = np.clip(np.dot(tau_achieved, tau_des) / (tau_ach_mag * t_mag), -1, 1)
            direction_error_deg = np.degrees(np.arccos(cos_angle))
        else:
            direction_error_deg = 0.0 if tau_ach_mag < 1e-12 else 90.0
        
        magnitude_ratio = tau_ach_mag / t_mag if t_mag > 1e-12 else 0.0
        
        return AllocationResult(
            u_rw=u_rw, u_mtq=u_mtq,
            tau_achieved=tau_achieved, tau_desired=tau_des,
            alpha=alpha, direction_error_deg=direction_error_deg,
            magnitude_ratio=magnitude_ratio, solve_time_us=solve_time,
            solver_success=res.success, method=self.name
        )


class QPAllocator(TorqueAllocator):
    """
    Quadratic Program allocator: minimize ||τ_achieved - τ_desired||².
    
    Solves:
        min ||A_total @ u - τ_des||²
        s.t. -u_max ≤ u ≤ u_max
    
    Properties:
    - Finds "closest" achievable torque in Euclidean sense
    - Does NOT preserve direction - may add perpendicular components
    - Always finds a feasible solution (u=0 is always feasible)
    """
    
    def __init__(self):
        super().__init__("QP")
    
    def allocate(self, tau_des: np.ndarray, b_body: np.ndarray,
                 A_rw: np.ndarray, A_mtq_axes: np.ndarray,
                 u_rw_max: np.ndarray, u_mtq_max: np.ndarray,
                 omega: Optional[np.ndarray] = None,
                 h_rw: Optional[np.ndarray] = None,
                 J_rw: Optional[np.ndarray] = None) -> AllocationResult:
        import time
        start = time.perf_counter()
        
        tau_des = np.asarray(tau_des, float).reshape(3,)
        t_mag = np.linalg.norm(tau_des)
        
        n_rw = A_rw.shape[1] if A_rw.size > 0 else 0
        n_mtq = A_mtq_axes.shape[1] if A_mtq_axes.size > 0 else 0
        
        if t_mag < 1e-12:
            return AllocationResult(
                u_rw=np.zeros(n_rw), u_mtq=np.zeros(n_mtq),
                tau_achieved=np.zeros(3), tau_desired=tau_des,
                alpha=1.0, direction_error_deg=0.0, magnitude_ratio=1.0,
                solve_time_us=0.0, solver_success=True, method=self.name
            )
        
        # Build A_mtq
        if n_mtq > 0:
            A_mtq = -skewsym(b_body) @ A_mtq_axes
        else:
            A_mtq = np.zeros((3, 0))
        
        A_total = np.hstack([A_rw, A_mtq])
        n_act = n_rw + n_mtq
        
        if n_act == 0:
            return AllocationResult(
                u_rw=np.zeros(0), u_mtq=np.zeros(0),
                tau_achieved=np.zeros(3), tau_desired=tau_des,
                alpha=0.0, direction_error_deg=90.0, magnitude_ratio=0.0,
                solve_time_us=0.0, solver_success=False, method=self.name
            )
        
        # Bounds
        lb = np.concatenate([-u_rw_max, -u_mtq_max])
        ub = np.concatenate([u_rw_max, u_mtq_max])
        
        # Solve bounded least squares (BVLS is more robust for this problem)
        res = lsq_linear(A_total, tau_des, bounds=(lb, ub), method='bvls')
        
        solve_time = (time.perf_counter() - start) * 1e6
        
        if res.success:
            u_sol = res.x
            u_rw = u_sol[:n_rw]
            u_mtq = u_sol[n_rw:]
            tau_achieved = A_total @ u_sol
        else:
            u_rw = np.zeros(n_rw)
            u_mtq = np.zeros(n_mtq)
            tau_achieved = np.zeros(3)
        
        # Compute metrics
        tau_ach_mag = np.linalg.norm(tau_achieved)
        if tau_ach_mag > 1e-12 and t_mag > 1e-12:
            cos_angle = np.clip(np.dot(tau_achieved, tau_des) / (tau_ach_mag * t_mag), -1, 1)
            direction_error_deg = np.degrees(np.arccos(cos_angle))
            alpha = np.dot(tau_achieved, tau_des / t_mag) / t_mag
            alpha = max(0.0, alpha)
        else:
            direction_error_deg = 0.0 if tau_ach_mag < 1e-12 else 90.0
            alpha = 0.0
        
        magnitude_ratio = tau_ach_mag / t_mag if t_mag > 1e-12 else 0.0
        
        return AllocationResult(
            u_rw=u_rw, u_mtq=u_mtq,
            tau_achieved=tau_achieved, tau_desired=tau_des,
            alpha=alpha, direction_error_deg=direction_error_deg,
            magnitude_ratio=magnitude_ratio, solve_time_us=solve_time,
            solver_success=res.success, method=self.name
        )


class QPCAllocator(TorqueAllocator):
    """
    QP with Lyapunov Constraint: prevents adding energy when trying to damp.
    
    Solves:
        min ||A_total @ u - τ_des||²
        s.t. -u_max ≤ u ≤ u_max
             ω · (A_total @ u) ≤ max(0, ω · τ_des)  [constraint variant A]
    
    The constraint ensures:
    - When damping (ω · τ_des < 0): don't add energy (ω · τ_ach ≤ 0)
    - When accelerating (ω · τ_des > 0): can add up to requested energy
    
    Properties:
    - Preserves Lyapunov stability intent of control law
    - May sacrifice torque magnitude to maintain stability
    """
    
    def __init__(self, variant: str = "A"):
        """
        variant options:
        - "A": ω · τ ≤ max(0, ω · τ_des) 
        - "B": ω · τ ≤ 0 when ω · τ_des < 0, unconstrained otherwise
        - "C": ω · τ ≤ ω · τ_des always (track energy intent exactly)
        - "D": Two-sided: min(0, ω · τ_des) ≤ ω · τ ≤ max(0, ω · τ_des)
        """
        super().__init__(f"QPC-{variant}")
        self.variant = variant
    
    def allocate(self, tau_des: np.ndarray, b_body: np.ndarray,
                 A_rw: np.ndarray, A_mtq_axes: np.ndarray,
                 u_rw_max: np.ndarray, u_mtq_max: np.ndarray,
                 omega: Optional[np.ndarray] = None,
                 h_rw: Optional[np.ndarray] = None,
                 J_rw: Optional[np.ndarray] = None) -> AllocationResult:
        import time
        start = time.perf_counter()
        
        tau_des = np.asarray(tau_des, float).reshape(3,)
        t_mag = np.linalg.norm(tau_des)
        
        n_rw = A_rw.shape[1] if A_rw.size > 0 else 0
        n_mtq = A_mtq_axes.shape[1] if A_mtq_axes.size > 0 else 0
        
        # If no omega provided, fall back to unconstrained QP
        if omega is None:
            omega = np.zeros(3)
        omega = np.asarray(omega, float).reshape(3,)
        omega_mag = np.linalg.norm(omega)
        
        if t_mag < 1e-12:
            return AllocationResult(
                u_rw=np.zeros(n_rw), u_mtq=np.zeros(n_mtq),
                tau_achieved=np.zeros(3), tau_desired=tau_des,
                alpha=1.0, direction_error_deg=0.0, magnitude_ratio=1.0,
                solve_time_us=0.0, solver_success=True, method=self.name
            )
        
        # Build A_mtq
        if n_mtq > 0:
            A_mtq = -skewsym(b_body) @ A_mtq_axes
        else:
            A_mtq = np.zeros((3, 0))
        
        A_total = np.hstack([A_rw, A_mtq])
        n_act = n_rw + n_mtq
        
        if n_act == 0:
            return AllocationResult(
                u_rw=np.zeros(0), u_mtq=np.zeros(0),
                tau_achieved=np.zeros(3), tau_desired=tau_des,
                alpha=0.0, direction_error_deg=90.0, magnitude_ratio=0.0,
                solve_time_us=0.0, solver_success=False, method=self.name
            )
        
        # Bounds
        lb = np.concatenate([-u_rw_max, -u_mtq_max])
        ub = np.concatenate([u_rw_max, u_mtq_max])
        
        # Compute energy terms
        omega_dot_tau_des = np.dot(omega, tau_des)
        
        # Define constraint based on variant
        C = omega @ A_total  # gradient of ω · τ w.r.t. u
        
        constraints = []
        
        if self.variant == "A":
            # ω · τ ≤ max(0, ω · τ_des)
            ub_constraint = max(0.0, omega_dot_tau_des)
            constraints.append({
                "type": "ineq",
                "fun": lambda u, C=C, ub=ub_constraint: ub - C @ u,
                "jac": lambda u, C=C: -C
            })
        
        elif self.variant == "B":
            # Only constrain when trying to damp
            if omega_dot_tau_des < 0:
                constraints.append({
                    "type": "ineq",
                    "fun": lambda u, C=C: -C @ u,  # ω · τ ≤ 0
                    "jac": lambda u, C=C: -C
                })
        
        elif self.variant == "C":
            # ω · τ ≤ ω · τ_des always
            constraints.append({
                "type": "ineq",
                "fun": lambda u, C=C, ub=omega_dot_tau_des: ub - C @ u,
                "jac": lambda u, C=C: -C
            })
        
        elif self.variant == "D":
            # Two-sided: don't overshoot in either direction
            lb_constraint = min(0.0, omega_dot_tau_des)
            ub_constraint = max(0.0, omega_dot_tau_des)
            constraints.append({
                "type": "ineq",
                "fun": lambda u, C=C, ub=ub_constraint: ub - C @ u,
                "jac": lambda u, C=C: -C
            })
            constraints.append({
                "type": "ineq",
                "fun": lambda u, C=C, lb=lb_constraint: C @ u - lb,
                "jac": lambda u, C=C: C
            })
        
        # Objective function
        def objective(u):
            r = A_total @ u - tau_des
            return 0.5 * np.dot(r, r)
        
        def gradient(u):
            return A_total.T @ (A_total @ u - tau_des)
        
        # Better initial guess: start from unconstrained QP solution
        res_unconstrained = lsq_linear(A_total, tau_des, bounds=(lb, ub), method='bvls')
        u0 = res_unconstrained.x if res_unconstrained.success else np.zeros(n_act)
        
        # Check if unconstrained solution already satisfies constraint
        constraint_satisfied = True
        if constraints:
            for c in constraints:
                if c['fun'](u0) < -1e-9:  # Violated
                    constraint_satisfied = False
                    break
        
        if constraint_satisfied:
            # Unconstrained solution is feasible - use it
            u_sol = u0
            res = type('Result', (), {'success': True, 'x': u0})()
        else:
            # Need to solve constrained problem
            bounds_obj = Bounds(lb, ub)
            res = minimize(objective, u0, jac=gradient, method='SLSQP',
                          constraints=constraints, bounds=bounds_obj,
                          options={'ftol': 1e-12, 'maxiter': 200})
        
        solve_time = (time.perf_counter() - start) * 1e6
        
        if res.success:
            u_sol = res.x
            u_rw = u_sol[:n_rw]
            u_mtq = u_sol[n_rw:]
            tau_achieved = A_total @ u_sol
        else:
            # Fall back to LP if QPC fails
            lp = LPAllocator()
            fallback = lp.allocate(tau_des, b_body, A_rw, A_mtq_axes,
                                   u_rw_max, u_mtq_max)
            u_rw = fallback.u_rw
            u_mtq = fallback.u_mtq
            tau_achieved = fallback.tau_achieved
        
        # Compute metrics
        tau_ach_mag = np.linalg.norm(tau_achieved)
        if tau_ach_mag > 1e-12 and t_mag > 1e-12:
            cos_angle = np.clip(np.dot(tau_achieved, tau_des) / (tau_ach_mag * t_mag), -1, 1)
            direction_error_deg = np.degrees(np.arccos(cos_angle))
            alpha = np.dot(tau_achieved, tau_des / t_mag) / t_mag
            alpha = max(0.0, alpha)
        else:
            direction_error_deg = 0.0 if tau_ach_mag < 1e-12 else 90.0
            alpha = 0.0
        
        magnitude_ratio = tau_ach_mag / t_mag if t_mag > 1e-12 else 0.0
        
        return AllocationResult(
            u_rw=u_rw, u_mtq=u_mtq,
            tau_achieved=tau_achieved, tau_desired=tau_des,
            alpha=alpha, direction_error_deg=direction_error_deg,
            magnitude_ratio=magnitude_ratio, solve_time_us=solve_time,
            solver_success=res.success, method=self.name
        )


class MinNormAllocator(TorqueAllocator):
    """
    Minimum Norm allocator: minimize ||u||² subject to achieving τ_des exactly.
    
    For overactuated systems only. Falls back to QP if exact torque not achievable.
    
    Solves:
        min ||u||²
        s.t. A_total @ u = τ_des
             -u_max ≤ u ≤ u_max
    
    If infeasible, falls back to QP.
    """
    
    def __init__(self):
        super().__init__("MinNorm")
    
    def allocate(self, tau_des: np.ndarray, b_body: np.ndarray,
                 A_rw: np.ndarray, A_mtq_axes: np.ndarray,
                 u_rw_max: np.ndarray, u_mtq_max: np.ndarray,
                 omega: Optional[np.ndarray] = None,
                 h_rw: Optional[np.ndarray] = None,
                 J_rw: Optional[np.ndarray] = None) -> AllocationResult:
        import time
        from scipy.optimize import minimize, LinearConstraint
        start = time.perf_counter()
        
        tau_des = np.asarray(tau_des, float).reshape(3,)
        t_mag = np.linalg.norm(tau_des)
        
        n_rw = A_rw.shape[1] if A_rw.size > 0 else 0
        n_mtq = A_mtq_axes.shape[1] if A_mtq_axes.size > 0 else 0
        
        if t_mag < 1e-12:
            return AllocationResult(
                u_rw=np.zeros(n_rw), u_mtq=np.zeros(n_mtq),
                tau_achieved=np.zeros(3), tau_desired=tau_des,
                alpha=1.0, direction_error_deg=0.0, magnitude_ratio=1.0,
                solve_time_us=0.0, solver_success=True, method=self.name
            )
        
        # Build A_mtq
        if n_mtq > 0:
            A_mtq = -skewsym(b_body) @ A_mtq_axes
        else:
            A_mtq = np.zeros((3, 0))
        
        A_total = np.hstack([A_rw, A_mtq])
        n_act = n_rw + n_mtq
        
        if n_act == 0:
            return AllocationResult(
                u_rw=np.zeros(0), u_mtq=np.zeros(0),
                tau_achieved=np.zeros(3), tau_desired=tau_des,
                alpha=0.0, direction_error_deg=90.0, magnitude_ratio=0.0,
                solve_time_us=0.0, solver_success=False, method=self.name
            )
        
        # Bounds
        lb = np.concatenate([-u_rw_max, -u_mtq_max])
        ub = np.concatenate([u_rw_max, u_mtq_max])
        
        # Try exact torque matching with minimum norm
        def objective(u):
            return 0.5 * np.dot(u, u)
        
        def gradient(u):
            return u
        
        # Equality constraint: A @ u = tau_des
        eq_constraint = LinearConstraint(A_total, tau_des, tau_des)
        
        u0 = np.zeros(n_act)
        bounds_obj = Bounds(lb, ub)
        
        res = minimize(objective, u0, jac=gradient, method='SLSQP',
                      constraints={'type': 'eq', 'fun': lambda u: A_total @ u - tau_des,
                                  'jac': lambda u: A_total},
                      bounds=bounds_obj)
        
        solve_time = (time.perf_counter() - start) * 1e6
        
        # Check if solution actually achieves desired torque
        if res.success:
            tau_achieved = A_total @ res.x
            if np.linalg.norm(tau_achieved - tau_des) < 1e-6:
                u_rw = res.x[:n_rw]
                u_mtq = res.x[n_rw:]
            else:
                # Fallback to QP
                qp = QPAllocator()
                fallback = qp.allocate(tau_des, b_body, A_rw, A_mtq_axes,
                                       u_rw_max, u_mtq_max)
                return fallback
        else:
            # Fallback to QP
            qp = QPAllocator()
            fallback = qp.allocate(tau_des, b_body, A_rw, A_mtq_axes,
                                   u_rw_max, u_mtq_max)
            return fallback
        
        # Compute metrics
        tau_ach_mag = np.linalg.norm(tau_achieved)
        direction_error_deg = 0.0  # Exact match if we got here
        alpha = 1.0
        magnitude_ratio = 1.0
        
        return AllocationResult(
            u_rw=u_rw, u_mtq=u_mtq,
            tau_achieved=tau_achieved, tau_desired=tau_des,
            alpha=alpha, direction_error_deg=direction_error_deg,
            magnitude_ratio=magnitude_ratio, solve_time_us=solve_time,
            solver_success=True, method=self.name
        )


def compare_allocators_single(tau_des: np.ndarray, b_body: np.ndarray,
                               A_rw: np.ndarray, A_mtq_axes: np.ndarray,
                               u_rw_max: np.ndarray, u_mtq_max: np.ndarray,
                               omega: Optional[np.ndarray] = None,
                               allocators: Optional[List[TorqueAllocator]] = None
                               ) -> Dict[str, AllocationResult]:
    """
    Compare multiple allocation methods on a single torque request.
    
    Returns a dictionary mapping method name to AllocationResult.
    """
    if allocators is None:
        allocators = [
            LPAllocator(),
            QPAllocator(),
            QPCAllocator("A"),
            QPCAllocator("B"),
            QPCAllocator("C"),
            QPCAllocator("D"),
        ]
    
    results = {}
    for alloc in allocators:
        result = alloc.allocate(tau_des, b_body, A_rw, A_mtq_axes,
                               u_rw_max, u_mtq_max, omega=omega)
        results[alloc.name] = result
    
    return results


def generate_test_scenarios(n_scenarios: int = 100,
                            seed: int = 42) -> List[Dict]:
    """
    Generate random test scenarios for allocation comparison.
    
    Returns list of dicts with:
    - tau_des: desired torque
    - b_body: magnetic field in body frame
    - omega: angular velocity
    - scenario_type: 'damping', 'accelerating', 'mixed'
    """
    np.random.seed(seed)
    scenarios = []
    
    for i in range(n_scenarios):
        # Random B-field direction (normalized, then scaled to ~30 μT)
        b_dir = np.random.randn(3)
        b_dir = b_dir / np.linalg.norm(b_dir)
        b_body = b_dir * 30e-6  # 30 μT typical LEO
        
        # Random omega
        omega = np.random.randn(3) * 0.02  # ~1 deg/s scale
        
        # Generate tau_des based on scenario type
        scenario_type = np.random.choice(['damping', 'accelerating', 'mixed'])
        
        if scenario_type == 'damping':
            # tau_des opposing omega
            tau_des = -np.random.uniform(0.5, 2.0) * omega / (np.linalg.norm(omega) + 1e-12)
            tau_des *= np.random.uniform(1e-6, 1e-4)  # Scale to realistic torque
        elif scenario_type == 'accelerating':
            # tau_des aligned with omega
            tau_des = np.random.uniform(0.5, 2.0) * omega / (np.linalg.norm(omega) + 1e-12)
            tau_des *= np.random.uniform(1e-6, 1e-4)
        else:
            # Random direction
            tau_des = np.random.randn(3)
            tau_des = tau_des / np.linalg.norm(tau_des) * np.random.uniform(1e-6, 1e-4)
        
        scenarios.append({
            'tau_des': tau_des,
            'b_body': b_body,
            'omega': omega,
            'scenario_type': scenario_type
        })
    
    return scenarios


def run_allocation_comparison(scenarios: List[Dict],
                              A_rw: np.ndarray,
                              A_mtq_axes: np.ndarray,
                              u_rw_max: np.ndarray,
                              u_mtq_max: np.ndarray,
                              allocators: Optional[List[TorqueAllocator]] = None
                              ) -> Dict[str, List[AllocationResult]]:
    """
    Run allocation comparison across multiple scenarios.
    
    Returns dict mapping method name to list of results.
    """
    if allocators is None:
        allocators = [
            LPAllocator(),
            QPAllocator(),
            QPCAllocator("A"),
            QPCAllocator("B"),
        ]
    
    all_results = {alloc.name: [] for alloc in allocators}
    
    for scenario in tqdm(scenarios, desc="Running allocation comparison"):
        results = compare_allocators_single(
            tau_des=scenario['tau_des'],
            b_body=scenario['b_body'],
            A_rw=A_rw,
            A_mtq_axes=A_mtq_axes,
            u_rw_max=u_rw_max,
            u_mtq_max=u_mtq_max,
            omega=scenario['omega'],
            allocators=allocators
        )
        
        for name, result in results.items():
            all_results[name].append(result)
    
    return all_results


def summarize_results(all_results: Dict[str, List[AllocationResult]]) -> Dict:
    """
    Compute summary statistics across all results for each method.
    """
    summary = {}
    
    for method_name, results in all_results.items():
        direction_errors = [r.direction_error_deg for r in results]
        magnitude_ratios = [r.magnitude_ratio for r in results]
        alphas = [r.alpha for r in results]
        solve_times = [r.solve_time_us for r in results]
        success_rate = sum(1 for r in results if r.solver_success) / len(results)
        
        summary[method_name] = {
            'direction_error_deg': {
                'mean': np.mean(direction_errors),
                'std': np.std(direction_errors),
                'max': np.max(direction_errors),
                'median': np.median(direction_errors)
            },
            'magnitude_ratio': {
                'mean': np.mean(magnitude_ratios),
                'std': np.std(magnitude_ratios),
                'min': np.min(magnitude_ratios),
                'median': np.median(magnitude_ratios)
            },
            'alpha': {
                'mean': np.mean(alphas),
                'std': np.std(alphas),
                'min': np.min(alphas),
                'median': np.median(alphas)
            },
            'solve_time_us': {
                'mean': np.mean(solve_times),
                'std': np.std(solve_times),
                'max': np.max(solve_times),
                'median': np.median(solve_times)
            },
            'success_rate': success_rate
        }
    
    return summary


if __name__ == "__main__":
    # Example usage: Compare allocators on a 3MTQ+1RW configuration
    print("=" * 60)
    print("Allocation Method Comparison Framework")
    print("=" * 60)
    
    # Define actuator configuration: 3MTQ + 1RW
    # MTQ axes (body frame)
    A_mtq_axes = np.eye(3)  # Orthogonal MTQs along x, y, z
    u_mtq_max = np.array([0.2, 0.2, 0.2])  # Am²
    
    # RW axis (along z)
    A_rw = np.array([[0], [0], [1.0]])  # Single RW along z
    u_rw_max = np.array([0.001])  # Nm
    
    print(f"\nActuator Configuration:")
    print(f"  MTQs: 3 orthogonal, max dipole {u_mtq_max[0]} Am²")
    print(f"  RWs: 1 along z-axis, max torque {u_rw_max[0]*1000:.1f} mNm")
    
    # Generate test scenarios
    print(f"\nGenerating test scenarios...")
    scenarios = generate_test_scenarios(n_scenarios=500, seed=42)
    
    # Define allocators to compare
    allocators = [
        LPAllocator(),
        QPAllocator(),
        QPCAllocator("A"),
        QPCAllocator("B"),
        QPCAllocator("C"),
        QPCAllocator("D"),
    ]
    
    # Run comparison
    print(f"\nRunning allocation comparison...")
    all_results = run_allocation_comparison(
        scenarios=scenarios,
        A_rw=A_rw,
        A_mtq_axes=A_mtq_axes,
        u_rw_max=u_rw_max,
        u_mtq_max=u_mtq_max,
        allocators=allocators
    )
    
    # Summarize
    summary = summarize_results(all_results)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    for method, stats in summary.items():
        print(f"\n{method}:")
        print(f"  Direction Error: {stats['direction_error_deg']['mean']:.2f}° ± {stats['direction_error_deg']['std']:.2f}° (max {stats['direction_error_deg']['max']:.2f}°)")
        print(f"  Magnitude Ratio: {stats['magnitude_ratio']['mean']:.3f} ± {stats['magnitude_ratio']['std']:.3f}")
        print(f"  Alpha: {stats['alpha']['mean']:.3f} ± {stats['alpha']['std']:.3f}")
        print(f"  Solve Time: {stats['solve_time_us']['mean']:.1f} μs ± {stats['solve_time_us']['std']:.1f} μs")
        print(f"  Success Rate: {stats['success_rate']*100:.1f}%")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"/home/pmckeen/Generalized_ADCS/research/allocation_comparison_{timestamp}.json"
    
    # Convert numpy arrays to lists for JSON serialization
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
