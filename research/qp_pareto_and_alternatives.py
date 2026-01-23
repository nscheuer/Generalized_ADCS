"""
Testing Pareto Constraints and Exploring Better Mathematical Options
====================================================================

1. Test Pareto (per-axis) constraints in closed-loop
2. Explore more mathematically principled alternatives
"""

import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
from typing import Tuple, Optional, Callable
import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from ADCS.helpers.math_helpers import skewsym


def setup_system():
    """Standard 3MTQ + 1RW test system."""
    B = np.array([20e-6, 15e-6, 10e-6])
    A_rw = np.array([[0], [0], [1.0]])
    A_mtq = -skewsym(B) @ np.eye(3)
    A = np.hstack([A_rw, A_mtq])
    
    lb = np.array([-0.001, -0.2, -0.2, -0.2])
    ub = np.array([0.001, 0.2, 0.2, 0.2])
    
    return A, lb, ub, B


def lp_solve(tau_des, A, lb, ub):
    """LP baseline: max α s.t. τ = α·τ̂_des."""
    n = len(lb)
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-12:
        return np.zeros(n), np.zeros(3), 1.0
    
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
            alpha = t_mag
        return u, A @ u, alpha / t_mag
    return np.zeros(n), np.zeros(3), 0.0


def qp_unconstrained(tau_des, A, lb, ub):
    """Standard QP: min ||τ - τ_des||²."""
    n = len(lb)
    u = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(A @ u - tau_des))
    constraints = [u >= lb, u <= ub]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    return u.value if u.value is not None else np.zeros(n)


def qp_pareto(tau_des, A, lb, ub):
    """QP with Pareto constraint: each axis at least as good as LP."""
    n = len(lb)
    u_lp, tau_lp, _ = lp_solve(tau_des, A, lb, ub)
    
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(tau - tau_des))
    
    constraints = [u >= lb, u <= ub]
    for i in range(3):
        if tau_des[i] > 1e-12:
            constraints.append(tau[i] >= tau_lp[i] - 1e-12)
        elif tau_des[i] < -1e-12:
            constraints.append(tau[i] <= tau_lp[i] + 1e-12)
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    if u.value is not None:
        return u.value
    return u_lp  # Fallback


def qp_projection_dom(tau_des, A, lb, ub):
    """QP with projection dominance: τ·τ̂ ≥ ||τ_LP||."""
    n = len(lb)
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-12:
        return np.zeros(n)
    
    tau_hat = tau_des / t_mag
    u_lp, tau_lp, _ = lp_solve(tau_des, A, lb, ub)
    proj_lp = np.dot(tau_lp, tau_hat)
    
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(tau - tau_des))
    constraints = [
        u >= lb, u <= ub,
        tau @ tau_hat >= proj_lp - 1e-12
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    return u.value if u.value is not None else u_lp


def qp_direction_cone(tau_des, A, lb, ub, theta_max_deg=30):
    """QP with direction cone: angle(τ, τ_des) ≤ θ_max."""
    n = len(lb)
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-12:
        return np.zeros(n)
    
    u_lp, tau_lp, _ = lp_solve(tau_des, A, lb, ub)
    
    u = cp.Variable(n)
    tau = A @ u
    
    # τ · τ_des ≥ cos(θ_max) * ||τ|| * ||τ_des||
    # This is a second-order cone constraint
    cos_theta = np.cos(np.radians(theta_max_deg))
    
    objective = cp.Minimize(cp.sum_squares(tau - tau_des))
    constraints = [
        u >= lb, u <= ub,
        tau @ tau_des >= cos_theta * cp.norm(tau) * t_mag
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    return u.value if u.value is not None else u_lp


# ==================== NEW MATHEMATICAL OPTIONS ====================

def qp_weighted_by_achievability(tau_des, A, lb, ub):
    """
    Weight the objective by inverse achievability.
    
    Instead of min ||τ - τ_des||², use:
        min Σ wᵢ(τᵢ - τ_des_i)²
    
    where wᵢ = 1 / (max achievable |τᵢ|)²
    
    This normalizes each axis by what's achievable.
    """
    n = len(lb)
    
    # Compute max achievable torque on each axis
    max_tau = np.zeros(3)
    for i in range(3):
        # Max τᵢ
        c = np.zeros(n)
        for j in range(n):
            c[j] = -A[i, j]  # Maximize A[i,:] @ u
        res = linprog(c, bounds=list(zip(lb, ub)), method='highs')
        if res.success:
            max_tau[i] = max(abs(-res.fun), 1e-12)
    
    # Weights: inverse square of achievability
    w = 1.0 / (max_tau ** 2 + 1e-24)
    w = w / np.sum(w)  # Normalize
    
    u = cp.Variable(n)
    tau = A @ u
    
    # Weighted objective
    objective = cp.Minimize(cp.sum([w[i] * cp.square(tau[i] - tau_des[i]) for i in range(3)]))
    constraints = [u >= lb, u <= ub]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    return u.value if u.value is not None else np.zeros(n)


def qp_normalized_error(tau_des, A, lb, ub):
    """
    Minimize normalized error: ||τ/τ_max - τ_des/τ_max||²
    
    This is equivalent to weighting, but conceptually cleaner.
    Each axis is normalized to [0,1] range.
    """
    n = len(lb)
    
    # Compute max achievable torque on each axis
    max_tau = np.zeros(3)
    for i in range(3):
        c = np.zeros(n)
        for j in range(n):
            c[j] = -A[i, j]
        res = linprog(c, bounds=list(zip(lb, ub)), method='highs')
        if res.success:
            max_tau[i] = max(abs(-res.fun), 1e-12)
    
    # Normalized desired
    tau_des_norm = tau_des / max_tau
    
    u = cp.Variable(n)
    tau = A @ u
    tau_norm = tau / max_tau  # Element-wise division
    
    objective = cp.Minimize(cp.sum_squares(tau_norm - tau_des_norm))
    constraints = [u >= lb, u <= ub]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    return u.value if u.value is not None else np.zeros(n)


def qp_ratio_matching(tau_des, A, lb, ub):
    """
    Match the RATIO of achieved to desired across axes.
    
    Find α and τ that minimize:
        ||τ - α·τ_des||²
    subject to τ achievable.
    
    This is LP + residual minimization.
    """
    n = len(lb)
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-12:
        return np.zeros(n)
    
    # First, find optimal α via LP
    u_lp, tau_lp, alpha = lp_solve(tau_des, A, lb, ub)
    
    # Now minimize ||τ - α·τ_des||² 
    # But this is just QP with target = α·τ_des!
    tau_target = alpha * tau_des
    
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(tau - tau_target))
    constraints = [u >= lb, u <= ub]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    return u.value if u.value is not None else u_lp


def qp_max_min_ratio(tau_des, A, lb, ub):
    """
    Maximize the minimum achievement ratio across axes.
    
    max min_i (τᵢ / τ_des_i)  for τ_des_i > 0
    
    This is a max-min (Chebyshev) formulation.
    """
    n = len(lb)
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-12:
        return np.zeros(n)
    
    u = cp.Variable(n)
    alpha = cp.Variable()  # The minimum ratio
    tau = A @ u
    
    objective = cp.Maximize(alpha)
    constraints = [u >= lb, u <= ub, alpha >= 0]
    
    for i in range(3):
        if abs(tau_des[i]) > 1e-12:
            # τᵢ / τ_des_i ≥ α (for positive τ_des)
            # τᵢ ≥ α * τ_des_i
            constraints.append(tau[i] * np.sign(tau_des[i]) >= alpha * abs(tau_des[i]))
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    return u.value if u.value is not None else np.zeros(n)


def qp_lexicographic(tau_des, A, lb, ub):
    """
    Lexicographic optimization: first maximize projection, then minimize perpendicular.
    
    Stage 1: max τ·τ̂_des s.t. bounds
    Stage 2: min ||τ_perp||² s.t. τ·τ̂ = τ_parallel_max
    """
    n = len(lb)
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-12:
        return np.zeros(n)
    
    tau_hat = tau_des / t_mag
    
    # Stage 1: Maximize projection
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Maximize(tau @ tau_hat)
    constraints = [u >= lb, u <= ub]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    if u.value is None:
        return np.zeros(n)
    
    tau_parallel_max = (A @ u.value) @ tau_hat
    
    # Stage 2: Minimize perpendicular given max parallel
    u2 = cp.Variable(n)
    tau2 = A @ u2
    tau_perp = tau2 - (tau2 @ tau_hat) * tau_hat
    
    objective2 = cp.Minimize(cp.sum_squares(tau_perp))
    constraints2 = [
        u2 >= lb, u2 <= ub,
        tau2 @ tau_hat >= tau_parallel_max - 1e-10
    ]
    prob2 = cp.Problem(objective2, constraints2)
    prob2.solve(solver=cp.ECOS)
    
    return u2.value if u2.value is not None else u.value


def qp_ellipsoidal_norm(tau_des, A, lb, ub):
    """
    Use ellipsoidal norm based on reachable set.
    
    The reachable set R has a natural "shape" - use its inverse
    as the metric for measuring error.
    
    For polytope R, approximate with inscribed ellipsoid.
    """
    n = len(lb)
    
    # Approximate: compute covariance of vertices
    # For box constraints, vertices are corners
    # Too expensive for large n, so use axis-aligned approximation
    
    # Compute achievable range on each axis
    ranges = np.zeros(3)
    for i in range(3):
        c = np.zeros(n)
        for j in range(n):
            c[j] = -A[i, j]
        res_max = linprog(c, bounds=list(zip(lb, ub)), method='highs')
        res_min = linprog(-c, bounds=list(zip(lb, ub)), method='highs')
        if res_max.success and res_min.success:
            ranges[i] = (-res_max.fun) - (-res_min.fun)
    
    # Weight by inverse range squared (like Mahalanobis distance)
    W = np.diag(1.0 / (ranges ** 2 + 1e-24))
    
    u = cp.Variable(n)
    tau = A @ u
    error = tau - tau_des
    
    # ||error||_W² = error^T W error
    objective = cp.Minimize(cp.quad_form(error, W))
    constraints = [u >= lb, u <= ub]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    return u.value if u.value is not None else np.zeros(n)


# ==================== TESTING ====================

def test_all_methods():
    """Compare all methods."""
    print("=" * 90)
    print("COMPARING QP CONSTRAINT OPTIONS")
    print("=" * 90)
    
    A, lb, ub, B = setup_system()
    n = len(lb)
    
    methods = [
        ("LP (baseline)", lambda td: lp_solve(td, A, lb, ub)[0]),
        ("QP unconstrained", lambda td: qp_unconstrained(td, A, lb, ub)),
        ("QP Pareto", lambda td: qp_pareto(td, A, lb, ub)),
        ("QP Proj Dom", lambda td: qp_projection_dom(td, A, lb, ub)),
        ("QP Cone 30°", lambda td: qp_direction_cone(td, A, lb, ub, 30)),
        ("QP Cone 15°", lambda td: qp_direction_cone(td, A, lb, ub, 15)),
        ("QP Weighted", lambda td: qp_weighted_by_achievability(td, A, lb, ub)),
        ("QP Normalized", lambda td: qp_normalized_error(td, A, lb, ub)),
        ("QP Ratio Match", lambda td: qp_ratio_matching(td, A, lb, ub)),
        ("QP MaxMin Ratio", lambda td: qp_max_min_ratio(td, A, lb, ub)),
        ("QP Lexicographic", lambda td: qp_lexicographic(td, A, lb, ub)),
        ("QP Ellipsoidal", lambda td: qp_ellipsoidal_norm(td, A, lb, ub)),
    ]
    
    test_cases = [
        ("Balanced [10,10,10]", np.array([10e-6, 10e-6, 10e-6])),
        ("Heavy z [1,1,100]", np.array([1e-6, 1e-6, 100e-6])),
        ("Heavy xy [100,100,1]", np.array([100e-6, 100e-6, 1e-6])),
        ("Asymmetric [50,10,10]", np.array([50e-6, 10e-6, 10e-6])),
        ("Small [1,1,1]", np.array([1e-6, 1e-6, 1e-6])),
    ]
    
    for tc_name, tau_des in test_cases:
        print(f"\n{'='*90}")
        print(f"TEST CASE: {tc_name}")
        print(f"τ_des = {tau_des * 1e6} μNm")
        print(f"{'='*90}")
        
        tau_hat = tau_des / np.linalg.norm(tau_des)
        
        # Get LP baseline for reference
        _, tau_lp, alpha_lp = lp_solve(tau_des, A, lb, ub)
        print(f"LP baseline: τ = {tau_lp*1e6} μNm, α = {alpha_lp:.4f}")
        
        print(f"\n{'Method':<18} {'τ (μNm)':<35} {'‖τ‖':>8} {'proj':>8} {'dir°':>6} {'ratios':<20}")
        print("-" * 100)
        
        for name, method in methods:
            try:
                u = method(tau_des)
                if u is None:
                    print(f"{name:<18} FAILED")
                    continue
                    
                tau = A @ u
                tau_norm = np.linalg.norm(tau)
                proj = np.dot(tau, tau_hat)
                
                if tau_norm > 1e-15:
                    dir_err = np.degrees(np.arccos(np.clip(proj/tau_norm, -1, 1)))
                else:
                    dir_err = 0
                
                # Achievement ratios per axis
                ratios = []
                for i in range(3):
                    if abs(tau_des[i]) > 1e-15:
                        ratios.append(tau[i] / tau_des[i])
                    else:
                        ratios.append(0)
                
                tau_str = f"[{tau[0]*1e6:8.3f},{tau[1]*1e6:8.3f},{tau[2]*1e6:8.3f}]"
                ratio_str = f"[{ratios[0]:.3f},{ratios[1]:.3f},{ratios[2]:.3f}]"
                
                print(f"{name:<18} {tau_str:<35} {tau_norm*1e6:>8.3f} {proj*1e6:>8.3f} {dir_err:>6.1f} {ratio_str:<20}")
                
            except Exception as e:
                print(f"{name:<18} ERROR: {e}")
    
    return


def analyze_mathematical_properties():
    """Analyze the mathematical properties of each method."""
    print("\n" + "=" * 90)
    print("MATHEMATICAL ANALYSIS OF METHODS")
    print("=" * 90)
    
    print("""
METHOD ANALYSIS:
================

1. LP (baseline)
   Objective: max α s.t. τ = α·τ̂_des
   Properties:
   - Guarantees τ ∝ τ_des (perfect direction)
   - Maximizes "progress" in desired direction
   - All axes get same fraction α of what they asked for
   
   Mathematical form: Linear program
   
   
2. QP Unconstrained
   Objective: min ||τ - τ_des||²
   Properties:
   - Minimizes Euclidean distance in torque space
   - PROBLEM: Treats all axes equally in absolute units
   - With heterogeneous authority, ignores weak axes
   
   Mathematical form: Quadratic program (convex)


3. QP Pareto
   Objective: min ||τ - τ_des||²  s.t. τᵢ ≥ τ_LP_i (when τ_des_i > 0)
   Properties:
   - Never worse than LP on any axis
   - Can improve on LP if possible
   - Convex constraints
   
   Mathematical form: QP with linear constraints
   
   ISSUE: Very restrictive - essentially forces near-LP solution


4. QP Projection Dominance  
   Objective: min ||τ - τ_des||²  s.t. τ·τ̂_des ≥ proj(τ_LP)
   Properties:
   - Total progress at least as good as LP
   - ALLOWS trading between axes
   - Doesn't prevent axis ignoring!
   
   Mathematical form: QP with linear constraint
   

5. QP Direction Cone
   Objective: min ||τ - τ_des||²  s.t. angle(τ, τ_des) ≤ θ_max
   Properties:
   - Bounds direction error directly
   - More restrictive than proj dominance
   - Tunable parameter θ_max
   
   Mathematical form: SOCP (second-order cone)


6. QP Weighted by Achievability
   Objective: min Σ wᵢ(τᵢ - τ_des_i)²  where wᵢ ∝ 1/τ_max_i²
   Properties:
   - Normalizes each axis by what's achievable
   - Hard-to-achieve axes get more weight
   - No constraints beyond bounds
   
   Mathematical form: Weighted QP (still convex)
   
   THIS IS MATHEMATICALLY PRINCIPLED!


7. QP Normalized Error
   Same as weighted, different formulation:
   Objective: min ||(τ - τ_des) / τ_max||²
   
   
8. QP Ratio Matching
   Two-stage:
   Stage 1: Find α* = max α s.t. τ = α·τ_des achievable
   Stage 2: min ||τ - α*·τ_des||²
   
   Properties:
   - Target is the scaled-down desired
   - Minimizes residual error from LP solution
   - Essentially same as LP if LP is tight
   

9. QP MaxMin Ratio (Chebyshev)
   Objective: max min_i (τᵢ/τ_des_i)
   Properties:
   - Maximizes worst-case achievement ratio
   - Balanced across axes by definition
   - Equivalent to LP for our problem!
   
   Mathematical form: Linear program (can be reformulated)


10. QP Lexicographic
    Stage 1: max τ·τ̂_des
    Stage 2: min ||τ_perp||² s.t. τ·τ̂ = max
    Properties:
    - First maximizes useful torque
    - Then minimizes perpendicular component
    - Principled priority ordering
    
    Mathematical form: Sequential optimization


11. QP Ellipsoidal Norm
    Objective: min (τ-τ_des)ᵀ W (τ-τ_des) where W based on reachable set
    Properties:
    - Uses geometry of reachable set
    - Mahalanobis-like distance
    - Adapts to actuator configuration
    
    Mathematical form: Weighted QP


MOST MATHEMATICALLY PRINCIPLED OPTIONS:
=======================================

A) WEIGHTED QP (Option 6/7)
   - Uses achievability as natural metric
   - No arbitrary parameters
   - Convex, efficient to solve
   
B) LEXICOGRAPHIC (Option 10)
   - Clear priority: useful torque first
   - Then clean up perpendicular
   - Two-stage but well-defined

C) MAXMIN RATIO (Option 9)  
   - Fairness criterion (Rawlsian)
   - Equivalent to LP for single-direction problem
   - Natural for "balanced achievement"
""")
    
    return


def deep_dive_weighted_qp():
    """Deep analysis of weighted QP approach."""
    print("\n" + "=" * 90)
    print("DEEP DIVE: WEIGHTED QP")
    print("=" * 90)
    
    A, lb, ub, B = setup_system()
    
    print("""
THE WEIGHTED QP APPROACH
========================

Standard QP: min ||τ - τ_des||² = Σᵢ (τᵢ - τ_des_i)²

Problem: This treats 1 μNm error on x-axis same as 1 μNm error on z-axis.
         But x-axis can only achieve ~5 μNm, while z-axis can achieve ~1000 μNm!
         
Solution: Weight by inverse achievability.

Weighted QP: min Σᵢ wᵢ(τᵢ - τ_des_i)²

where wᵢ = 1 / τ_max_i²

This means:
- 1 μNm error on x-axis (max ~5 μNm) contributes: (1/5)² = 0.04
- 1 μNm error on z-axis (max ~1000 μNm) contributes: (1/1000)² = 0.000001

So x-axis errors are weighted 40,000× more than z-axis errors!
""")
    
    # Compute max achievable on each axis
    n = len(lb)
    max_tau = np.zeros(3)
    for i in range(3):
        c = np.zeros(n)
        for j in range(n):
            c[j] = -A[i, j]
        res = linprog(c, bounds=list(zip(lb, ub)), method='highs')
        if res.success:
            max_tau[i] = abs(-res.fun)
    
    print(f"Max achievable torques: {max_tau * 1e6} μNm")
    print(f"Weights (unnormalized): {1/max_tau**2}")
    
    w = 1 / (max_tau ** 2)
    w_norm = w / np.sum(w)
    print(f"Weights (normalized): {w_norm}")
    print(f"Weight ratios: x/z = {w_norm[0]/w_norm[2]:.0f}×, y/z = {w_norm[1]/w_norm[2]:.0f}×")
    
    # Test
    tau_des = np.array([10e-6, 10e-6, 10e-6])
    
    u_std = qp_unconstrained(tau_des, A, lb, ub)
    u_wgt = qp_weighted_by_achievability(tau_des, A, lb, ub)
    u_lp, tau_lp, _ = lp_solve(tau_des, A, lb, ub)
    
    print(f"\nFor τ_des = [10, 10, 10] μNm:")
    print(f"  Standard QP:  τ = {(A @ u_std) * 1e6} μNm")
    print(f"  Weighted QP:  τ = {(A @ u_wgt) * 1e6} μNm")
    print(f"  LP:           τ = {tau_lp * 1e6} μNm")
    
    # Achievement ratios
    tau_std = A @ u_std
    tau_wgt = A @ u_wgt
    
    print(f"\nAchievement ratios (τ/τ_des):")
    print(f"  Standard QP:  {tau_std / tau_des}")
    print(f"  Weighted QP:  {tau_wgt / tau_des}")
    print(f"  LP:           {tau_lp / tau_des}")
    
    return


def test_closed_loop_simple():
    """Simple closed-loop test comparing methods."""
    print("\n" + "=" * 90)
    print("SIMPLE CLOSED-LOOP COMPARISON")
    print("=" * 90)
    
    A, lb, ub, B = setup_system()
    
    # Simple attitude dynamics (just integrating torque for now)
    # dω/dt = J⁻¹ τ
    J = np.diag([0.01, 0.01, 0.005])  # Small satellite
    J_inv = np.linalg.inv(J)
    
    dt = 0.1
    t_end = 60.0
    n_steps = int(t_end / dt)
    
    # Initial conditions
    omega_0 = np.array([0.05, 0.05, 0.05])  # rad/s
    
    methods = [
        ("LP", lambda td: lp_solve(td, A, lb, ub)[0]),
        ("QP", lambda td: qp_unconstrained(td, A, lb, ub)),
        ("QP Pareto", lambda td: qp_pareto(td, A, lb, ub)),
        ("QP Weighted", lambda td: qp_weighted_by_achievability(td, A, lb, ub)),
        ("QP MaxMin", lambda td: qp_max_min_ratio(td, A, lb, ub)),
        ("QP Lexicographic", lambda td: qp_lexicographic(td, A, lb, ub)),
    ]
    
    results = {}
    
    for name, method in methods:
        omega = omega_0.copy()
        omega_history = [omega.copy()]
        
        for i in range(n_steps):
            # Simple damping control law
            tau_des = -0.001 * omega  # Proportional damping
            
            # Allocate
            try:
                u = method(tau_des)
                if u is None:
                    u = np.zeros(len(lb))
            except:
                u = np.zeros(len(lb))
            
            tau = A @ u
            
            # Integrate
            omega = omega + dt * J_inv @ tau
            omega_history.append(omega.copy())
        
        omega_history = np.array(omega_history)
        results[name] = omega_history
    
    print(f"\nFinal angular velocities after {t_end}s (initial: {omega_0} rad/s):")
    print(f"{'Method':<18} {'ω_final (rad/s)':<35} {'|ω|':>10}")
    print("-" * 65)
    
    for name in results:
        omega_final = results[name][-1]
        omega_mag = np.linalg.norm(omega_final)
        print(f"{name:<18} [{omega_final[0]:8.5f},{omega_final[1]:8.5f},{omega_final[2]:8.5f}] {omega_mag:>10.6f}")
    
    return results


if __name__ == "__main__":
    np.random.seed(42)
    
    test_all_methods()
    analyze_mathematical_properties()
    deep_dive_weighted_qp()
    test_closed_loop_simple()
    
    print("\n" + "=" * 90)
    print("CONCLUSIONS")
    print("=" * 90)
    print("""
BEST MATHEMATICAL OPTIONS:
==========================

1. WEIGHTED QP (by achievability)
   - Most principled: uses actuator geometry as metric
   - No arbitrary parameters 
   - Result: Equal "effort" across axes
   
2. LP (baseline)
   - Guarantees direction preservation
   - Natural fairness (all axes get same fraction)
   - Already mathematically optimal for proportional achievement

3. MAXMIN RATIO
   - Fairness criterion: maximize worst-case axis
   - Equivalent to LP for single-direction problems
   - More computation for same result

4. LEXICOGRAPHIC
   - Clear priorities: useful torque first
   - Then minimize unwanted perpendicular
   - Good when direction matters most

RECOMMENDATION:
===============
Use WEIGHTED QP as primary method:
- Adapts to actuator geometry automatically
- No tuning parameters needed
- Mathematically principled (Mahalanobis-like distance)

Fall back to LP when weighted QP has numerical issues.
""")
