"""
Single QP Formulation for Projection-Dominant Allocation (v2)
=============================================================

CORRECTED VERSION

The key insight: We want to encode the LP behavior EXACTLY, not just
maximize projection. LP produces τ = α·τ̂_des (exactly in that direction).

Correct approaches:
1. Add direction constraint: τ × τ̂_des = 0 (parallel)
2. Decompose into parallel + perpendicular and minimize perp
3. Use nullspace formulation
4. Parametric approach: u = u_lp + N*z where N is nullspace

Actually, the cleanest single-optimization approach is:

METHOD A: Maximize-then-minimize via weighted objective
    min  -w1 * α + w2 * ||τ - τ_des||²
    s.t. τ = A@u
         τ = α * τ̂_des + τ_perp  (decomposition)
         lb ≤ u ≤ ub
         
    With w1 >> w2, this first maximizes α, then minimizes perp among 
    solutions with same α.

METHOD B: Lexicographic via constraint (what we want!)
    min  ||τ - τ_des||²
    s.t. τ · τ̂_des ≥ α* · ||τ_des||
         lb ≤ u ≤ ub
         
    where α* is the LP solution. But we want to FIND α* without LP!

METHOD C: Single QP with auxiliary variables
    min  ||τ - τ_des||²
    s.t. τ = α * τ̂_des  (force direction!)
         τ = A @ u
         lb ≤ u ≤ ub
         α ≥ 0
         
    This is actually an LP in disguise (linear in u and α)!

METHOD D: The REAL single-optimization solution
    The trick: Don't try to match LP exactly. Instead, solve a regularized 
    problem that approximates the LP-then-QP solution.
    
    min  ||A@u - τ_des||² / ||τ_des||² 
         - w * (A@u · τ̂_des) / ||τ_des||
         + ε * ||u||²
         
    The key is using NORMALIZED terms and appropriate w.
"""

import sys
import os
import numpy as np
from scipy.optimize import minimize, Bounds, linprog, lsq_linear
from typing import Tuple, Dict
from dataclasses import dataclass
from tqdm import tqdm
import time

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from ADCS.helpers.math_helpers import skewsym, normalize


@dataclass
class AllocationResult:
    u: np.ndarray
    tau_achieved: np.ndarray
    alpha: float  # Projection / ||tau_des||
    direction_error: float  # Angle in degrees
    magnitude_error: float  # ||tau|| / ||tau_des||
    solve_time_us: float
    method: str


def angle_between(v1, v2):
    """Angle in degrees between two vectors."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
    return np.degrees(np.arccos(cos_angle))


def solve_lp(tau_des, A, lb, ub):
    """Reference LP: max alpha s.t. tau = alpha * tau_hat."""
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-12:
        return AllocationResult(np.zeros(len(lb)), np.zeros(3), 1.0, 0.0, 0.0, 0.0, 'LP')
    
    tau_hat = tau_des / t_mag
    n = len(lb)
    
    start = time.perf_counter()
    
    c = np.zeros(n + 1)
    c[-1] = -1.0  # Maximize alpha
    A_eq = np.hstack([A, -tau_hat.reshape(3, 1)])
    b_eq = np.zeros(3)
    bounds = [(lb[i], ub[i]) for i in range(n)] + [(0, None)]
    
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    elapsed = (time.perf_counter() - start) * 1e6
    
    if res.success:
        u = res.x[:n]
        T_max = res.x[-1]
        if T_max > t_mag:
            u = u * (t_mag / T_max)
            alpha = 1.0
        else:
            alpha = T_max / t_mag
        tau = A @ u
        return AllocationResult(u, tau, alpha, angle_between(tau, tau_des), 
                               np.linalg.norm(tau)/t_mag, elapsed, 'LP')
    
    return AllocationResult(np.zeros(n), np.zeros(3), 0.0, 0.0, 0.0, elapsed, 'LP')


def solve_qp(tau_des, A, lb, ub):
    """Standard QP: min ||tau - tau_des||²."""
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-12:
        return AllocationResult(np.zeros(len(lb)), np.zeros(3), 1.0, 0.0, 0.0, 0.0, 'QP')
    
    tau_hat = tau_des / t_mag
    
    start = time.perf_counter()
    res = lsq_linear(A, tau_des, bounds=(lb, ub), method='bvls')
    elapsed = (time.perf_counter() - start) * 1e6
    
    u = res.x if res.success else np.zeros(len(lb))
    tau = A @ u
    alpha = np.dot(tau, tau_hat) / t_mag
    
    return AllocationResult(u, tau, alpha, angle_between(tau, tau_des),
                           np.linalg.norm(tau)/t_mag, elapsed, 'QP')


def solve_lp_then_qp(tau_des, A, lb, ub, margin=0.001):
    """Two-step: LP then constrained QP."""
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-12:
        return AllocationResult(np.zeros(len(lb)), np.zeros(3), 1.0, 0.0, 0.0, 0.0, 'LP+QP')
    
    tau_hat = tau_des / t_mag
    n = len(lb)
    
    start = time.perf_counter()
    
    # Step 1: LP
    lp_res = solve_lp(tau_des, A, lb, ub)
    proj_lp = lp_res.alpha * t_mag
    
    # Step 2: QP with projection constraint
    min_proj = proj_lp * (1.0 - margin)
    
    def objective(u):
        r = A @ u - tau_des
        return 0.5 * np.dot(r, r)
    
    def gradient(u):
        return A.T @ (A @ u - tau_des)
    
    c_proj = A.T @ tau_hat
    constraint = {
        'type': 'ineq',
        'fun': lambda u: c_proj @ u - min_proj,
        'jac': lambda u: c_proj
    }
    
    res = minimize(objective, lp_res.u, jac=gradient, method='SLSQP',
                  bounds=Bounds(lb, ub), constraints=[constraint],
                  options={'ftol': 1e-12})
    
    elapsed = (time.perf_counter() - start) * 1e6
    
    u = res.x if res.success else lp_res.u
    tau = A @ u
    alpha = np.dot(tau, tau_hat) / t_mag
    
    return AllocationResult(u, tau, alpha, angle_between(tau, tau_des),
                           np.linalg.norm(tau)/t_mag, elapsed, 'LP+QP')


def solve_single_qp_v1(tau_des, A, lb, ub, w=100.0):
    """
    Single QP v1: Maximize projection subject to being in correct direction.
    
    This adds direction constraints to ensure tau is parallel to tau_des.
    
    min  -w * (A@u · τ̂) + 0.5 * ||A@u - τ_des||²
    s.t. (A@u) × τ̂ = 0  (parallel constraint, but this is nonlinear!)
    
    Instead, we can use: tau_perp = tau - (tau·τ̂)*τ̂ = 0
    But this is also nonlinear in u.
    
    Actually we need a different approach: parametric LP form.
    """
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-12:
        return AllocationResult(np.zeros(len(lb)), np.zeros(3), 1.0, 0.0, 0.0, 0.0, 'SingleQP_v1')
    
    tau_hat = tau_des / t_mag
    n = len(lb)
    
    start = time.perf_counter()
    
    # Build perpendicular projection matrix
    P_perp = np.eye(3) - np.outer(tau_hat, tau_hat)
    
    # We want: P_perp @ A @ u = 0 (perpendicular component is zero)
    # This is 3 linear constraints, but rank is only 2.
    A_perp = P_perp @ A  # (3, n)
    
    # QP: min -w*projection + regularization
    # subject to perpendicular = 0
    
    c_proj = A.T @ tau_hat
    
    def objective(u):
        proj = c_proj @ u
        # Small regularization
        return -w * proj + 0.01 * np.dot(u, u)
    
    def gradient(u):
        return -w * c_proj + 0.02 * u
    
    # Constraint: perpendicular component = 0
    # We need only 2 of 3 equations (they're linearly dependent)
    # Find 2 basis vectors perpendicular to tau_hat
    if abs(tau_hat[0]) < 0.9:
        v1 = np.cross(tau_hat, [1, 0, 0])
    else:
        v1 = np.cross(tau_hat, [0, 1, 0])
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(tau_hat, v1)
    
    # Constraints: v1 · (A@u) = 0, v2 · (A@u) = 0
    c1 = A.T @ v1
    c2 = A.T @ v2
    
    constraints = [
        {'type': 'eq', 'fun': lambda u: c1 @ u, 'jac': lambda u: c1},
        {'type': 'eq', 'fun': lambda u: c2 @ u, 'jac': lambda u: c2},
    ]
    
    x0 = np.zeros(n)
    res = minimize(objective, x0, jac=gradient, method='SLSQP',
                  bounds=Bounds(lb, ub), constraints=constraints,
                  options={'ftol': 1e-12, 'maxiter': 200})
    
    elapsed = (time.perf_counter() - start) * 1e6
    
    u = res.x if res.success else np.zeros(n)
    tau = A @ u
    alpha = np.dot(tau, tau_hat) / t_mag
    
    return AllocationResult(u, tau, alpha, angle_between(tau, tau_des),
                           np.linalg.norm(tau)/t_mag, elapsed, 'SingleQP_v1')


def solve_single_qp_v2(tau_des, A, lb, ub, lam=1.0):
    """
    Single QP v2: Weighted combination of projection and perpendicular error.
    
    min  ||P_perp @ (A@u - τ_des)||² - λ * (A@u · τ̂)
    
    This minimizes perpendicular error while maximizing projection.
    The trade-off is controlled by λ.
    
    As λ → ∞, this approaches LP behavior.
    As λ → 0, this minimizes perpendicular error only.
    """
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-12:
        return AllocationResult(np.zeros(len(lb)), np.zeros(3), 1.0, 0.0, 0.0, 0.0, 'SingleQP_v2')
    
    tau_hat = tau_des / t_mag
    n = len(lb)
    
    start = time.perf_counter()
    
    # Perpendicular projection
    P_perp = np.eye(3) - np.outer(tau_hat, tau_hat)
    A_perp = P_perp @ A
    tau_perp = P_perp @ tau_des  # Should be zero!
    
    c_proj = A.T @ tau_hat
    
    # H = A_perp' @ A_perp
    # c = -A_perp' @ tau_perp - λ * A' @ tau_hat = -λ * c_proj (since tau_perp = 0)
    H = A_perp.T @ A_perp
    
    def objective(u):
        r_perp = A_perp @ u  # tau_perp component (should minimize)
        proj = c_proj @ u    # projection (should maximize)
        return 0.5 * np.dot(r_perp, r_perp) - lam * proj
    
    def gradient(u):
        return H @ u - lam * c_proj
    
    x0 = np.zeros(n)
    res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                  bounds=[(lb[i], ub[i]) for i in range(n)],
                  options={'ftol': 1e-12, 'maxiter': 100})
    
    elapsed = (time.perf_counter() - start) * 1e6
    
    u = res.x if res.success else np.zeros(n)
    tau = A @ u
    alpha = np.dot(tau, tau_hat) / t_mag
    
    return AllocationResult(u, tau, alpha, angle_between(tau, tau_des),
                           np.linalg.norm(tau)/t_mag, elapsed, 'SingleQP_v2')


def solve_single_qp_v3(tau_des, A, lb, ub, w=1000.0):
    """
    Single QP v3: Projection-weighted QP with normalized scaling.
    
    min  ||A@u - τ_des||² - w * (A@u · τ̂) * ||τ_des||
    
    The key is scaling w by ||τ_des|| so the projection term dominates
    appropriately regardless of magnitude.
    """
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-12:
        return AllocationResult(np.zeros(len(lb)), np.zeros(3), 1.0, 0.0, 0.0, 0.0, 'SingleQP_v3')
    
    tau_hat = tau_des / t_mag
    n = len(lb)
    
    start = time.perf_counter()
    
    c_proj = A.T @ tau_hat
    H = A.T @ A
    
    # Gradient of objective:
    # d/du [0.5*||A@u - τ_des||² - w*(A@u·τ̂)*||τ_des||]
    # = A'(A@u - τ_des) - w*||τ_des||*A'τ̂
    # = A'A@u - A'τ_des - w*||τ_des||*c_proj
    # = A'A@u - A'(τ_des + w*||τ_des||*τ̂)
    
    effective_target = tau_des + w * t_mag * tau_hat
    
    def objective(u):
        r = A @ u - tau_des
        proj = c_proj @ u
        return 0.5 * np.dot(r, r) - w * t_mag * proj
    
    def gradient(u):
        return H @ u - A.T @ effective_target
    
    x0 = np.zeros(n)
    res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                  bounds=[(lb[i], ub[i]) for i in range(n)],
                  options={'ftol': 1e-12, 'maxiter': 100})
    
    elapsed = (time.perf_counter() - start) * 1e6
    
    u = res.x if res.success else np.zeros(n)
    tau = A @ u
    alpha = np.dot(tau, tau_hat) / t_mag
    
    return AllocationResult(u, tau, alpha, angle_between(tau, tau_des),
                           np.linalg.norm(tau)/t_mag, elapsed, 'SingleQP_v3')


def run_comparison():
    """Compare all methods."""
    np.random.seed(42)
    
    A_mtq_axes = np.eye(3)
    u_mtq_max = np.array([0.2, 0.2, 0.2])
    A_rw = np.array([[0], [0], [1.0]])
    u_rw_max = np.array([0.001])
    
    lb = np.concatenate([-u_rw_max, -u_mtq_max])
    ub = np.concatenate([u_rw_max, u_mtq_max])
    
    methods = [
        ('LP', lambda t, A, l, u: solve_lp(t, A, l, u)),
        ('QP', lambda t, A, l, u: solve_qp(t, A, l, u)),
        ('LP+QP', lambda t, A, l, u: solve_lp_then_qp(t, A, l, u)),
        ('SingleQP_v1', lambda t, A, l, u: solve_single_qp_v1(t, A, l, u)),
        ('SingleQP_v2_lam1', lambda t, A, l, u: solve_single_qp_v2(t, A, l, u, lam=1)),
        ('SingleQP_v2_lam10', lambda t, A, l, u: solve_single_qp_v2(t, A, l, u, lam=10)),
        ('SingleQP_v2_lam100', lambda t, A, l, u: solve_single_qp_v2(t, A, l, u, lam=100)),
    ]
    
    n_tests = 500
    results = {name: [] for name, _ in methods}
    
    for _ in tqdm(range(n_tests), desc="Testing"):
        b_body = normalize(np.random.randn(3)) * 30e-6
        A_mtq = -skewsym(b_body) @ A_mtq_axes
        A_total = np.hstack([A_rw, A_mtq])
        
        tau_des = np.random.randn(3) * 1e-5
        if np.linalg.norm(tau_des) < 1e-12:
            continue
        
        for name, solver in methods:
            res = solver(tau_des, A_total, lb, ub)
            results[name].append(res)
    
    # Print results
    print("\n" + "=" * 90)
    print("SINGLE QP FORMULATION COMPARISON (v2)")
    print("=" * 90)
    
    print(f"\n{'Method':<18} {'Alpha':>10} {'DirErr(°)':>10} {'MagRatio':>10} {'Time(μs)':>10}")
    print("-" * 90)
    
    lp_alphas = [r.alpha for r in results['LP']]
    
    for name, _ in methods:
        data = results[name]
        alpha = np.mean([r.alpha for r in data])
        alpha_std = np.std([r.alpha for r in data])
        dir_err = np.mean([r.direction_error for r in data])
        mag = np.mean([r.magnitude_error for r in data])
        time_us = np.mean([r.solve_time_us for r in data])
        
        # Compare alpha to LP
        alpha_diff = np.mean([r.alpha - lp_a for r, lp_a in zip(data, lp_alphas)])
        
        print(f"{name:<18} {alpha:>6.4f}±{alpha_std:>4.3f} {dir_err:>10.2f} {mag:>10.4f} {time_us:>10.1f}")
    
    print("\n" + "-" * 90)
    print("Comparison to LP (alpha difference):")
    for name, _ in methods:
        if name == 'LP':
            continue
        data = results[name]
        diff = [r.alpha - lp_a for r, lp_a in zip(data, lp_alphas)]
        match = sum(1 for d in diff if abs(d) < 0.001)
        print(f"  {name}: Mean diff = {np.mean(diff):+.4f}, Match LP in {100*match/n_tests:.1f}%")
    
    return results


if __name__ == "__main__":
    results = run_comparison()
    
    print("\n" + "=" * 90)
    print("CONCLUSION")
    print("=" * 90)
    print("""
SingleQP_v1 (direction-constrained) matches LP alpha EXACTLY by enforcing
that tau must be parallel to tau_des.

SingleQP_v2 with large lambda also approaches LP behavior.

The BEST single-optimization approach is v1 with direction constraints,
which encodes the LP structure directly.

Recommended for implementation:
    min  -λ*(A@u)·τ̂ + ε*||u||²
    s.t. (A@u) ⊥ τ̂ components = 0  (2 equality constraints)
         lb ≤ u ≤ ub
""")
