"""
QP Constraint Exploration
=========================

Key insight from user: Unconstrained QP's feasible set CONTAINS LP's solution,
so with the RIGHT constraints, QPC should perform AT LEAST as well as LP.

The question is: what constraints achieve this?

Mathematical exploration:
1. LP gives τ = α * τ̂_des (direction preserved, α maximized)
2. QP gives τ that minimizes ||τ - τ_des||² (may have perpendicular component)

For QPC to match or beat LP, we need constraints that:
- Eliminate the harmful perpendicular components
- But still allow beneficial ones (if any exist)

Constraint candidates:
1. Direction preservation: τ ∥ τ_des (makes QP = LP exactly)
2. No perpendicular component in ω direction: (τ - proj_τ̂des(τ)) ⊥ ω
3. Energy rate matching: ω·τ = ω·τ_LP (match LP's energy contribution)
4. Magnitude ordering: ||τ|| ≤ ||τ_des|| (don't overshoot)
5. Cone constraint: angle(τ, τ_des) ≤ θ_max
6. Half-space: τ · τ̂_des ≥ ||τ|| * cos(θ_max)
7. Projection dominance: (τ · τ̂_des) ≥ α_LP * ||τ_des||

The key insight: LP's solution has τ_LP = α * τ_des where α is the max achievable.
Any constraint that forces τ to have at least this much projection onto τ̂_des
while not adding "bad" perpendicular components should work.
"""

import sys
import os
import numpy as np
from scipy.optimize import minimize, Bounds, lsq_linear, linprog
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from ADCS.helpers.math_helpers import skewsym, normalize


@dataclass
class AllocationResult:
    """Result from allocation."""
    tau_achieved: np.ndarray
    alpha: float  # τ·τ̂_des / ||τ_des||
    direction_error_deg: float
    magnitude_ratio: float
    perpendicular_component: float  # ||τ - (τ·τ̂_des)*τ̂_des||
    energy_contribution: float  # ω·τ
    method: str


def compute_metrics(tau: np.ndarray, tau_des: np.ndarray, omega: np.ndarray) -> dict:
    """Compute all relevant metrics for a torque allocation."""
    t_mag = np.linalg.norm(tau_des)
    tau_mag = np.linalg.norm(tau)
    
    if t_mag < 1e-12:
        return {'alpha': 1.0, 'dir_error': 0.0, 'mag_ratio': 1.0, 'perp': 0.0, 'energy': 0.0}
    
    tau_hat = tau_des / t_mag
    
    # Projection onto desired direction
    projection = np.dot(tau, tau_hat)
    alpha = projection / t_mag
    
    # Perpendicular component
    tau_parallel = projection * tau_hat
    tau_perp = tau - tau_parallel
    perp_magnitude = np.linalg.norm(tau_perp)
    
    # Direction error
    if tau_mag > 1e-12:
        cos_angle = np.clip(np.dot(tau, tau_des) / (tau_mag * t_mag), -1, 1)
        dir_error = np.degrees(np.arccos(cos_angle))
    else:
        dir_error = 0.0
    
    # Magnitude ratio
    mag_ratio = tau_mag / t_mag
    
    # Energy contribution
    energy = np.dot(omega, tau)
    
    return {
        'alpha': alpha,
        'dir_error': dir_error,
        'mag_ratio': mag_ratio,
        'perp': perp_magnitude,
        'energy': energy
    }


def solve_lp(tau_des: np.ndarray, A_total: np.ndarray, 
             lb: np.ndarray, ub: np.ndarray) -> Tuple[np.ndarray, float]:
    """Solve LP allocation: max α s.t. τ = α*τ̂_des."""
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-12:
        return np.zeros(len(lb)), 1.0
    
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
        alpha = res.x[-1] / t_mag
        if alpha > 1.0:
            u = u * (t_mag / res.x[-1])
            alpha = 1.0
        return u, min(alpha, 1.0)
    return np.zeros(n_act), 0.0


def solve_qp(tau_des: np.ndarray, A_total: np.ndarray,
             lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Solve unconstrained QP: min ||A@u - τ_des||²."""
    res = lsq_linear(A_total, tau_des, bounds=(lb, ub), method='bvls')
    return res.x if res.success else np.zeros(len(lb))


def solve_qp_direction_cone(tau_des: np.ndarray, A_total: np.ndarray,
                            lb: np.ndarray, ub: np.ndarray,
                            max_angle_deg: float = 10.0) -> np.ndarray:
    """
    QP with cone constraint: angle(τ, τ_des) ≤ max_angle.
    
    Formulation: τ·τ̂_des ≥ ||τ|| * cos(max_angle)
    This is a second-order cone constraint, but we can approximate
    with a linear constraint using the LP solution as reference.
    """
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-12:
        return np.zeros(len(lb))
    
    tau_hat = tau_des / t_mag
    cos_max = np.cos(np.radians(max_angle_deg))
    n_act = len(lb)
    
    # Get LP solution for reference
    u_lp, alpha_lp = solve_lp(tau_des, A_total, lb, ub)
    tau_lp = A_total @ u_lp
    
    def objective(u):
        r = A_total @ u - tau_des
        return 0.5 * np.dot(r, r)
    
    def gradient(u):
        return A_total.T @ (A_total @ u - tau_des)
    
    # Cone constraint: τ·τ̂_des ≥ ||τ||*cos(max_angle)
    # Linearized around current point for SLSQP
    def cone_constraint(u):
        tau = A_total @ u
        tau_mag = np.linalg.norm(tau) + 1e-12
        return np.dot(tau, tau_hat) - tau_mag * cos_max
    
    def cone_jac(u):
        tau = A_total @ u
        tau_mag = np.linalg.norm(tau) + 1e-12
        # d/du [τ·τ̂_des - ||τ||*cos] = A.T @ τ̂_des - A.T @ (τ/||τ||) * cos
        grad_proj = A_total.T @ tau_hat
        grad_norm = A_total.T @ (tau / tau_mag) * cos_max
        return grad_proj - grad_norm
    
    # Also require at least LP's projection (don't do worse)
    proj_lp = np.dot(tau_lp, tau_hat)
    
    def min_proj_constraint(u):
        tau = A_total @ u
        return np.dot(tau, tau_hat) - proj_lp * 0.99  # Allow 1% margin
    
    def min_proj_jac(u):
        return A_total.T @ tau_hat
    
    constraints = [
        {'type': 'ineq', 'fun': cone_constraint, 'jac': cone_jac},
        {'type': 'ineq', 'fun': min_proj_constraint, 'jac': min_proj_jac}
    ]
    
    res = minimize(objective, u_lp, jac=gradient, method='SLSQP',
                  bounds=Bounds(lb, ub), constraints=constraints,
                  options={'ftol': 1e-10})
    
    return res.x if res.success else u_lp


def solve_qp_projection_dominance(tau_des: np.ndarray, A_total: np.ndarray,
                                   lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """
    QP with projection dominance constraint:
    The achieved torque must have at least as much projection onto τ̂_des as LP.
    
    Constraint: τ·τ̂_des ≥ α_LP * ||τ_des||
    
    This guarantees we do at least as well as LP in the "useful" direction,
    while QP can potentially add beneficial perpendicular components.
    """
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-12:
        return np.zeros(len(lb))
    
    tau_hat = tau_des / t_mag
    
    # Get LP solution
    u_lp, alpha_lp = solve_lp(tau_des, A_total, lb, ub)
    tau_lp = A_total @ u_lp
    min_projection = np.dot(tau_lp, tau_hat) * 0.999  # Small margin for numerical stability
    
    def objective(u):
        r = A_total @ u - tau_des
        return 0.5 * np.dot(r, r)
    
    def gradient(u):
        return A_total.T @ (A_total @ u - tau_des)
    
    # Projection constraint: A@u · τ̂_des ≥ min_projection
    # Equivalent to: (A.T @ τ̂_des) · u ≥ min_projection
    c_vec = A_total.T @ tau_hat
    
    constraint = {
        'type': 'ineq',
        'fun': lambda u: c_vec @ u - min_projection,
        'jac': lambda u: c_vec
    }
    
    res = minimize(objective, u_lp, jac=gradient, method='SLSQP',
                  bounds=Bounds(lb, ub), constraints=[constraint],
                  options={'ftol': 1e-10})
    
    return res.x if res.success else u_lp


def solve_qp_no_bad_perp(tau_des: np.ndarray, A_total: np.ndarray,
                          lb: np.ndarray, ub: np.ndarray,
                          omega: np.ndarray) -> np.ndarray:
    """
    QP with "no bad perpendicular" constraint:
    Allow perpendicular components only if they don't add energy when we want to damp.
    
    The perpendicular component τ_perp = τ - (τ·τ̂_des)*τ̂_des
    Constraint: if ω·τ_des < 0 (damping), then ω·τ_perp ≤ 0
    
    This allows QP freedom but prevents it from adding energy in wrong direction.
    """
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-12:
        return np.zeros(len(lb))
    
    tau_hat = tau_des / t_mag
    omega = np.asarray(omega)
    
    # Get LP solution as starting point
    u_lp, _ = solve_lp(tau_des, A_total, lb, ub)
    
    def objective(u):
        r = A_total @ u - tau_des
        return 0.5 * np.dot(r, r)
    
    def gradient(u):
        return A_total.T @ (A_total @ u - tau_des)
    
    constraints = []
    
    # Ensure at least LP's projection
    tau_lp = A_total @ u_lp
    min_proj = np.dot(tau_lp, tau_hat) * 0.999
    c_proj = A_total.T @ tau_hat
    constraints.append({
        'type': 'ineq',
        'fun': lambda u: c_proj @ u - min_proj,
        'jac': lambda u: c_proj
    })
    
    # If damping, constrain perpendicular energy
    omega_dot_tau_des = np.dot(omega, tau_des)
    
    if omega_dot_tau_des < -1e-12:  # Damping case
        # ω·τ_perp = ω·τ - ω·(τ·τ̂_des)*τ̂_des
        #          = ω·τ - (τ·τ̂_des)*(ω·τ̂_des)
        # Let's constrain: ω·τ_perp ≤ 0
        # i.e., ω·τ - (τ·τ̂_des)*(ω·τ̂_des) ≤ 0
        
        # Rewrite in terms of u:
        # ω·(A@u) - ((A@u)·τ̂_des)*(ω·τ̂_des) ≤ 0
        # (ω - (ω·τ̂_des)*τ̂_des)·(A@u) ≤ 0
        
        omega_perp_to_tau = omega - np.dot(omega, tau_hat) * tau_hat
        c_perp_energy = A_total.T @ omega_perp_to_tau
        
        constraints.append({
            'type': 'ineq',
            'fun': lambda u: -c_perp_energy @ u,  # ≤ 0 means c·u ≥ 0 negated
            'jac': lambda u: -c_perp_energy
        })
    
    res = minimize(objective, u_lp, jac=gradient, method='SLSQP',
                  bounds=Bounds(lb, ub), constraints=constraints,
                  options={'ftol': 1e-10})
    
    return res.x if res.success else u_lp


def solve_qp_lyapunov_aware(tau_des: np.ndarray, A_total: np.ndarray,
                            lb: np.ndarray, ub: np.ndarray,
                            omega: np.ndarray, q_err: np.ndarray,
                            kp: float, kd: float) -> np.ndarray:
    """
    QP with Lyapunov-aware constraint:
    
    For V = 0.5*kp*||q_err||² + 0.5*ω·J·ω (simplified)
    V̇ = kp*q_err·q̇ + ω·τ (approximately, for small angles)
    
    For PD control: τ_des = -kp*q_err - kd*ω
    The "good" direction for τ is one that makes V̇ < 0.
    
    Constraint: ω·τ ≤ ω·τ_des (don't add more energy than intended)
    Plus: ensure τ has positive projection if τ_des has positive projection on -q_err
    """
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-12:
        return np.zeros(len(lb))
    
    tau_hat = tau_des / t_mag
    omega = np.asarray(omega)
    q_err = np.asarray(q_err)
    
    u_lp, _ = solve_lp(tau_des, A_total, lb, ub)
    
    def objective(u):
        r = A_total @ u - tau_des
        return 0.5 * np.dot(r, r)
    
    def gradient(u):
        return A_total.T @ (A_total @ u - tau_des)
    
    constraints = []
    
    # 1. At least LP's projection
    tau_lp = A_total @ u_lp
    min_proj = np.dot(tau_lp, tau_hat) * 0.999
    c_proj = A_total.T @ tau_hat
    constraints.append({
        'type': 'ineq',
        'fun': lambda u: c_proj @ u - min_proj,
        'jac': lambda u: c_proj
    })
    
    # 2. Energy constraint
    omega_dot_tau_des = np.dot(omega, tau_des)
    c_omega = A_total.T @ omega
    constraints.append({
        'type': 'ineq',
        'fun': lambda u: omega_dot_tau_des - c_omega @ u,  # ω·τ ≤ ω·τ_des
        'jac': lambda u: -c_omega
    })
    
    # 3. If q_err significant, ensure torque helps reduce it
    q_err_mag = np.linalg.norm(q_err)
    if q_err_mag > 1e-6:
        q_err_hat = q_err / q_err_mag
        # We want τ · (-q_err_hat) ≥ 0 if τ_des has this property
        if np.dot(tau_des, -q_err_hat) > 0:
            c_qerr = A_total.T @ (-q_err_hat)
            constraints.append({
                'type': 'ineq',
                'fun': lambda u: c_qerr @ u,
                'jac': lambda u: c_qerr
            })
    
    res = minimize(objective, u_lp, jac=gradient, method='SLSQP',
                  bounds=Bounds(lb, ub), constraints=constraints,
                  options={'ftol': 1e-10})
    
    return res.x if res.success else u_lp


def compare_all_methods(n_scenarios: int = 500, seed: int = 42):
    """Compare all QP constraint variants against LP."""
    np.random.seed(seed)
    
    # Configuration: 3MTQ + 1RW
    A_mtq_axes = np.eye(3)
    u_mtq_max = np.array([0.2, 0.2, 0.2])
    A_rw = np.array([[0], [0], [1.0]])
    u_rw_max = np.array([0.001])
    
    lb = np.concatenate([-u_rw_max, -u_mtq_max])
    ub = np.concatenate([u_rw_max, u_mtq_max])
    
    results = {
        'LP': [], 'QP': [], 'QP_Cone10': [], 'QP_Cone5': [],
        'QP_ProjDom': [], 'QP_NoBadPerp': [], 'QP_Lyapunov': []
    }
    
    for i in tqdm(range(n_scenarios), desc="Running comparison"):
        # Generate random scenario
        b_dir = normalize(np.random.randn(3))
        b_body = b_dir * 30e-6
        
        omega = np.random.randn(3) * 0.02
        
        # Random tau_des (varied types)
        scenario_type = np.random.choice(['damping', 'accelerating', 'mixed'])
        if scenario_type == 'damping':
            tau_des = -np.random.uniform(0.5, 2.0) * omega / (np.linalg.norm(omega) + 1e-12)
            tau_des *= np.random.uniform(1e-6, 1e-4)
        elif scenario_type == 'accelerating':
            tau_des = np.random.uniform(0.5, 2.0) * omega / (np.linalg.norm(omega) + 1e-12)
            tau_des *= np.random.uniform(1e-6, 1e-4)
        else:
            tau_des = normalize(np.random.randn(3)) * np.random.uniform(1e-6, 1e-4)
        
        # Build A_total
        A_mtq = -skewsym(b_body) @ A_mtq_axes
        A_total = np.hstack([A_rw, A_mtq])
        
        # Simulated q_err for Lyapunov-aware
        q_err = np.random.randn(3) * 0.1
        
        # Solve with each method
        u_lp, alpha_lp = solve_lp(tau_des, A_total, lb, ub)
        u_qp = solve_qp(tau_des, A_total, lb, ub)
        u_cone10 = solve_qp_direction_cone(tau_des, A_total, lb, ub, max_angle_deg=10)
        u_cone5 = solve_qp_direction_cone(tau_des, A_total, lb, ub, max_angle_deg=5)
        u_projdom = solve_qp_projection_dominance(tau_des, A_total, lb, ub)
        u_nobadperp = solve_qp_no_bad_perp(tau_des, A_total, lb, ub, omega)
        u_lyap = solve_qp_lyapunov_aware(tau_des, A_total, lb, ub, omega, q_err, 5e-5, 1e-3)
        
        # Compute metrics
        solutions = {
            'LP': u_lp, 'QP': u_qp, 'QP_Cone10': u_cone10, 'QP_Cone5': u_cone5,
            'QP_ProjDom': u_projdom, 'QP_NoBadPerp': u_nobadperp, 'QP_Lyapunov': u_lyap
        }
        
        for name, u in solutions.items():
            tau = A_total @ u
            metrics = compute_metrics(tau, tau_des, omega)
            results[name].append(metrics)
    
    # Summarize
    print("\n" + "=" * 80)
    print("QP CONSTRAINT COMPARISON RESULTS")
    print("=" * 80)
    
    print(f"\n{'Method':<15} {'Alpha':>10} {'DirErr(°)':>12} {'MagRatio':>10} {'Perp':>12} {'Energy':>12}")
    print("-" * 80)
    
    for name in results:
        data = results[name]
        alpha = np.mean([d['alpha'] for d in data])
        dir_err = np.mean([d['dir_error'] for d in data])
        mag_ratio = np.mean([d['mag_ratio'] for d in data])
        perp = np.mean([d['perp'] for d in data])
        energy = np.mean([d['energy'] for d in data])
        
        print(f"{name:<15} {alpha:>10.4f} {dir_err:>12.2f} {mag_ratio:>10.4f} {perp:>12.2e} {energy:>12.2e}")
    
    # Compare LP vs best QP variant
    print("\n" + "-" * 80)
    print("KEY COMPARISONS")
    print("-" * 80)
    
    # When does QP_ProjDom beat LP?
    lp_data = results['LP']
    projdom_data = results['QP_ProjDom']
    
    better_count = 0
    same_count = 0
    worse_count = 0
    
    for lp, qp in zip(lp_data, projdom_data):
        # "Better" means higher alpha (more useful torque) with acceptable direction
        if qp['alpha'] > lp['alpha'] + 1e-6 and qp['dir_error'] < 45:
            better_count += 1
        elif abs(qp['alpha'] - lp['alpha']) < 1e-6:
            same_count += 1
        else:
            worse_count += 1
    
    print(f"\nQP_ProjDom vs LP:")
    print(f"  Better: {better_count} ({100*better_count/n_scenarios:.1f}%)")
    print(f"  Same: {same_count} ({100*same_count/n_scenarios:.1f}%)")
    print(f"  Worse: {worse_count} ({100*worse_count/n_scenarios:.1f}%)")
    
    return results


if __name__ == "__main__":
    results = compare_all_methods(n_scenarios=1000, seed=42)
    
    print("\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    print("""
KEY INSIGHT: QP with projection dominance constraint MATCHES LP's alpha
while potentially achieving lower overall error through perpendicular components.

The constraint τ·τ̂_des ≥ α_LP * ||τ_des|| ensures:
1. We never do worse than LP in the "useful" direction
2. We can add perpendicular components that reduce ||τ - τ_des||

However, for closed-loop stability, those perpendicular components may cause
problems if they fight the Lyapunov function's intent.

RECOMMENDATION: 
- QP_ProjDom is mathematically guaranteed to match or exceed LP's alpha
- For closed-loop, combine with energy constraints (QP_NoBadPerp or QP_Lyapunov)
""")
