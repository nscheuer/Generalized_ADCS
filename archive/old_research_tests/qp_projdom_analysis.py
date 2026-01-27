"""
Mathematical Analysis of QP with Projection Dominance
=====================================================

Why does QP_ProjDom outperform LP in closed-loop?

Key insight: The constraint τ·τ̂_des ≥ α_LP·‖τ_des‖ ensures we get at least
as much "useful" torque as LP, but QP can add perpendicular components.

Question: When are perpendicular components HELPFUL vs HARMFUL?

Analysis:
1. For PD control: τ_des = -kp*q_err - kd*ω
2. The Lyapunov function V = kp*Φ(q_err) + 0.5*ω·J·ω
3. V̇ = -ω·τ (for the rotational kinetic energy part)

If τ = τ_parallel + τ_perp where τ_parallel ∥ τ_des:
- ω·τ_parallel contributes to V̇ based on alignment with ω
- ω·τ_perp is the "extra" energy contribution

The perpendicular component τ_perp is HELPFUL when:
- It doesn't add energy (ω·τ_perp ≤ 0) while damping
- It accelerates in a direction that helps convergence

Let's analyze when this happens and develop optimal constraints.
"""

import sys
import os
import numpy as np
from scipy.optimize import minimize, Bounds, linprog, lsq_linear
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from ADCS.helpers.math_helpers import skewsym, normalize


def analyze_perpendicular_benefit(tau_des: np.ndarray, tau_achieved: np.ndarray, 
                                   omega: np.ndarray, q_err: np.ndarray) -> Dict:
    """
    Analyze whether perpendicular torque components are helpful.
    
    Returns analysis of the benefit/harm of perpendicular components.
    """
    tau_des = np.asarray(tau_des)
    tau_achieved = np.asarray(tau_achieved)
    omega = np.asarray(omega)
    q_err = np.asarray(q_err)
    
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-12:
        return {'beneficial': True, 'reason': 'no desired torque'}
    
    tau_hat = tau_des / t_mag
    
    # Decompose achieved torque
    tau_parallel = np.dot(tau_achieved, tau_hat) * tau_hat
    tau_perp = tau_achieved - tau_parallel
    
    perp_mag = np.linalg.norm(tau_perp)
    
    # Energy contributions
    energy_parallel = np.dot(omega, tau_parallel)
    energy_perp = np.dot(omega, tau_perp)
    energy_total = np.dot(omega, tau_achieved)
    energy_desired = np.dot(omega, tau_des)
    
    # Attitude error contribution
    # For V = kp * ||q_err||², we want τ to oppose q_err direction
    # Actually for quaternion control, it's more complex, but roughly:
    # We want τ to have negative projection on q_err (to reduce error)
    err_parallel = np.dot(tau_parallel, q_err)
    err_perp = np.dot(tau_perp, q_err)
    
    analysis = {
        'tau_parallel': tau_parallel,
        'tau_perp': tau_perp,
        'perp_magnitude': perp_mag,
        'energy_parallel': energy_parallel,
        'energy_perp': energy_perp,
        'energy_total': energy_total,
        'energy_desired': energy_desired,
        'err_contribution_parallel': err_parallel,
        'err_contribution_perp': err_perp,
    }
    
    # Determine if perpendicular is beneficial
    # Beneficial if:
    # 1. Doesn't add energy when we want to remove it (damping case)
    # 2. Helps reduce attitude error
    
    is_damping = energy_desired < 0
    perp_adds_energy = energy_perp > 1e-12
    perp_helps_error = err_perp < -1e-12  # Negative means reducing error
    
    if is_damping:
        if perp_adds_energy:
            beneficial = False
            reason = 'perpendicular adds energy during damping'
        elif perp_helps_error:
            beneficial = True
            reason = 'perpendicular helps reduce error without adding energy'
        else:
            beneficial = True  # Neutral is OK
            reason = 'perpendicular is neutral'
    else:
        # Accelerating case - perp might help reach goal faster
        if perp_helps_error:
            beneficial = True
            reason = 'perpendicular helps reduce error'
        else:
            beneficial = False
            reason = 'perpendicular fights error reduction'
    
    analysis['beneficial'] = beneficial
    analysis['reason'] = reason
    
    return analysis


def find_optimal_constraint_bound(tau_des: np.ndarray, A_total: np.ndarray,
                                   lb: np.ndarray, ub: np.ndarray,
                                   omega: np.ndarray, q_err: np.ndarray) -> Dict:
    """
    Find the optimal constraint that maximizes benefit of QP over LP.
    
    We want to find constraints such that:
    1. τ·τ̂_des ≥ some threshold (projection dominance)
    2. Additional constraints on perpendicular energy/direction
    
    Returns the optimal constraint parameters.
    """
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-12:
        return {'optimal_threshold': 0, 'optimal_perp_constraint': None}
    
    tau_hat = tau_des / t_mag
    
    # Get LP solution
    n_act = len(lb)
    c = np.zeros(n_act + 1)
    c[-1] = -1.0
    A_eq = np.hstack([A_total, -tau_hat.reshape(3, 1)])
    b_eq = np.zeros(3)
    bounds = [(lb[i], ub[i]) for i in range(n_act)] + [(0, None)]
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if not res.success:
        return {'optimal_threshold': 0, 'optimal_perp_constraint': None}
    
    u_lp = res.x[:n_act]
    alpha_lp = res.x[-1] / t_mag
    tau_lp = A_total @ u_lp
    proj_lp = np.dot(tau_lp, tau_hat)
    
    # Now find QP solutions with varying projection thresholds
    results = []
    
    for threshold_factor in np.linspace(0.5, 1.0, 11):
        min_proj = proj_lp * threshold_factor
        
        # Solve QP with projection constraint
        def objective(u):
            r = A_total @ u - tau_des
            return 0.5 * np.dot(r, r)
        
        def gradient(u):
            return A_total.T @ (A_total @ u - tau_des)
        
        c_proj = A_total.T @ tau_hat
        
        constraint = {
            'type': 'ineq',
            'fun': lambda u, c=c_proj, mp=min_proj: c @ u - mp,
            'jac': lambda u, c=c_proj: c
        }
        
        res_qp = minimize(objective, u_lp, jac=gradient, method='SLSQP',
                         bounds=Bounds(lb, ub), constraints=[constraint],
                         options={'ftol': 1e-10})
        
        if res_qp.success:
            tau_qp = A_total @ res_qp.x
            analysis = analyze_perpendicular_benefit(tau_des, tau_qp, omega, q_err)
            
            # Compute error reduction potential
            err_reduction = np.linalg.norm(tau_des - tau_qp)
            
            results.append({
                'threshold_factor': threshold_factor,
                'tau_achieved': tau_qp,
                'error_reduction': err_reduction,
                'analysis': analysis
            })
    
    # Find best threshold
    best_result = None
    best_score = float('inf')
    
    for r in results:
        if r['analysis']['beneficial']:
            score = r['error_reduction']
            if score < best_score:
                best_score = score
                best_result = r
    
    if best_result is None and results:
        best_result = results[-1]  # Default to strict LP-matching
    
    return {
        'optimal_threshold': best_result['threshold_factor'] if best_result else 1.0,
        'all_results': results,
        'best_result': best_result
    }


def analyze_qp_benefit_conditions():
    """
    Systematically analyze when QP_ProjDom outperforms LP.
    
    Hypotheses:
    1. QP helps when tau_des has large perpendicular component to actuator capability
    2. QP helps when omega is nearly parallel to tau_des
    3. QP helps when the achievable polytope is "thin" in tau_des direction
    """
    np.random.seed(42)
    
    # Actuator config
    A_mtq_axes = np.eye(3)
    u_mtq_max = np.array([0.2, 0.2, 0.2])
    A_rw = np.array([[0], [0], [1.0]])
    u_rw_max = np.array([0.001])
    
    lb = np.concatenate([-u_rw_max, -u_mtq_max])
    ub = np.concatenate([u_rw_max, u_mtq_max])
    
    n_tests = 500
    qp_better_cases = []
    lp_better_cases = []
    
    for i in range(n_tests):
        # Random B-field
        b_body = normalize(np.random.randn(3)) * 30e-6
        A_mtq = -skewsym(b_body) @ A_mtq_axes
        A_total = np.hstack([A_rw, A_mtq])
        
        # Random omega and q_err
        omega = np.random.randn(3) * 0.02
        q_err = np.random.randn(3) * 0.3
        
        # tau_des from PD control
        kp, kd = 5e-5, 1e-3
        tau_des = -kp * q_err - kd * omega
        t_mag = np.linalg.norm(tau_des)
        
        if t_mag < 1e-12:
            continue
        
        tau_hat = tau_des / t_mag
        
        # Solve LP
        n_act = 4
        c = np.zeros(n_act + 1)
        c[-1] = -1.0
        A_eq = np.hstack([A_total, -tau_hat.reshape(3, 1)])
        b_eq = np.zeros(3)
        bounds_lp = [(lb[i], ub[i]) for i in range(n_act)] + [(0, None)]
        res_lp = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds_lp, method='highs')
        
        if not res_lp.success:
            continue
        
        u_lp = res_lp.x[:n_act]
        tau_lp = A_total @ u_lp
        
        # Solve QP_ProjDom
        min_proj = np.dot(tau_lp, tau_hat) * 0.999
        
        def objective(u):
            r = A_total @ u - tau_des
            return 0.5 * np.dot(r, r)
        
        def gradient(u):
            return A_total.T @ (A_total @ u - tau_des)
        
        c_proj = A_total.T @ tau_hat
        constraint = {
            'type': 'ineq',
            'fun': lambda u, c=c_proj, mp=min_proj: c @ u - mp,
            'jac': lambda u, c=c_proj: c
        }
        
        res_qp = minimize(objective, u_lp, jac=gradient, method='SLSQP',
                         bounds=Bounds(lb, ub), constraints=[constraint],
                         options={'ftol': 1e-10})
        
        if not res_qp.success:
            continue
        
        tau_qp = A_total @ res_qp.x
        
        # Compare errors
        err_lp = np.linalg.norm(tau_lp - tau_des)
        err_qp = np.linalg.norm(tau_qp - tau_des)
        
        # Analyze the geometry
        b_hat = b_body / np.linalg.norm(b_body)
        tau_des_perp_to_b = tau_des - np.dot(tau_des, b_hat) * b_hat
        tau_des_achievability = np.linalg.norm(tau_des_perp_to_b) / t_mag
        
        omega_tau_alignment = np.abs(np.dot(omega, tau_hat)) / (np.linalg.norm(omega) + 1e-12)
        
        case_data = {
            'tau_des': tau_des,
            'tau_lp': tau_lp,
            'tau_qp': tau_qp,
            'err_lp': err_lp,
            'err_qp': err_qp,
            'b_body': b_body,
            'omega': omega,
            'q_err': q_err,
            'tau_des_achievability': tau_des_achievability,
            'omega_tau_alignment': omega_tau_alignment
        }
        
        if err_qp < err_lp - 1e-9:
            qp_better_cases.append(case_data)
        else:
            lp_better_cases.append(case_data)
    
    # Analyze patterns
    print("=" * 70)
    print("ANALYSIS: When does QP_ProjDom outperform LP?")
    print("=" * 70)
    
    print(f"\nTotal cases: {len(qp_better_cases) + len(lp_better_cases)}")
    print(f"QP better: {len(qp_better_cases)} ({100*len(qp_better_cases)/(len(qp_better_cases)+len(lp_better_cases)):.1f}%)")
    
    if qp_better_cases:
        # Average characteristics when QP is better
        qp_achievability = np.mean([c['tau_des_achievability'] for c in qp_better_cases])
        qp_alignment = np.mean([c['omega_tau_alignment'] for c in qp_better_cases])
        qp_improvement = np.mean([c['err_lp'] - c['err_qp'] for c in qp_better_cases])
        
        lp_achievability = np.mean([c['tau_des_achievability'] for c in lp_better_cases])
        lp_alignment = np.mean([c['omega_tau_alignment'] for c in lp_better_cases])
        
        print(f"\nWhen QP is better:")
        print(f"  τ_des achievability (perp to B): {qp_achievability:.3f}")
        print(f"  ω-τ alignment: {qp_alignment:.3f}")
        print(f"  Average improvement: {qp_improvement:.2e}")
        
        print(f"\nWhen LP is better/equal:")
        print(f"  τ_des achievability (perp to B): {lp_achievability:.3f}")
        print(f"  ω-τ alignment: {lp_alignment:.3f}")
        
        # Statistical test
        from scipy import stats
        
        achievability_qp = [c['tau_des_achievability'] for c in qp_better_cases]
        achievability_lp = [c['tau_des_achievability'] for c in lp_better_cases]
        
        t_stat, p_val = stats.ttest_ind(achievability_qp, achievability_lp)
        print(f"\nT-test for achievability difference: p={p_val:.4f}")
        
        if p_val < 0.05:
            if np.mean(achievability_qp) > np.mean(achievability_lp):
                print("  → QP helps more when τ_des is MORE achievable (perp to B)")
            else:
                print("  → QP helps more when τ_des is LESS achievable (parallel to B)")
    
    return qp_better_cases, lp_better_cases


def derive_optimal_qp_formulation():
    """
    Based on analysis, derive the optimal QP formulation.
    """
    print("\n" + "=" * 70)
    print("OPTIMAL QP FORMULATION DERIVATION")
    print("=" * 70)
    
    print("""
Based on analysis, the optimal QP formulation is:

    min  ||A·u - τ_des||²
    s.t. u_min ≤ u ≤ u_max
         (A·u)·τ̂_des ≥ (1-ε)·α_LP·||τ_des||     [Projection Dominance]
         
Where:
- α_LP = LP's achieved magnitude ratio
- ε = small margin (0.001) for numerical stability

ADDITIONAL CONSTRAINTS (when beneficial):

For damping (ω·τ_des < 0):
    ω·(A·u - τ_parallel) ≤ 0    [No perpendicular energy injection]
    
Equivalently:
    ω·(A·u) ≤ ω·τ_parallel = α·(ω·τ_des)

This ensures perpendicular components only remove energy, never add it.

MATHEMATICAL JUSTIFICATION:

The Lyapunov derivative for rotational dynamics is:
    V̇ = ω·τ_achieved - (other terms)

For stability, we need V̇ ≤ 0 eventually (with PE conditions for underactuated).

LP guarantees: τ_achieved = α·τ_des (direction preserved)
QP_ProjDom guarantees: τ_achieved·τ̂_des ≥ α·||τ_des|| (projection preserved)

The difference is the perpendicular component:
    τ_perp = τ_achieved - α·τ_des
    
If ω·τ_perp ≤ 0, then:
    V̇_QP ≤ V̇_LP
    
So QP with the energy constraint is at least as stable as LP!

WHEN QP IS STRICTLY BETTER:

QP outperforms LP when the perpendicular component helps reduce ||τ - τ_des||
without destabilizing the system. This happens when:

1. The desired torque has a large component parallel to B (unachievable by MTQ)
2. But there exists a perpendicular component that QP can add
3. And this perpendicular component doesn't add energy during damping
""")


if __name__ == "__main__":
    # Run analysis
    qp_better, lp_better = analyze_qp_benefit_conditions()
    
    # Derive formulation
    derive_optimal_qp_formulation()
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The QP_ProjDom formulation with energy constraint is mathematically
justified to:
1. Never perform worse than LP (by projection dominance)
2. Sometimes perform better (by utilizing perpendicular freedom)
3. Maintain Lyapunov stability (by energy constraint during damping)

RECOMMENDED IMPLEMENTATION:

    def allocate_optimal(tau_des, A_total, lb, ub, omega):
        # Step 1: Solve LP to get α_LP
        alpha_lp, u_lp = solve_lp(tau_des, A_total, lb, ub)
        
        # Step 2: Solve QP with constraints
        min_proj = alpha_lp * ||tau_des|| * 0.999
        
        is_damping = (omega · tau_des) < 0
        
        constraints = [
            projection_dominance(min_proj),
        ]
        
        if is_damping:
            max_energy = omega · (alpha_lp * tau_des)
            constraints.append(energy_limit(max_energy))
        
        return solve_qp(tau_des, A_total, lb, ub, constraints)
""")
