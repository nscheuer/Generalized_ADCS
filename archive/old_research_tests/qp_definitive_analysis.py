"""
Definitive Analysis: What's the Best QP Formulation?
====================================================

The previous tests revealed unexpected results. Let's analyze carefully.
"""

import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
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


def analyze_previous_results():
    """Analyze why some methods performed unexpectedly."""
    print("=" * 80)
    print("ANALYZING UNEXPECTED RESULTS")
    print("=" * 80)
    
    print("""
OBSERVATIONS FROM PREVIOUS TEST:
================================

1. LP (baseline): τ = [2, 2, 2] μNm for τ_des = [10, 10, 10]
   - Direction: 0° (perfect)
   - Achievement: 20% on all axes
   - This is the FAIREST solution

2. QP Unconstrained: τ = [0.01, 0.09, 10.6] μNm
   - Direction: 54.4°
   - Achievement: [0.1%, 0.9%, 106%]
   - IGNORES x,y completely

3. QP Pareto: τ = [2, 2, 14.4] μNm  
   - Direction: 43.6°
   - Achievement: [20%, 20%, 144%]
   - Maintains LP on x,y, BOOSTS z
   - Actually pretty good!

4. QP Weighted: τ = [0.03, 0.01, 0.9] μNm
   - Direction: 53.1°
   - Achievement: [0.3%, 0.1%, 9%]
   - WORSE than LP! What happened?

5. QP Cone (15°): τ = [1.43, 1.46, 1.57] μNm
   - Direction: 2.3°
   - Achievement: [14%, 15%, 16%]
   - Good direction, but LOWER magnitude than LP!

6. MaxMin Ratio: τ = [2, 2, 423] μNm
   - Direction: 54.4°
   - Achievement: [20%, 20%, 4232%]
   - Crazy z overshoot!


KEY INSIGHT: The problem formulations have subtle issues!
""")


def understand_maxmin_failure():
    """Understand why MaxMin Ratio overshoots z."""
    print("\n" + "=" * 80)
    print("UNDERSTANDING MAXMIN RATIO FAILURE")
    print("=" * 80)
    
    print("""
MaxMin formulation: max α s.t. τᵢ ≥ α·τ_des_i for all i

For τ_des = [10, 10, 10]:
    τ_x ≥ α·10
    τ_y ≥ α·10  
    τ_z ≥ α·10

The max achievable α is limited by x,y (which max out at ~5 μNm).
So α_max ≈ 0.2, giving τ_x = τ_y = 2 μNm.

BUT: The constraint on z is just τ_z ≥ 2 μNm.
     Since we're MAXIMIZING α with no upper bound on τ_z,
     the solver can push τ_z to ANY value ≥ 2!
     
     And with objective max α, there's no reason NOT to use
     more z-torque (it doesn't hurt α).

FIX: We need EQUALITY or RATIO constraints, not just lower bounds!
""")
    
    A, lb, ub, B = setup_system()
    n = len(lb)
    tau_des = np.array([10e-6, 10e-6, 10e-6])
    
    # Corrected MaxMin: also bound from above
    print("\nCorrected MaxMin (with upper bound):")
    u = cp.Variable(n)
    alpha = cp.Variable()
    tau = A @ u
    
    objective = cp.Maximize(alpha)
    constraints = [u >= lb, u <= ub, alpha >= 0, alpha <= 2.0]  # Cap alpha
    
    for i in range(3):
        if abs(tau_des[i]) > 1e-12:
            # Lower AND upper bound
            constraints.append(tau[i] >= alpha * tau_des[i] - 1e-12)
            constraints.append(tau[i] <= 2 * alpha * tau_des[i] + 1e-12)  # Allow 2x overshoot max
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    if u.value is not None:
        tau_result = A @ u.value
        print(f"  α = {alpha.value:.4f}")
        print(f"  τ = {tau_result * 1e6} μNm")
        print(f"  Ratios: {tau_result / tau_des}")


def understand_weighted_failure():
    """Understand why weighted QP performed poorly."""
    print("\n" + "=" * 80)
    print("UNDERSTANDING WEIGHTED QP FAILURE")
    print("=" * 80)
    
    print("""
Weighted QP: min Σ wᵢ(τᵢ - τ_des_i)²

where wᵢ = 1/τ_max_i²

For our system:
    τ_max = [5, 6, 1007] μNm
    w = [0.59, 0.41, 0.00001]  (normalized)

So the objective is:
    0.59·(τ_x - 10)² + 0.41·(τ_y - 10)² + 0.00001·(τ_z - 10)²

Since z has near-zero weight, the solver IGNORES z error!
It focuses entirely on x,y, giving tiny torques.

THE PROBLEM: Weighting by 1/τ_max² doesn't account for the COUPLING
             between actuators and torque axes.

INSIGHT: The "achievable" metric should be about the REACHABLE SET,
         not just the max torque on each axis independently.
""")
    
    A, lb, ub, B = setup_system()
    n = len(lb)
    
    # Show the reachable set better
    print("\nReachable set analysis:")
    print("  Torque is τ = A @ u where u ∈ [lb, ub]")
    print("  A maps actuator space to torque space")
    print()
    
    # Max on each axis (independently)
    for i, axis in enumerate(['x', 'y', 'z']):
        c = np.zeros(n)
        for j in range(n):
            c[j] = -A[i, j]
        res = linprog(c, bounds=list(zip(lb, ub)), method='highs')
        print(f"  Max |τ_{axis}| = {abs(-res.fun)*1e6:.2f} μNm")
    
    # But these maxes can't all be achieved simultaneously!
    print("\nBUT: To achieve max τ_x, we need specific MTQ commands")
    print("     that may not be compatible with achieving max τ_y!")


def correct_weighted_qp():
    """Develop a corrected weighted QP approach."""
    print("\n" + "=" * 80)
    print("CORRECTED WEIGHTED QP: Use LP Achievement as Target")
    print("=" * 80)
    
    print("""
BETTER APPROACH: Weight by LP achievement, not raw authority

For each τ_des direction, compute:
    α_LP = max achievable scaling factor
    τ_LP = α_LP · τ_des

Then minimize error from this achievable target:
    min ||τ - τ_LP||²

This is basically "ratio matching" but framed as weighted QP.

EVEN BETTER: Normalize by what's achievable IN THAT DIRECTION

For direction τ̂_des:
    τ_achievable_max = max α s.t. α·τ̂_des ∈ R
    
Then: min ||τ/τ_achievable_max - τ_des/τ_achievable_max||²
    = min ||τ - τ_des||² / τ_achievable_max²

This is a SCALAR weight that depends on the DIRECTION, not just axes.
""")
    
    A, lb, ub, B = setup_system()
    n = len(lb)
    
    def lp_solve(tau_des):
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
            return u, A @ u, alpha
        return np.zeros(n), np.zeros(3), 0.0
    
    # Test with direction-normalized QP
    tau_des = np.array([10e-6, 10e-6, 10e-6])
    u_lp, tau_lp, alpha_lp = lp_solve(tau_des)
    
    print(f"\nFor τ_des = {tau_des*1e6} μNm:")
    print(f"LP gives: τ_LP = {tau_lp*1e6} μNm (α = {alpha_lp:.4f})")
    
    # Direction-normalized QP: minimize ||τ - τ_des||² / α_LP²
    # Equivalent to: minimize ||τ - τ_des||² with target scaled to achievable
    
    u = cp.Variable(n)
    tau = A @ u
    
    # Use τ_LP as target instead of τ_des
    objective = cp.Minimize(cp.sum_squares(tau - tau_lp))
    constraints = [u >= lb, u <= ub]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    tau_result = A @ u.value
    print(f"Direction-normalized QP: τ = {tau_result*1e6} μNm")
    print(f"  (This should match LP since τ_LP is achievable)")


def derive_optimal_formulation():
    """Derive the mathematically optimal formulation."""
    print("\n" + "=" * 80)
    print("DERIVING THE OPTIMAL FORMULATION")
    print("=" * 80)
    
    print("""
GOAL: Find τ that:
    1. Is in the reachable set R
    2. Best matches τ_des in some meaningful sense
    3. Doesn't "ignore" any axis

WHAT WE WANT:
    - If τ_des is achievable: τ = τ_des (exact match)
    - If τ_des is not achievable: τ = best approximation
    
WHAT IS "BEST APPROXIMATION"?

Option A: Closest in Euclidean distance
    min ||τ - τ_des||²
    PROBLEM: Ignores weak axes
    
Option B: Closest in direction (LP)
    max α s.t. τ = α·τ_des
    GOOD: Perfect direction, fair scaling
    LIMITATION: Doesn't use extra capacity
    
Option C: LP + perpendicular utilization
    Stage 1: Find τ_∥ = α_LP · τ̂_des (LP)
    Stage 2: Add τ_⊥ to utilize remaining capacity
    
    This is LEXICOGRAPHIC but needs care about what τ_⊥ to add.

Option D: Pareto improvement over LP
    min ||τ - τ_des||² s.t. τᵢ ≥ τ_LP_i (sign-aware)
    
    This CAN improve on LP without making any axis worse.
    But it's biased toward improving z (easiest).


THE KEY INSIGHT:
================

LP is optimal for DIRECTION preservation.
QP is optimal for MAGNITUDE matching.

The question is: which matters more for control?

For ATTITUDE CONTROL:
    - Direction matters for stability (τ·ω term)
    - Magnitude matters for convergence speed
    
For RATE DAMPING (τ_des = -k·ω):
    - τ·ω < 0 is essential (don't accelerate!)
    - Larger |τ| means faster damping
    
So the priority should be:
    1. Ensure τ·ω ≤ τ_des·ω (energy constraint from QPC)
    2. Maximize useful torque ||τ|| subject to direction bound
    3. Minimize perpendicular if there's slack
""")
    
    return


def optimal_hybrid_formulation():
    """Implement the optimal hybrid formulation."""
    print("\n" + "=" * 80)
    print("OPTIMAL HYBRID FORMULATION")
    print("=" * 80)
    
    A, lb, ub, B = setup_system()
    n = len(lb)
    
    def lp_solve(tau_des):
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
            return u, A @ u, alpha
        return np.zeros(n), np.zeros(3), 0.0
    
    def optimal_hybrid(tau_des, omega=None, theta_max_deg=30):
        """
        Optimal hybrid formulation:
        1. Get LP baseline for direction reference
        2. Maximize projection along τ_des
        3. Subject to direction cone constraint
        4. (Optional) energy constraint if omega provided
        """
        t_mag = np.linalg.norm(tau_des)
        if t_mag < 1e-12:
            return np.zeros(n), np.zeros(3)
        
        tau_hat = tau_des / t_mag
        u_lp, tau_lp, alpha_lp = lp_solve(tau_des)
        
        u = cp.Variable(n)
        tau = A @ u
        
        # Objective: maximize projection (= useful torque)
        # BUT also penalize perpendicular slightly
        tau_proj = tau @ tau_hat
        tau_perp_sq = cp.sum_squares(tau - tau_proj * tau_hat)
        
        # Weighted objective: maximize projection, penalize perp
        # Higher lambda = stricter direction
        lambda_perp = 0.1
        objective = cp.Maximize(tau_proj - lambda_perp * tau_perp_sq)
        
        # Constraints
        cos_theta = np.cos(np.radians(theta_max_deg))
        constraints = [
            u >= lb, u <= ub,
            tau @ tau_des >= cos_theta * cp.norm(tau) * t_mag  # Direction cone
        ]
        
        # Energy constraint if omega provided
        if omega is not None:
            P_des = np.dot(tau_des, omega)
            if P_des < 0:  # Trying to damp
                constraints.append(tau @ omega <= 0)  # Don't accelerate
        
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        
        if u.value is not None:
            return u.value, A @ u.value
        return u_lp, tau_lp
    
    # Test cases
    print("\nTesting optimal hybrid formulation:")
    print("-" * 70)
    
    test_cases = [
        ("Balanced [10,10,10]", np.array([10e-6, 10e-6, 10e-6]), None),
        ("Heavy z [1,1,100]", np.array([1e-6, 1e-6, 100e-6]), None),
        ("With damping ω=[0.01,0.01,0.01]", np.array([10e-6, 10e-6, 10e-6]), np.array([0.01, 0.01, 0.01])),
    ]
    
    for name, tau_des, omega in test_cases:
        u_lp, tau_lp, _ = lp_solve(tau_des)
        u_hyb, tau_hyb = optimal_hybrid(tau_des, omega)
        
        tau_hat = tau_des / np.linalg.norm(tau_des)
        
        proj_lp = np.dot(tau_lp, tau_hat)
        proj_hyb = np.dot(tau_hyb, tau_hat)
        
        dir_lp = np.degrees(np.arccos(np.clip(proj_lp / (np.linalg.norm(tau_lp) + 1e-15), -1, 1)))
        dir_hyb = np.degrees(np.arccos(np.clip(proj_hyb / (np.linalg.norm(tau_hyb) + 1e-15), -1, 1)))
        
        print(f"\n{name}:")
        print(f"  LP:     τ = {tau_lp*1e6} μNm, proj = {proj_lp*1e6:.3f}, dir = {dir_lp:.1f}°")
        print(f"  Hybrid: τ = {tau_hyb*1e6} μNm, proj = {proj_hyb*1e6:.3f}, dir = {dir_hyb:.1f}°")
        
        if omega is not None:
            print(f"  Energy check: τ·ω = {np.dot(tau_hyb, omega)*1e9:.2f} nW (want ≤ 0)")


def final_recommendation():
    """Final recommendation based on analysis."""
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)
    
    print("""
CONCLUSION: LP IS THE RIGHT ANSWER FOR MOST CASES
=================================================

After all this analysis, the mathematically best allocation is:

    FOR GENERAL USE: LP
    
    max α  s.t.  τ = α · τ_des,  lb ≤ u ≤ ub

Why:
1. Guarantees DIRECTION preservation (τ ∝ τ_des)
2. FAIR: All axes get same fraction of what they asked for
3. STABLE: No risk of axis ignoring causing instability
4. SIMPLE: Linear program, fast and reliable

WHEN TO USE QP VARIANTS:
========================

1. QP + Pareto (τᵢ ≥ τ_LP_i):
   When you want to use EXTRA CAPACITY on easy axes.
   Example: z-axis can do more, use it to converge faster.
   Trade-off: Direction error, potential overshoot.

2. QP + Direction Cone:
   When you want MORE MAGNITUDE but bounded direction error.
   Use θ_max = 15-30° for reasonable trade-off.
   
3. QP + Energy Constraint (QPC):
   ALWAYS use during damping to prevent energy injection.
   This is orthogonal to the other constraints.


THE SIMPLE ANSWER:
==================

    IF achievable(τ_des):
        return τ_des
    ELSE:
        return LP(τ_des)  # Scaled proportionally

Anything fancier needs a clear justification for the trade-off.


FOR THE PAPER:
==============

LP provides:
- Optimal direction preservation
- Fair allocation across axes
- Mathematical simplicity

QP provides:
- Better magnitude when direction is not critical
- But requires constraints to prevent axis ignoring:
  * Energy constraint (always)
  * Direction cone or Pareto bounds

Recommendation: LP as default, QP+Energy+Pareto for aggressive control.
""")


if __name__ == "__main__":
    np.random.seed(42)
    
    analyze_previous_results()
    understand_maxmin_failure()
    understand_weighted_failure()
    correct_weighted_qp()
    derive_optimal_formulation()
    optimal_hybrid_formulation()
    final_recommendation()
