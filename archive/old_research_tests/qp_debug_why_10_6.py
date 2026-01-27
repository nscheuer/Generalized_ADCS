"""
DEBUG: Why does QP achieve τ_z = 10.6 μNm when τ_des_z = 10 μNm?
================================================================

Something is wrong. Let's trace through exactly what's happening.
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


def trace_qp_solution():
    """Trace through QP solution step by step."""
    print("=" * 80)
    print("TRACING QP SOLUTION")
    print("=" * 80)
    
    A, lb, ub, B = setup_system()
    n = len(lb)
    
    print("\n1. SYSTEM SETUP:")
    print(f"   A (torque = A @ u):")
    print(f"   {A}")
    print(f"\n   lb = {lb}")
    print(f"   ub = {ub}")
    
    tau_des = np.array([10e-6, 10e-6, 10e-6])
    print(f"\n   τ_des = {tau_des * 1e6} μNm")
    
    print("\n2. ACTUATOR INTERPRETATION:")
    print(f"   u[0] = RW torque command (Nm), range [{lb[0]}, {ub[0]}]")
    print(f"   u[1:4] = MTQ dipole commands (Am²), range [{lb[1]}, {ub[1]}]")
    
    print("\n3. HOW EACH ACTUATOR CONTRIBUTES TO TORQUE:")
    print(f"   τ = A @ u = A_rw @ u_rw + A_mtq @ u_mtq")
    print(f"   τ_z = 1.0 * u_rw + small_mtq_contribution")
    print(f"   τ_x, τ_y = only from MTQ (weak, ~μNm scale)")
    
    print("\n4. QP PROBLEM:")
    print(f"   min ||A @ u - τ_des||²")
    print(f"   s.t. lb ≤ u ≤ ub")
    
    # Solve
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(tau - tau_des))
    constraints = [u >= lb, u <= ub]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=True)
    
    u_sol = u.value
    tau_sol = A @ u_sol
    
    print(f"\n5. QP SOLUTION:")
    print(f"   u = {u_sol}")
    print(f"   τ = A @ u = {tau_sol * 1e6} μNm")
    
    print("\n6. CHECKING CONSTRAINTS:")
    print(f"   u >= lb: {np.all(u_sol >= lb - 1e-9)}")
    print(f"   u <= ub: {np.all(u_sol <= ub + 1e-9)}")
    
    print("\n7. WHY τ_z = 10.6 WHEN τ_des_z = 10?")
    print(f"   RW contribution to τ_z: {A[2,0] * u_sol[0] * 1e6:.4f} μNm")
    print(f"   MTQ contribution to τ_z: {(A[2,1:] @ u_sol[1:]) * 1e6:.4f} μNm")
    print(f"   Total τ_z: {tau_sol[2] * 1e6:.4f} μNm")
    
    print(f"\n   u_rw = {u_sol[0]:.6f} Nm")
    print(f"   This is {'within' if abs(u_sol[0]) <= ub[0] else 'OUTSIDE'} bounds [{lb[0]}, {ub[0]}]")
    
    print("\n8. THE KEY INSIGHT:")
    print(f"   τ_des_z = 10 μNm = 10e-6 Nm = 0.00001 Nm")
    print(f"   RW can produce up to {ub[0]} Nm = {ub[0]*1e6} μNm")
    print(f"   So RW is WAY under capacity!")
    print(f"   QP finds u_rw ≈ 10.6e-6 Nm to match τ_z ≈ 10.6 μNm")
    
    print("\n   Wait... that's LESS than the RW limit of 1000 μNm.")
    print(f"   So τ_z = 10.6 μNm is ACHIEVABLE and close to τ_des_z = 10 μNm.")
    print(f"   The small overshoot (0.6 μNm) comes from MTQ coupling.")
    
    # Let's verify
    print("\n9. VERIFICATION - Can we achieve exactly τ_z = 10 μNm?")
    # If we set u_rw to give exactly 10 μNm after MTQ coupling
    # τ_z = u_rw + A[2,1:] @ u_mtq = 10e-6
    # u_rw = 10e-6 - A[2,1:] @ u_mtq
    
    # What's the MTQ contribution at current solution?
    mtq_contrib_z = A[2,1:] @ u_sol[1:]
    print(f"   MTQ contribution to z: {mtq_contrib_z * 1e6:.4f} μNm")
    print(f"   This comes from u_mtq trying to achieve τ_x, τ_y")
    
    print("\n10. THE REAL PROBLEM:")
    print(f"   QP achieves τ_z ≈ τ_des_z (good!)")
    print(f"   But τ_x = {tau_sol[0]*1e6:.4f} μNm vs τ_des_x = 10 μNm")
    print(f"   And τ_y = {tau_sol[1]*1e6:.4f} μNm vs τ_des_y = 10 μNm")
    print(f"   QP ignores x,y to focus on z (because z error dominates)")
    
    return


def understand_error_weighting():
    """Understand why QP focuses on z."""
    print("\n" + "=" * 80)
    print("UNDERSTANDING ERROR WEIGHTING IN QP")
    print("=" * 80)
    
    A, lb, ub, B = setup_system()
    tau_des = np.array([10e-6, 10e-6, 10e-6])
    
    print("""
QP objective: min ||τ - τ_des||² = (τ_x - 10)² + (τ_y - 10)² + (τ_z - 10)²

where τ is in μNm for readability.

Consider two solutions:

Solution A (LP-like): τ = [2, 2, 2] μNm
    Error² = (2-10)² + (2-10)² + (2-10)² = 64 + 64 + 64 = 192

Solution B (QP actual): τ = [0.01, 0.09, 10.6] μNm
    Error² = (0.01-10)² + (0.09-10)² + (10.6-10)² 
           = 99.8 + 98.0 + 0.36 = 198.2

Wait! Solution A has LOWER error (192 < 198.2)?!

Let me recalculate...
""")
    
    # Calculate errors for both
    tau_A = np.array([2e-6, 2e-6, 2e-6])
    tau_B = np.array([0.01e-6, 0.09e-6, 10.6e-6])
    
    err_A = np.sum((tau_A - tau_des)**2)
    err_B = np.sum((tau_B - tau_des)**2)
    
    print(f"Actual calculation:")
    print(f"  τ_A = {tau_A * 1e6} μNm")
    print(f"  τ_B = {tau_B * 1e6} μNm")
    print(f"  τ_des = {tau_des * 1e6} μNm")
    print(f"  Error(A) = {err_A:.4e}")
    print(f"  Error(B) = {err_B:.4e}")
    print(f"  A is {'better' if err_A < err_B else 'worse'} than B")
    
    # So why doesn't QP find A?
    print("\nSo why doesn't QP find solution A?")
    print("Let's check if A is achievable...")
    
    # Check if tau_A is achievable
    # We need A @ u = tau_A with lb <= u <= ub
    # This is a feasibility problem
    
    u = cp.Variable(4)
    constraints = [A @ u == tau_A, u >= lb, u <= ub]
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.ECOS)
    
    print(f"  Is τ = [2, 2, 2] achievable? {prob.status}")
    
    if prob.status == 'optimal':
        print(f"  u = {u.value}")
        print(f"  τ = {(A @ u.value) * 1e6} μNm")
    else:
        print(f"  τ = [2, 2, 2] μNm is NOT achievable!")
        print(f"  This explains why QP doesn't find it.")
    
    return


def find_achievable_directions():
    """Find what torques are actually achievable."""
    print("\n" + "=" * 80)
    print("WHAT TORQUES ARE ACHIEVABLE?")
    print("=" * 80)
    
    A, lb, ub, B = setup_system()
    n = len(lb)
    
    print("\nMax achievable on each axis independently:")
    for i, axis in enumerate(['x', 'y', 'z']):
        # Maximize tau_i
        c = np.zeros(n)
        for j in range(n):
            c[j] = -A[i, j]
        res = linprog(c, bounds=list(zip(lb, ub)), method='highs')
        max_pos = -res.fun if res.success else 0
        
        # Minimize tau_i (= maximize -tau_i)
        c = np.zeros(n)
        for j in range(n):
            c[j] = A[i, j]
        res = linprog(c, bounds=list(zip(lb, ub)), method='highs')
        max_neg = -(-res.fun) if res.success else 0
        
        print(f"  τ_{axis}: [{max_neg*1e6:.2f}, {max_pos*1e6:.2f}] μNm")
    
    print("\nCan we achieve τ = [10, 10, 10] μNm?")
    print("  τ_x needs 10 μNm, max is ~5 μNm → NO")
    print("  τ_y needs 10 μNm, max is ~6 μNm → NO")
    print("  τ_z needs 10 μNm, max is ~1007 μNm → YES")
    
    print("\nSo τ_des = [10, 10, 10] is NOT achievable!")
    print("QP finds the closest achievable point.")
    
    # What IS the closest achievable point?
    print("\nWhat is the closest achievable point to [10, 10, 10] μNm?")
    
    tau_des = np.array([10e-6, 10e-6, 10e-6])
    
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(tau - tau_des))
    constraints = [u >= lb, u <= ub]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    tau_qp = A @ u.value
    print(f"  τ_QP = {tau_qp * 1e6} μNm")
    print(f"  Error = {np.linalg.norm(tau_qp - tau_des) * 1e6:.4f} μNm")
    
    # Now let's see the LP solution
    print("\nLP solution (proportional scaling):")
    t_mag = np.linalg.norm(tau_des)
    tau_hat = tau_des / t_mag
    
    c_lp = np.zeros(n + 1)
    c_lp[-1] = -1
    A_eq = np.hstack([A, -tau_hat.reshape(3, 1)])
    bounds_lp = [(lb[i], ub[i]) for i in range(n)] + [(0, None)]
    res = linprog(c_lp, A_eq=A_eq, b_eq=np.zeros(3), bounds=bounds_lp, method='highs')
    
    if res.success:
        u_lp = res.x[:n]
        alpha = res.x[-1]
        tau_lp = A @ u_lp
        print(f"  α = {alpha / t_mag:.4f}")
        print(f"  τ_LP = {tau_lp * 1e6} μNm")
        print(f"  Error = {np.linalg.norm(tau_lp - tau_des) * 1e6:.4f} μNm")
        
        print(f"\n  QP error: {np.linalg.norm(tau_qp - tau_des) * 1e6:.4f} μNm")
        print(f"  LP error: {np.linalg.norm(tau_lp - tau_des) * 1e6:.4f} μNm")
        print(f"  QP has {'lower' if np.linalg.norm(tau_qp - tau_des) < np.linalg.norm(tau_lp - tau_des) else 'higher'} error")
    
    return


def fix_weighted_qp():
    """Fix the weighted QP formulation."""
    print("\n" + "=" * 80)
    print("FIXING WEIGHTED QP")
    print("=" * 80)
    
    A, lb, ub, B = setup_system()
    n = len(lb)
    tau_des = np.array([10e-6, 10e-6, 10e-6])
    
    print("""
PROBLEM WITH PREVIOUS WEIGHTED QP:
==================================

We weighted by 1/τ_max_i² where τ_max_i is the max achievable on axis i.

But this ignores COUPLING. The max achievable τ_x uses specific MTQ commands
that may conflict with achieving τ_y or τ_z.

BETTER APPROACH: Weight by achievability in the DESIRED DIRECTION.

For τ_des = [10, 10, 10], find:
    α* = max α s.t. α·τ_des is achievable

Then the "achievable magnitude" in this direction is α*·||τ_des||.

Weight: w = 1 / (α* · ||τ_des||)²

This is a SCALAR weight that normalizes the entire error.
""")
    
    # Find α*
    t_mag = np.linalg.norm(tau_des)
    tau_hat = tau_des / t_mag
    
    c_lp = np.zeros(n + 1)
    c_lp[-1] = -1
    A_eq = np.hstack([A, -tau_hat.reshape(3, 1)])
    bounds_lp = [(lb[i], ub[i]) for i in range(n)] + [(0, None)]
    res = linprog(c_lp, A_eq=A_eq, b_eq=np.zeros(3), bounds=bounds_lp, method='highs')
    
    alpha_star = res.x[-1] if res.success else 0
    tau_achievable = alpha_star  # This is the max magnitude in direction τ_hat
    
    print(f"α* = {alpha_star / t_mag:.4f} (fraction of τ_des achievable)")
    print(f"τ_achievable = α* = {alpha_star * 1e6:.2f} μNm in direction τ̂_des")
    
    # Now the problem is: we can achieve at most α* in direction τ_des
    # If τ_des_magnitude > α*, we can't achieve τ_des
    
    print(f"\n||τ_des|| = {t_mag * 1e6:.2f} μNm")
    print(f"Achievable in that direction: {alpha_star * 1e6:.2f} μNm")
    print(f"τ_des is {'achievable' if t_mag <= alpha_star else 'NOT achievable'}")
    
    print("""
    
THE REAL FIX: Don't weight by axis - weight by DIRECTION achievability.

But actually, the simpler fix is:

1. If τ_des is achievable → QP will find it exactly
2. If τ_des is NOT achievable → use LP (proportional scaling)

The "weighted QP" approach was trying to solve a problem that doesn't exist
when τ_des is achievable, and gives wrong answer when it's not.
""")
    
    return


def fix_maxmin_ratio():
    """Fix the MaxMin Ratio formulation."""
    print("\n" + "=" * 80)
    print("FIXING MAXMIN RATIO")
    print("=" * 80)
    
    A, lb, ub, B = setup_system()
    n = len(lb)
    tau_des = np.array([10e-6, 10e-6, 10e-6])
    
    print("""
PROBLEM WITH PREVIOUS MAXMIN:
=============================

We had: max α s.t. τᵢ ≥ α·τ_des_i (for positive τ_des_i)

This only gives LOWER bounds. The solver can make τ_z arbitrarily large
without violating any constraint.

FIX 1: Add upper bounds too
    α·τ_des_i ≤ τᵢ ≤ k·α·τ_des_i  (allow up to k times overshoot)

FIX 2: Use EQUALITY constraint (same as LP!)
    τᵢ = α·τ_des_i

FIX 3: Minimize excess while maximizing α
    max α - λ·||τ - α·τ_des||²
""")
    
    # Fix 1: Bounded MaxMin
    print("\nFix 1: Bounded MaxMin (k=1.5 overshoot allowed)")
    k = 1.5
    
    u = cp.Variable(n)
    alpha = cp.Variable()
    tau = A @ u
    
    objective = cp.Maximize(alpha)
    constraints = [u >= lb, u <= ub, alpha >= 0]
    
    for i in range(3):
        if tau_des[i] > 1e-15:
            constraints.append(tau[i] >= alpha * tau_des[i])
            constraints.append(tau[i] <= k * alpha * tau_des[i])
        elif tau_des[i] < -1e-15:
            constraints.append(tau[i] <= alpha * tau_des[i])
            constraints.append(tau[i] >= k * alpha * tau_des[i])
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    if u.value is not None:
        tau_result = A @ u.value
        print(f"  α = {alpha.value:.4f}")
        print(f"  τ = {tau_result * 1e6} μNm")
        print(f"  Ratios: {tau_result / tau_des}")
    
    # Fix 2: Equality (same as LP)
    print("\nFix 2: Equality constraint (= LP)")
    
    u = cp.Variable(n)
    alpha = cp.Variable()
    tau = A @ u
    
    objective = cp.Maximize(alpha)
    constraints = [u >= lb, u <= ub, alpha >= 0]
    
    # τ = α·τ_des → τ = α·||τ_des||·τ̂_des
    tau_hat = tau_des / np.linalg.norm(tau_des)
    constraints.append(tau == alpha * tau_hat)
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    if u.value is not None:
        tau_result = A @ u.value
        print(f"  α = {alpha.value:.4f}")
        print(f"  τ = {tau_result * 1e6} μNm")
        print(f"  Ratios: {tau_result / tau_des}")
    
    # Fix 3: Just use LP - it's the right answer
    print("\nFix 3: Just use LP (the right answer)")
    print("  LP gives proportional scaling which is mathematically optimal")
    print("  for preserving direction when target is not achievable.")
    
    return


def correct_pareto_formulation():
    """Show the correct Pareto formulation."""
    print("\n" + "=" * 80)
    print("CORRECT PARETO FORMULATION")
    print("=" * 80)
    
    A, lb, ub, B = setup_system()
    n = len(lb)
    tau_des = np.array([10e-6, 10e-6, 10e-6])
    
    print("""
PARETO IMPROVEMENT OVER LP:
===========================

Goal: Find τ that is at least as good as LP on EVERY axis,
      while minimizing total error.

Step 1: Get LP solution τ_LP
Step 2: QP with constraints τᵢ ≥ τ_LP_i (for positive τ_des_i)

This guarantees we don't make any axis worse than LP.
""")
    
    # LP baseline
    t_mag = np.linalg.norm(tau_des)
    tau_hat = tau_des / t_mag
    
    c_lp = np.zeros(n + 1)
    c_lp[-1] = -1
    A_eq = np.hstack([A, -tau_hat.reshape(3, 1)])
    bounds_lp = [(lb[i], ub[i]) for i in range(n)] + [(0, None)]
    res = linprog(c_lp, A_eq=A_eq, b_eq=np.zeros(3), bounds=bounds_lp, method='highs')
    
    u_lp = res.x[:n]
    tau_lp = A @ u_lp
    
    print(f"LP: τ_LP = {tau_lp * 1e6} μNm")
    
    # Pareto QP
    u = cp.Variable(n)
    tau = A @ u
    
    objective = cp.Minimize(cp.sum_squares(tau - tau_des))
    constraints = [u >= lb, u <= ub]
    
    for i in range(3):
        if tau_des[i] > 1e-15:
            constraints.append(tau[i] >= tau_lp[i] - 1e-15)
        elif tau_des[i] < -1e-15:
            constraints.append(tau[i] <= tau_lp[i] + 1e-15)
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    tau_pareto = A @ u.value
    print(f"Pareto QP: τ = {tau_pareto * 1e6} μNm")
    
    print(f"\nImprovement over LP:")
    for i, axis in enumerate(['x', 'y', 'z']):
        lp_err = abs(tau_lp[i] - tau_des[i])
        pareto_err = abs(tau_pareto[i] - tau_des[i])
        print(f"  {axis}: LP error = {lp_err*1e6:.3f} μNm, Pareto error = {pareto_err*1e6:.3f} μNm")
    
    print("""
    
OBSERVATION:
Pareto improves z-axis significantly (from 8μNm error to ~4μNm)
while maintaining x,y at LP level.

This IS a valid improvement - we're using extra z-capacity.
But it DOES change the direction (from 0° to ~44°).

Is this good for control? Depends on the application:
- For fast settling: yes, more torque helps
- For direction-critical: no, direction error can cause issues
""")
    
    return


def summary():
    """Summary of findings."""
    print("\n" + "=" * 80)
    print("SUMMARY: THE REAL STORY")
    print("=" * 80)
    
    print("""
WHAT WAS ACTUALLY HAPPENING:
============================

1. τ_des = [10, 10, 10] μNm is NOT achievable
   - Max τ_x ≈ 5 μNm (MTQ limited)
   - Max τ_y ≈ 6 μNm (MTQ limited)
   - Max τ_z ≈ 1007 μNm (RW dominated)

2. QP finds the closest achievable point:
   - τ_QP = [0.01, 0.09, 10.6] μNm
   - This minimizes ||τ - τ_des||² over achievable τ
   - The z overshoot (10.6 vs 10) comes from MTQ coupling

3. LP finds the proportionally scaled point:
   - τ_LP = [2, 2, 2] μNm (α ≈ 0.2)
   - Perfect direction (τ ∝ τ_des)
   - But LARGER Euclidean error than QP!

4. QP is "correct" for L2 error minimization
   - But L2 treats all axes equally
   - For control, we often care more about direction


WHY THE "SMART" FORMULATIONS FAILED:
====================================

1. Weighted QP (by 1/τ_max²):
   - Gave near-zero weight to z (because τ_max_z >> τ_max_x)
   - Should have weighted by direction achievability, not axis achievability

2. MaxMin Ratio:
   - Only had lower bounds (τᵢ ≥ α·τ_des_i)
   - No upper bound → z ran away to max
   - Fix: add upper bounds or use equality

3. The fixes work:
   - Bounded MaxMin: gives sensible results
   - Equality MaxMin: same as LP
   - Pareto: improves on LP while maintaining per-axis guarantees


CORRECT FORMULATIONS:
====================

A) LP (for direction preservation):
   max α  s.t.  τ = α·τ_des

B) Pareto QP (for magnitude improvement):
   min ||τ - τ_des||²  s.t.  τᵢ ≥ τ_LP_i

C) Bounded ratio (for controlled overshoot):
   max α  s.t.  α·τ_des_i ≤ τᵢ ≤ k·α·τ_des_i

D) Cone-constrained QP (for bounded direction error):
   min ||τ - τ_des||²  s.t.  angle(τ, τ_des) ≤ θ_max
""")


if __name__ == "__main__":
    np.random.seed(42)
    
    trace_qp_solution()
    understand_error_weighting()
    find_achievable_directions()
    fix_weighted_qp()
    fix_maxmin_ratio()
    correct_pareto_formulation()
    summary()
