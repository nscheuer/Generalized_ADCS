"""
QP Solver Bug Investigation
===========================

QP is returning a suboptimal solution. [2,2,2] is achievable and has lower
error than what QP finds. Let's understand why.
"""

import numpy as np
import cvxpy as cp
from scipy.optimize import minimize, linprog, least_squares
import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from ADCS.helpers.math_helpers import skewsym


def setup_system():
    B = np.array([20e-6, 15e-6, 10e-6])
    A_rw = np.array([[0], [0], [1.0]])
    A_mtq = -skewsym(B) @ np.eye(3)
    A = np.hstack([A_rw, A_mtq])
    lb = np.array([-0.001, -0.2, -0.2, -0.2])
    ub = np.array([0.001, 0.2, 0.2, 0.2])
    return A, lb, ub


def investigate_qp_bug():
    """Investigate why QP doesn't find optimal."""
    print("=" * 80)
    print("INVESTIGATING QP SOLVER BUG")
    print("=" * 80)
    
    A, lb, ub = setup_system()
    n = len(lb)
    tau_des = np.array([10e-6, 10e-6, 10e-6])
    
    print(f"\nτ_des = {tau_des * 1e6} μNm")
    
    # The achievable point with lower error
    tau_222 = np.array([2e-6, 2e-6, 2e-6])
    err_222 = np.linalg.norm(tau_222 - tau_des)
    print(f"\nKnown achievable: τ = [2, 2, 2] μNm, error = {err_222 * 1e6:.4f} μNm")
    
    # Find u that achieves [2,2,2]
    u_222 = cp.Variable(n)
    constraints = [A @ u_222 == tau_222, u_222 >= lb, u_222 <= ub]
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.ECOS)
    u_at_222 = u_222.value
    print(f"u for [2,2,2]: {u_at_222}")
    print(f"Verification: A @ u = {(A @ u_at_222) * 1e6} μNm")
    
    # Now solve QP with CVXPY
    print("\n" + "-" * 50)
    print("CVXPY QP Solution:")
    
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(tau - tau_des))
    constraints = [u >= lb, u <= ub]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=True)
    
    u_cvxpy = u.value
    tau_cvxpy = A @ u_cvxpy
    err_cvxpy = np.linalg.norm(tau_cvxpy - tau_des)
    
    print(f"\nCVXPY result:")
    print(f"  u = {u_cvxpy}")
    print(f"  τ = {tau_cvxpy * 1e6} μNm")
    print(f"  error = {err_cvxpy * 1e6:.4f} μNm")
    
    # Check objective value at both points
    obj_222 = 0.5 * np.sum((tau_222 - tau_des)**2)
    obj_cvxpy = 0.5 * np.sum((tau_cvxpy - tau_des)**2)
    
    print(f"\nObjective comparison:")
    print(f"  At [2,2,2]: {obj_222:.4e}")
    print(f"  At CVXPY:   {obj_cvxpy:.4e}")
    print(f"  [2,2,2] is {'better' if obj_222 < obj_cvxpy else 'worse'}")
    
    # Try different solvers
    print("\n" + "-" * 50)
    print("TRYING DIFFERENT SOLVERS:")
    
    solvers = [cp.ECOS, cp.SCS, cp.OSQP]
    for solver in solvers:
        try:
            u = cp.Variable(n)
            tau = A @ u
            objective = cp.Minimize(cp.sum_squares(tau - tau_des))
            constraints = [u >= lb, u <= ub]
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=solver)
            
            if u.value is not None:
                tau_sol = A @ u.value
                err = np.linalg.norm(tau_sol - tau_des)
                print(f"\n{solver}: τ = {tau_sol * 1e6}, error = {err * 1e6:.4f} μNm")
            else:
                print(f"\n{solver}: FAILED")
        except Exception as e:
            print(f"\n{solver}: ERROR - {e}")
    
    # Try scipy.optimize
    print("\n" + "-" * 50)
    print("SCIPY.OPTIMIZE:")
    
    def objective(u):
        tau = A @ u
        return 0.5 * np.sum((tau - tau_des)**2)
    
    def gradient(u):
        tau = A @ u
        return A.T @ (tau - tau_des)
    
    # Try from different starting points
    starts = [
        np.zeros(n),
        u_at_222,  # Start at known good point
        0.5 * (lb + ub),  # Center of box
    ]
    
    for i, x0 in enumerate(starts):
        res = minimize(objective, x0, method='L-BFGS-B', jac=gradient,
                      bounds=list(zip(lb, ub)))
        tau_sol = A @ res.x
        err = np.linalg.norm(tau_sol - tau_des)
        print(f"\nStart {i}: τ = {tau_sol * 1e6}, error = {err * 1e6:.4f} μNm")
        print(f"  x = {res.x}")
    
    return


def understand_the_geometry():
    """Understand why solvers find different solutions."""
    print("\n" + "=" * 80)
    print("UNDERSTANDING THE GEOMETRY")
    print("=" * 80)
    
    A, lb, ub = setup_system()
    n = len(lb)
    tau_des = np.array([10e-6, 10e-6, 10e-6])
    
    print("""
The problem: min ||A·u - τ_des||² s.t. lb ≤ u ≤ ub

This is a convex QP. There should be a UNIQUE global minimum.
But we're seeing different solutions from different solvers/starts.

Possible explanations:
1. Numerical precision issues (A has very different scales)
2. Multiple local minima (impossible for convex QP)
3. The reachable set R has a "flat" region where many u give similar τ

Let me check the condition number of AᵀA...
""")
    
    ATA = A.T @ A
    print(f"AᵀA =\n{ATA}")
    
    eigvals = np.linalg.eigvalsh(ATA)
    print(f"\nEigenvalues of AᵀA: {eigvals}")
    print(f"Condition number: {max(eigvals) / min(eigvals):.2e}")
    
    print("""
    
The condition number is HUGE because:
- RW column [0,0,1] has magnitude 1
- MTQ columns have magnitude ~10⁻⁵

This means the problem is ILL-CONDITIONED in actuator space.

But wait - the QP is in TORQUE space, not actuator space.
Let me check AAᵀ instead...
""")
    
    AAT = A @ A.T
    print(f"AAᵀ =\n{AAT}")
    
    eigvals_aat = np.linalg.eigvalsh(AAT)
    print(f"\nEigenvalues of AAᵀ: {eigvals_aat}")
    print(f"Condition number: {max(eigvals_aat) / min(eigvals_aat):.2e}")
    
    print("""
    
AAᵀ is also extremely ill-conditioned!
- Large eigenvalue ~1 (from RW z-axis)
- Tiny eigenvalues ~10⁻¹⁰ (from MTQ x,y)

This causes numerical issues when solvers try to minimize in torque space.
""")
    
    return


def fix_with_scaling():
    """Fix by proper scaling."""
    print("\n" + "=" * 80)
    print("FIX: PROPER SCALING")
    print("=" * 80)
    
    A, lb, ub = setup_system()
    n = len(lb)
    tau_des = np.array([10e-6, 10e-6, 10e-6])
    
    print("""
The fix: Scale the problem so all quantities are O(1).

Option 1: Scale τ by 1e6 (work in μNm)
Option 2: Scale each axis of τ separately
Option 3: Scale actuator commands to [−1, 1]
""")
    
    # Option 1: Scale τ
    print("\nOption 1: Scale τ by 1e6")
    SCALE = 1e6
    
    u = cp.Variable(n)
    tau_scaled = SCALE * A @ u
    tau_des_scaled = SCALE * tau_des
    objective = cp.Minimize(cp.sum_squares(tau_scaled - tau_des_scaled))
    constraints = [u >= lb, u <= ub]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    tau_sol = A @ u.value
    err = np.linalg.norm(tau_sol - tau_des)
    print(f"  τ = {tau_sol * 1e6} μNm, error = {err * 1e6:.4f} μNm")
    
    # Option 3: Scale actuators
    print("\nOption 3: Scale actuators to [-1, 1]")
    
    # u_scaled = u / u_max, so u = u_scaled * u_max
    u_max = np.array([0.001, 0.2, 0.2, 0.2])
    A_scaled = A * u_max  # A_scaled @ u_scaled = A @ u
    
    u_scaled = cp.Variable(n)
    tau = A_scaled @ u_scaled
    objective = cp.Minimize(cp.sum_squares(tau - tau_des))
    constraints = [u_scaled >= -1, u_scaled <= 1]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    u_sol = u_scaled.value * u_max
    tau_sol = A @ u_sol
    err = np.linalg.norm(tau_sol - tau_des)
    print(f"  τ = {tau_sol * 1e6} μNm, error = {err * 1e6:.4f} μNm")
    
    # Combined scaling
    print("\nCombined scaling (actuators AND torque):")
    
    SCALE_TAU = 1e6
    A_scaled = SCALE_TAU * A * u_max
    tau_des_scaled = SCALE_TAU * tau_des
    
    u_scaled = cp.Variable(n)
    tau_scaled = A_scaled @ u_scaled
    objective = cp.Minimize(cp.sum_squares(tau_scaled - tau_des_scaled))
    constraints = [u_scaled >= -1, u_scaled <= 1]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    u_sol = u_scaled.value * u_max
    tau_sol = A @ u_sol
    err = np.linalg.norm(tau_sol - tau_des)
    print(f"  τ = {tau_sol * 1e6} μNm, error = {err * 1e6:.4f} μNm")
    print(f"  u = {u_sol}")
    
    # Verify this is the same as [2,2,2]
    print(f"\n  Compare to [2,2,2]: error would be {np.linalg.norm(np.array([2e-6,2e-6,2e-6]) - tau_des) * 1e6:.4f} μNm")
    
    return


def correct_qp_implementation():
    """Correct QP implementation with scaling."""
    print("\n" + "=" * 80)
    print("CORRECT QP IMPLEMENTATION")
    print("=" * 80)
    
    A, lb, ub = setup_system()
    n = len(lb)
    
    def solve_qp_scaled(tau_des):
        """QP with proper scaling."""
        SCALE = 1e6
        
        u = cp.Variable(n)
        tau_scaled = SCALE * A @ u
        tau_des_scaled = SCALE * tau_des
        objective = cp.Minimize(cp.sum_squares(tau_scaled - tau_des_scaled))
        constraints = [u >= lb, u <= ub]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        
        return u.value, A @ u.value
    
    def solve_lp(tau_des):
        """LP for comparison."""
        t_mag = np.linalg.norm(tau_des)
        if t_mag < 1e-15:
            return np.zeros(n), np.zeros(3)
        tau_hat = tau_des / t_mag
        
        c = np.zeros(n + 1)
        c[-1] = -1
        A_eq = np.hstack([A, -tau_hat.reshape(3, 1)])
        bounds = [(lb[i], ub[i]) for i in range(n)] + [(0, None)]
        
        res = linprog(c, A_eq=A_eq, b_eq=np.zeros(3), bounds=bounds, method='highs')
        
        if res.success:
            u = res.x[:n]
            return u, A @ u
        return np.zeros(n), np.zeros(3)
    
    # Test cases
    test_cases = [
        ("Balanced [10,10,10]", np.array([10e-6, 10e-6, 10e-6])),
        ("Heavy z [1,1,100]", np.array([1e-6, 1e-6, 100e-6])),
        ("Small [1,1,1]", np.array([1e-6, 1e-6, 1e-6])),
        ("Achievable [2,2,2]", np.array([2e-6, 2e-6, 2e-6])),
    ]
    
    print(f"\n{'Test':<25} {'LP τ (μNm)':<25} {'QP τ (μNm)':<25} {'LP err':>10} {'QP err':>10}")
    print("-" * 100)
    
    for name, tau_des in test_cases:
        _, tau_lp = solve_lp(tau_des)
        _, tau_qp = solve_qp_scaled(tau_des)
        
        err_lp = np.linalg.norm(tau_lp - tau_des) * 1e6
        err_qp = np.linalg.norm(tau_qp - tau_des) * 1e6
        
        tau_lp_str = f"[{tau_lp[0]*1e6:.2f},{tau_lp[1]*1e6:.2f},{tau_lp[2]*1e6:.2f}]"
        tau_qp_str = f"[{tau_qp[0]*1e6:.2f},{tau_qp[1]*1e6:.2f},{tau_qp[2]*1e6:.2f}]"
        
        better = "QP" if err_qp < err_lp else "LP" if err_lp < err_qp else "TIE"
        
        print(f"{name:<25} {tau_lp_str:<25} {tau_qp_str:<25} {err_lp:>10.4f} {err_qp:>10.4f} {better}")
    
    return solve_qp_scaled


if __name__ == "__main__":
    np.random.seed(42)
    
    investigate_qp_bug()
    understand_the_geometry()
    fix_with_scaling()
    solve_qp_scaled = correct_qp_implementation()
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
THE PROBLEM: Numerical ill-conditioning
    - A has entries spanning 5 orders of magnitude (1 vs 10⁻⁵)
    - AAᵀ condition number ~ 10⁹
    - Solvers struggle to find the true minimum

THE FIX: Scale the problem
    - Scale τ by 1e6 (work in μNm units)
    - Or scale actuator commands to [-1, 1]
    - Both improve conditioning dramatically

RESULT: With proper scaling, QP correctly finds the minimum error solution
""")
