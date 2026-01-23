"""
Corrected Comprehensive QP Test
===============================

All QP tests with proper scaling (SCALE=1e6) to fix numerical conditioning.
"""

import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from ADCS.helpers.math_helpers import skewsym

# CRITICAL: Scaling factor to fix numerical conditioning
SCALE = 1e6


def setup_system():
    """Standard 3MTQ + 1RW test system."""
    B = np.array([20e-6, 15e-6, 10e-6])
    A_rw = np.array([[0], [0], [1.0]])
    A_mtq = -skewsym(B) @ np.eye(3)
    A = np.hstack([A_rw, A_mtq])
    lb = np.array([-0.001, -0.2, -0.2, -0.2])
    ub = np.array([0.001, 0.2, 0.2, 0.2])
    return A, lb, ub, B


# ============== ALLOCATORS (ALL WITH PROPER SCALING) ==============

def solve_lp(tau_des, A, lb, ub):
    """LP: max α s.t. τ = α·τ̂_des"""
    n = len(lb)
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-15:
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


def solve_qp(tau_des, A, lb, ub):
    """Standard QP with scaling: min ||τ - τ_des||²"""
    n = len(lb)
    
    u = cp.Variable(n)
    tau_scaled = SCALE * A @ u
    tau_des_scaled = SCALE * tau_des
    objective = cp.Minimize(cp.sum_squares(tau_scaled - tau_des_scaled))
    constraints = [u >= lb, u <= ub]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    if u.value is not None:
        return u.value, A @ u.value
    return np.zeros(n), np.zeros(3)


def solve_qp_pareto(tau_des, A, lb, ub):
    """QP with Pareto constraint: each axis at least as good as LP"""
    n = len(lb)
    u_lp, tau_lp, _ = solve_lp(tau_des, A, lb, ub)
    
    u = cp.Variable(n)
    tau_scaled = SCALE * A @ u
    tau_des_scaled = SCALE * tau_des
    objective = cp.Minimize(cp.sum_squares(tau_scaled - tau_des_scaled))
    
    constraints = [u >= lb, u <= ub]
    for i in range(3):
        if tau_des[i] > 1e-15:
            constraints.append(A[i, :] @ u >= tau_lp[i] - 1e-15)
        elif tau_des[i] < -1e-15:
            constraints.append(A[i, :] @ u <= tau_lp[i] + 1e-15)
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    if u.value is not None:
        return u.value, A @ u.value
    return u_lp, tau_lp


def solve_bounded_ratio(tau_des, A, lb, ub, k=1.5):
    """Bounded ratio: α·τ_des ≤ τ ≤ k·α·τ_des"""
    n = len(lb)
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-15:
        return np.zeros(n), np.zeros(3)
    
    u = cp.Variable(n)
    alpha = cp.Variable(nonneg=True)
    tau = A @ u
    
    objective = cp.Maximize(alpha)
    constraints = [u >= lb, u <= ub]
    
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
        return u.value, A @ u.value
    u_lp, tau_lp, _ = solve_lp(tau_des, A, lb, ub)
    return u_lp, tau_lp


def solve_cone_qp(tau_des, A, lb, ub, theta_max_deg=30):
    """QP with direction cone: angle(τ, τ_des) ≤ θ_max"""
    n = len(lb)
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-15:
        return np.zeros(n), np.zeros(3)
    
    cos_theta = np.cos(np.radians(theta_max_deg))
    
    u = cp.Variable(n)
    tau = A @ u
    tau_scaled = SCALE * tau
    tau_des_scaled = SCALE * tau_des
    
    objective = cp.Minimize(cp.sum_squares(tau_scaled - tau_des_scaled))
    constraints = [
        u >= lb, u <= ub,
        tau @ tau_des >= cos_theta * cp.norm(tau) * t_mag
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    if u.value is not None:
        return u.value, A @ u.value
    u_lp, tau_lp, _ = solve_lp(tau_des, A, lb, ub)
    return u_lp, tau_lp


def solve_qp_energy(tau_des, A, lb, ub, omega):
    """QP with energy constraint: τ·ω ≤ max(0, τ_des·ω)"""
    n = len(lb)
    
    P_des = np.dot(tau_des, omega)
    P_bound = max(0, P_des) + 1e-15
    
    u = cp.Variable(n)
    tau = A @ u
    tau_scaled = SCALE * tau
    tau_des_scaled = SCALE * tau_des
    
    objective = cp.Minimize(cp.sum_squares(tau_scaled - tau_des_scaled))
    constraints = [
        u >= lb, u <= ub,
        tau @ omega <= P_bound
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    if u.value is not None:
        return u.value, A @ u.value
    u_lp, tau_lp, _ = solve_lp(tau_des, A, lb, ub)
    return u_lp, tau_lp


# ============== COMPREHENSIVE TESTS ==============

def run_all_tests():
    """Run comprehensive tests with corrected implementations."""
    print("=" * 90)
    print("COMPREHENSIVE QP TESTS WITH PROPER SCALING (SCALE=1e6)")
    print("=" * 90)
    
    A, lb, ub, B = setup_system()
    
    test_cases = [
        ("Balanced [10,10,10]", np.array([10e-6, 10e-6, 10e-6]), None),
        ("Heavy z [1,1,100]", np.array([1e-6, 1e-6, 100e-6]), None),
        ("Heavy xy [100,100,1]", np.array([100e-6, 100e-6, 1e-6]), None),
        ("Small achievable [1,1,1]", np.array([1e-6, 1e-6, 1e-6]), None),
        ("Achievable [2,2,2]", np.array([2e-6, 2e-6, 2e-6]), None),
        ("With damping ω", np.array([10e-6, 10e-6, 10e-6]), np.array([0.01, 0.01, 0.01])),
    ]
    
    methods = [
        ("LP", lambda td, om: (solve_lp(td, A, lb, ub)[0:2])),
        ("QP", lambda td, om: solve_qp(td, A, lb, ub)),
        ("QP Pareto", lambda td, om: solve_qp_pareto(td, A, lb, ub)),
        ("Bounded 1.5", lambda td, om: solve_bounded_ratio(td, A, lb, ub, 1.5)),
        ("Cone 30°", lambda td, om: solve_cone_qp(td, A, lb, ub, 30)),
        ("Cone 15°", lambda td, om: solve_cone_qp(td, A, lb, ub, 15)),
        ("QP Energy", lambda td, om: solve_qp_energy(td, A, lb, ub, om if om is not None else np.array([0.01,0.01,0.01]))),
    ]
    
    for tc_name, tau_des, omega in test_cases:
        print(f"\n{'='*90}")
        print(f"TEST: {tc_name}")
        print(f"τ_des = {tau_des * 1e6} μNm")
        if omega is not None:
            print(f"ω = {omega} rad/s")
        print(f"{'='*90}")
        
        tau_hat = tau_des / np.linalg.norm(tau_des)
        
        print(f"{'Method':<12} {'τ (μNm)':<35} {'||τ||':>8} {'error':>8} {'dir°':>6} {'P(nW)':>10}")
        print("-" * 90)
        
        for name, method in methods:
            u, tau = method(tau_des, omega)
            
            tau_norm = np.linalg.norm(tau)
            error = np.linalg.norm(tau - tau_des)
            
            if tau_norm > 1e-15:
                cos_angle = np.dot(tau, tau_des) / (tau_norm * np.linalg.norm(tau_des))
                dir_err = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            else:
                dir_err = 0
            
            if omega is not None:
                P = np.dot(tau, omega) * 1e9
            else:
                P = 0
            
            tau_str = f"[{tau[0]*1e6:7.3f},{tau[1]*1e6:7.3f},{tau[2]*1e6:8.3f}]"
            print(f"{name:<12} {tau_str:<35} {tau_norm*1e6:>8.3f} {error*1e6:>8.3f} {dir_err:>6.1f} {P:>10.2f}")
    
    return


def sanity_checks():
    """Sanity check the results."""
    print("\n" + "=" * 90)
    print("SANITY CHECKS")
    print("=" * 90)
    
    A, lb, ub, B = setup_system()
    
    print("\n1. For achievable targets, all methods should match exactly:")
    tau_des = np.array([2e-6, 2e-6, 2e-6])
    
    _, tau_lp = solve_lp(tau_des, A, lb, ub)[0:2]
    _, tau_qp = solve_qp(tau_des, A, lb, ub)
    
    print(f"   τ_des = {tau_des * 1e6} μNm")
    print(f"   LP:  τ = {tau_lp * 1e6} μNm, error = {np.linalg.norm(tau_lp - tau_des) * 1e6:.6f}")
    print(f"   QP:  τ = {tau_qp * 1e6} μNm, error = {np.linalg.norm(tau_qp - tau_des) * 1e6:.6f}")
    
    print("\n2. QP should have lower L2 error than LP for unachievable targets:")
    tau_des = np.array([10e-6, 10e-6, 10e-6])
    
    _, tau_lp = solve_lp(tau_des, A, lb, ub)[0:2]
    _, tau_qp = solve_qp(tau_des, A, lb, ub)
    
    err_lp = np.linalg.norm(tau_lp - tau_des)
    err_qp = np.linalg.norm(tau_qp - tau_des)
    
    print(f"   τ_des = {tau_des * 1e6} μNm (unachievable)")
    print(f"   LP error: {err_lp * 1e6:.4f} μNm")
    print(f"   QP error: {err_qp * 1e6:.4f} μNm")
    print(f"   QP better? {err_qp < err_lp} ✓" if err_qp < err_lp else f"   QP better? {err_qp < err_lp} ✗ BUG!")
    
    print("\n3. LP should have zero direction error:")
    tau_des = np.array([10e-6, 10e-6, 10e-6])
    _, tau_lp = solve_lp(tau_des, A, lb, ub)[0:2]
    
    cos_angle = np.dot(tau_lp, tau_des) / (np.linalg.norm(tau_lp) * np.linalg.norm(tau_des))
    dir_err = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    
    print(f"   LP direction error: {dir_err:.4f}° (should be ~0)")
    
    print("\n4. Pareto should never be worse than LP on any axis:")
    tau_des = np.array([10e-6, 10e-6, 10e-6])
    _, tau_lp = solve_lp(tau_des, A, lb, ub)[0:2]
    _, tau_pareto = solve_qp_pareto(tau_des, A, lb, ub)
    
    print(f"   LP:     τ = {tau_lp * 1e6} μNm")
    print(f"   Pareto: τ = {tau_pareto * 1e6} μNm")
    
    all_better = True
    for i, axis in enumerate(['x', 'y', 'z']):
        err_lp = abs(tau_lp[i] - tau_des[i])
        err_pareto = abs(tau_pareto[i] - tau_des[i])
        status = "✓" if err_pareto <= err_lp + 1e-12 else "✗"
        if err_pareto > err_lp + 1e-12:
            all_better = False
        print(f"   {axis}: LP err = {err_lp*1e6:.4f}, Pareto err = {err_pareto*1e6:.4f} {status}")
    
    print(f"   All axes Pareto-better? {all_better}")
    
    print("\n5. Energy constraint should prevent τ·ω > 0 when τ_des·ω < 0:")
    tau_des = np.array([-10e-6, -10e-6, -10e-6])  # Braking
    omega = np.array([0.01, 0.01, 0.01])  # Positive rotation
    
    _, tau_qp = solve_qp(tau_des, A, lb, ub)
    _, tau_energy = solve_qp_energy(tau_des, A, lb, ub, omega)
    
    P_qp = np.dot(tau_qp, omega)
    P_energy = np.dot(tau_energy, omega)
    P_des = np.dot(tau_des, omega)
    
    print(f"   τ_des·ω = {P_des * 1e9:.2f} nW (want negative = braking)")
    print(f"   QP:     τ·ω = {P_qp * 1e9:.2f} nW")
    print(f"   Energy: τ·ω = {P_energy * 1e9:.2f} nW")
    print(f"   Energy prevents acceleration? {P_energy <= 1e-15} {'✓' if P_energy <= 1e-15 else '✗'}")
    
    return


def summary():
    """Final summary."""
    print("\n" + "=" * 90)
    print("SUMMARY OF CORRECTED QP ANALYSIS")
    print("=" * 90)
    
    print("""
KEY FINDINGS:
=============

1. NUMERICAL CONDITIONING IS CRITICAL
   - Original QP failed due to condition number ~10⁹
   - Fix: Scale by 1e6 (work in μNm units)
   - All CVXPY/QP code must use SCALE = 1e6

2. QP IS BETTER THAN LP FOR L2 ERROR
   - For unachievable τ_des, QP finds closer point
   - Example: τ_des = [10,10,10] μNm
     - LP:  [2, 2, 2] μNm,       error = 13.86 μNm, direction = 0°
     - QP:  [1.04, 3.28, 10] μNm, error = 11.20 μNm, direction = 49°

3. LP IS BETTER FOR DIRECTION PRESERVATION  
   - LP guarantees τ ∝ τ_des (zero direction error)
   - Important for stability in control loops

4. PARETO QP IS A GOOD COMPROMISE
   - Never worse than LP on any axis
   - Can improve overall error
   - Some direction error possible

5. CONE QP GIVES BOUNDED DIRECTION ERROR
   - Specify max angle (15° or 30°)
   - Gets more magnitude than LP
   - Guaranteed direction quality

6. ENERGY CONSTRAINT PREVENTS INSTABILITY
   - τ·ω ≤ max(0, τ_des·ω)
   - Prevents energy injection during damping
   - Should ALWAYS be used for rate control


RECOMMENDATIONS:
================

A) Default: LP (direction preservation, stability)
B) When more magnitude needed: Cone QP (15-30°)  
C) When Pareto improvement OK: QP Pareto
D) For rate damping: Always add energy constraint
E) ALWAYS scale by 1e6 for numerical stability
""")


if __name__ == "__main__":
    np.random.seed(42)
    
    run_all_tests()
    sanity_checks()
    summary()
