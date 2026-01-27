"""
QP Mathematical Rethink
=======================

Let's think about this problem from first principles, mathematically.
"""

import numpy as np
import cvxpy as cp
from scipy.optimize import linprog, minimize
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


def mathematical_analysis():
    """Think through the math carefully."""
    print("=" * 80)
    print("MATHEMATICAL ANALYSIS FROM FIRST PRINCIPLES")
    print("=" * 80)
    
    A, lb, ub, B = setup_system()
    n = len(lb)
    
    print("""
THE PROBLEM:
============

Given: 
    - Actuator matrix A ∈ ℝ³ˣ⁴ (maps commands to torque)
    - Box constraints: lb ≤ u ≤ ub
    - Desired torque: τ_des ∈ ℝ³

Find: u* that produces τ = A·u* "best matching" τ_des

The reachable set R = {A·u : lb ≤ u ≤ ub} is a POLYTOPE in ℝ³.
""")
    
    print("Let's examine the structure of A:")
    print(f"\nA = \n{A}")
    print(f"\nA[:, 0] (RW column): {A[:, 0]}")
    print(f"A[:, 1:] (MTQ columns):\n{A[:, 1:]}")
    
    print("""
    
KEY OBSERVATION:
================

The RW column is [0, 0, 1]. This means:
    - RW ONLY affects τ_z
    - τ_x, τ_y come ONLY from MTQ
    
MTQ columns come from -skew(B) @ I:
    - τ_x = B_z·m_y - B_y·m_z = 10e-6·m_y - 15e-6·m_z
    - τ_y = -B_z·m_x + B_x·m_z = -10e-6·m_x + 20e-6·m_z  
    - τ_z = B_y·m_x - B_x·m_y + u_rw = 15e-6·m_x - 20e-6·m_y + u_rw

With |m_i| ≤ 0.2 Am² and B ~ 10-20 μT:
    - Max |τ_x| from MTQ ≈ 0.2 × (10+15) μT = 5 μNm
    - Max |τ_y| from MTQ ≈ 0.2 × (10+20) μT = 6 μNm
    - Max |τ_z| from RW alone = 1000 μNm (dominates MTQ contribution)
""")
    
    # Verify max achievable torques
    print("\nVERIFYING MAX ACHIEVABLE TORQUES:")
    print("-" * 50)
    
    for i, axis in enumerate(['x', 'y', 'z']):
        # Maximize A[i,:] @ u
        c = -A[i, :]  # Negative for maximization
        bounds = list(zip(lb, ub))
        res_max = linprog(c, bounds=bounds, method='highs')
        res_min = linprog(-c, bounds=bounds, method='highs')
        
        tau_max = -res_max.fun if res_max.success else 0
        tau_min = res_min.fun if res_min.success else 0
        
        print(f"τ_{axis} ∈ [{tau_min*1e6:.2f}, {tau_max*1e6:.2f}] μNm")
        
        if res_max.success:
            u_at_max = res_max.x
            print(f"  At max: u = [{u_at_max[0]:.4f}, {u_at_max[1]:.3f}, {u_at_max[2]:.3f}, {u_at_max[3]:.3f}]")
    
    return A, lb, ub


def analyze_specific_case(A, lb, ub):
    """Analyze τ_des = [10, 10, 10] μNm carefully."""
    print("\n" + "=" * 80)
    print("ANALYZING τ_des = [10, 10, 10] μNm")
    print("=" * 80)
    
    tau_des = np.array([10e-6, 10e-6, 10e-6])
    n = len(lb)
    
    print(f"\nτ_des = {tau_des * 1e6} μNm")
    print(f"||τ_des|| = {np.linalg.norm(tau_des) * 1e6:.2f} μNm")
    
    # Check if achievable
    print("\n1. IS τ_des ACHIEVABLE?")
    print("-" * 50)
    
    u = cp.Variable(n)
    constraints = [A @ u == tau_des, u >= lb, u <= ub]
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.ECOS)
    
    if prob.status == 'optimal':
        print(f"YES! τ_des is achievable.")
        print(f"u = {u.value}")
        print(f"τ = {(A @ u.value) * 1e6} μNm")
    else:
        print(f"NO. τ_des is NOT achievable. Status: {prob.status}")
    
    # LP solution
    print("\n2. LP SOLUTION (proportional scaling)")
    print("-" * 50)
    
    t_mag = np.linalg.norm(tau_des)
    tau_hat = tau_des / t_mag
    
    # max α s.t. A·u = α·τ̂, lb ≤ u ≤ ub
    c = np.zeros(n + 1)
    c[-1] = -1  # Maximize α
    A_eq = np.hstack([A, -tau_hat.reshape(3, 1)])
    bounds_lp = [(lb[i], ub[i]) for i in range(n)] + [(0, None)]
    
    res = linprog(c, A_eq=A_eq, b_eq=np.zeros(3), bounds=bounds_lp, method='highs')
    
    if res.success:
        u_lp = res.x[:n]
        alpha_lp = res.x[-1]
        tau_lp = A @ u_lp
        
        print(f"α_LP = {alpha_lp * 1e6:.4f} μNm (in direction τ̂)")
        print(f"u_LP = {u_lp}")
        print(f"τ_LP = {tau_lp * 1e6} μNm")
        print(f"||τ_LP - τ_des|| = {np.linalg.norm(tau_lp - tau_des) * 1e6:.4f} μNm")
        
        # Check which constraints are active
        active_lb = np.isclose(u_lp, lb, atol=1e-6)
        active_ub = np.isclose(u_lp, ub, atol=1e-6)
        print(f"Active lower bounds: {np.where(active_lb)[0].tolist()}")
        print(f"Active upper bounds: {np.where(active_ub)[0].tolist()}")
    
    # QP solution
    print("\n3. QP SOLUTION (min ||τ - τ_des||²)")
    print("-" * 50)
    
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(tau - tau_des))
    constraints = [u >= lb, u <= ub]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    u_qp = u.value
    tau_qp = A @ u_qp
    
    print(f"u_QP = {u_qp}")
    print(f"τ_QP = {tau_qp * 1e6} μNm")
    print(f"||τ_QP - τ_des|| = {np.linalg.norm(tau_qp - tau_des) * 1e6:.4f} μNm")
    
    active_lb = np.isclose(u_qp, lb, atol=1e-6)
    active_ub = np.isclose(u_qp, ub, atol=1e-6)
    print(f"Active lower bounds: {np.where(active_lb)[0].tolist()}")
    print(f"Active upper bounds: {np.where(active_ub)[0].tolist()}")
    
    # Compare errors
    print("\n4. COMPARISON")
    print("-" * 50)
    
    err_lp = np.linalg.norm(tau_lp - tau_des)
    err_qp = np.linalg.norm(tau_qp - tau_des)
    
    print(f"LP error:  {err_lp * 1e6:.4f} μNm")
    print(f"QP error:  {err_qp * 1e6:.4f} μNm")
    print(f"QP finds {'better' if err_qp < err_lp else 'worse'} solution by L2 metric")
    
    # The KEY question: why doesn't QP find [2, 2, 2]?
    print("\n5. SANITY CHECK: Is [2, 2, 2] μNm achievable?")
    print("-" * 50)
    
    tau_test = np.array([2e-6, 2e-6, 2e-6])
    
    u = cp.Variable(n)
    constraints = [A @ u == tau_test, u >= lb, u <= ub]
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.ECOS)
    
    if prob.status == 'optimal':
        print(f"YES! τ = [2, 2, 2] μNm is achievable.")
        u_222 = u.value
        print(f"u = {u_222}")
        print(f"Verification: A @ u = {(A @ u_222) * 1e6} μNm")
        
        # Error if we used this
        err_222 = np.linalg.norm(tau_test - tau_des)
        print(f"||[2,2,2] - [10,10,10]|| = {err_222 * 1e6:.4f} μNm")
        print(f"This is {'less' if err_222 < err_qp else 'more'} than QP error of {err_qp * 1e6:.4f} μNm")
    else:
        print(f"NO! τ = [2, 2, 2] μNm is NOT achievable. Status: {prob.status}")
    
    return tau_lp, tau_qp, tau_des


def understand_qp_behavior(A, lb, ub):
    """Understand why QP gives what it gives."""
    print("\n" + "=" * 80)
    print("UNDERSTANDING QP BEHAVIOR")
    print("=" * 80)
    
    tau_des = np.array([10e-6, 10e-6, 10e-6])
    n = len(lb)
    
    print("""
QP solves: min ||A·u - τ_des||² s.t. lb ≤ u ≤ ub

The Lagrangian is:
    L = ½||A·u - τ_des||² + λ_lb·(lb - u) + λ_ub·(u - ub)

KKT conditions:
    Aᵀ(A·u - τ_des) + λ_ub - λ_lb = 0
    λ_lb ≥ 0, λ_ub ≥ 0
    λ_lb·(lb - u) = 0, λ_ub·(u - ub) = 0

Let's solve this explicitly...
""")
    
    # Get QP solution
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(tau - tau_des))
    constraints = [u >= lb, u <= ub]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    u_qp = u.value
    tau_qp = A @ u_qp
    
    # Check KKT gradient
    gradient = A.T @ (tau_qp - tau_des)
    print(f"u_QP = {u_qp}")
    print(f"τ_QP = {tau_qp * 1e6} μNm")
    print(f"Gradient Aᵀ(τ - τ_des) = {gradient}")
    
    # Interpret gradient at each bound
    print("\nGradient interpretation:")
    for i in range(n):
        if np.isclose(u_qp[i], lb[i], atol=1e-6):
            print(f"  u[{i}] at lower bound: λ_lb[{i}] = {-gradient[i]:.6f}")
        elif np.isclose(u_qp[i], ub[i], atol=1e-6):
            print(f"  u[{i}] at upper bound: λ_ub[{i}] = {gradient[i]:.6f}")
        else:
            print(f"  u[{i}] interior: gradient[{i}] = {gradient[i]:.6f} (should be ~0)")
    
    print("""
    
KEY INSIGHT:
============

The gradient at u_QP tells us the "price" of relaxing each bound.

If u[i] is at a bound and gradient[i] ≠ 0, then:
    - Moving u[i] away from bound would IMPROVE the objective
    - But we can't because of the constraint
    
If u[i] is interior and gradient[i] ≈ 0:
    - We're at a local minimum w.r.t. u[i]
    - Could move either way without improving much
""")
    
    # Now let's understand geometrically
    print("\n" + "=" * 80)
    print("GEOMETRIC UNDERSTANDING")
    print("=" * 80)
    
    print("""
The reachable set R = {A·u : lb ≤ u ≤ ub} is a polytope.

With A ∈ ℝ³ˣ⁴ and box constraints, R is the image of a 4D box through A.

Key observation: The columns of A span at most rank(A) = 3 dimensions.
So R is a 3D polytope (assuming full row rank).

Let's compute the vertices of R by looking at corner points of the box.
""")
    
    # Compute some vertices
    from itertools import product
    
    vertices = []
    for corner in product([0, 1], repeat=n):
        u_corner = np.array([lb[i] if c == 0 else ub[i] for i, c in enumerate(corner)])
        tau_corner = A @ u_corner
        vertices.append(tau_corner)
    
    vertices = np.array(vertices)
    print(f"Number of vertices: {len(vertices)} (2^{n} = {2**n})")
    
    # Find extreme points in each direction
    print("\nExtreme τ values at vertices:")
    for i, axis in enumerate(['x', 'y', 'z']):
        min_val = np.min(vertices[:, i])
        max_val = np.max(vertices[:, i])
        print(f"  τ_{axis} ∈ [{min_val*1e6:.2f}, {max_val*1e6:.2f}] μNm")
    
    # Distance from τ_des to each vertex
    print(f"\nDistance from τ_des = {tau_des * 1e6} μNm to vertices:")
    distances = [np.linalg.norm(v - tau_des) for v in vertices]
    min_dist_idx = np.argmin(distances)
    print(f"  Closest vertex: {vertices[min_dist_idx] * 1e6} μNm")
    print(f"  Distance: {distances[min_dist_idx] * 1e6:.4f} μNm")
    
    # QP finds closest point in R to τ_des
    print(f"\nQP solution: {tau_qp * 1e6} μNm")
    print(f"QP distance: {np.linalg.norm(tau_qp - tau_des) * 1e6:.4f} μNm")
    
    print("""
    
The QP solution might be:
1. At a vertex of R
2. On an edge of R  
3. On a face of R
4. In the interior of R (only if τ_des ∈ R)

Since τ_des is NOT in R, QP finds the closest point on the boundary of R.
""")
    
    return


def rerun_all_tests():
    """Re-run all QP variants with correct understanding."""
    print("\n" + "=" * 80)
    print("RE-RUNNING ALL TESTS WITH CORRECT IMPLEMENTATIONS")
    print("=" * 80)
    
    A, lb, ub, B = setup_system()
    n = len(lb)
    
    def solve_lp(tau_des):
        """LP: max α s.t. τ = α·τ̂_des"""
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
            # Scale down if over-achieving
            if alpha > t_mag:
                u = u * (t_mag / alpha)
            return u, A @ u, alpha / t_mag
        return np.zeros(n), np.zeros(3), 0.0
    
    def solve_qp(tau_des):
        """Standard QP: min ||τ - τ_des||²"""
        u = cp.Variable(n)
        tau = A @ u
        objective = cp.Minimize(cp.sum_squares(tau - tau_des))
        constraints = [u >= lb, u <= ub]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        return u.value if u.value is not None else np.zeros(n)
    
    def solve_qp_pareto(tau_des):
        """QP with Pareto constraint: don't make any axis worse than LP"""
        u_lp, tau_lp, _ = solve_lp(tau_des)
        
        u = cp.Variable(n)
        tau = A @ u
        objective = cp.Minimize(cp.sum_squares(tau - tau_des))
        constraints = [u >= lb, u <= ub]
        
        # Pareto: each axis at least as good as LP
        for i in range(3):
            if tau_des[i] > 1e-15:
                constraints.append(tau[i] >= tau_lp[i] - 1e-15)
            elif tau_des[i] < -1e-15:
                constraints.append(tau[i] <= tau_lp[i] + 1e-15)
        
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        return u.value if u.value is not None else u_lp
    
    def solve_bounded_ratio(tau_des, k=1.5):
        """Bounded ratio: α·τ_des ≤ τ ≤ k·α·τ_des"""
        t_mag = np.linalg.norm(tau_des)
        if t_mag < 1e-15:
            return np.zeros(n)
        
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
        return u.value if u.value is not None else np.zeros(n)
    
    def solve_cone_qp(tau_des, theta_max_deg=30):
        """QP with direction cone constraint"""
        t_mag = np.linalg.norm(tau_des)
        if t_mag < 1e-15:
            return np.zeros(n)
        
        cos_theta = np.cos(np.radians(theta_max_deg))
        
        u = cp.Variable(n)
        tau = A @ u
        
        objective = cp.Minimize(cp.sum_squares(tau - tau_des))
        constraints = [
            u >= lb, u <= ub,
            tau @ tau_des >= cos_theta * cp.norm(tau) * t_mag
        ]
        
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        
        if u.value is not None:
            return u.value
        # Fallback to LP
        u_lp, _, _ = solve_lp(tau_des)
        return u_lp
    
    # Test cases
    test_cases = [
        ("Balanced [10,10,10]", np.array([10e-6, 10e-6, 10e-6])),
        ("Heavy z [1,1,100]", np.array([1e-6, 1e-6, 100e-6])),
        ("Heavy xy [100,100,1]", np.array([100e-6, 100e-6, 1e-6])),
        ("Small [1,1,1]", np.array([1e-6, 1e-6, 1e-6])),
        ("Achievable [2,2,2]", np.array([2e-6, 2e-6, 2e-6])),
    ]
    
    methods = [
        ("LP", solve_lp),
        ("QP", lambda td: (solve_qp(td), None, None)),
        ("QP Pareto", lambda td: (solve_qp_pareto(td), None, None)),
        ("Bounded k=1.5", lambda td: (solve_bounded_ratio(td, 1.5), None, None)),
        ("Cone 30°", lambda td: (solve_cone_qp(td, 30), None, None)),
        ("Cone 15°", lambda td: (solve_cone_qp(td, 15), None, None)),
    ]
    
    for tc_name, tau_des in test_cases:
        print(f"\n{'='*70}")
        print(f"TEST: {tc_name}")
        print(f"τ_des = {tau_des * 1e6} μNm")
        print(f"{'='*70}")
        
        tau_hat = tau_des / np.linalg.norm(tau_des)
        
        print(f"{'Method':<15} {'τ (μNm)':<35} {'||τ||':>8} {'error':>8} {'dir°':>6} {'α':>6}")
        print("-" * 85)
        
        for name, method in methods:
            result = method(tau_des)
            if len(result) == 3:
                u, tau, alpha = result
                if tau is None:
                    tau = A @ u
                    alpha = np.dot(tau, tau_hat) / np.linalg.norm(tau_des)
            else:
                u = result
                tau = A @ u
                alpha = np.dot(tau, tau_hat) / np.linalg.norm(tau_des)
            
            tau_norm = np.linalg.norm(tau)
            error = np.linalg.norm(tau - tau_des)
            
            if tau_norm > 1e-15:
                cos_angle = np.dot(tau, tau_des) / (tau_norm * np.linalg.norm(tau_des))
                dir_err = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            else:
                dir_err = 0
            
            tau_str = f"[{tau[0]*1e6:7.3f},{tau[1]*1e6:7.3f},{tau[2]*1e6:8.3f}]"
            print(f"{name:<15} {tau_str:<35} {tau_norm*1e6:>8.3f} {error*1e6:>8.3f} {dir_err:>6.1f} {alpha:>6.3f}")
    
    return


def final_summary():
    """Final summary of correct understanding."""
    print("\n" + "=" * 80)
    print("FINAL SUMMARY: CORRECT UNDERSTANDING")
    print("=" * 80)
    
    print("""
WHAT WE NOW UNDERSTAND:
=======================

1. THE REACHABLE SET
   R = {A·u : lb ≤ u ≤ ub} is a 3D polytope
   
   For 3MTQ + 1RW with z-axis wheel:
   - τ_x ∈ [-5, +5] μNm (MTQ only)
   - τ_y ∈ [-6, +6] μNm (MTQ only)
   - τ_z ∈ [-1007, +1007] μNm (RW dominates)
   
   The polytope is HIGHLY ASYMMETRIC: thin in x,y, huge in z.

2. QP FINDS THE CLOSEST POINT IN R TO τ_des
   
   For τ_des = [10, 10, 10] μNm (outside R):
   - QP finds the closest point on the BOUNDARY of R
   - This point has τ_z ≈ τ_des_z because z is easy
   - But τ_x, τ_y are at their limits (~5-6 μNm max)
   
   The QP solution [~0, ~0, ~10] makes sense geometrically:
   - It's on the face of R where x,y are maxed out
   - Moving τ_z to 10 gets us closest to [10,10,10]

3. LP FINDS THE PROPORTIONAL SCALING
   
   LP forces τ ∝ τ_des, giving τ = α·τ_des
   The maximum α is limited by the tightest constraint (x or y)
   
   For τ_des = [10, 10, 10], α ≈ 0.2 gives τ = [2, 2, 2]

4. WHICH IS "BETTER"?
   
   L2 error: QP wins (closer to τ_des in Euclidean distance)
   Direction: LP wins (τ ∝ τ_des, zero direction error)
   
   For CONTROL, direction often matters more than magnitude!
   - Wrong direction can inject energy (destabilize)
   - Proportional scaling preserves stability properties

5. THE "SMART" FORMULATIONS
   
   - Weighted QP failed because weights were axis-based, not direction-based
   - MaxMin failed because only lower bounds (no upper)
   - Bounded ratio works: forces α·τ_des ≤ τ ≤ k·α·τ_des
   - Cone QP works: bounds direction error directly


RECOMMENDATIONS:
================

A) DEFAULT: Use LP
   - Preserves direction perfectly
   - Fair allocation across all axes
   - Stable control behavior

B) WHEN MORE MAGNITUDE IS NEEDED: Use Cone QP
   - Specify max direction error θ_max (e.g., 15-30°)
   - Gets more magnitude while bounding direction error
   
C) WHEN PARETO IMPROVEMENT IS OK: Use QP + Pareto
   - Each axis at least as good as LP
   - Uses extra capacity on "easy" axes
   - Direction error possible but bounded

D) FOR RATE DAMPING: Add energy constraint τ·ω ≤ 0
   - Prevents energy injection
   - Works with any of the above
""")


if __name__ == "__main__":
    np.random.seed(42)
    
    A, lb, ub = mathematical_analysis()
    analyze_specific_case(A, lb, ub)
    understand_qp_behavior(A, lb, ub)
    rerun_all_tests()
    final_summary()
