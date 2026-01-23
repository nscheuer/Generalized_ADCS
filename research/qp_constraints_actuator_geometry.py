"""
QP Constraints Analysis: What Actually Makes Sense Given Actuator Geometry?
==========================================================================

Key questions:
1. When would QP return negative value (opposite sign) anyway?
2. Which of the 12 constraints matter for different actuator configurations?
3. Mathematical analysis of each constraint's applicability
4. Test best candidates across varied geometries
"""

import numpy as np
from scipy.optimize import minimize, Bounds, linprog
import cvxpy as cp
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from ADCS.helpers.math_helpers import skewsym, normalize


# ============== ACTUATOR GEOMETRY ANALYSIS ==============

def analyze_when_qp_goes_negative():
    """
    Analyze when QP (min ||τ - τ_des||²) would produce opposite sign torque.
    
    Key insight: QP produces τ_i with wrong sign when:
    1. τ_des_i is small relative to other axes
    2. The reachable set is shifted/tilted such that closest point has opposite sign
    3. Actuator coupling creates this geometry
    """
    print("=" * 80)
    print("WHEN DOES QP PRODUCE OPPOSITE SIGN TORQUE?")
    print("=" * 80)
    
    print("""
MATHEMATICAL ANALYSIS:
---------------------

QP solves: min ||A·u - τ_des||²
           s.t. lb ≤ u ≤ ub

The solution is the closest point in the reachable set R = {A·u : lb ≤ u ≤ ub}
to τ_des.

Sign flip occurs when:
  τ_QP_i · τ_des_i < 0  (opposite signs)

This happens geometrically when τ_des is outside R and the closest point
is on the "other side" of the origin for some axis.


CASE 1: Shifted reachable set
-----------------------------
If R doesn't include origin (bias), QP may give wrong sign.

Example: A = [[1], [1], [1]], lb = [0.1], ub = [1]
         τ_des = [0, 0, 0.05]
         Achievable: τ ∈ {[t, t, t] : 0.1 ≤ t ≤ 1}
         QP: τ = [0.1, 0.1, 0.1] (minimum achievable)
         
         All positive even though τ_des_z is small positive.
         No sign flip here because R is "above" origin.


CASE 2: Tilted/rotated reachable set
------------------------------------
More common case with MTQ + RW.

Example: 
  A_rw = [[0], [0], [1]]  (z-axis RW)
  A_mtq = -skew(B) @ I   (perpendicular to B)
  
  If B = [1, 0, 0]:
    A_mtq produces torque in y-z plane only
    τ_x can only come from... nothing! (underactuated)
    
  If τ_des = [1, 0, 0]:
    QP wants τ_x > 0 but can't achieve it
    Closest achievable might have τ_x = 0 or even small negative
    depending on A_mtq structure.


CASE 3: The "axis stealing" problem
-----------------------------------
When τ_des has components in orthogonal directions but actuators
have coupling.

Example: τ_des = [1, 0, 10]
         System can easily do z (RW), hard to do x (MTQ limited)
         
QP might produce τ = [0, 0, 10] - stealing all authority for z,
giving nothing to x. This isn't a sign flip but is the "ignoring axes" problem.

For sign flip:
  τ_des = [1, 0, 10]
  If somehow achieving τ_z = 10 created τ_x = -0.5 (coupling),
  QP would do it to minimize total error.
""")
    
    return


def test_sign_flip_scenarios():
    """Test specific scenarios where sign flip might occur."""
    print("\n" + "=" * 80)
    print("TESTING SIGN FLIP SCENARIOS")
    print("=" * 80)
    
    def qp_solve(tau_des, A, lb, ub):
        n = len(lb)
        u = cp.Variable(n)
        tau = A @ u
        objective = cp.Minimize(cp.sum_squares(tau - tau_des))
        constraints = [u >= lb, u <= ub]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        return A @ u.value if u.value is not None else None
    
    # Scenario 1: 1 RW + 3 MTQ (our standard underactuated case)
    print("\n1. 3MTQ + 1RW (z-axis)")
    A_rw = np.array([[0], [0], [1.0]])
    B = np.array([20e-6, 15e-6, 10e-6])
    A_mtq = -skewsym(B) @ np.eye(3)
    A = np.hstack([A_rw, A_mtq])
    
    lb = np.array([-0.001, -0.2, -0.2, -0.2])
    ub = np.array([0.001, 0.2, 0.2, 0.2])
    
    test_cases = [
        ("Balanced", np.array([1e-5, 1e-5, 1e-5])),
        ("Heavy z", np.array([1e-5, 1e-5, 10e-5])),
        ("Light z", np.array([10e-5, 10e-5, 1e-5])),
        ("Negative x", np.array([-1e-5, 1e-5, 1e-5])),
        ("All negative", np.array([-1e-5, -1e-5, -1e-5])),
        ("Along B", B / np.linalg.norm(B) * 1e-5),
    ]
    
    print(f"{'Case':<15} {'τ_des':<30} {'τ_QP':<35} {'Sign flip?':<10}")
    print("-" * 95)
    
    for name, tau_des in test_cases:
        tau_qp = qp_solve(tau_des, A, lb, ub)
        if tau_qp is not None:
            sign_flips = []
            for i in range(3):
                if abs(tau_des[i]) > 1e-12 and tau_qp[i] * tau_des[i] < 0:
                    sign_flips.append(['x', 'y', 'z'][i])
            flip_str = ','.join(sign_flips) if sign_flips else 'No'
            
            tau_des_str = f"[{tau_des[0]*1e6:6.2f},{tau_des[1]*1e6:6.2f},{tau_des[2]*1e6:6.2f}]"
            tau_qp_str = f"[{tau_qp[0]*1e6:8.4f},{tau_qp[1]*1e6:8.4f},{tau_qp[2]*1e6:8.4f}]"
            print(f"{name:<15} {tau_des_str:<30} {tau_qp_str:<35} {flip_str:<10}")
    
    # Scenario 2: 3 MTQ only (no RW)
    print("\n2. 3MTQ only (severely underactuated)")
    A = A_mtq
    lb = np.array([-0.2, -0.2, -0.2])
    ub = np.array([0.2, 0.2, 0.2])
    
    test_cases = [
        ("Along B", B / np.linalg.norm(B) * 1e-5),
        ("Perp to B", np.cross(B, [1,0,0]) / np.linalg.norm(np.cross(B, [1,0,0])) * 1e-5),
        ("45° to B", (B/np.linalg.norm(B) + np.array([1,0,0])) * 0.5e-5),
    ]
    
    print(f"{'Case':<15} {'τ_des':<30} {'τ_QP':<35} {'Sign flip?':<10}")
    print("-" * 95)
    
    for name, tau_des in test_cases:
        tau_qp = qp_solve(tau_des, A, lb, ub)
        if tau_qp is not None:
            sign_flips = []
            for i in range(3):
                if abs(tau_des[i]) > 1e-12 and tau_qp[i] * tau_des[i] < 0:
                    sign_flips.append(['x', 'y', 'z'][i])
            flip_str = ','.join(sign_flips) if sign_flips else 'No'
            
            tau_des_str = f"[{tau_des[0]*1e6:6.2f},{tau_des[1]*1e6:6.2f},{tau_des[2]*1e6:6.2f}]"
            tau_qp_str = f"[{tau_qp[0]*1e6:8.4f},{tau_qp[1]*1e6:8.4f},{tau_qp[2]*1e6:8.4f}]"
            print(f"{name:<15} {tau_des_str:<30} {tau_qp_str:<35} {flip_str:<10}")
    
    # Scenario 3: Coupled actuators (e.g., canted RWs)
    print("\n3. 4RW Pyramid (canted, may have coupling)")
    theta = np.radians(54.74)
    A_rw = np.array([
        [np.sin(theta), 0, -np.sin(theta), 0],
        [0, np.sin(theta), 0, -np.sin(theta)],
        [np.cos(theta), np.cos(theta), np.cos(theta), np.cos(theta)]
    ])
    A = A_rw
    lb = np.array([-0.001, -0.001, -0.001, -0.001])
    ub = np.array([0.001, 0.001, 0.001, 0.001])
    
    test_cases = [
        ("Pure x", np.array([1e-5, 0, 0])),
        ("Pure z", np.array([0, 0, 1e-5])),
        ("Diagonal", np.array([1e-5, 1e-5, 1e-5])),
        ("Asymmetric", np.array([2e-5, 0.5e-5, 0.5e-5])),
    ]
    
    print(f"{'Case':<15} {'τ_des':<30} {'τ_QP':<35} {'Sign flip?':<10}")
    print("-" * 95)
    
    for name, tau_des in test_cases:
        tau_qp = qp_solve(tau_des, A, lb, ub)
        if tau_qp is not None:
            sign_flips = []
            for i in range(3):
                if abs(tau_des[i]) > 1e-12 and tau_qp[i] * tau_des[i] < 0:
                    sign_flips.append(['x', 'y', 'z'][i])
            flip_str = ','.join(sign_flips) if sign_flips else 'No'
            
            tau_des_str = f"[{tau_des[0]*1e6:6.2f},{tau_des[1]*1e6:6.2f},{tau_des[2]*1e6:6.2f}]"
            tau_qp_str = f"[{tau_qp[0]*1e6:8.4f},{tau_qp[1]*1e6:8.4f},{tau_qp[2]*1e6:8.4f}]"
            print(f"{name:<15} {tau_des_str:<30} {tau_qp_str:<35} {flip_str:<10}")
    
    return


def analyze_constraint_applicability():
    """Analyze which constraints make sense for different geometries."""
    print("\n" + "=" * 80)
    print("CONSTRAINT APPLICABILITY BY ACTUATOR GEOMETRY")
    print("=" * 80)
    
    print("""
Constraint analysis for different actuator configurations:

┌────────────────────────────────┬───────────┬───────────┬───────────┬───────────┐
│ Constraint                     │ 3MTQ+1RW  │ 3MTQ+3RW  │ 4RW       │ 3MTQ only │
├────────────────────────────────┼───────────┼───────────┼───────────┼───────────┤
│ 1. No Sign Flip                │ Rare      │ Very rare │ Very rare │ Possible  │
│ 2. Proportionality Bounds      │ CRITICAL  │ Useful    │ Useful    │ CRITICAL  │
│ 3. Energy Bound (τ·ω)          │ CRITICAL  │ CRITICAL  │ CRITICAL  │ CRITICAL  │
│ 4. Lyapunov Bound              │ = #3      │ = #3      │ = #3      │ = #3      │
│ 5. Perp Magnitude Bound        │ Useful    │ Less need │ Less need │ Useful    │
│ 6. Component Error Bound       │ Useful    │ Less need │ Less need │ Useful    │
│ 7. Projection Dominance        │ CRITICAL  │ Useful    │ Less need │ CRITICAL  │
│ 8. Pareto Improvement          │ CRITICAL  │ Useful    │ Less need │ CRITICAL  │
│ 9. Error-State Weighted        │ Useful    │ Useful    │ Useful    │ Useful    │
│ 10. Rate Limited               │ Useful    │ Useful    │ Useful    │ Useful    │
│ 11. Momentum Aware             │ Useful    │ Useful    │ Useful    │ Useful    │
│ 12. Controllability Weighted   │ CRITICAL  │ Useful    │ Less need │ CRITICAL  │
└────────────────────────────────┴───────────┴───────────┴───────────┴───────────┘

Key observations:

1. SIGN FLIP (Option 1):
   - Rare for most geometries when reachable set includes origin
   - More likely with MTQ-only (rank-deficient) or biased actuators
   - Worth checking but often not the main problem

2. ENERGY BOUND (Option 3):
   - ALWAYS relevant - this is about physics, not geometry
   - The QPC already implements this!
   - Key constraint: τ·ω ≤ τ_des·ω when damping

3. PROPORTIONALITY (Option 2) & PROJECTION DOMINANCE (Option 7):
   - Critical for underactuated systems
   - Prevents "ignoring axes" problem
   - Less needed when fully actuated

4. CONTROLLABILITY WEIGHTING (Option 12):
   - Adapts to the geometry automatically
   - Weights hard-to-achieve axes appropriately
   - Good for heterogeneous systems
""")
    
    return


# ============== BEST CANDIDATES IMPLEMENTATION ==============

def implement_best_candidates():
    """Implement and test the most promising constraint combinations."""
    print("\n" + "=" * 80)
    print("IMPLEMENTING BEST CONSTRAINT CANDIDATES")
    print("=" * 80)
    
    # Setup test system
    A_rw = np.array([[0], [0], [1.0]])
    B = np.array([20e-6, 15e-6, 10e-6])
    A_mtq = -skewsym(B) @ np.eye(3)
    A = np.hstack([A_rw, A_mtq])
    
    u_rw_max = np.array([0.001])
    u_mtq_max = np.array([0.2, 0.2, 0.2])
    lb = np.concatenate([-u_rw_max, -u_mtq_max])
    ub = np.concatenate([u_rw_max, u_mtq_max])
    
    n = len(lb)
    
    def lp_baseline(tau_des, A, lb, ub):
        """LP for comparison."""
        t_mag = np.linalg.norm(tau_des)
        if t_mag < 1e-12:
            return np.zeros(n)
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
            return u
        return np.zeros(n)
    
    # Candidate A: Energy-constrained QP (like QPC)
    def qp_energy_constrained(tau_des, A, lb, ub, omega):
        """QP with energy constraint: τ·ω ≤ max(0, τ_des·ω)"""
        u = cp.Variable(n)
        tau = A @ u
        
        objective = cp.Minimize(cp.sum_squares(tau - tau_des))
        
        P_des = tau_des @ omega
        P_bound = max(0, P_des)  # Don't accelerate if trying to brake
        
        constraints = [
            u >= lb, u <= ub,
            tau @ omega <= P_bound + 1e-12
        ]
        
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        
        return u.value if u.value is not None else lp_baseline(tau_des, A, lb, ub)
    
    # Candidate B: Projection dominance + energy
    def qp_proj_dom_energy(tau_des, A, lb, ub, omega):
        """QP with projection dominance and energy constraint."""
        t_mag = np.linalg.norm(tau_des)
        if t_mag < 1e-12:
            return np.zeros(n)
        tau_hat = tau_des / t_mag
        
        # Get LP baseline
        u_lp = lp_baseline(tau_des, A, lb, ub)
        tau_lp = A @ u_lp
        alpha_lp = np.dot(tau_lp, tau_hat)
        
        u = cp.Variable(n)
        tau = A @ u
        
        objective = cp.Minimize(cp.sum_squares(tau - tau_des))
        
        P_des = tau_des @ omega
        P_bound = max(0, P_des)
        
        constraints = [
            u >= lb, u <= ub,
            tau @ omega <= P_bound + 1e-12,  # Energy
            tau @ tau_hat >= alpha_lp - 1e-12  # Projection dominance
        ]
        
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        
        return u.value if u.value is not None else u_lp
    
    # Candidate C: Proportionality + energy
    def qp_proportional_energy(tau_des, A, lb, ub, omega, k=3.0):
        """QP with proportionality and energy constraints."""
        t_mag = np.linalg.norm(tau_des)
        if t_mag < 1e-12:
            return np.zeros(n)
        tau_hat = tau_des / t_mag
        
        u = cp.Variable(n)
        alpha = cp.Variable(nonneg=True)
        tau = A @ u
        
        objective = cp.Maximize(alpha)
        
        P_des = tau_des @ omega
        P_bound = max(0, P_des)
        
        constraints = [
            u >= lb, u <= ub,
            tau @ omega <= P_bound + 1e-12,  # Energy
        ]
        
        # Proportionality: tau_i between alpha*tau_des_i and k*alpha*tau_des_i
        eps = 1e-12
        for i in range(3):
            if abs(tau_des[i]) > eps:
                if tau_des[i] > 0:
                    constraints.append(tau[i] >= alpha * tau_des[i])
                    constraints.append(tau[i] <= k * alpha * tau_des[i])
                else:
                    constraints.append(tau[i] <= alpha * tau_des[i])
                    constraints.append(tau[i] >= k * alpha * tau_des[i])
        
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        
        return u.value if u.value is not None else lp_baseline(tau_des, A, lb, ub)
    
    # Candidate D: Hybrid (projection dominance + proportionality bounds on perp component)
    def qp_hybrid(tau_des, A, lb, ub, omega, perp_limit=0.5):
        """
        Hybrid: 
        - Projection dominance (at least as good as LP)
        - Energy constraint (don't inject)
        - Perpendicular magnitude bound (don't go crazy off-axis)
        """
        t_mag = np.linalg.norm(tau_des)
        if t_mag < 1e-12:
            return np.zeros(n)
        tau_hat = tau_des / t_mag
        
        u_lp = lp_baseline(tau_des, A, lb, ub)
        tau_lp = A @ u_lp
        alpha_lp = np.dot(tau_lp, tau_hat)
        
        u = cp.Variable(n)
        tau = A @ u
        tau_proj = tau @ tau_hat
        tau_perp = tau - tau_proj * tau_hat
        
        objective = cp.Maximize(tau_proj)
        
        P_des = tau_des @ omega
        P_bound = max(0, P_des)
        
        constraints = [
            u >= lb, u <= ub,
            tau @ omega <= P_bound + 1e-12,  # Energy
            tau_proj >= alpha_lp - 1e-12,  # Projection dominance
            cp.norm(tau_perp) <= perp_limit * tau_proj,  # Perp bound
        ]
        
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        
        return u.value if u.value is not None else u_lp
    
    # Test scenarios
    print("\nTest setup: 3MTQ + 1RW (z-axis)")
    print("Testing different τ_des and ω combinations\n")
    
    test_cases = [
        {
            'name': 'Balanced, damping',
            'tau_des': np.array([1e-5, 1e-5, 1e-5]),
            'omega': np.array([0.01, 0.01, 0.01]),
        },
        {
            'name': 'Heavy z, damping',
            'tau_des': np.array([1e-5, 1e-5, 10e-5]),
            'omega': np.array([0.01, 0.01, 0.05]),
        },
        {
            'name': 'Light z, damping',
            'tau_des': np.array([10e-5, 10e-5, 1e-5]),
            'omega': np.array([0.05, 0.05, 0.01]),
        },
        {
            'name': 'Cross-axis, damping',
            'tau_des': np.array([1e-5, -1e-5, 1e-5]),
            'omega': np.array([0.01, -0.01, 0.01]),
        },
        {
            'name': 'Accelerating (slew)',
            'tau_des': np.array([1e-5, 1e-5, 5e-5]),
            'omega': np.array([-0.01, -0.01, -0.02]),
        },
    ]
    
    methods = [
        ('LP', lambda td, A, lb, ub, om: lp_baseline(td, A, lb, ub)),
        ('QP Energy', qp_energy_constrained),
        ('QP ProjDom+E', qp_proj_dom_energy),
        ('QP Prop+E', qp_proportional_energy),
        ('QP Hybrid', qp_hybrid),
    ]
    
    for tc in test_cases:
        print(f"\n{'='*70}")
        print(f"Case: {tc['name']}")
        print(f"τ_des = {tc['tau_des']*1e6} μNm")
        print(f"ω = {tc['omega']} rad/s")
        print(f"P_des = τ_des·ω = {np.dot(tc['tau_des'], tc['omega'])*1e9:.2f} nW")
        print(f"{'='*70}")
        
        tau_des = tc['tau_des']
        omega = tc['omega']
        tau_hat = tau_des / np.linalg.norm(tau_des)
        P_des = np.dot(tau_des, omega)
        
        print(f"{'Method':<15} {'τ (μNm)':<35} {'proj':>8} {'P (nW)':>10} {'dir_err':>8}")
        print("-" * 80)
        
        for name, method in methods:
            u = method(tau_des, A, lb, ub, omega)
            tau = A @ u
            
            proj = np.dot(tau, tau_hat)
            P = np.dot(tau, omega)
            
            tau_norm = np.linalg.norm(tau)
            if tau_norm > 1e-12:
                dir_err = np.degrees(np.arccos(np.clip(proj/tau_norm, -1, 1)))
            else:
                dir_err = 0
            
            tau_str = f"[{tau[0]*1e6:8.3f},{tau[1]*1e6:8.3f},{tau[2]*1e6:8.3f}]"
            
            # Flag problematic cases
            flags = []
            if P > P_des + 1e-12 and P_des < 0:  # Injecting energy when should damp
                flags.append("⚠️")
            if proj < np.dot(A @ lp_baseline(tau_des, A, lb, ub), tau_hat) - 1e-12:
                flags.append("↓")  # Worse than LP
            
            flag_str = ''.join(flags)
            
            print(f"{name:<15} {tau_str:<35} {proj*1e6:>8.3f} {P*1e9:>10.2f} {dir_err:>8.1f}° {flag_str}")
    
    return


def mathematical_formulation():
    """Present the mathematical formulation for arbitrary geometry."""
    print("\n" + "=" * 80)
    print("MATHEMATICAL FORMULATION FOR ARBITRARY ACTUATOR GEOMETRY")
    print("=" * 80)
    
    print("""
GENERAL TORQUE ALLOCATION PROBLEM
=================================

Given:
  - Actuator matrix A ∈ ℝ^(3×n) mapping commands u to torque τ = A·u
  - Bounds: lb ≤ u ≤ ub
  - Desired torque: τ_des ∈ ℝ³
  - Current angular velocity: ω ∈ ℝ³

Find: u* that "best" achieves τ_des


PROPOSED FORMULATION: Constrained QP with Adaptive Fallback
===========================================================

Stage 1: Compute LP baseline
----------------------------
  Solve:  max α
          s.t. A·u = α·τ̂_des
               lb ≤ u ≤ ub
               α ≥ 0
  
  Result: α_LP, u_LP, τ_LP = A·u_LP


Stage 2: Constrained QP
-----------------------
  Solve:  min ||A·u - τ_des||²
          s.t. lb ≤ u ≤ ub
               (A·u)·ω ≤ max(0, τ_des·ω)     [Energy constraint]
               (A·u)·τ̂_des ≥ α_LP            [Projection dominance]

  If infeasible or QP fails: return u_LP


PROPERTIES:
-----------
1. NEVER WORSE THAN LP: Projection dominance ensures this
2. ENERGY SAFE: Won't inject energy when trying to damp
3. HANDLES ARBITRARY GEOMETRY: Works for any A matrix
4. GRACEFUL DEGRADATION: Falls back to LP if needed


WHEN EACH CONSTRAINT ACTIVATES:
-------------------------------
1. Energy constraint activates when:
   - QP would produce τ that accelerates rotation
   - τ_des wants to slow down (τ_des·ω < 0)
   - Unconstrained QP would have τ·ω > 0

2. Projection dominance activates when:
   - QP would sacrifice useful torque for smaller error
   - Usually when τ_des has components in hard-to-achieve directions
   - Prevents "ignoring axes" problem


OPTIONAL ADDITIONAL CONSTRAINTS:
--------------------------------
For specific applications, add:

a) Sign preservation (if needed):
   sign(τᵢ) = sign(τ_des_i) for all i with |τ_des_i| > ε

b) Perpendicular bound (if oscillation is concern):
   ||τ_perp|| ≤ k·||τ_parallel||

c) Rate limiting (if chatter is concern):
   ||τ - τ_prev|| ≤ Δτ_max


IMPLEMENTATION NOTES:
--------------------
1. Scale problem for numerical stability (multiply by 1e6 or similar)
2. Use warm start from LP solution
3. Set reasonable tolerances (1e-12 for constraint satisfaction)
4. Always have LP fallback ready
""")
    
    return


def test_across_geometries():
    """Test the best formulation across different actuator geometries."""
    print("\n" + "=" * 80)
    print("TESTING ACROSS DIFFERENT ACTUATOR GEOMETRIES")
    print("=" * 80)
    
    def qp_best(tau_des, A, lb, ub, omega):
        """Best candidate: Projection dominance + Energy constraint."""
        n = len(lb)
        t_mag = np.linalg.norm(tau_des)
        
        if t_mag < 1e-12:
            return np.zeros(n), "trivial"
        
        tau_hat = tau_des / t_mag
        
        # LP baseline
        c = np.zeros(n + 1)
        c[-1] = -1
        A_eq = np.hstack([A, -tau_hat.reshape(3, 1)])
        bounds_lp = [(lb[i], ub[i]) for i in range(n)] + [(0, None)]
        res_lp = linprog(c, A_eq=A_eq, b_eq=np.zeros(3), bounds=bounds_lp, method='highs')
        
        if not res_lp.success:
            return np.zeros(n), "LP failed"
        
        u_lp = res_lp.x[:n]
        alpha_lp = res_lp.x[-1]
        if alpha_lp > t_mag:
            u_lp = u_lp * (t_mag / alpha_lp)
            alpha_lp = t_mag
        
        tau_lp = A @ u_lp
        proj_lp = np.dot(tau_lp, tau_hat)
        
        # QP with constraints
        u = cp.Variable(n)
        tau = A @ u
        
        P_des = np.dot(tau_des, omega)
        P_bound = max(0, P_des) + 1e-15
        
        objective = cp.Minimize(cp.sum_squares(tau - tau_des))
        constraints = [
            u >= lb, u <= ub,
            tau @ omega <= P_bound,
            tau @ tau_hat >= proj_lp - 1e-12
        ]
        
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
            if u.value is not None:
                return u.value, "QP"
        except:
            pass
        
        return u_lp, "LP fallback"
    
    # Test configurations
    configs = [
        {
            'name': '3MTQ + 1RW (z)',
            'A': np.hstack([
                np.array([[0], [0], [1.0]]),
                -skewsym(np.array([20e-6, 15e-6, 10e-6])) @ np.eye(3)
            ]),
            'lb': np.array([-0.001, -0.2, -0.2, -0.2]),
            'ub': np.array([0.001, 0.2, 0.2, 0.2]),
        },
        {
            'name': '3MTQ + 3RW',
            'A': np.hstack([
                np.eye(3),
                -skewsym(np.array([20e-6, 15e-6, 10e-6])) @ np.eye(3)
            ]),
            'lb': np.array([-0.001, -0.001, -0.001, -0.2, -0.2, -0.2]),
            'ub': np.array([0.001, 0.001, 0.001, 0.2, 0.2, 0.2]),
        },
        {
            'name': '4RW Pyramid',
            'A': np.array([
                [np.sin(np.radians(54.74)), 0, -np.sin(np.radians(54.74)), 0],
                [0, np.sin(np.radians(54.74)), 0, -np.sin(np.radians(54.74))],
                [np.cos(np.radians(54.74))]*4
            ]),
            'lb': np.array([-0.001]*4),
            'ub': np.array([0.001]*4),
        },
        {
            'name': '3MTQ only',
            'A': -skewsym(np.array([20e-6, 15e-6, 10e-6])) @ np.eye(3),
            'lb': np.array([-0.2, -0.2, -0.2]),
            'ub': np.array([0.2, 0.2, 0.2]),
        },
    ]
    
    # Test cases
    test_cases = [
        ('Balanced damp', np.array([1e-5, 1e-5, 1e-5]), np.array([0.01, 0.01, 0.01])),
        ('Heavy z damp', np.array([1e-5, 1e-5, 10e-5]), np.array([0.01, 0.01, 0.05])),
        ('Slew (accel)', np.array([1e-5, 1e-5, 5e-5]), np.array([-0.01, -0.01, -0.02])),
    ]
    
    for cfg in configs:
        print(f"\n{'='*70}")
        print(f"Configuration: {cfg['name']}")
        print(f"{'='*70}")
        
        A = cfg['A']
        lb = cfg['lb']
        ub = cfg['ub']
        
        print(f"{'Case':<15} {'Method':>10} {'proj(μNm)':>12} {'P(nW)':>10} {'P_des':>10} {'Safe?':>6}")
        print("-" * 70)
        
        for tc_name, tau_des, omega in test_cases:
            u, method = qp_best(tau_des, A, lb, ub, omega)
            tau = A @ u
            
            tau_hat = tau_des / np.linalg.norm(tau_des)
            proj = np.dot(tau, tau_hat)
            P = np.dot(tau, omega)
            P_des = np.dot(tau_des, omega)
            
            # Check safety: P ≤ max(0, P_des)
            safe = P <= max(0, P_des) + 1e-10
            safe_str = "✓" if safe else "✗"
            
            print(f"{tc_name:<15} {method:>10} {proj*1e6:>12.3f} {P*1e9:>10.2f} {P_des*1e9:>10.2f} {safe_str:>6}")
    
    return


if __name__ == "__main__":
    np.random.seed(42)
    
    analyze_when_qp_goes_negative()
    test_sign_flip_scenarios()
    analyze_constraint_applicability()
    implement_best_candidates()
    mathematical_formulation()
    test_across_geometries()
    
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATIONS")
    print("=" * 80)
    print("""
RECOMMENDED CONSTRAINT SET:
===========================

1. ENERGY CONSTRAINT (Always use)
   τ·ω ≤ max(0, τ_des·ω)
   
   - Prevents energy injection when damping
   - Already in QPC implementation
   - Works for all geometries

2. PROJECTION DOMINANCE (Always use)
   τ·τ̂_des ≥ α_LP
   
   - Guarantees at least LP performance
   - Prevents "ignoring axes" problem
   - Safe fallback to LP if infeasible

3. SIGN CONSTRAINT (Optional, rarely needed)
   sign(τᵢ) = sign(τ_des_i)
   
   - Only needed if sign flips observed
   - Usually not a problem with symmetric bounds
   - May be needed for biased actuators

4. PERPENDICULAR BOUND (Optional, for oscillation)
   ||τ_perp|| ≤ k·||τ_parallel||
   
   - Helps if perpendicular torque causes oscillation
   - k ~ 0.3-0.5 typical
   - Can be omitted if not observing problems


IMPLEMENTATION:
===============
1. Try QP with constraints 1+2
2. If infeasible, fall back to LP
3. Always safe, often better than LP alone
4. Add constraint 3 or 4 only if needed for specific geometry
""")
