"""
QP Constraint Options: Finding When Perpendicular Torque Causes Problems
========================================================================

Goal: Define constraints that allow QP flexibility while preventing
the cases where perpendicular torque causes instability/problems.

We want to identify the ACTUAL failure modes and constrain those directly,
not use arbitrary direction cones.
"""

import numpy as np
from scipy.optimize import minimize, Bounds, linprog
import cvxpy as cp
import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from ADCS.helpers.math_helpers import skewsym, normalize, rot_mat, quat_inv, quat_mult


def setup_test_system():
    """Create test system for experiments."""
    # 3MTQ + 1RW (underactuated)
    A_rw = np.array([[0], [0], [1.0]])
    b = np.array([25e-6, 15e-6, 10e-6])
    A_mtq = -skewsym(b) @ np.eye(3)
    A = np.hstack([A_rw, A_mtq])
    
    u_rw_max = np.array([0.001])
    u_mtq_max = np.array([0.2, 0.2, 0.2])
    lb = np.concatenate([-u_rw_max, -u_mtq_max])
    ub = np.concatenate([u_rw_max, u_mtq_max])
    
    return A, lb, ub, A_rw, A_mtq


def allocate_lp(tau_des, A, lb, ub):
    """Baseline LP allocation."""
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-12:
        return np.zeros(len(lb))
    tau_hat = tau_des / t_mag
    n = len(lb)
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


def analyze_failure_modes():
    """Identify when and why QP fails."""
    print("=" * 80)
    print("FAILURE MODE ANALYSIS: When does perpendicular torque cause problems?")
    print("=" * 80)
    
    print("""
FAILURE MODE 1: Ignoring critical axes
--------------------------------------
When τ_des has components on hard-to-achieve axes, QP may ignore them
and put all effort into easy axes.

Example: τ_des = [10, 10, 10], system can easily do z but not x,y
QP: [0, 0, 100] - x,y errors grow indefinitely

Detection: Check if any axis with significant τ_des gets near-zero τ


FAILURE MODE 2: Wrong sign on some axis
---------------------------------------
QP might produce torque in OPPOSITE direction on some axis to minimize
total error (if it can't achieve positive values on all axes).

Example: τ_des = [10, 10, 10], but achievable region has τ_x < 0
QP might give [-5, 0, 15] to minimize ||τ - τ_des||

Detection: Check sign(τ_i) != sign(τ_des_i) for significant components


FAILURE MODE 3: Overshooting easy axes
--------------------------------------
QP might overshoot the desired value on easy axes while undershooting
on hard axes, causing oscillation.

Example: τ_des = [1, 1, 1], system can do 1000 on z
QP might give [0.5, 0.5, 1.5] - z is overshooting

Detection: |τ_i| > |τ_des_i| while other axes are undershoot


FAILURE MODE 4: Energy injection
--------------------------------
If τ has component along ω that's larger than τ_des, it might inject
energy into the system rather than remove it.

Damping term in τ_des: -K_d * ω (opposes motion)
If τ · ω > τ_des · ω, we're adding energy

Detection: ω · τ > ω · τ_des + tolerance


FAILURE MODE 5: Lyapunov function increase
------------------------------------------
The perpendicular component might cause V̇ > 0.

For V = (1/2)e'Ke + (1/2)ω'Jω:
V̇ depends on τ in a specific way.

Detection: Compute V̇ with τ vs V̇ with τ_des, flag if worse


FAILURE MODE 6: Angular momentum growth
---------------------------------------
Perpendicular torque might accumulate angular momentum in wrong direction.

If τ_perp consistently points one way, ∫τ_perp dt grows.

Detection: Track τ_perp direction, flag if consistent


FAILURE MODE 7: Excitation of flexible modes
--------------------------------------------
Perpendicular high-frequency torque might excite structural modes.

Detection: High-pass filter τ_perp, check magnitude


FAILURE MODE 8: Gyroscopic coupling amplification
-------------------------------------------------
Perpendicular torque might amplify gyroscopic precession instead of
canceling it.

Detection: Compare τ_perp direction to ω × (Jω + h) direction
""")
    
    return


def option_1_no_sign_flip():
    """OPTION 1: No sign flip constraint."""
    print("\n" + "=" * 80)
    print("OPTION 1: No Sign Flip Constraint")
    print("=" * 80)
    
    print("""
Constraint: sign(τ_i) = sign(τ_des_i) for all i where |τ_des_i| > threshold

Implementation:
    For each axis i with τ_des_i > ε:   τ_i ≥ 0
    For each axis i with τ_des_i < -ε:  τ_i ≤ 0

This prevents the allocator from "helping" one axis by hurting another.
""")
    
    A, lb, ub, _, _ = setup_test_system()
    tau_des = np.array([1e-5, 1e-5, 1e-5])
    
    n = len(lb)
    u = cp.Variable(n)
    tau = A @ u
    
    objective = cp.Minimize(cp.sum_squares(tau - tau_des))
    
    # Sign constraints
    eps = 1e-10
    constraints = [u >= lb, u <= ub]
    for i in range(3):
        if tau_des[i] > eps:
            constraints.append(tau[i] >= 0)
        elif tau_des[i] < -eps:
            constraints.append(tau[i] <= 0)
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    if u.value is not None:
        tau_result = A @ u.value
        print(f"τ_des = {tau_des * 1e6} μNm")
        print(f"τ_qp  = {tau_result * 1e6} μNm")
        print(f"All signs match: {all(np.sign(tau_result[i]) == np.sign(tau_des[i]) for i in range(3) if abs(tau_des[i]) > eps)}")
    
    return


def option_2_proportionality_bounds():
    """OPTION 2: Proportionality bounds."""
    print("\n" + "=" * 80)
    print("OPTION 2: Proportionality Bounds")
    print("=" * 80)
    
    print("""
Constraint: All achieved ratios τ_i/τ_des_i within factor k of each other

If one axis achieves 50%, all must achieve between 50%/k and 50%*k.

Implementation:
    Let r_i = τ_i / τ_des_i
    max(r) ≤ k * min(r)  for k ~ 2-3

This prevents "all-in on one axis" behavior.
""")
    
    A, lb, ub, _, _ = setup_test_system()
    tau_des = np.array([1e-5, 1e-5, 1e-5])
    
    n = len(lb)
    k = 2.0  # Proportionality factor
    
    # This is tricky with CVXPY due to division
    # Reformulate: τ_i * τ_des_j ≤ k * τ_j * τ_des_i for all i,j
    # For same-sign, this becomes bilinear - need to linearize
    
    # Simpler: use auxiliary variable α for minimum ratio
    # τ_i ≥ α * τ_des_i for all i (if τ_des_i > 0)
    # τ_i ≤ k * α * τ_des_i for all i
    
    u = cp.Variable(n)
    alpha = cp.Variable(nonneg=True)
    tau = A @ u
    
    objective = cp.Maximize(alpha)  # Maximize the minimum ratio
    
    constraints = [u >= lb, u <= ub]
    eps = 1e-12
    for i in range(3):
        if abs(tau_des[i]) > eps:
            # τ_i between α*τ_des_i and k*α*τ_des_i
            if tau_des[i] > 0:
                constraints.append(tau[i] >= alpha * tau_des[i])
                constraints.append(tau[i] <= k * alpha * tau_des[i])
            else:
                constraints.append(tau[i] <= alpha * tau_des[i])
                constraints.append(tau[i] >= k * alpha * tau_des[i])
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    if u.value is not None:
        tau_result = A @ u.value
        print(f"τ_des = {tau_des * 1e6} μNm")
        print(f"τ_qp  = {tau_result * 1e6} μNm")
        print(f"α (min ratio) = {alpha.value:.4f}")
        ratios = tau_result / tau_des
        print(f"Ratios: {ratios}")
        print(f"Max/min ratio: {max(ratios)/min(ratios):.2f} (should be ≤ {k})")
    
    return


def option_3_energy_bound():
    """OPTION 3: Energy injection bound."""
    print("\n" + "=" * 80)
    print("OPTION 3: Energy Injection Bound")
    print("=" * 80)
    
    print("""
Constraint: Don't inject more energy than τ_des would

Energy injection rate: P = τ · ω
Desired: P_des = τ_des · ω (should be negative for damping)

Constraint: τ · ω ≤ τ_des · ω + margin

This directly addresses the stability concern.
""")
    
    A, lb, ub, _, _ = setup_test_system()
    tau_des = np.array([1e-5, 1e-5, 1e-5])
    omega = np.array([0.01, -0.015, 0.005])  # Current angular velocity
    
    n = len(lb)
    u = cp.Variable(n)
    tau = A @ u
    
    # Minimize error while constraining energy injection
    objective = cp.Minimize(cp.sum_squares(tau - tau_des))
    
    P_des = tau_des @ omega
    margin = 0.1 * abs(P_des) if abs(P_des) > 1e-15 else 1e-10
    
    constraints = [
        u >= lb, 
        u <= ub,
        tau @ omega <= P_des + margin  # Don't inject more energy
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    if u.value is not None:
        tau_result = A @ u.value
        print(f"ω = {omega}")
        print(f"τ_des = {tau_des * 1e6} μNm")
        print(f"τ_qp  = {tau_result * 1e6} μNm")
        print(f"P_des = τ_des · ω = {P_des * 1e9:.4f} nW")
        print(f"P_qp  = τ_qp · ω  = {(tau_result @ omega) * 1e9:.4f} nW")
    
    return


def option_4_lyapunov_bound():
    """OPTION 4: Lyapunov derivative bound."""
    print("\n" + "=" * 80)
    print("OPTION 4: Lyapunov Derivative Bound")
    print("=" * 80)
    
    print("""
Constraint: V̇(τ) ≤ V̇(τ_des) + margin

For standard Lyapunov V = (1/2)e'Ke + (1/2)ω'Jω:
V̇ = e'K*ė + ω'J*ω̇ = ... + ω'τ (plus other terms)

The term ω'(τ - τ_des) is the difference.

Constraint: ω'τ ≤ ω'τ_des + margin

(Same as energy bound for this formulation!)
""")
    
    print("Same as Option 3 for standard Lyapunov function.")
    return


def option_5_perpendicular_magnitude_bound():
    """OPTION 5: Bound on perpendicular component magnitude."""
    print("\n" + "=" * 80)
    print("OPTION 5: Perpendicular Magnitude Bound")
    print("=" * 80)
    
    print("""
Constraint: ||τ_perp|| ≤ k * ||τ_parallel||

Where τ_parallel = (τ · τ̂_des) * τ̂_des
      τ_perp = τ - τ_parallel

This allows some perpendicular component but bounds it relative to useful torque.
""")
    
    A, lb, ub, _, _ = setup_test_system()
    tau_des = np.array([1e-5, 1e-5, 1e-5])
    tau_hat = tau_des / np.linalg.norm(tau_des)
    
    n = len(lb)
    k = 0.5  # Allow perpendicular up to 50% of parallel
    
    u = cp.Variable(n)
    tau = A @ u
    
    # τ_parallel_mag = τ · τ̂
    tau_parallel_mag = tau @ tau_hat
    
    # τ_perp = τ - τ_parallel_mag * τ̂
    tau_perp = tau - tau_parallel_mag * tau_hat
    
    objective = cp.Maximize(tau_parallel_mag)
    
    constraints = [
        u >= lb, 
        u <= ub,
        cp.norm(tau_perp) <= k * tau_parallel_mag,
        tau_parallel_mag >= 0
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    if u.value is not None:
        tau_result = A @ u.value
        proj = tau_result @ tau_hat
        tau_perp_result = tau_result - proj * tau_hat
        
        print(f"τ_des = {tau_des * 1e6} μNm")
        print(f"τ_qp  = {tau_result * 1e6} μNm")
        print(f"τ_parallel = {proj * 1e6:.4f} μNm")
        print(f"||τ_perp|| = {np.linalg.norm(tau_perp_result) * 1e6:.4f} μNm")
        print(f"Ratio: {np.linalg.norm(tau_perp_result) / proj:.2f} (should be ≤ {k})")
    
    return


def option_6_component_wise_error_bound():
    """OPTION 6: Per-axis error bounds."""
    print("\n" + "=" * 80)
    print("OPTION 6: Component-wise Error Bounds")
    print("=" * 80)
    
    print("""
Constraint: |τ_i - τ_des_i| ≤ max(ε, β * |τ_des_i|) for each i

This bounds the error on each axis proportionally to the desired value.
Prevents massive error on one axis to reduce another.
""")
    
    A, lb, ub, _, _ = setup_test_system()
    tau_des = np.array([1e-5, 1e-5, 1e-5])
    
    n = len(lb)
    beta = 0.5  # Allow 50% error per axis
    eps = 1e-7  # Minimum absolute tolerance
    
    u = cp.Variable(n)
    tau = A @ u
    
    objective = cp.Minimize(cp.sum_squares(tau - tau_des))
    
    constraints = [u >= lb, u <= ub]
    for i in range(3):
        bound = max(eps, beta * abs(tau_des[i]))
        constraints.append(tau[i] - tau_des[i] <= bound)
        constraints.append(tau[i] - tau_des[i] >= -bound)
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    if u.value is not None:
        tau_result = A @ u.value
        print(f"τ_des = {tau_des * 1e6} μNm")
        print(f"τ_qp  = {tau_result * 1e6} μNm")
        print(f"Per-axis errors: {(tau_result - tau_des) / tau_des * 100}%")
    
    return


def option_7_projection_dominance():
    """OPTION 7: Projection dominance (LP+QP)."""
    print("\n" + "=" * 80)
    print("OPTION 7: Projection Dominance (LP+QP)")
    print("=" * 80)
    
    print("""
Two-stage approach:
1. LP finds α_max (max projection with exact direction)
2. QP maximizes projection with constraint: τ · τ̂ ≥ α_max

This guarantees at least LP performance while allowing QP flexibility.
""")
    
    A, lb, ub, _, _ = setup_test_system()
    tau_des = np.array([1e-5, 1e-5, 1e-5])
    tau_hat = tau_des / np.linalg.norm(tau_des)
    
    # Stage 1: LP
    u_lp = allocate_lp(tau_des, A, lb, ub)
    tau_lp = A @ u_lp
    alpha_lp = tau_lp @ tau_hat
    
    # Stage 2: QP with projection constraint
    n = len(lb)
    u = cp.Variable(n)
    tau = A @ u
    
    objective = cp.Maximize(tau @ tau_hat)
    constraints = [
        u >= lb, 
        u <= ub,
        tau @ tau_hat >= alpha_lp  # At least as good as LP
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    if u.value is not None:
        tau_result = A @ u.value
        print(f"τ_des = {tau_des * 1e6} μNm")
        print(f"τ_LP  = {tau_lp * 1e6} μNm (proj = {alpha_lp * 1e6:.4f})")
        print(f"τ_QP  = {tau_result * 1e6} μNm (proj = {(tau_result @ tau_hat) * 1e6:.4f})")
    
    return


def option_8_pareto_constraint():
    """OPTION 8: Pareto improvement constraint."""
    print("\n" + "=" * 80)
    print("OPTION 8: Pareto Improvement Constraint")
    print("=" * 80)
    
    print("""
Constraint: Don't make ANY axis worse than LP while improving others.

For each axis i: τ_i should be at least as good as τ_LP_i
(considering sign - closer to τ_des_i is better)
""")
    
    A, lb, ub, _, _ = setup_test_system()
    tau_des = np.array([1e-5, 1e-5, 1e-5])
    
    # LP baseline
    u_lp = allocate_lp(tau_des, A, lb, ub)
    tau_lp = A @ u_lp
    
    n = len(lb)
    u = cp.Variable(n)
    tau = A @ u
    
    objective = cp.Minimize(cp.sum_squares(tau - tau_des))
    
    constraints = [u >= lb, u <= ub]
    # Pareto: don't make any axis error worse than LP
    for i in range(3):
        lp_error = abs(tau_lp[i] - tau_des[i])
        constraints.append(cp.abs(tau[i] - tau_des[i]) <= lp_error + 1e-12)
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    if u.value is not None:
        tau_result = A @ u.value
        print(f"τ_des = {tau_des * 1e6} μNm")
        print(f"τ_LP  = {tau_lp * 1e6} μNm")
        print(f"τ_QP  = {tau_result * 1e6} μNm")
        print(f"LP errors: {np.abs(tau_lp - tau_des) * 1e6}")
        print(f"QP errors: {np.abs(tau_result - tau_des) * 1e6}")
    
    return


def option_9_error_weighted():
    """OPTION 9: Error-state weighted allocation."""
    print("\n" + "=" * 80)
    print("OPTION 9: Error-State Weighted Allocation")
    print("=" * 80)
    
    print("""
Weight the torque error by the current attitude error magnitude.

If e_x is large, τ_x error matters more.
Minimize: Σ w_i * (τ_i - τ_des_i)²

where w_i = |e_i| / ||e|| (or based on controller gains)

This prioritizes axes that need the most correction.
""")
    
    A, lb, ub, _, _ = setup_test_system()
    tau_des = np.array([1e-5, 1e-5, 1e-5])
    
    # Attitude error (example)
    e = np.array([0.1, 0.02, 0.01])  # Large x-error
    weights = np.abs(e) / (np.linalg.norm(e) + 1e-10)
    weights = weights / np.sum(weights)
    
    n = len(lb)
    u = cp.Variable(n)
    tau = A @ u
    
    # Weighted error
    weighted_error = sum(weights[i] * cp.square(tau[i] - tau_des[i]) for i in range(3))
    objective = cp.Minimize(weighted_error)
    
    constraints = [u >= lb, u <= ub]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    if u.value is not None:
        tau_result = A @ u.value
        print(f"Attitude error e = {e} (x is large)")
        print(f"Weights: {weights}")
        print(f"τ_des = {tau_des * 1e6} μNm")
        print(f"τ_qp  = {tau_result * 1e6} μNm")
    
    return


def option_10_rate_limited():
    """OPTION 10: Rate-limited perpendicular torque."""
    print("\n" + "=" * 80)
    print("OPTION 10: Rate-Limited Perpendicular Torque")
    print("=" * 80)
    
    print("""
Constraint: ||τ_perp - τ_perp_prev|| ≤ Δτ_max

Don't let perpendicular component change too fast.
Prevents rapid switching/oscillation in the unused direction.
""")
    
    print("(Requires state from previous timestep - demonstration skipped)")
    return


def option_11_momentum_aware():
    """OPTION 11: Momentum-aware constraint."""
    print("\n" + "=" * 80)
    print("OPTION 11: Momentum-Aware Constraint")
    print("=" * 80)
    
    print("""
Constraint: Perpendicular torque shouldn't systematically build momentum.

If τ_perp consistently points one direction over time, it builds h.

Track: h_perp_accumulated = ∫ τ_perp dt
Constraint: |τ_perp · h_perp_dir| limited when h_perp large

This prevents the allocator from continuously pushing momentum one way.
""")
    
    print("(Requires integration over time - demonstration skipped)")
    return


def option_12_controllability_weighted():
    """OPTION 12: Controllability-weighted allocation."""
    print("\n" + "=" * 80)
    print("OPTION 12: Controllability-Weighted Allocation")
    print("=" * 80)
    
    print("""
Weight errors inversely by controllability.

Axes that are hard to control should have tighter constraints
because errors there are harder to fix.

Compute controllability Gramian W = ∫ e^{At} B B' e^{A't} dt
Use diagonal of W^{-1} as weights.
""")
    
    A, lb, ub, _, _ = setup_test_system()
    tau_des = np.array([1e-5, 1e-5, 1e-5])
    
    # For underactuated system, z is highly controllable, x,y less so
    # Approximate controllability weights (inverse of controllability)
    controllability = np.array([0.1, 0.1, 10.0])  # z is easy, x,y are hard
    weights = 1.0 / controllability
    weights = weights / np.sum(weights)
    
    n = len(lb)
    u = cp.Variable(n)
    tau = A @ u
    
    weighted_error = sum(weights[i] * cp.square(tau[i] - tau_des[i]) for i in range(3))
    objective = cp.Minimize(weighted_error)
    
    constraints = [u >= lb, u <= ub]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    if u.value is not None:
        tau_result = A @ u.value
        print(f"Controllability: {controllability} (z is easy)")
        print(f"Weights (inverse): {weights}")
        print(f"τ_des = {tau_des * 1e6} μNm")
        print(f"τ_qp  = {tau_result * 1e6} μNm")
    
    return


def summary():
    """Print summary of all options."""
    print("\n" + "=" * 80)
    print("SUMMARY: 12 QP CONSTRAINT OPTIONS")
    print("=" * 80)
    
    print("""
┌─────┬────────────────────────────────┬─────────────────────────────────────────┐
│  #  │ Name                           │ Key Idea                                │
├─────┼────────────────────────────────┼─────────────────────────────────────────┤
│  1  │ No Sign Flip                   │ sign(τ_i) = sign(τ_des_i)               │
│  2  │ Proportionality Bounds         │ max(ratio) ≤ k * min(ratio)             │
│  3  │ Energy Injection Bound         │ τ·ω ≤ τ_des·ω + margin                  │
│  4  │ Lyapunov Derivative Bound      │ V̇(τ) ≤ V̇(τ_des) + margin               │
│  5  │ Perpendicular Magnitude Bound  │ ||τ_perp|| ≤ k * ||τ_parallel||         │
│  6  │ Component-wise Error Bound     │ |τ_i - τ_des_i| ≤ β*|τ_des_i|          │
│  7  │ Projection Dominance (LP+QP)   │ τ·τ̂ ≥ α_LP                              │
│  8  │ Pareto Improvement             │ Don't make any axis worse than LP       │
│  9  │ Error-State Weighted           │ Weight by attitude error magnitude      │
│ 10  │ Rate-Limited Perp              │ ||Δτ_perp|| ≤ max rate                  │
│ 11  │ Momentum-Aware                 │ Limit systematic h buildup              │
│ 12  │ Controllability-Weighted       │ Weight hard-to-control axes more        │
└─────┴────────────────────────────────┴─────────────────────────────────────────┘

RECOMMENDATIONS:
- Options 1, 2, 6 prevent "ignoring axes" directly
- Options 3, 4 address stability through energy/Lyapunov
- Options 7, 8 guarantee at least LP performance
- Options 9, 12 adapt to the current state/system
- Options 10, 11 address temporal effects

HYBRID APPROACH: Combine multiple constraints:
- Start with Option 7 (projection dominance) as baseline
- Add Option 3 (energy bound) for stability guarantee
- Add Option 1 (sign constraint) to prevent reversal
""")


if __name__ == "__main__":
    np.random.seed(42)
    
    analyze_failure_modes()
    option_1_no_sign_flip()
    option_2_proportionality_bounds()
    option_3_energy_bound()
    option_4_lyapunov_bound()
    option_5_perpendicular_magnitude_bound()
    option_6_component_wise_error_bound()
    option_7_projection_dominance()
    option_8_pareto_constraint()
    option_9_error_weighted()
    option_10_rate_limited()
    option_11_momentum_aware()
    option_12_controllability_weighted()
    summary()
