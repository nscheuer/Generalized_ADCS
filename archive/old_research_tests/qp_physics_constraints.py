"""
Physics-Based QP Constraints
============================

Deriving constraints from dynamics and energy, not just geometry.

The question: What physical principles should constrain torque allocation?

Key insight: The controller computes τ_des for a REASON - it's trying to:
1. Reduce attitude error (potential energy in the Lyapunov sense)
2. Reduce rate error (kinetic energy)
3. Follow a desired trajectory

If we can't achieve τ_des exactly, what physics guarantees do we need?
"""

import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from ADCS.helpers.math_helpers import skewsym

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


# ============================================================================
# PHYSICS ANALYSIS
# ============================================================================

def analyze_control_physics():
    """
    What is the controller actually trying to do?
    
    For a PD controller: τ_des = -K_p * θ_err - K_d * ω
    
    The Lyapunov function is typically:
        V = (1/2) θ_err' K_p θ_err + (1/2) ω' J ω
    
    Its derivative:
        V̇ = θ_err' K_p θ̇_err + ω' J ω̇
          = θ_err' K_p ω_err + ω' τ  (for small angles, θ̇ ≈ ω)
          = θ_err' K_p ω + ω' τ      (if ω_des = 0)
    
    For the desired torque τ_des = -K_p θ_err - K_d ω:
        V̇_des = θ_err' K_p ω + ω' (-K_p θ_err - K_d ω)
              = θ_err' K_p ω - ω' K_p θ_err - ω' K_d ω
              = -ω' K_d ω ≤ 0
    
    So τ_des guarantees V̇ ≤ 0 (energy dissipation).
    
    QUESTION: If we use τ ≠ τ_des, what happens to V̇?
    
        V̇_actual = θ_err' K_p ω + ω' τ
    
    For stability, we need V̇_actual ≤ 0, or at least V̇_actual ≤ V̇_des.
    """
    print("=" * 80)
    print("PHYSICS OF TORQUE ALLOCATION")
    print("=" * 80)
    
    print("""
LYAPUNOV ANALYSIS FOR PD CONTROL
================================

Standard Lyapunov function:
    V = (1/2) θ' K_p θ + (1/2) ω' J ω

Time derivative:
    V̇ = θ' K_p ω + ω' J ω̇
      = θ' K_p ω + ω' τ

For τ_des = -K_p θ - K_d ω:
    V̇_des = θ' K_p ω + ω' (-K_p θ - K_d ω)
          = θ' K_p ω - ω' K_p θ - ω' K_d ω
          = -ω' K_d ω ≤ 0  ✓

KEY INSIGHT: The cross term θ' K_p ω cancels with -ω' K_p θ!


WHAT IF τ ≠ τ_des?
==================

    V̇_actual = θ' K_p ω + ω' τ

Let τ = τ_des + δτ (allocation error):
    V̇_actual = θ' K_p ω + ω' (τ_des + δτ)
             = V̇_des + ω' δτ
             = -ω' K_d ω + ω' δτ

For stability, need:
    V̇_actual ≤ 0
    -ω' K_d ω + ω' δτ ≤ 0
    ω' δτ ≤ ω' K_d ω

PHYSICAL CONSTRAINT #1: ω' δτ ≤ ω' K_d ω
=========================================
The allocation error's power injection must not exceed the damping power.

Equivalently: ω' τ ≤ ω' τ_des + ω' K_d ω
            = ω' (-K_p θ - K_d ω) + ω' K_d ω
            = -ω' K_p θ

So: ω' τ ≤ -ω' K_p θ = -Σ K_p[i] ω[i] θ[i]

This is the MAXIMUM POWER the torque can inject while maintaining V̇ ≤ 0.
""")
    return


def analyze_component_physics():
    """
    What about per-axis constraints?
    
    The issue: QP might give τ = [1, 3, 10] when τ_des = [10, 10, 10].
    This has lower L2 error but completely different axis distribution.
    
    Does this matter physically?
    """
    print("""
PER-AXIS PHYSICS
================

Consider the attitude dynamics on each axis (linearized):
    J_i θ̈_i = τ_i

For axis i, the "local Lyapunov" is:
    V_i = (1/2) K_p θ_i² + (1/2) J_i ω_i²
    
    V̇_i = K_p θ_i ω_i + J_i ω_i (τ_i / J_i)
        = K_p θ_i ω_i + ω_i τ_i
        = ω_i (K_p θ_i + τ_i)

For τ_des,i = -K_p θ_i - K_d ω_i:
    V̇_i,des = ω_i (K_p θ_i - K_p θ_i - K_d ω_i) = -K_d ω_i² ≤ 0

If τ_i ≠ τ_des,i:
    V̇_i = ω_i (K_p θ_i + τ_i)
    
For V̇_i ≤ 0, need:
    ω_i (K_p θ_i + τ_i) ≤ 0

Case 1: ω_i > 0
    K_p θ_i + τ_i ≤ 0
    τ_i ≤ -K_p θ_i

Case 2: ω_i < 0  
    K_p θ_i + τ_i ≥ 0
    τ_i ≥ -K_p θ_i

Combined: sign(τ_i + K_p θ_i) = -sign(ω_i)
         Or equivalently: (τ_i + K_p θ_i) * ω_i ≤ 0

PHYSICAL CONSTRAINT #2 (per-axis): (τ_i + K_p θ_i) ω_i ≤ 0
==========================================================
Each axis must not inject energy faster than it dissipates.

Note: If τ_des,i = -K_p θ_i - K_d ω_i, then:
    τ_des,i + K_p θ_i = -K_d ω_i
    (τ_des,i + K_p θ_i) ω_i = -K_d ω_i² ≤ 0 ✓


ALTERNATIVE FORMULATION
=======================
Since τ_des = -K_p θ - K_d ω, we have K_p θ = -τ_des - K_d ω

The constraint (τ + K_p θ) · ω ≤ 0 becomes:
    (τ - τ_des - K_d ω) · ω ≤ 0
    τ · ω - τ_des · ω - K_d ||ω||² ≤ 0
    τ · ω ≤ τ_des · ω + K_d ||ω||²

But τ_des · ω = (-K_p θ - K_d ω) · ω = -K_p θ · ω - K_d ||ω||²

So: τ · ω ≤ -K_p θ · ω - K_d ||ω||² + K_d ||ω||²
    τ · ω ≤ -K_p θ · ω = -(K_p θ) · ω

PHYSICAL CONSTRAINT #1 (recovered): τ · ω ≤ -(K_p θ) · ω
========================================================
""")
    return


def derive_passivity_constraint():
    """
    Passivity-based constraint derivation.
    """
    print("""
PASSIVITY CONSTRAINT
====================

The spacecraft attitude dynamics can be viewed as a passive system:
- Storage function: V = (1/2) ω' J ω (kinetic energy)
- Supply rate: s = ω' τ (power input)

Passivity: V̇ ≤ s, i.e., ω' J ω̇ ≤ ω' τ

For rigid body: J ω̇ = τ - ω × (J ω) [Euler's equation]
    ω' J ω̇ = ω' τ - ω' (ω × J ω)
            = ω' τ - 0  [since ω ⊥ (ω × anything)]
            = ω' τ

So passivity is automatically satisfied: V̇ = ω' τ = s ✓

But for CONTROL, we want V̇ < 0 (energy dissipation).

DAMPING INJECTION: τ = τ_ff - K_d ω
where τ_ff might be for trajectory tracking.

Then: V̇ = ω' (τ_ff - K_d ω) = ω' τ_ff - K_d ||ω||²

For V̇ ≤ 0: ω' τ_ff ≤ K_d ||ω||²

PHYSICAL CONSTRAINT #3: ω' τ ≤ K_d ||ω||²
=========================================
Power injection bounded by damping capacity.

If τ_des already includes damping (τ_des = -K_p θ - K_d ω):
    ω' τ_des = -K_p (θ · ω) - K_d ||ω||²

The constraint becomes:
    ω' τ ≤ ω' τ_des + 2 K_d ||ω||²  [allowing some margin]

Or more conservatively:
    ω' τ ≤ ω' τ_des  [don't inject more power than intended]
""")
    return


def derive_trajectory_constraint():
    """
    What if we're tracking a trajectory, not just regulating?
    """
    print("""
TRAJECTORY TRACKING PHYSICS
===========================

For trajectory tracking, τ_des typically includes:
    τ_des = τ_ff + τ_fb
    
where:
    τ_ff = J ω̇_des + ω × (J ω)  [feedforward]
    τ_fb = -K_p θ_err - K_d ω_err  [feedback]

The Lyapunov function is:
    V = (1/2) θ_err' K_p θ_err + (1/2) ω_err' J ω_err

    V̇ = θ_err' K_p ω_err + ω_err' J ω̇_err
    
where ω̇_err = ω̇ - ω̇_des = (τ - τ_ff)/J - some terms

This gets complicated, but the key insight remains:

PHYSICAL CONSTRAINT #4: Don't fight the feedforward
===================================================
The feedforward τ_ff is computed to follow the trajectory.
The feedback τ_fb corrects errors.

If we must reduce torque, we should:
1. PRESERVE the feedforward direction (trajectory following)
2. REDUCE the feedback magnitude proportionally

Mathematically:
    τ = α τ_ff + β τ_fb
    where α ≈ 1 (preserve feedforward)
    and β ≤ 1 (scale feedback if needed)

This suggests: If τ_des = τ_ff + τ_fb and we can only achieve |τ| < |τ_des|,
prefer solutions that maintain τ_ff and reduce τ_fb.
""")
    return


# ============================================================================
# PHYSICS-BASED QP FORMULATIONS
# ============================================================================

def solve_lp(tau_des, A, lb, ub):
    """LP baseline."""
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


def qp_lyapunov_stable(tau_des, A, lb, ub, omega, theta, K_p, K_d):
    """
    QP with Lyapunov stability constraint.
    
    Constraint: V̇ = θ' K_p ω + ω' τ ≤ 0
    
    This ensures the closed-loop system is stable regardless of allocation.
    """
    n = len(lb)
    
    # Compute the "spring" term
    spring_power = np.dot(K_p * theta, omega)  # θ' K_p ω (element-wise K_p)
    
    u = cp.Variable(n)
    tau = A @ u
    tau_s = SCALE * tau
    tau_des_s = SCALE * tau_des
    
    objective = cp.Minimize(cp.sum_squares(tau_s - tau_des_s))
    constraints = [
        u >= lb, 
        u <= ub,
        # V̇ = spring_power + ω' τ ≤ 0
        spring_power + omega @ tau <= 0
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    if u.value is not None:
        return u.value, A @ u.value
    return solve_lp(tau_des, A, lb, ub)[0:2]


def qp_power_bounded(tau_des, A, lb, ub, omega, K_d):
    """
    QP with power injection bound.
    
    Constraint: ω' τ ≤ ω' τ_des + margin
    
    Don't inject more power than the controller intended.
    The margin allows for some approximation error.
    """
    n = len(lb)
    
    P_des = np.dot(omega, tau_des)
    # Allow up to 10% more power injection, or zero if τ_des is already braking
    margin = 0.1 * abs(P_des) if P_des > 0 else 0
    P_max = P_des + margin
    
    u = cp.Variable(n)
    tau = A @ u
    tau_s = SCALE * tau
    tau_des_s = SCALE * tau_des
    
    objective = cp.Minimize(cp.sum_squares(tau_s - tau_des_s))
    constraints = [
        u >= lb, 
        u <= ub,
        omega @ tau <= P_max
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    if u.value is not None:
        return u.value, A @ u.value
    return solve_lp(tau_des, A, lb, ub)[0:2]


def qp_damping_preserved(tau_des, A, lb, ub, omega, K_d):
    """
    QP that preserves damping on each axis.
    
    If ω_i > 0 and τ_des,i < 0 (braking), require τ_i ≤ 0.
    If ω_i < 0 and τ_des,i > 0 (braking), require τ_i ≥ 0.
    
    This prevents the allocator from accelerating when it should brake.
    """
    n = len(lb)
    
    u = cp.Variable(n)
    tau = A @ u
    tau_s = SCALE * tau
    tau_des_s = SCALE * tau_des
    
    objective = cp.Minimize(cp.sum_squares(tau_s - tau_des_s))
    constraints = [u >= lb, u <= ub]
    
    for i in range(3):
        # Check if this axis is in "braking" mode
        if omega[i] > 1e-6 and tau_des[i] < -1e-12:
            # Positive velocity, negative torque desired = braking
            # Require τ_i ≤ 0 (don't accelerate)
            constraints.append(tau[i] <= 0)
        elif omega[i] < -1e-6 and tau_des[i] > 1e-12:
            # Negative velocity, positive torque desired = braking
            # Require τ_i ≥ 0 (don't accelerate)
            constraints.append(tau[i] >= 0)
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    if u.value is not None:
        return u.value, A @ u.value
    return solve_lp(tau_des, A, lb, ub)[0:2]


def qp_per_axis_lyapunov(tau_des, A, lb, ub, omega, theta, K_p):
    """
    QP with per-axis Lyapunov constraint.
    
    For each axis: (τ_i + K_p θ_i) ω_i ≤ 0
    
    This ensures each axis individually doesn't inject energy.
    More restrictive than global Lyapunov but gives better per-axis behavior.
    """
    n = len(lb)
    
    u = cp.Variable(n)
    tau = A @ u
    tau_s = SCALE * tau
    tau_des_s = SCALE * tau_des
    
    objective = cp.Minimize(cp.sum_squares(tau_s - tau_des_s))
    constraints = [u >= lb, u <= ub]
    
    for i in range(3):
        if abs(omega[i]) > 1e-6:
            # (τ_i + K_p θ_i) ω_i ≤ 0
            # If ω_i > 0: τ_i + K_p θ_i ≤ 0, so τ_i ≤ -K_p θ_i
            # If ω_i < 0: τ_i + K_p θ_i ≥ 0, so τ_i ≥ -K_p θ_i
            if omega[i] > 0:
                constraints.append(tau[i] <= -K_p[i] * theta[i])
            else:
                constraints.append(tau[i] >= -K_p[i] * theta[i])
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    if u.value is not None:
        return u.value, A @ u.value
    return solve_lp(tau_des, A, lb, ub)[0:2]


def qp_work_bounded(tau_des, A, lb, ub, omega, dt=0.1):
    """
    QP with work (energy) bound over the timestep.
    
    Work = τ · Δθ ≈ τ · (ω · dt)
    
    Constraint: τ · ω ≤ τ_des · ω
    
    Don't do more work than intended.
    """
    n = len(lb)
    
    W_des = np.dot(tau_des, omega) * dt
    
    u = cp.Variable(n)
    tau = A @ u
    tau_s = SCALE * tau
    tau_des_s = SCALE * tau_des
    
    objective = cp.Minimize(cp.sum_squares(tau_s - tau_des_s))
    constraints = [
        u >= lb, 
        u <= ub,
        tau @ omega * dt <= W_des + 1e-12  # Don't exceed intended work
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    if u.value is not None:
        return u.value, A @ u.value
    return solve_lp(tau_des, A, lb, ub)[0:2]


def qp_angular_momentum_rate(tau_des, A, lb, ub, omega, J):
    """
    QP that bounds the angular momentum rate error.
    
    ḣ = τ (in body frame, ignoring external torques)
    
    We want ḣ ≈ ḣ_des = τ_des
    
    Constraint: ||ḣ - ḣ_des||_J^{-1} ≤ tolerance
    
    This weights the momentum error by the inverse inertia,
    so axes with small inertia (fast response) are weighted more.
    """
    n = len(lb)
    
    J_inv = np.linalg.inv(J)
    
    u = cp.Variable(n)
    tau = A @ u
    tau_s = SCALE * tau
    tau_des_s = SCALE * tau_des
    
    # Weight by J^{-1}: error in ω̇ space
    # ||J^{-1}(τ - τ_des)|| = ||ω̇ - ω̇_des||
    omega_dot_err = J_inv @ (tau - tau_des)
    
    objective = cp.Minimize(cp.sum_squares(SCALE * omega_dot_err))
    constraints = [u >= lb, u <= ub]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    if u.value is not None:
        return u.value, A @ u.value
    return solve_lp(tau_des, A, lb, ub)[0:2]


# ============================================================================
# COMPREHENSIVE TESTS
# ============================================================================

def test_physics_constraints():
    """Test all physics-based constraints."""
    print("\n" + "=" * 100)
    print("TESTING PHYSICS-BASED QP CONSTRAINTS")
    print("=" * 100)
    
    A, lb, ub, B = setup_system()
    
    # Control parameters
    J = np.diag([0.01, 0.01, 0.005])
    K_p = np.array([0.001, 0.001, 0.001])
    K_d = np.array([0.01, 0.01, 0.01])
    
    # Test scenarios
    scenarios = [
        {
            "name": "Pure damping (ω > 0, θ = 0)",
            "theta": np.array([0.0, 0.0, 0.0]),
            "omega": np.array([0.05, 0.05, 0.05]),
        },
        {
            "name": "Regulation (θ > 0, ω = 0)", 
            "theta": np.array([0.1, 0.1, 0.1]),
            "omega": np.array([0.0, 0.0, 0.0]),
        },
        {
            "name": "Mixed (θ > 0, ω > 0, same sign)",
            "theta": np.array([0.1, 0.1, 0.1]),
            "omega": np.array([0.02, 0.02, 0.02]),
        },
        {
            "name": "Converging (θ > 0, ω < 0)",
            "theta": np.array([0.1, 0.1, 0.1]),
            "omega": np.array([-0.02, -0.02, -0.02]),
        },
        {
            "name": "Asymmetric",
            "theta": np.array([0.2, 0.05, 0.01]),
            "omega": np.array([0.01, 0.03, 0.05]),
        },
    ]
    
    methods = [
        ("LP", lambda td, th, om: solve_lp(td, A, lb, ub)[0:2]),
        ("QP (unconstrained)", lambda td, th, om: (
            qp_power_bounded(td, A, lb, ub, np.zeros(3), K_d)  # No power constraint = unconstrained
        )),
        ("QP Lyapunov", lambda td, th, om: qp_lyapunov_stable(td, A, lb, ub, om, th, K_p, K_d)),
        ("QP Power Bounded", lambda td, th, om: qp_power_bounded(td, A, lb, ub, om, K_d)),
        ("QP Damping Preserved", lambda td, th, om: qp_damping_preserved(td, A, lb, ub, om, K_d)),
        ("QP Per-Axis Lyap", lambda td, th, om: qp_per_axis_lyapunov(td, A, lb, ub, om, th, K_p)),
        ("QP Work Bounded", lambda td, th, om: qp_work_bounded(td, A, lb, ub, om)),
        ("QP ω̇ Weighted", lambda td, th, om: qp_angular_momentum_rate(td, A, lb, ub, om, J)),
    ]
    
    for scenario in scenarios:
        theta = scenario["theta"]
        omega = scenario["omega"]
        tau_des = -K_p * theta - K_d * omega
        
        print(f"\n{'='*100}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"θ = {theta}")
        print(f"ω = {omega}")
        print(f"τ_des = {tau_des * 1e6} μNm")
        print(f"{'='*100}")
        
        # Compute physics quantities for τ_des
        V_dot_des = np.dot(K_p * theta, omega) + np.dot(omega, tau_des)
        P_des = np.dot(omega, tau_des)
        
        print(f"\nPhysics for τ_des:")
        print(f"  V̇_des = θ'K_pω + ω'τ = {V_dot_des:.6e} (should be ≤ 0)")
        print(f"  P_des = ω'τ = {P_des:.6e} W")
        print()
        
        print(f"{'Method':<20} {'τ (μNm)':<40} {'error':>8} {'V̇':>12} {'P':>12} {'Stable?':>8}")
        print("-" * 110)
        
        for name, method in methods:
            try:
                u, tau = method(tau_des, theta, omega)
                
                error = np.linalg.norm(tau - tau_des) * 1e6
                V_dot = np.dot(K_p * theta, omega) + np.dot(omega, tau)
                P = np.dot(omega, tau)
                stable = "Yes" if V_dot <= 1e-12 else "No"
                
                tau_str = f"[{tau[0]*1e6:8.3f},{tau[1]*1e6:8.3f},{tau[2]*1e6:8.3f}]"
                print(f"{name:<20} {tau_str:<40} {error:>8.3f} {V_dot:>12.2e} {P:>12.2e} {stable:>8}")
                
            except Exception as e:
                print(f"{name:<20} ERROR: {str(e)[:50]}")
    
    return


def closed_loop_physics_test():
    """Closed-loop test with physics constraints."""
    print("\n" + "=" * 100)
    print("CLOSED-LOOP TEST: PHYSICS-BASED CONSTRAINTS")
    print("=" * 100)
    
    A, lb, ub, B = setup_system()
    
    J = np.diag([0.01, 0.01, 0.005])
    J_inv = np.linalg.inv(J)
    K_p = np.array([0.001, 0.001, 0.001])
    K_d = np.array([0.01, 0.01, 0.01])
    
    dt = 0.1
    t_end = 120.0
    n_steps = int(t_end / dt)
    
    # Initial conditions
    theta_0 = np.array([0.3, 0.2, 0.1])  # 17°, 11°, 6°
    omega_0 = np.array([0.0, 0.0, 0.0])
    
    methods = [
        ("LP", lambda td, th, om: solve_lp(td, A, lb, ub)[0]),
        ("QP Lyapunov", lambda td, th, om: qp_lyapunov_stable(td, A, lb, ub, om, th, K_p, K_d)[0]),
        ("QP Power Bounded", lambda td, th, om: qp_power_bounded(td, A, lb, ub, om, K_d)[0]),
        ("QP Damping Preserved", lambda td, th, om: qp_damping_preserved(td, A, lb, ub, om, K_d)[0]),
        ("QP Per-Axis Lyap", lambda td, th, om: qp_per_axis_lyapunov(td, A, lb, ub, om, th, K_p)[0]),
    ]
    
    print(f"\nSimulating {t_end}s regulation from θ₀ = {np.degrees(theta_0)} deg")
    print(f"PD gains: K_p = {K_p}, K_d = {K_d}")
    print()
    
    results = {}
    
    for name, method in methods:
        theta = theta_0.copy()
        omega = omega_0.copy()
        
        theta_history = [theta.copy()]
        omega_history = [omega.copy()]
        V_history = [0.5 * np.dot(K_p * theta, theta) + 0.5 * np.dot(omega, J @ omega)]
        
        for i in range(n_steps):
            tau_des = -K_p * theta - K_d * omega
            
            try:
                u = method(tau_des, theta, omega)
                if u is None:
                    u = np.zeros(len(lb))
            except:
                u = np.zeros(len(lb))
            
            tau = A @ u
            
            # Simple Euler integration (linearized attitude)
            omega_new = omega + dt * J_inv @ tau
            theta_new = theta + dt * omega
            
            theta = theta_new
            omega = omega_new
            
            theta_history.append(theta.copy())
            omega_history.append(omega.copy())
            V = 0.5 * np.dot(K_p * theta, theta) + 0.5 * np.dot(omega, J @ omega)
            V_history.append(V)
        
        results[name] = {
            'theta': np.array(theta_history),
            'omega': np.array(omega_history),
            'V': np.array(V_history),
        }
    
    print(f"{'Method':<22} {'θ_final (deg)':<30} {'|θ|':>8} {'|ω|':>10} {'V_final':>12} {'V_mono?':>8}")
    print("-" * 100)
    
    for name in results:
        theta_final = results[name]['theta'][-1]
        omega_final = results[name]['omega'][-1]
        V_final = results[name]['V'][-1]
        V_history = results[name]['V']
        
        # Check if V is monotonically decreasing
        V_mono = all(V_history[i+1] <= V_history[i] + 1e-12 for i in range(len(V_history)-1))
        
        theta_deg = np.degrees(theta_final)
        theta_str = f"[{theta_deg[0]:6.2f},{theta_deg[1]:6.2f},{theta_deg[2]:6.2f}]"
        print(f"{name:<22} {theta_str:<30} {np.degrees(np.linalg.norm(theta_final)):>8.2f} {np.linalg.norm(omega_final):>10.6f} {V_final:>12.2e} {'Yes' if V_mono else 'No':>8}")
    
    return results


def summary():
    """Final summary of physics-based constraints."""
    print("\n" + "=" * 100)
    print("SUMMARY: PHYSICS-BASED QP CONSTRAINTS")
    print("=" * 100)
    
    print("""
DERIVED CONSTRAINTS FROM PHYSICS:
=================================

1. GLOBAL LYAPUNOV STABILITY
   Constraint: V̇ = θ'K_p ω + ω'τ ≤ 0
   Meaning: Total energy (potential + kinetic) must not increase
   Use when: Need guaranteed stability regardless of allocation error

2. POWER INJECTION BOUND  
   Constraint: ω'τ ≤ ω'τ_des + margin
   Meaning: Don't inject more power than the controller intended
   Use when: Worried about energy injection causing instability

3. DAMPING PRESERVATION
   Constraint: If braking desired (ω_i · τ_des,i < 0), enforce sign(τ_i) = sign(τ_des,i)
   Meaning: Never accelerate when you should be braking
   Use when: Rate damping is critical (detumbling, settling)

4. PER-AXIS LYAPUNOV
   Constraint: (τ_i + K_p θ_i) ω_i ≤ 0 for each axis
   Meaning: Each axis individually dissipates energy
   Use when: Want axis-by-axis stability guarantees

5. WORK BOUNDED
   Constraint: ∫τ·ω dt ≤ ∫τ_des·ω dt
   Meaning: Don't do more mechanical work than intended
   Use when: Energy budget matters (thermal, power)

6. ANGULAR ACCELERATION WEIGHTED
   Objective: min ||J⁻¹(τ - τ_des)||²
   Meaning: Match angular acceleration, not torque
   Use when: Attitude dynamics response matters more than raw torque


KEY INSIGHT:
============
The "best" constraint depends on what failure mode you're protecting against:

- Fear instability? → Lyapunov constraint
- Fear energy injection? → Power bound
- Fear overshoot? → Damping preservation
- Fear axis coupling? → Per-axis Lyapunov
- Fear wasted energy? → Work bound
- Fear slow response? → ω̇ weighting


RECOMMENDED PHYSICS-BASED CONFIGURATION:
========================================

For general attitude control:
    QP + Lyapunov stability + Damping preservation

This ensures:
1. V̇ ≤ 0 (global stability)
2. Never accelerate when braking (no overshoot)
3. Minimize L2 torque error (best tracking within constraints)
""")


if __name__ == "__main__":
    np.random.seed(42)
    
    # Theory
    analyze_control_physics()
    analyze_component_physics()
    derive_passivity_constraint()
    derive_trajectory_constraint()
    
    # Tests
    test_physics_constraints()
    closed_loop_physics_test()
    
    # Summary
    summary()
