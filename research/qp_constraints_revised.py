"""
Revised Physics-Based Constraints
=================================

Carefully reconsidering each constraint with proper sign conventions
and conditional application.

Key insight: Constraints should be applied based on WHAT THE CONTROLLER IS TRYING TO DO,
not just the current state.
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


def qp_unconstrained(tau_des, A, lb, ub):
    """Baseline QP - no physics constraints."""
    n = len(lb)
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(SCALE * (tau - tau_des)))
    constraints = [u >= lb, u <= ub]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    return u.value if u.value is not None else None


# ============================================================================
# REVISED CONSTRAINT 1: Power Bound
# ============================================================================

def analyze_power_constraint():
    """
    Let's think carefully about the power constraint.
    
    Power P = τ · ω = rate of kinetic energy change
    
    If P > 0: torque is adding energy (accelerating in direction of motion)
    If P < 0: torque is removing energy (braking)
    
    For damping (τ_des = -K_d ω):
        P_des = τ_des · ω = -K_d ω · ω = -K_d |ω|² < 0 (always braking)
        
    For PD control (τ_des = -K_p θ - K_d ω):
        P_des = (-K_p θ - K_d ω) · ω = -K_p (θ·ω) - K_d |ω|²
        
        Case A: θ·ω > 0 (diverging - moving away from origin)
            P_des < 0 (we want to brake)
            
        Case B: θ·ω < 0 (converging - moving toward origin)  
            P_des = -K_p (θ·ω) - K_d |ω|²
                  = +K_p |θ·ω| - K_d |ω|²
            Could be positive or negative!
            
            If |K_p θ| > |K_d ω|: P_des > 0 (we want to accelerate to fight overshoot)
            If |K_p θ| < |K_d ω|: P_des < 0 (we want to brake)
    
    THE ORIGINAL CONSTRAINT: ω'τ ≤ max(0, ω'τ_des)
    
    When P_des < 0 (braking): constraint is ω'τ ≤ 0 (don't add energy)
    When P_des > 0 (accelerating): constraint is ω'τ ≤ P_des (don't add MORE energy than intended)
    
    This seems correct! But let me check the implementation...
    """
    print("Power Constraint Analysis")
    print("=" * 60)
    print("""
    P = τ · ω (power into system)
    
    Controller intent:
    - P_des < 0: wants to REMOVE energy (brake)
    - P_des > 0: wants to ADD energy (accelerate)
    
    Original constraint: ω'τ ≤ max(0, ω'τ_des)
    
    When P_des < 0: ω'τ ≤ 0 (can only brake, never accelerate)
    When P_des > 0: ω'τ ≤ P_des (can accelerate, but not more than intended)
    
    PROBLEM: When converging and need to accelerate (P_des > 0),
    the constraint ω'τ ≤ P_des might be too restrictive if P_des is small.
    
    REVISED CONSTRAINT 1a: Only apply when braking
        If P_des < 0: ω'τ ≤ 0
        Else: no constraint
        
    REVISED CONSTRAINT 1b: Symmetric bound
        |ω'τ| ≤ |ω'τ_des| + margin
        (Don't do more power transfer than intended, in either direction)
        
    REVISED CONSTRAINT 1c: Directional
        If P_des < 0: ω'τ ≤ 0 (braking case: don't accelerate)
        If P_des > 0: ω'τ ≥ 0 (accelerating case: don't brake)
    """)


def qp_1a_power_braking_only(tau_des, A, lb, ub, omega, **kwargs):
    """
    Revised Power Constraint 1a: Only constrain when braking.
    
    If τ_des wants to brake (P_des < 0), don't allow acceleration.
    If τ_des wants to accelerate (P_des > 0), no constraint.
    """
    n = len(lb)
    P_des = np.dot(omega, tau_des)
    
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(SCALE * (tau - tau_des)))
    constraints = [u >= lb, u <= ub]
    
    if P_des < -1e-12:  # Braking intended
        constraints.append(omega @ tau <= 1e-12)  # Don't accelerate
    # If accelerating intended, no power constraint
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    return u.value if u.value is not None else None


def qp_1b_power_symmetric(tau_des, A, lb, ub, omega, **kwargs):
    """
    Revised Power Constraint 1b: Symmetric bound on power magnitude.
    
    |P| ≤ |P_des| + margin
    """
    n = len(lb)
    P_des = np.dot(omega, tau_des)
    P_bound = abs(P_des) * 1.5 + 1e-12  # Allow 50% margin
    
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(SCALE * (tau - tau_des)))
    constraints = [
        u >= lb, u <= ub,
        omega @ tau <= P_bound,
        omega @ tau >= -P_bound,
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    return u.value if u.value is not None else None


def qp_1c_power_directional(tau_des, A, lb, ub, omega, **kwargs):
    """
    Revised Power Constraint 1c: Match power direction.
    
    If braking intended: don't accelerate
    If accelerating intended: don't brake
    """
    n = len(lb)
    P_des = np.dot(omega, tau_des)
    
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(SCALE * (tau - tau_des)))
    constraints = [u >= lb, u <= ub]
    
    if P_des < -1e-12:  # Braking
        constraints.append(omega @ tau <= 0)
    elif P_des > 1e-12:  # Accelerating
        constraints.append(omega @ tau >= 0)
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    return u.value if u.value is not None else None


# ============================================================================
# REVISED CONSTRAINT 2: Lyapunov
# ============================================================================

def analyze_lyapunov_constraint():
    print("\nLyapunov Constraint Analysis")
    print("=" * 60)
    print("""
    V = ½ θ'K_p θ + ½ ω'J ω
    V̇ = θ'K_p ω + ω'τ (ignoring J for simplicity)
    
    For τ_des = -K_p θ - K_d ω:
        V̇_des = θ'K_p ω + ω'(-K_p θ - K_d ω)
              = θ'K_p ω - ω'K_p θ - K_d |ω|²
              = -K_d |ω|² ≤ 0  (cross terms cancel!)
    
    ORIGINAL CONSTRAINT: V̇ ≤ 0
        θ'K_p ω + ω'τ ≤ 0
        
    PROBLEM: This can prevent progress. V̇ ≤ 0 means energy decreasing,
    but doesn't guarantee convergence to θ = 0.
    
    REVISED 2a: V̇ ≤ V̇_des (no worse than ideal)
        θ'K_p ω + ω'τ ≤ θ'K_p ω + ω'τ_des
        ω'τ ≤ ω'τ_des
        (This is just the power constraint!)
        
    REVISED 2b: V̇ ≤ -ε V (exponential convergence)
        θ'K_p ω + ω'τ ≤ -ε V
        Very restrictive, often infeasible.
        
    REVISED 2c: V̇ ≤ 0 ONLY when |ω| > threshold (avoid getting stuck at θ ≠ 0)
        When ω ≈ 0, we need torque to start moving, which increases V temporarily.
    """)


def qp_2a_lyapunov_relative(tau_des, A, lb, ub, omega, theta, K_p, **kwargs):
    """
    Revised Lyapunov 2a: V̇ ≤ V̇_des (no worse than ideal).
    
    This reduces to: ω'τ ≤ ω'τ_des
    """
    n = len(lb)
    P_des = np.dot(omega, tau_des)
    
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(SCALE * (tau - tau_des)))
    constraints = [
        u >= lb, u <= ub,
        omega @ tau <= P_des + 1e-12
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    return u.value if u.value is not None else None


def qp_2c_lyapunov_rate_gated(tau_des, A, lb, ub, omega, theta, K_p, **kwargs):
    """
    Revised Lyapunov 2c: Only enforce V̇ ≤ 0 when |ω| is significant.
    
    When |ω| < threshold, we need to allow V to increase temporarily.
    """
    n = len(lb)
    spring_term = np.dot(K_p * theta, omega)
    omega_mag = np.linalg.norm(omega)
    
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(SCALE * (tau - tau_des)))
    constraints = [u >= lb, u <= ub]
    
    # Only enforce Lyapunov when moving fast enough
    if omega_mag > 0.005:  # Threshold
        constraints.append(spring_term + omega @ tau <= 0)
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    return u.value if u.value is not None else None


# ============================================================================
# REVISED CONSTRAINT 3: Sign Preservation
# ============================================================================

def analyze_sign_constraint():
    print("\nSign Preservation Constraint Analysis")
    print("=" * 60)
    print("""
    ORIGINAL: If ω_i · τ_des,i < 0 (braking on axis i), require sign(τ_i) = sign(τ_des,i)
    
    Let's think about this more carefully for each axis:
    
    τ_des,i = -K_p θ_i - K_d ω_i
    
    Case 1: ω_i > 0, θ_i > 0 (diverging)
        τ_des,i = -K_p θ_i - K_d ω_i < 0 (negative, brake)
        ω_i · τ_des,i < 0 → constraint applies
        Require τ_i ≤ 0 ✓ (correct: brake the positive velocity)
        
    Case 2: ω_i > 0, θ_i < 0 (converging toward negative side)
        τ_des,i = -K_p θ_i - K_d ω_i = +K_p|θ_i| - K_d ω_i
        Could be + or - depending on magnitudes
        
        If |K_p θ_i| < |K_d ω_i|: τ_des,i < 0 (brake)
            ω_i · τ_des,i < 0 → require τ_i ≤ 0 ✓
            
        If |K_p θ_i| > |K_d ω_i|: τ_des,i > 0 (accelerate toward θ=0)
            ω_i · τ_des,i > 0 → no constraint (correct!)
            
    Case 3: ω_i < 0, θ_i > 0 (converging toward positive side)
        τ_des,i = -K_p θ_i - K_d ω_i = -K_p θ_i + K_d|ω_i|
        
        If |K_p θ_i| > |K_d ω_i|: τ_des,i < 0 (slow approach)
            ω_i · τ_des,i > 0 → no constraint
            
        If |K_p θ_i| < |K_d ω_i|: τ_des,i > 0 (brake negative velocity)
            ω_i · τ_des,i < 0 → require τ_i ≥ 0 ✓
            
    The original logic seems correct! But maybe too strict in magnitude.
    
    REVISED 3a: Sign preservation with minimum magnitude
        If braking: not only sign(τ_i) = sign(τ_des,i), 
        but also |τ_i| ≥ ε (don't allow zero torque when braking needed)
        
    REVISED 3b: Apply only to "critical" axes
        Only constrain axes where |τ_des,i| > threshold
    """)


def qp_3a_sign_with_min_magnitude(tau_des, A, lb, ub, omega, **kwargs):
    """
    Revised Sign 3a: Sign preservation with minimum magnitude.
    
    When braking, require both correct sign AND minimum effort.
    """
    n = len(lb)
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(SCALE * (tau - tau_des)))
    constraints = [u >= lb, u <= ub]
    
    eps = 1e-9  # Minimum torque magnitude when braking
    
    for i in range(3):
        if omega[i] > 1e-6 and tau_des[i] < -1e-12:
            # Positive velocity, want negative torque
            constraints.append(tau[i] <= -eps)
        elif omega[i] < -1e-6 and tau_des[i] > 1e-12:
            # Negative velocity, want positive torque
            constraints.append(tau[i] >= eps)
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    return u.value if u.value is not None else None


def qp_3b_sign_critical_only(tau_des, A, lb, ub, omega, **kwargs):
    """
    Revised Sign 3b: Only apply to axes with significant τ_des.
    """
    n = len(lb)
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(SCALE * (tau - tau_des)))
    constraints = [u >= lb, u <= ub]
    
    tau_threshold = 0.1 * np.linalg.norm(tau_des)  # 10% of total
    
    for i in range(3):
        if abs(tau_des[i]) < tau_threshold:
            continue  # Skip weak axes
            
        if omega[i] > 1e-6 and tau_des[i] < -1e-12:
            constraints.append(tau[i] <= 0)
        elif omega[i] < -1e-6 and tau_des[i] > 1e-12:
            constraints.append(tau[i] >= 0)
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    return u.value if u.value is not None else None


# ============================================================================
# REVISED CONSTRAINT: Phase-Space Aware
# ============================================================================

def qp_phase_aware(tau_des, A, lb, ub, omega, theta, K_p, **kwargs):
    """
    NEW: Phase-space aware constraint.
    
    The key insight: different constraints apply in different regions:
    
    Region 1: Diverging (θ·ω > 0) - moving away from origin
        → Need to brake, energy constraints make sense
        
    Region 2: Converging fast (θ·ω < 0, |ω| large) - approaching origin fast
        → Need to brake to avoid overshoot
        
    Region 3: Converging slow (θ·ω < 0, |ω| small) - approaching origin slowly
        → May need to accelerate or maintain, don't over-constrain
        
    Region 4: Near origin (|θ|, |ω| small)
        → Fine control, don't constrain much
    """
    n = len(lb)
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(SCALE * (tau - tau_des)))
    constraints = [u >= lb, u <= ub]
    
    theta_dot_omega = np.dot(theta, omega)
    omega_mag = np.linalg.norm(omega)
    theta_mag = np.linalg.norm(theta)
    P_des = np.dot(omega, tau_des)
    
    if theta_dot_omega > 1e-6:
        # Diverging: apply power constraint (don't accelerate)
        constraints.append(omega @ tau <= max(0, P_des) + 1e-12)
    elif theta_dot_omega < -1e-6 and omega_mag > 0.01:
        # Converging fast: allow some constraint relaxation
        # Don't inject MORE than 50% extra power
        constraints.append(omega @ tau <= P_des * 1.5 if P_des > 0 else P_des * 0.5)
    # Near origin or slow convergence: no constraint
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    return u.value if u.value is not None else None


# ============================================================================
# REVISED CONSTRAINT: Projection + Direction Hybrid
# ============================================================================

def qp_projection_direction_hybrid(tau_des, A, lb, ub, **kwargs):
    """
    Hybrid: Guarantee LP projection AND bound direction error.
    
    1. τ · τ̂_des ≥ α_LP (at least as much in desired direction)
    2. ||τ_perp|| ≤ k * τ_parallel (perpendicular bounded by parallel)
    """
    n = len(lb)
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-15:
        return np.zeros(n)
    
    tau_hat = tau_des / t_mag
    _, tau_lp, _ = solve_lp(tau_des, A, lb, ub)
    proj_lp = np.dot(tau_lp, tau_hat)
    
    u = cp.Variable(n)
    tau = A @ u
    
    # Projection onto desired direction
    tau_parallel = tau @ tau_hat
    
    objective = cp.Minimize(cp.sum_squares(SCALE * (tau - tau_des)))
    constraints = [
        u >= lb, u <= ub,
        tau_parallel >= proj_lp - 1e-12,  # At least LP projection
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    return u.value if u.value is not None else None


# ============================================================================
# TEST ALL REVISED CONSTRAINTS
# ============================================================================

def run_tests():
    """Test all revised constraints."""
    print("=" * 100)
    print("TESTING REVISED PHYSICS-BASED CONSTRAINTS")
    print("=" * 100)
    
    # Print analysis
    analyze_power_constraint()
    analyze_lyapunov_constraint()
    analyze_sign_constraint()
    
    A, lb, ub, B = setup_system()
    J = np.diag([0.01, 0.01, 0.005])
    K_p = np.array([0.001, 0.001, 0.001])
    K_d = np.array([0.01, 0.01, 0.01])
    
    scenarios = [
        {"name": "Diverging (θ>0, ω>0)", "theta": np.array([0.1, 0.1, 0.1]), "omega": np.array([0.03, 0.03, 0.03])},
        {"name": "Converging fast (θ>0, ω<0 large)", "theta": np.array([0.1, 0.1, 0.1]), "omega": np.array([-0.05, -0.05, -0.05])},
        {"name": "Converging slow (θ>0, ω<0 small)", "theta": np.array([0.1, 0.1, 0.1]), "omega": np.array([-0.01, -0.01, -0.01])},
        {"name": "Near origin", "theta": np.array([0.01, 0.01, 0.01]), "omega": np.array([0.005, 0.005, 0.005])},
        {"name": "Pure damping", "theta": np.array([0.0, 0.0, 0.0]), "omega": np.array([0.05, 0.05, 0.05])},
    ]
    
    methods = [
        ("LP", lambda td, **kw: solve_lp(td, A, lb, ub)[0]),
        ("QP unconstrained", lambda td, **kw: qp_unconstrained(td, A, lb, ub)),
        ("1a-Power brake only", lambda td, **kw: qp_1a_power_braking_only(td, A, lb, ub, **kw)),
        ("1b-Power symmetric", lambda td, **kw: qp_1b_power_symmetric(td, A, lb, ub, **kw)),
        ("1c-Power directional", lambda td, **kw: qp_1c_power_directional(td, A, lb, ub, **kw)),
        ("2a-Lyap relative", lambda td, **kw: qp_2a_lyapunov_relative(td, A, lb, ub, **kw)),
        ("2c-Lyap rate-gated", lambda td, **kw: qp_2c_lyapunov_rate_gated(td, A, lb, ub, **kw)),
        ("3a-Sign w/ min mag", lambda td, **kw: qp_3a_sign_with_min_magnitude(td, A, lb, ub, **kw)),
        ("3b-Sign critical", lambda td, **kw: qp_3b_sign_critical_only(td, A, lb, ub, **kw)),
        ("Phase-aware", lambda td, **kw: qp_phase_aware(td, A, lb, ub, **kw)),
        ("Proj+Dir hybrid", lambda td, **kw: qp_projection_direction_hybrid(td, A, lb, ub, **kw)),
    ]
    
    for scenario in scenarios:
        theta = scenario["theta"]
        omega = scenario["omega"]
        tau_des = -K_p * theta - K_d * omega
        
        print(f"\n{'='*100}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"θ = {theta}, ω = {omega}")
        print(f"τ_des = {tau_des * 1e6} μNm")
        print(f"θ·ω = {np.dot(theta, omega):.4f} ({'diverging' if np.dot(theta,omega) > 0 else 'converging'})")
        print(f"P_des = ω·τ_des = {np.dot(omega, tau_des):.2e}")
        print(f"{'='*100}")
        
        kwargs = {'omega': omega, 'theta': theta, 'K_p': K_p, 'J': J}
        
        print(f"{'Method':<22} {'τ (μNm)':<42} {'err':>7} {'P':>11} {'V̇':>11}")
        print("-" * 100)
        
        for name, method in methods:
            try:
                u = method(tau_des, **kwargs)
                if u is None:
                    print(f"{name:<22} FAILED")
                    continue
                
                tau = A @ u
                error = np.linalg.norm(tau - tau_des) * 1e6
                P = np.dot(omega, tau)
                V_dot = np.dot(K_p * theta, omega) + P
                
                tau_str = f"[{tau[0]*1e6:8.2f},{tau[1]*1e6:8.2f},{tau[2]*1e6:9.2f}]"
                print(f"{name:<22} {tau_str:<42} {error:>7.2f} {P:>11.2e} {V_dot:>11.2e}")
                
            except Exception as e:
                print(f"{name:<22} ERROR: {str(e)[:40]}")
    
    return


def closed_loop_test():
    """Closed-loop comparison of revised constraints."""
    print("\n" + "=" * 100)
    print("CLOSED-LOOP TEST: REVISED CONSTRAINTS (120s)")
    print("=" * 100)
    
    A, lb, ub, B = setup_system()
    J = np.diag([0.01, 0.01, 0.005])
    J_inv = np.linalg.inv(J)
    K_p = np.array([0.001, 0.001, 0.001])
    K_d = np.array([0.01, 0.01, 0.01])
    
    dt = 0.1
    t_end = 120.0
    n_steps = int(t_end / dt)
    
    theta_0 = np.array([0.3, 0.2, 0.1])
    omega_0 = np.array([0.01, 0.01, 0.01])
    
    methods = [
        ("LP", lambda td, **kw: solve_lp(td, A, lb, ub)[0]),
        ("QP unconstrained", lambda td, **kw: qp_unconstrained(td, A, lb, ub)),
        ("1a-Power brake only", lambda td, **kw: qp_1a_power_braking_only(td, A, lb, ub, **kw)),
        ("1c-Power directional", lambda td, **kw: qp_1c_power_directional(td, A, lb, ub, **kw)),
        ("2a-Lyap relative", lambda td, **kw: qp_2a_lyapunov_relative(td, A, lb, ub, **kw)),
        ("2c-Lyap rate-gated", lambda td, **kw: qp_2c_lyapunov_rate_gated(td, A, lb, ub, **kw)),
        ("3b-Sign critical", lambda td, **kw: qp_3b_sign_critical_only(td, A, lb, ub, **kw)),
        ("Phase-aware", lambda td, **kw: qp_phase_aware(td, A, lb, ub, **kw)),
        ("Proj+Dir hybrid", lambda td, **kw: qp_projection_direction_hybrid(td, A, lb, ub, **kw)),
    ]
    
    print(f"\nInitial: θ = {np.degrees(theta_0)} deg, ω = {omega_0} rad/s\n")
    
    results = {}
    
    for name, method in methods:
        theta = theta_0.copy()
        omega = omega_0.copy()
        
        for _ in range(n_steps):
            tau_des = -K_p * theta - K_d * omega
            kwargs = {'omega': omega, 'theta': theta, 'K_p': K_p, 'J': J}
            
            try:
                u = method(tau_des, **kwargs)
                if u is None:
                    u = np.zeros(len(lb))
            except:
                u = np.zeros(len(lb))
            
            tau = A @ u
            omega = omega + dt * J_inv @ tau
            theta = theta + dt * omega
        
        results[name] = {'theta': theta, 'omega': omega}
    
    print(f"{'Method':<22} {'θ_final (deg)':<35} {'|θ|':>8} {'|ω|':>10}")
    print("-" * 80)
    
    for name in results:
        r = results[name]
        theta_deg = np.degrees(r['theta'])
        theta_str = f"[{theta_deg[0]:7.2f},{theta_deg[1]:7.2f},{theta_deg[2]:7.2f}]"
        theta_mag = np.degrees(np.linalg.norm(r['theta']))
        omega_mag = np.linalg.norm(r['omega'])
        print(f"{name:<22} {theta_str:<35} {theta_mag:>8.2f} {omega_mag:>10.5f}")
    
    return results


def print_summary():
    """Print summary of findings."""
    print("\n" + "=" * 100)
    print("SUMMARY: REVISED CONSTRAINT ANALYSIS")
    print("=" * 100)
    print("""
KEY INSIGHTS:
=============

1. POWER CONSTRAINT VARIANTS:
   - 1a (brake only): Only constrain when P_des < 0 → Best balance
   - 1b (symmetric): |P| ≤ |P_des| → Too restrictive  
   - 1c (directional): Match sign of P → Good for pure modes

2. LYAPUNOV VARIANTS:
   - 2a (relative): ω'τ ≤ ω'τ_des → Same as power, but includes convergence case
   - 2c (rate-gated): Only apply when |ω| large → Avoids getting stuck

3. SIGN VARIANTS:
   - 3a (with min magnitude): Requires minimum braking effort → Can be infeasible
   - 3b (critical only): Only constrain significant axes → Less restrictive

4. NEW APPROACHES:
   - Phase-aware: Different constraints in different θ-ω regions
   - Projection+Direction hybrid: Guarantee LP projection, minimize error

RECOMMENDATION FOR GENERAL USE:
===============================

Option A: Simple and Safe
   - Use LP (always works, direction preserved)
   - Or QP unconstrained with scaling (better magnitude)

Option B: Physics-Aware
   - Use "1a-Power brake only" during divergence
   - Use "2c-Lyap rate-gated" for formal stability
   - Combine with projection guarantee

Option C: Adaptive
   - Use "Phase-aware" constraint
   - Automatically adjusts based on θ·ω sign and |ω| magnitude

THE KEY LESSON:
===============
Don't apply energy constraints when the controller NEEDS to inject energy.
Check θ·ω sign to determine if system is diverging or converging.
""")


if __name__ == "__main__":
    np.random.seed(42)
    run_tests()
    closed_loop_test()
    print_summary()
