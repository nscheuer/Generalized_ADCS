"""
Test 10 Physics-Based Constraints for Torque Allocation
========================================================

Implementing and comparing all the constraint options.
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


# ============================================================================
# THE 10 CONSTRAINT IMPLEMENTATIONS
# ============================================================================

def qp_unconstrained(tau_des, A, lb, ub, **kwargs):
    """Baseline: No physics constraints, just actuator bounds."""
    n = len(lb)
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(SCALE * (tau - tau_des)))
    constraints = [u >= lb, u <= ub]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    return u.value if u.value is not None else None


def qp_1_power_bound(tau_des, A, lb, ub, omega, **kwargs):
    """
    Constraint 1: Power bound (QPC's approach)
    ω'τ ≤ max(0, ω'τ_des)
    """
    n = len(lb)
    P_des = np.dot(omega, tau_des)
    P_max = max(0, P_des) + 1e-15
    
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(SCALE * (tau - tau_des)))
    constraints = [
        u >= lb, u <= ub,
        omega @ tau <= P_max
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    return u.value if u.value is not None else None


def qp_2_global_lyapunov(tau_des, A, lb, ub, omega, theta, K_p, **kwargs):
    """
    Constraint 2: Global Lyapunov stability
    V̇ = θ'K_p ω + ω'τ ≤ 0
    """
    n = len(lb)
    spring_term = np.dot(K_p * theta, omega)
    
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(SCALE * (tau - tau_des)))
    constraints = [
        u >= lb, u <= ub,
        spring_term + omega @ tau <= 0
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    return u.value if u.value is not None else None


def qp_3_sign_preservation(tau_des, A, lb, ub, omega, **kwargs):
    """
    Constraint 3: Per-axis sign preservation (damping direction)
    If ω_i · τ_des,i < 0 (braking), require sign(τ_i) = sign(τ_des,i)
    """
    n = len(lb)
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(SCALE * (tau - tau_des)))
    constraints = [u >= lb, u <= ub]
    
    for i in range(3):
        if omega[i] > 1e-6 and tau_des[i] < -1e-12:
            # Positive velocity, negative torque desired = braking
            constraints.append(tau[i] <= 0)
        elif omega[i] < -1e-6 and tau_des[i] > 1e-12:
            # Negative velocity, positive torque desired = braking
            constraints.append(tau[i] >= 0)
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    return u.value if u.value is not None else None


def qp_4_per_axis_lyapunov(tau_des, A, lb, ub, omega, theta, K_p, **kwargs):
    """
    Constraint 4: Per-axis Lyapunov
    (τ_i + K_p θ_i) ω_i ≤ 0 for each axis
    """
    n = len(lb)
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(SCALE * (tau - tau_des)))
    constraints = [u >= lb, u <= ub]
    
    for i in range(3):
        if abs(omega[i]) > 1e-6:
            if omega[i] > 0:
                constraints.append(tau[i] <= -K_p[i] * theta[i])
            else:
                constraints.append(tau[i] >= -K_p[i] * theta[i])
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    return u.value if u.value is not None else None


def qp_5_momentum_rate_bound(tau_des, A, lb, ub, J, **kwargs):
    """
    Constraint 5: Angular momentum rate bound
    ||τ|| ≤ ||τ_des|| * k (don't change momentum faster than intended)
    """
    n = len(lb)
    k = 1.5  # Allow 50% overshoot
    tau_des_mag = np.linalg.norm(tau_des)
    
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(SCALE * (tau - tau_des)))
    constraints = [
        u >= lb, u <= ub,
        cp.norm(tau) <= k * tau_des_mag + 1e-12
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    return u.value if u.value is not None else None


def qp_6_work_budget(tau_des, A, lb, ub, omega, **kwargs):
    """
    Constraint 6: Work budget (same as power, really)
    τ·ω ≤ τ_des·ω (don't do more work than intended)
    """
    n = len(lb)
    W_des = np.dot(tau_des, omega)
    
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(SCALE * (tau - tau_des)))
    constraints = [
        u >= lb, u <= ub,
        tau @ omega <= W_des + 1e-15
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    return u.value if u.value is not None else None


def qp_7_projection_guarantee(tau_des, A, lb, ub, **kwargs):
    """
    Constraint 7: Projection guarantee (LP+QP hybrid)
    τ·τ̂_des ≥ α_LP * ||τ_des|| (at least as much projection as LP)
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
    objective = cp.Minimize(cp.sum_squares(SCALE * (tau - tau_des)))
    constraints = [
        u >= lb, u <= ub,
        tau @ tau_hat >= proj_lp - 1e-12
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    return u.value if u.value is not None else None


def qp_8_feedforward_preservation(tau_des, A, lb, ub, tau_ff, **kwargs):
    """
    Constraint 8: Feedforward preservation
    First maximize feedforward component, then minimize total error.
    τ·τ̂_ff ≥ β * ||τ_ff|| where β is maximized first.
    """
    n = len(lb)
    ff_mag = np.linalg.norm(tau_ff)
    if ff_mag < 1e-15:
        return qp_unconstrained(tau_des, A, lb, ub)
    
    tau_ff_hat = tau_ff / ff_mag
    
    # First: find max achievable feedforward
    _, tau_lp_ff, _ = solve_lp(tau_ff, A, lb, ub)
    proj_ff_max = np.dot(tau_lp_ff, tau_ff_hat)
    
    # Then: minimize error subject to achieving at least 90% of max feedforward
    beta = 0.9
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(SCALE * (tau - tau_des)))
    constraints = [
        u >= lb, u <= ub,
        tau @ tau_ff_hat >= beta * proj_ff_max - 1e-12
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    return u.value if u.value is not None else None


def qp_9_cone_constraint(tau_des, A, lb, ub, theta_max_deg=30, **kwargs):
    """
    Constraint 9: Direction cone constraint
    angle(τ, τ_des) ≤ θ_max
    """
    n = len(lb)
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-15:
        return np.zeros(n)
    
    cos_theta = np.cos(np.radians(theta_max_deg))
    
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(SCALE * (tau - tau_des)))
    constraints = [
        u >= lb, u <= ub,
        tau @ tau_des >= cos_theta * cp.norm(tau) * t_mag
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    return u.value if u.value is not None else None


def qp_10_combined_safe(tau_des, A, lb, ub, omega, **kwargs):
    """
    Constraint 10: Combined safe allocation
    - Power bound: ω'τ ≤ max(0, ω'τ_des)
    - Sign preservation per axis
    """
    n = len(lb)
    P_des = np.dot(omega, tau_des)
    P_max = max(0, P_des) + 1e-15
    
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(SCALE * (tau - tau_des)))
    constraints = [
        u >= lb, u <= ub,
        omega @ tau <= P_max
    ]
    
    # Add sign preservation
    for i in range(3):
        if omega[i] > 1e-6 and tau_des[i] < -1e-12:
            constraints.append(tau[i] <= 0)
        elif omega[i] < -1e-6 and tau_des[i] > 1e-12:
            constraints.append(tau[i] >= 0)
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    return u.value if u.value is not None else None


# ============================================================================
# TEST FRAMEWORK
# ============================================================================

def run_scenario_tests():
    """Test all constraints across various scenarios."""
    print("=" * 120)
    print("TESTING 10 PHYSICS-BASED CONSTRAINTS")
    print("=" * 120)
    
    A, lb, ub, B = setup_system()
    
    # Controller parameters
    J = np.diag([0.01, 0.01, 0.005])
    K_p = np.array([0.001, 0.001, 0.001])
    K_d = np.array([0.01, 0.01, 0.01])
    
    # Define test scenarios
    scenarios = [
        {
            "name": "Damping only (ω > 0, θ = 0)",
            "theta": np.array([0.0, 0.0, 0.0]),
            "omega": np.array([0.05, 0.05, 0.05]),
            "tau_ff": np.array([0.0, 0.0, 0.0]),  # No feedforward
        },
        {
            "name": "Regulation (θ > 0, ω ≈ 0)",
            "theta": np.array([0.1, 0.1, 0.1]),
            "omega": np.array([0.001, 0.001, 0.001]),
            "tau_ff": np.array([0.0, 0.0, 0.0]),
        },
        {
            "name": "Diverging (θ > 0, ω > 0, same sign)",
            "theta": np.array([0.1, 0.1, 0.1]),
            "omega": np.array([0.03, 0.03, 0.03]),
            "tau_ff": np.array([0.0, 0.0, 0.0]),
        },
        {
            "name": "Converging (θ > 0, ω < 0)",
            "theta": np.array([0.1, 0.1, 0.1]),
            "omega": np.array([-0.03, -0.03, -0.03]),
            "tau_ff": np.array([0.0, 0.0, 0.0]),
        },
        {
            "name": "Slew with feedforward",
            "theta": np.array([0.5, 0.3, 0.1]),  # Large error
            "omega": np.array([0.0, 0.0, 0.0]),
            "tau_ff": np.array([50e-6, 50e-6, 200e-6]),  # Trajectory torque
        },
        {
            "name": "Asymmetric state",
            "theta": np.array([0.2, 0.05, 0.01]),
            "omega": np.array([0.01, 0.04, 0.06]),
            "tau_ff": np.array([0.0, 0.0, 0.0]),
        },
    ]
    
    methods = [
        ("LP", lambda td, **kw: solve_lp(td, A, lb, ub)[0]),
        ("QP (no const)", lambda td, **kw: qp_unconstrained(td, A, lb, ub, **kw)),
        ("1-Power", lambda td, **kw: qp_1_power_bound(td, A, lb, ub, **kw)),
        ("2-Lyapunov", lambda td, **kw: qp_2_global_lyapunov(td, A, lb, ub, **kw)),
        ("3-Sign", lambda td, **kw: qp_3_sign_preservation(td, A, lb, ub, **kw)),
        ("4-PerAxis", lambda td, **kw: qp_4_per_axis_lyapunov(td, A, lb, ub, **kw)),
        ("5-MomRate", lambda td, **kw: qp_5_momentum_rate_bound(td, A, lb, ub, **kw)),
        ("6-Work", lambda td, **kw: qp_6_work_budget(td, A, lb, ub, **kw)),
        ("7-Proj", lambda td, **kw: qp_7_projection_guarantee(td, A, lb, ub, **kw)),
        ("8-FF", lambda td, **kw: qp_8_feedforward_preservation(td, A, lb, ub, **kw)),
        ("9-Cone30", lambda td, **kw: qp_9_cone_constraint(td, A, lb, ub, theta_max_deg=30, **kw)),
        ("10-Combined", lambda td, **kw: qp_10_combined_safe(td, A, lb, ub, **kw)),
    ]
    
    for scenario in scenarios:
        theta = scenario["theta"]
        omega = scenario["omega"]
        tau_ff = scenario["tau_ff"]
        
        # Compute τ_des from PD control + feedforward
        tau_fb = -K_p * theta - K_d * omega
        tau_des = tau_fb + tau_ff
        
        print(f"\n{'='*120}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"θ = {theta}, ω = {omega}")
        print(f"τ_des = {tau_des * 1e6} μNm (fb={tau_fb*1e6}, ff={tau_ff*1e6})")
        print(f"{'='*120}")
        
        # Compute reference values
        V_dot_des = np.dot(K_p * theta, omega) + np.dot(omega, tau_des)
        P_des = np.dot(omega, tau_des)
        
        print(f"Reference: V̇_des = {V_dot_des:.2e}, P_des = {P_des:.2e}")
        print()
        
        kwargs = {
            'omega': omega,
            'theta': theta,
            'K_p': K_p,
            'J': J,
            'tau_ff': tau_ff,
        }
        
        print(f"{'Method':<12} {'τ (μNm)':<42} {'err':>7} {'V̇':>11} {'P':>11} {'dir°':>6} {'Stab':>5}")
        print("-" * 105)
        
        for name, method in methods:
            try:
                u = method(tau_des, **kwargs)
                if u is None:
                    print(f"{name:<12} FAILED")
                    continue
                
                tau = A @ u
                error = np.linalg.norm(tau - tau_des) * 1e6
                V_dot = np.dot(K_p * theta, omega) + np.dot(omega, tau)
                P = np.dot(omega, tau)
                
                tau_norm = np.linalg.norm(tau)
                if tau_norm > 1e-15 and np.linalg.norm(tau_des) > 1e-15:
                    cos_angle = np.dot(tau, tau_des) / (tau_norm * np.linalg.norm(tau_des))
                    dir_err = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                else:
                    dir_err = 0
                
                stable = "Y" if V_dot <= 1e-12 else "N"
                
                tau_str = f"[{tau[0]*1e6:8.2f},{tau[1]*1e6:8.2f},{tau[2]*1e6:9.2f}]"
                print(f"{name:<12} {tau_str:<42} {error:>7.2f} {V_dot:>11.2e} {P:>11.2e} {dir_err:>6.1f} {stable:>5}")
                
            except Exception as e:
                print(f"{name:<12} ERROR: {str(e)[:40]}")
    
    return


def closed_loop_comparison():
    """Compare constraints in closed-loop simulation."""
    print("\n" + "=" * 120)
    print("CLOSED-LOOP COMPARISON (120s simulation)")
    print("=" * 120)
    
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
        ("QP (no const)", lambda td, **kw: qp_unconstrained(td, A, lb, ub, **kw)),
        ("1-Power", lambda td, **kw: qp_1_power_bound(td, A, lb, ub, **kw)),
        ("3-Sign", lambda td, **kw: qp_3_sign_preservation(td, A, lb, ub, **kw)),
        ("7-Proj", lambda td, **kw: qp_7_projection_guarantee(td, A, lb, ub, **kw)),
        ("10-Combined", lambda td, **kw: qp_10_combined_safe(td, A, lb, ub, **kw)),
    ]
    
    print(f"\nInitial: θ = {np.degrees(theta_0)} deg, ω = {omega_0} rad/s")
    print()
    
    results = {}
    
    for name, method in methods:
        theta = theta_0.copy()
        omega = omega_0.copy()
        
        theta_hist = [theta.copy()]
        omega_hist = [omega.copy()]
        V_hist = [0.5 * np.dot(K_p * theta, theta) + 0.5 * np.dot(omega, J @ omega)]
        unstable_count = 0
        
        for _ in range(n_steps):
            tau_des = -K_p * theta - K_d * omega
            
            kwargs = {
                'omega': omega,
                'theta': theta,
                'K_p': K_p,
                'J': J,
                'tau_ff': np.zeros(3),
            }
            
            try:
                u = method(tau_des, **kwargs)
                if u is None:
                    u = np.zeros(len(lb))
            except:
                u = np.zeros(len(lb))
            
            tau = A @ u
            
            # Check instantaneous stability
            V_dot = np.dot(K_p * theta, omega) + np.dot(omega, tau)
            if V_dot > 1e-10:
                unstable_count += 1
            
            # Integrate
            omega = omega + dt * J_inv @ tau
            theta = theta + dt * omega
            
            theta_hist.append(theta.copy())
            omega_hist.append(omega.copy())
            V = 0.5 * np.dot(K_p * theta, theta) + 0.5 * np.dot(omega, J @ omega)
            V_hist.append(V)
        
        results[name] = {
            'theta': np.array(theta_hist),
            'omega': np.array(omega_hist),
            'V': np.array(V_hist),
            'unstable_count': unstable_count,
        }
    
    print(f"{'Method':<14} {'θ_final (deg)':<32} {'|θ|':>7} {'|ω|':>9} {'V_final':>10} {'V̇>0 cnt':>9}")
    print("-" * 95)
    
    for name in results:
        r = results[name]
        theta_f = r['theta'][-1]
        omega_f = r['omega'][-1]
        V_f = r['V'][-1]
        
        theta_deg = np.degrees(theta_f)
        theta_str = f"[{theta_deg[0]:6.2f},{theta_deg[1]:6.2f},{theta_deg[2]:6.2f}]"
        print(f"{name:<14} {theta_str:<32} {np.degrees(np.linalg.norm(theta_f)):>7.2f} {np.linalg.norm(omega_f):>9.5f} {V_f:>10.2e} {r['unstable_count']:>9}")
    
    return results


def summary():
    """Print summary and recommendations."""
    print("\n" + "=" * 120)
    print("SUMMARY: PHYSICS-BASED CONSTRAINT COMPARISON")
    print("=" * 120)
    
    print("""
CONSTRAINT ANALYSIS:
====================

1. POWER BOUND (QPC's approach): ω'τ ≤ max(0, ω'τ_des)
   + Simple, always feasible
   + Controls total energy injection
   - Doesn't prevent per-axis acceleration
   - Can give τ = [+5, +5, -20] when τ_des = [-10, -10, -10]

2. GLOBAL LYAPUNOV: V̇ = θ'K_p ω + ω'τ ≤ 0
   + Formal stability guarantee
   + Considers full state (θ and ω)
   - Can get stuck (V̇ ≤ 0 ≠ convergence)
   - Requires knowing θ, K_p

3. SIGN PRESERVATION: sign(τ_i) = sign(τ_des,i) when braking
   + Per-axis guarantees
   + Simple and intuitive
   - Doesn't constrain magnitude
   - Can give τ = [-0.001, -0.001, -0.001] when τ_des = [-100, -100, -100]

4. PER-AXIS LYAPUNOV: (τ_i + K_p θ_i) ω_i ≤ 0
   + Strongest per-axis guarantee
   - Often infeasible (too restrictive)
   - Requires θ, K_p

5. MOMENTUM RATE BOUND: ||τ|| ≤ k||τ_des||
   + Simple magnitude bound
   - Doesn't care about direction
   - Rarely useful alone

6. WORK BUDGET: τ·ω ≤ τ_des·ω
   ~ Same as power bound
   + No worse than intended energy change
   
7. PROJECTION GUARANTEE: τ·τ̂_des ≥ α_LP
   + At least as good as LP in desired direction
   + Allows QP to improve magnitude
   - May still have large perpendicular component

8. FEEDFORWARD PRESERVATION: τ·τ̂_ff ≥ β||τ_ff||
   + Preserves trajectory tracking
   + Useful when τ_ff and τ_fb are separate
   - Requires knowing τ_ff

9. CONE CONSTRAINT: angle(τ, τ_des) ≤ θ_max
   + Direct direction bound
   + User-tunable
   - Can be infeasible for small θ_max

10. COMBINED (Power + Sign): Both constraints
    + Best of both worlds
    + Total energy bound AND per-axis sign
    - Slightly more restrictive


RECOMMENDATION:
===============

For general use, **COMBINED (Power + Sign)** is best:
- Prevents total energy injection (power bound)
- Prevents per-axis acceleration (sign preservation)
- Simple to implement
- Always feasible

If trajectory tracking is important, add:
- Projection guarantee (LP+QP hybrid) or
- Feedforward preservation

If formal stability proof needed:
- Use Global Lyapunov constraint
- But accept it may sacrifice convergence rate
""")


if __name__ == "__main__":
    np.random.seed(42)
    
    run_scenario_tests()
    closed_loop_comparison()
    summary()
