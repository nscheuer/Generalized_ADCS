"""
Deeper Analysis: Why Might Matching Torque Be Bad?
==================================================

Patrick's challenge: The desired torque is trying to do something useful
(counter disturbances, gyroscopic, move toward goal). Matching it as closely
as possible SHOULD be good. Why would preserving direction be better than
matching magnitude?

Let's reason through this carefully with specific cases.
"""

import numpy as np
from scipy.optimize import linprog, minimize, Bounds
import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from ADCS.helpers.math_helpers import skewsym, normalize, rot_mat, quat_inv, quat_mult


def quat_err(q, q_goal):
    q, q_goal = normalize(q), normalize(q_goal)
    qe = quat_mult(quat_inv(q_goal), q)
    if qe[0] < 0: qe = -qe
    return 2*qe[1:4]


def full_err_deg(q, q_goal):
    q, q_goal = normalize(q), normalize(q_goal)
    qe = quat_mult(quat_inv(q_goal), q)
    if qe[0] < 0: qe = -qe
    return np.degrees(2*np.arccos(np.clip(qe[0], -1, 1)))


# ============== ALLOCATORS ==============

def allocate_lp(tau_des, A, lb, ub):
    """LP: max projection along desired direction."""
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


def allocate_qp_min_error(tau_des, A, lb, ub):
    """QP: minimize ||τ - τ_des||² (closest to desired)."""
    n = len(lb)
    
    def objective(u):
        tau = A @ u
        return np.sum((tau - tau_des)**2)
    
    res = minimize(objective, np.zeros(n), bounds=Bounds(lb, ub),
                  method='SLSQP', options={'ftol': 1e-12})
    
    return res.x if res.success else np.zeros(n)


def allocate_qp_max_proj(tau_des, A, lb, ub):
    """QP: maximize τ·τ_des (max projection, no direction constraint)."""
    n = len(lb)
    
    def objective(u):
        tau = A @ u
        return -np.dot(tau, tau_des)  # Negative because minimizing
    
    res = minimize(objective, np.zeros(n), bounds=Bounds(lb, ub),
                  method='SLSQP', options={'ftol': 1e-12})
    
    return res.x if res.success else np.zeros(n)


# ============== ANALYSIS ==============

def analyze_single_allocation():
    """
    Analyze what happens with different allocators for a specific case.
    """
    print("=" * 80)
    print("CASE STUDY: What does each allocator actually produce?")
    print("=" * 80)
    
    # Underactuated system: 3MTQ + 1RW
    A_rw = np.array([[0], [0], [1.0]])
    b = np.array([20e-6, 15e-6, 10e-6])
    A_mtq = -skewsym(b) @ np.eye(3)
    A = np.hstack([A_rw, A_mtq])
    
    u_rw_max = np.array([0.001])
    u_mtq_max = np.array([0.2, 0.2, 0.2])
    lb = np.concatenate([-u_rw_max, -u_mtq_max])
    ub = np.concatenate([u_rw_max, u_mtq_max])
    
    # Desired torque that's NOT achievable exactly
    tau_des = np.array([1e-5, 1e-5, 1e-5])  # Equal in all axes
    tau_hat = tau_des / np.linalg.norm(tau_des)
    t_mag = np.linalg.norm(tau_des)
    
    print(f"\nτ_des = {tau_des * 1e6} μNm")
    print(f"|τ_des| = {t_mag * 1e6:.2f} μNm")
    
    # What can the system achieve?
    print(f"\nActuator capabilities:")
    print(f"  RW: z-axis only, max ±1000 μNm")
    print(f"  MTQ: perpendicular to B, limited by B magnitude")
    
    # LP allocation
    u_lp = allocate_lp(tau_des, A, lb, ub)
    tau_lp = A @ u_lp
    proj_lp = np.dot(tau_lp, tau_hat)
    err_lp = np.linalg.norm(tau_lp - tau_des)
    
    # QP min error
    u_qp = allocate_qp_min_error(tau_des, A, lb, ub)
    tau_qp = A @ u_qp
    proj_qp = np.dot(tau_qp, tau_hat)
    err_qp = np.linalg.norm(tau_qp - tau_des)
    
    # QP max projection
    u_qp_proj = allocate_qp_max_proj(tau_des, A, lb, ub)
    tau_qp_proj = A @ u_qp_proj
    proj_qp_proj = np.dot(tau_qp_proj, tau_hat)
    err_qp_proj = np.linalg.norm(tau_qp_proj - tau_des)
    
    print(f"\n{'Method':<20} {'τ achieved (μNm)':<30} {'Proj':>10} {'||τ-τ_des||':>12} {'Dir err':>10}")
    print("-" * 85)
    
    for name, tau, proj, err in [
        ('LP (direction)', tau_lp, proj_lp, err_lp),
        ('QP (min error)', tau_qp, proj_qp, err_qp),
        ('QP (max proj)', tau_qp_proj, proj_qp_proj, err_qp_proj),
    ]:
        tau_str = f"[{tau[0]*1e6:.2f}, {tau[1]*1e6:.2f}, {tau[2]*1e6:.2f}]"
        tau_norm = np.linalg.norm(tau)
        if tau_norm > 1e-12:
            cos_ang = np.dot(tau, tau_hat) / tau_norm
            dir_err = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))
        else:
            dir_err = 0
        print(f"{name:<20} {tau_str:<30} {proj*1e6:>10.2f} {err*1e6:>12.2f} {dir_err:>10.1f}°")
    
    print(f"\n" + "=" * 80)
    print("ANALYSIS: Why does min-error QP have direction error?")
    print("=" * 80)
    print(f"""
The desired torque τ_des = [10, 10, 10] μNm is NOT in the reachable set.

The reachable set is:
  τ = A_rw·u_rw + A_mtq·u_mtq
  
Where:
  - A_rw·u_rw gives [0, 0, τ_z] (z-axis only)
  - A_mtq·u_mtq gives torque ⊥ to B

The closest point in the reachable set to τ_des may NOT be in the 
direction of τ_des!

Imagine τ_des points "northeast" but the reachable set is a tilted plane.
The closest point on that plane might be to the "east" or "north" of the 
origin, not along the northeast line.
""")
    
    return tau_lp, tau_qp, tau_des


def analyze_lyapunov_decomposition():
    """
    Decompose the desired torque into components and understand
    what each allocator does to each component.
    """
    print("\n" + "=" * 80)
    print("LYAPUNOV FUNCTION DECOMPOSITION")
    print("=" * 80)
    
    print("""
For a PD attitude controller:
    
    τ_des = -K_p·e - K_d·ω + τ_gyro
    
Where:
    - K_p·e: Proportional term (points toward goal)
    - K_d·ω: Damping term (opposes motion)
    - τ_gyro = ω × (J·ω + h): Gyroscopic compensation
    
The Lyapunov function is:
    
    V = (1/2)·e^T·K_p·e + (1/2)·ω^T·J·ω
    
Taking the derivative:
    
    V̇ = e^T·K_p·ė + ω^T·J·ω̇
      = e^T·K_p·(-ω×e + ...) + ω^T·τ_net
      
If τ_net = τ_des (perfect tracking):
    V̇ = -K_d·||ω||² + (small cross terms) ≤ 0
    
If τ_net = τ_des + τ_error:
    V̇ = -K_d·||ω||² + ω^T·τ_error
    
The question: What τ_error is acceptable?
""")
    
    # Numerical example
    print("\nNumerical Example:")
    print("-" * 40)
    
    e = np.array([0.1, 0.05, -0.02])  # Attitude error
    omega = np.array([0.01, -0.015, 0.005])  # Angular velocity
    J = np.diag([0.022, 0.022, 0.004])
    h = np.array([0, 0, 0.003])  # RW momentum
    
    Kp, Kd = 5e-5, 1e-3
    
    # Desired torque components
    tau_prop = -Kp * e
    tau_damp = -Kd * omega
    tau_gyro = np.cross(omega, J @ omega + h)
    tau_des = tau_prop + tau_damp + tau_gyro
    
    print(f"Error e = {e}")
    print(f"Omega ω = {omega}")
    print(f"\nτ_proportional = {tau_prop * 1e6} μNm")
    print(f"τ_damping     = {tau_damp * 1e6} μNm")
    print(f"τ_gyroscopic  = {tau_gyro * 1e6} μNm")
    print(f"τ_des (total) = {tau_des * 1e6} μNm")
    
    # V̇ contribution from each
    print(f"\nV̇ contributions if τ = τ_des:")
    print(f"  From damping: -K_d·||ω||² = {-Kd * np.dot(omega, omega) * 1e6:.4f} μJ/s")
    
    # What if we have torque error?
    print(f"\nIf τ_achieved = τ_des + τ_error:")
    print(f"  Extra V̇ term: ω^T·τ_error")
    print(f"  For stability: ω^T·τ_error should be small or negative")
    
    # The key insight
    print(f"""
KEY INSIGHT:
-----------
The stability concern is ω^T·τ_error.

If τ_error is perpendicular to ω: ω^T·τ_error = 0 (SAFE!)
If τ_error is parallel to ω: ω^T·τ_error = ||ω||·||τ_error|| (DANGEROUS if wrong sign)

So the ACTUAL constraint should be:
    ω^T·(τ - τ_des) ≤ some_bound
    
NOT:
    τ must be parallel to τ_des

This is a weaker constraint! It allows more of the reachable set.
""")
    
    return omega, tau_des


def derive_stability_constraint():
    """
    Derive the correct stability-preserving constraint.
    """
    print("\n" + "=" * 80)
    print("DERIVING THE CORRECT STABILITY CONSTRAINT")
    print("=" * 80)
    
    print("""
From Lyapunov analysis:
    
    V̇ = -K_d·||ω||² + ω^T·(τ - τ_des)
    
For V̇ ≤ 0, we need:
    
    ω^T·(τ - τ_des) ≤ K_d·||ω||²
    
Rearranging:
    
    ω^T·τ ≤ ω^T·τ_des + K_d·||ω||²
    
This is a LINEAR constraint on u!
    
    ω^T·A·u ≤ ω^T·τ_des + K_d·||ω||²
    
New allocation formulation:
    
    min ||A·u - τ_des||²    (match as closely as possible)
    s.t. lb ≤ u ≤ ub
         ω^T·A·u ≤ ω^T·τ_des + K_d·||ω||²   (stability constraint)
         
Or equivalently, maximize projection while maintaining stability:
    
    max τ_des^T·(A·u)
    s.t. lb ≤ u ≤ ub
         ω^T·A·u ≤ ω^T·τ_des + K_d·||ω||²   (stability)
""")
    
    return


def test_stability_constrained_allocation():
    """
    Test the new stability-constrained allocation.
    """
    print("\n" + "=" * 80)
    print("TESTING STABILITY-CONSTRAINED ALLOCATION")
    print("=" * 80)
    
    # System setup
    A_rw = np.array([[0], [0], [1.0]])
    b = np.array([20e-6, 15e-6, 10e-6])
    A_mtq = -skewsym(b) @ np.eye(3)
    A = np.hstack([A_rw, A_mtq])
    
    u_rw_max = np.array([0.001])
    u_mtq_max = np.array([0.2, 0.2, 0.2])
    lb = np.concatenate([-u_rw_max, -u_mtq_max])
    ub = np.concatenate([u_rw_max, u_mtq_max])
    
    Kd = 1e-3
    
    # Test case
    omega = np.array([0.01, -0.015, 0.005])
    tau_des = np.array([1e-5, 1e-5, 1e-5])
    
    n = len(lb)
    
    def allocate_stability_constrained(tau_des, A, lb, ub, omega, Kd):
        """
        New allocation: minimize ||τ - τ_des||² subject to stability constraint.
        """
        n = len(lb)
        omega_norm_sq = np.dot(omega, omega)
        
        # RHS of stability constraint
        stability_bound = np.dot(omega, tau_des) + Kd * omega_norm_sq
        
        def objective(u):
            tau = A @ u
            return np.sum((tau - tau_des)**2)
        
        def stability_constraint(u):
            tau = A @ u
            # Want: ω^T·τ ≤ stability_bound
            return stability_bound - np.dot(omega, tau)
        
        res = minimize(objective, np.zeros(n), bounds=Bounds(lb, ub),
                      constraints={'type': 'ineq', 'fun': stability_constraint},
                      method='SLSQP', options={'ftol': 1e-12})
        
        return res.x if res.success else allocate_lp(tau_des, A, lb, ub)
    
    # Compare methods
    u_lp = allocate_lp(tau_des, A, lb, ub)
    u_qp = allocate_qp_min_error(tau_des, A, lb, ub)
    u_stab = allocate_stability_constrained(tau_des, A, lb, ub, omega, Kd)
    
    tau_lp = A @ u_lp
    tau_qp = A @ u_qp
    tau_stab = A @ u_stab
    
    print(f"ω = {omega}")
    print(f"τ_des = {tau_des * 1e6} μNm")
    print(f"K_d·||ω||² = {Kd * np.dot(omega, omega) * 1e6:.4f} μNm")
    
    print(f"\n{'Method':<25} {'||τ-τ_des||':>12} {'ω^T·(τ-τ_des)':>15} {'V̇ contrib':>12}")
    print("-" * 70)
    
    for name, tau in [('LP (direction)', tau_lp), 
                      ('QP (min error)', tau_qp),
                      ('Stability-constrained', tau_stab)]:
        err = np.linalg.norm(tau - tau_des)
        omega_err = np.dot(omega, tau - tau_des)
        v_dot = omega_err  # The problematic term
        
        print(f"{name:<25} {err*1e6:>12.4f} {omega_err*1e9:>15.4f} {v_dot*1e9:>12.4f}")
    
    print(f"""
OBSERVATIONS:
- LP may have larger ||τ-τ_des|| but controlled ω^T·(τ-τ_des)
- QP min-error has smaller ||τ-τ_des|| but may have bad ω^T·(τ-τ_des)
- Stability-constrained: minimizes error WHILE guaranteeing stability

The stability-constrained approach is THE RIGHT ANSWER because:
1. It directly constrains what matters for stability
2. It allows maximum flexibility within that constraint
3. It doesn't impose arbitrary direction requirements
""")
    
    return


def closed_loop_comparison():
    """
    Compare LP, QP, and stability-constrained in closed loop.
    """
    print("\n" + "=" * 80)
    print("CLOSED-LOOP COMPARISON: LP vs QP vs STABILITY-CONSTRAINED")
    print("=" * 80)
    
    # System
    J = np.diag([0.022, 0.022, 0.004])
    A_rw = np.array([[0], [0], [1.0]])
    A_mtq_axes = np.eye(3)
    u_rw_max = np.array([0.001])
    u_mtq_max = np.array([0.2, 0.2, 0.2])
    
    kp, kd = 5e-5, 1e-3
    
    q_goal = np.array([1, 0, 0, 0])
    
    def simulate(allocator_fn, name, tf=300, dt=1.0):
        axis = normalize([1, 1, 0.5])
        angle = 0.5
        q0 = normalize(np.concatenate([[np.cos(angle/2)], axis*np.sin(angle/2)]))
        omega0 = np.array([0.01, 0.01, 0.005])
        h0 = np.zeros(1)
        
        x = np.concatenate([omega0, q0, h0])
        
        err_hist = []
        v_dot_hist = []
        
        for k in range(int(tf/dt)):
            omega = x[0:3]
            q = normalize(x[3:7])
            h = x[7:8]
            
            t = k * dt
            phase = 2 * np.pi * t / 5400
            B = 30e-6 * np.array([np.cos(phase), 0.5*np.sin(phase), 0.3])
            b = rot_mat(q).T @ B
            
            A_mtq = -skewsym(b) @ A_mtq_axes
            A = np.hstack([A_rw, A_mtq])
            lb = np.concatenate([-u_rw_max, -u_mtq_max])
            ub = np.concatenate([u_rw_max, u_mtq_max])
            
            qe = quat_err(q, q_goal)
            h_vec = A_rw @ h
            tau_gyro = np.cross(omega, J @ omega + h_vec)
            tau_des = -kp * qe - kd * omega + tau_gyro
            
            u = allocator_fn(tau_des, A, lb, ub, omega, kd)
            
            tau = A @ u
            v_dot_term = np.dot(omega, tau - tau_des)
            
            err_hist.append(full_err_deg(q, q_goal))
            v_dot_hist.append(v_dot_term)
            
            u_rw, u_mtq = u[:1], u[1:]
            
            def deriv(state):
                w = state[:3]
                qu = normalize(state[3:7])
                hr = state[7:8]
                
                hrv = A_rw @ hr
                tau_total = A_rw @ u_rw + A_mtq @ u_mtq
                
                w_dot = np.linalg.solve(J, tau_total - np.cross(w, J @ w + hrv))
                
                W = np.zeros((4, 3))
                W[0, :] = -qu[1:4]
                W[1:4, :] = qu[0] * np.eye(3) + skewsym(qu[1:4])
                q_dot = 0.5 * W @ w
                
                h_dot = -u_rw
                
                return np.concatenate([w_dot, q_dot, h_dot])
            
            k1 = deriv(x)
            k2 = deriv(x + 0.5*dt*k1)
            k3 = deriv(x + 0.5*dt*k2)
            k4 = deriv(x + dt*k3)
            
            x = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            x[3:7] = normalize(x[3:7])
        
        return {
            'name': name,
            'final_error': err_hist[-1],
            'mean_error': np.mean(err_hist),
            'max_v_dot': np.max(v_dot_hist),
            'mean_v_dot': np.mean(v_dot_hist),
        }
    
    # Allocators (need to accept omega, kd even if not used)
    def lp_wrapper(tau_des, A, lb, ub, omega, kd):
        return allocate_lp(tau_des, A, lb, ub)
    
    def qp_wrapper(tau_des, A, lb, ub, omega, kd):
        return allocate_qp_min_error(tau_des, A, lb, ub)
    
    def stability_wrapper(tau_des, A, lb, ub, omega, kd):
        n = len(lb)
        omega_norm_sq = np.dot(omega, omega)
        stability_bound = np.dot(omega, tau_des) + kd * omega_norm_sq
        
        def objective(u):
            tau = A @ u
            return np.sum((tau - tau_des)**2)
        
        def stability_constraint(u):
            tau = A @ u
            return stability_bound - np.dot(omega, tau)
        
        res = minimize(objective, np.zeros(n), bounds=Bounds(lb, ub),
                      constraints={'type': 'ineq', 'fun': stability_constraint},
                      method='SLSQP', options={'ftol': 1e-12})
        
        return res.x if res.success else allocate_lp(tau_des, A, lb, ub)
    
    results = []
    for fn, name in [(lp_wrapper, 'LP (direction)'),
                     (qp_wrapper, 'QP (min error)'),
                     (stability_wrapper, 'Stability-constrained')]:
        r = simulate(fn, name)
        results.append(r)
    
    print(f"\n{'Method':<25} {'Final Err':>12} {'Mean Err':>12} {'Max ω·Δτ':>12}")
    print("-" * 65)
    for r in results:
        print(f"{r['name']:<25} {r['final_error']:>12.2f}° {r['mean_error']:>12.2f}° {r['max_v_dot']*1e9:>12.2f}")
    
    return results


if __name__ == "__main__":
    np.random.seed(42)
    
    analyze_single_allocation()
    analyze_lyapunov_decomposition()
    derive_stability_constraint()
    test_stability_constrained_allocation()
    results = closed_loop_comparison()
    
    print("\n" + "=" * 80)
    print("FINAL CONCLUSIONS")
    print("=" * 80)
    print("""
YOU ARE RIGHT! The direction constraint was too conservative.

The ACTUAL stability requirement is:
    ω^T·(τ - τ_des) ≤ K_d·||ω||²
    
This is a HYPERPLANE constraint, not a CONE constraint!

New optimal allocation:
    min ||τ - τ_des||²           (match desired as closely as possible)
    s.t. lb ≤ u ≤ ub             (actuator limits)
         ω^T·τ ≤ ω^T·τ_des + K_d·||ω||²   (stability)

This allows:
1. Matching the desired torque as closely as possible
2. Using more of the reachable set
3. Only restricting torque in the direction that matters for stability
4. The constraint adapts to the current state (ω), not arbitrary cone size
""")
