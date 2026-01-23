"""
Filtered Testing: Only configurations where unbounded control converges
======================================================================

First filter to cases where the control problem is solvable,
then compare allocation methods fairly.
"""

import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from ADCS.helpers.math_helpers import skewsym

SCALE = 1e6


# ============================================================================
# ACTUATOR CONFIGURATIONS
# ============================================================================

def config_3mtq_1rw(B_body, mtq_max=0.2, rw_max=0.001):
    """Standard 3MTQ + 1RW (z-axis)"""
    A_rw = np.array([[0], [0], [1.0]])
    A_mtq = -skewsym(B_body) @ np.eye(3)
    A = np.hstack([A_rw, A_mtq])
    lb = np.array([-rw_max, -mtq_max, -mtq_max, -mtq_max])
    ub = np.array([rw_max, mtq_max, mtq_max, mtq_max])
    return A, lb, ub, "3MTQ+1RW"


def config_3mtq_only(B_body, mtq_max=0.2):
    """3MTQ only (underactuated)"""
    A = -skewsym(B_body) @ np.eye(3)
    lb = np.array([-mtq_max, -mtq_max, -mtq_max])
    ub = np.array([mtq_max, mtq_max, mtq_max])
    return A, lb, ub, "3MTQ only"


def config_4rw_pyramid(rw_max=0.001):
    """4RW in pyramid configuration"""
    beta = np.radians(54.74)
    axes = np.array([
        [np.sin(beta), 0, np.cos(beta)],
        [0, np.sin(beta), np.cos(beta)],
        [-np.sin(beta), 0, np.cos(beta)],
        [0, -np.sin(beta), np.cos(beta)],
    ]).T
    A = axes
    lb = np.array([-rw_max, -rw_max, -rw_max, -rw_max])
    ub = np.array([rw_max, rw_max, rw_max, rw_max])
    return A, lb, ub, "4RW pyramid"


def config_4rw_3mtq(B_body, rw_max=0.001, mtq_max=0.2):
    """4RW pyramid + 3MTQ (fully redundant)"""
    beta = np.radians(54.74)
    rw_axes = np.array([
        [np.sin(beta), 0, np.cos(beta)],
        [0, np.sin(beta), np.cos(beta)],
        [-np.sin(beta), 0, np.cos(beta)],
        [0, -np.sin(beta), np.cos(beta)],
    ]).T
    A_rw = rw_axes
    A_mtq = -skewsym(B_body) @ np.eye(3)
    A = np.hstack([A_rw, A_mtq])
    lb = np.array([-rw_max]*4 + [-mtq_max]*3)
    ub = np.array([rw_max]*4 + [mtq_max]*3)
    return A, lb, ub, "4RW+3MTQ"


# ============================================================================
# ALLOCATORS
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
        return u, A @ u, min(1.0, alpha / t_mag)
    return np.zeros(n), np.zeros(3), 0.0


def qp_unconstrained(tau_des, A, lb, ub, **kwargs):
    """QP with no physics constraints."""
    n = len(lb)
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(SCALE * (tau - tau_des)))
    constraints = [u >= lb, u <= ub]
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
        return u.value if u.value is not None else None
    except:
        return None


def qp_1a_power_brake_only(tau_des, A, lb, ub, omega, **kwargs):
    """1a: Only constrain power when braking intended."""
    n = len(lb)
    P_des = np.dot(omega, tau_des)
    
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(SCALE * (tau - tau_des)))
    constraints = [u >= lb, u <= ub]
    
    if P_des < -1e-12:
        constraints.append(omega @ tau <= 1e-12)
    
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
        return u.value if u.value is not None else None
    except:
        return None


def qp_3b_sign_critical(tau_des, A, lb, ub, omega, **kwargs):
    """3b: Sign preservation on critical axes only."""
    n = len(lb)
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(SCALE * (tau - tau_des)))
    constraints = [u >= lb, u <= ub]
    
    tau_threshold = 0.1 * np.linalg.norm(tau_des) + 1e-12
    
    for i in range(3):
        if abs(tau_des[i]) < tau_threshold:
            continue
        if omega[i] > 1e-6 and tau_des[i] < -1e-12:
            constraints.append(tau[i] <= 0)
        elif omega[i] < -1e-6 and tau_des[i] > 1e-12:
            constraints.append(tau[i] >= 0)
    
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
        return u.value if u.value is not None else None
    except:
        return None


def qp_phase_aware(tau_des, A, lb, ub, omega, theta, **kwargs):
    """Phase-space aware constraint."""
    n = len(lb)
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(SCALE * (tau - tau_des)))
    constraints = [u >= lb, u <= ub]
    
    theta_dot_omega = np.dot(theta, omega)
    P_des = np.dot(omega, tau_des)
    
    if theta_dot_omega > 1e-6:  # Diverging
        constraints.append(omega @ tau <= max(0, P_des) + 1e-12)
    
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
        return u.value if u.value is not None else None
    except:
        return None


def qp_projection_hybrid(tau_des, A, lb, ub, **kwargs):
    """Projection guarantee (LP+QP hybrid)."""
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
        tau @ tau_hat >= proj_lp - 1e-12,
    ]
    
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
        return u.value if u.value is not None else None
    except:
        return None


def allocate_unbounded(tau_des, A):
    """Pseudo-inverse allocation (no bounds) - for testing convergence."""
    return np.linalg.pinv(A) @ tau_des


# ============================================================================
# SIMULATION
# ============================================================================

def simulate(method, A, lb, ub, J, K_p, K_d, theta_0, omega_0, dt=0.2, t_end=60.0, bounded=True):
    """Simulate closed-loop regulation."""
    J_inv = np.linalg.inv(J)
    n_steps = int(t_end / dt)
    
    theta = theta_0.copy()
    omega = omega_0.copy()
    
    for _ in range(n_steps):
        tau_des = -K_p * theta - K_d * omega
        
        if not bounded:
            # Unbounded allocation (pseudo-inverse)
            u = allocate_unbounded(tau_des, A)
            tau = A @ u
        else:
            kwargs = {'omega': omega, 'theta': theta, 'K_p': K_p}
            try:
                if method.__name__ == 'solve_lp':
                    u, tau, _ = method(tau_des, A, lb, ub)
                else:
                    u = method(tau_des, A, lb, ub, **kwargs)
                    if u is None:
                        u = np.zeros(len(lb))
                    tau = A @ u
            except:
                u = np.zeros(len(lb))
                tau = A @ u
        
        # Integrate
        omega = omega + dt * J_inv @ tau
        theta = theta + dt * omega
    
    return {
        'theta_final': theta,
        'omega_final': omega,
        'theta_mag_deg': np.degrees(np.linalg.norm(theta)),
        'converged': np.linalg.norm(theta) < 0.1,
    }


# ============================================================================
# TEST CONFIGURATIONS
# ============================================================================

def get_all_configs():
    """Generate test configurations."""
    configs = []
    
    # B-fields
    B_nominal = np.array([20e-6, 15e-6, 10e-6])
    B_along_z = np.array([1e-6, 1e-6, 30e-6])
    B_along_x = np.array([30e-6, 1e-6, 1e-6])
    B_weak = np.array([5e-6, 4e-6, 3e-6])
    B_strong = np.array([50e-6, 40e-6, 30e-6])
    
    # Base parameters
    J_nominal = np.diag([0.01, 0.01, 0.005])
    J_symmetric = np.diag([0.01, 0.01, 0.01])
    J_long_thin = np.diag([0.001, 0.001, 0.01])
    
    K_p_nom = np.array([0.001, 0.001, 0.001])
    K_d_nom = np.array([0.01, 0.01, 0.01])
    K_p_high = np.array([0.005, 0.005, 0.005])
    K_d_high = np.array([0.05, 0.05, 0.05])
    K_p_low = np.array([0.0005, 0.0005, 0.0005])
    K_d_low = np.array([0.005, 0.005, 0.005])
    
    # ICs
    IC_small = (np.array([0.05, 0.03, 0.02]), np.array([0.005, 0.005, 0.005]))
    IC_medium = (np.array([0.2, 0.15, 0.1]), np.array([0.01, 0.01, 0.01]))
    IC_large = (np.array([0.5, 0.4, 0.3]), np.array([0.02, 0.02, 0.02]))
    IC_large_angle_small_rate = (np.array([0.8, 0.6, 0.4]), np.array([0.001, 0.001, 0.001]))
    IC_small_angle_large_rate = (np.array([0.02, 0.02, 0.02]), np.array([0.05, 0.05, 0.05]))
    IC_default = (np.array([0.3, 0.2, 0.1]), np.array([0.01, 0.01, 0.01]))
    
    # ===== 3MTQ+1RW configurations =====
    
    # Varying B-field
    for B, B_name in [(B_nominal, "B_nom"), (B_along_z, "B_z"), (B_along_x, "B_x"), 
                       (B_weak, "B_weak"), (B_strong, "B_strong")]:
        A, lb, ub, _ = config_3mtq_1rw(B)
        configs.append({
            'name': f"3MTQ+1RW | {B_name}",
            'A': A, 'lb': lb, 'ub': ub,
            'J': J_nominal, 'K_p': K_p_nom, 'K_d': K_d_nom,
            'theta_0': IC_default[0], 'omega_0': IC_default[1],
        })
    
    # Varying bounds
    for bounds_name, mtq_max, rw_max in [("tight", 0.05, 0.0005), ("nominal", 0.2, 0.001), ("loose", 0.5, 0.005)]:
        A, lb, ub, _ = config_3mtq_1rw(B_nominal, mtq_max, rw_max)
        configs.append({
            'name': f"3MTQ+1RW | {bounds_name} bounds",
            'A': A, 'lb': lb, 'ub': ub,
            'J': J_nominal, 'K_p': K_p_nom, 'K_d': K_d_nom,
            'theta_0': IC_default[0], 'omega_0': IC_default[1],
        })
    
    # Varying ICs
    for IC_name, (theta_0, omega_0) in [("small_IC", IC_small), ("medium_IC", IC_medium), 
                                         ("large_IC", IC_large), ("large_ang_small_rate", IC_large_angle_small_rate),
                                         ("small_ang_large_rate", IC_small_angle_large_rate)]:
        A, lb, ub, _ = config_3mtq_1rw(B_nominal)
        configs.append({
            'name': f"3MTQ+1RW | {IC_name}",
            'A': A, 'lb': lb, 'ub': ub,
            'J': J_nominal, 'K_p': K_p_nom, 'K_d': K_d_nom,
            'theta_0': theta_0, 'omega_0': omega_0,
        })
    
    # Varying inertia
    for J_name, J in [("J_sym", J_symmetric), ("J_nom", J_nominal), ("J_long", J_long_thin)]:
        A, lb, ub, _ = config_3mtq_1rw(B_nominal)
        configs.append({
            'name': f"3MTQ+1RW | {J_name}",
            'A': A, 'lb': lb, 'ub': ub,
            'J': J, 'K_p': K_p_nom, 'K_d': K_d_nom,
            'theta_0': IC_default[0], 'omega_0': IC_default[1],
        })
    
    # Varying gains
    for gain_name, K_p, K_d in [("low_gain", K_p_low, K_d_low), ("nom_gain", K_p_nom, K_d_nom), 
                                 ("high_gain", K_p_high, K_d_high)]:
        A, lb, ub, _ = config_3mtq_1rw(B_nominal)
        configs.append({
            'name': f"3MTQ+1RW | {gain_name}",
            'A': A, 'lb': lb, 'ub': ub,
            'J': J_nominal, 'K_p': K_p, 'K_d': K_d,
            'theta_0': IC_default[0], 'omega_0': IC_default[1],
        })
    
    # ===== 4RW configurations =====
    for IC_name, (theta_0, omega_0) in [("small_IC", IC_small), ("medium_IC", IC_medium), ("large_IC", IC_large)]:
        A, lb, ub, _ = config_4rw_pyramid()
        configs.append({
            'name': f"4RW | {IC_name}",
            'A': A, 'lb': lb, 'ub': ub,
            'J': J_nominal, 'K_p': K_p_nom, 'K_d': K_d_nom,
            'theta_0': theta_0, 'omega_0': omega_0,
        })
    
    # ===== 4RW+3MTQ configurations =====
    for IC_name, (theta_0, omega_0) in [("small_IC", IC_small), ("medium_IC", IC_medium)]:
        A, lb, ub, _ = config_4rw_3mtq(B_nominal)
        configs.append({
            'name': f"4RW+3MTQ | {IC_name}",
            'A': A, 'lb': lb, 'ub': ub,
            'J': J_nominal, 'K_p': K_p_nom, 'K_d': K_d_nom,
            'theta_0': theta_0, 'omega_0': omega_0,
        })
    
    # ===== 3MTQ only (underactuated) =====
    for B, B_name in [(B_nominal, "B_nom"), (B_along_z, "B_z"), (B_strong, "B_strong")]:
        A, lb, ub, _ = config_3mtq_only(B)
        configs.append({
            'name': f"3MTQ | {B_name}",
            'A': A, 'lb': lb, 'ub': ub,
            'J': J_nominal, 'K_p': K_p_nom, 'K_d': K_d_nom,
            'theta_0': IC_default[0], 'omega_0': IC_default[1],
        })
    
    return configs


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 120)
    print("STEP 1: FILTER CONFIGURATIONS BY UNBOUNDED CONVERGENCE")
    print("=" * 120)
    
    configs = get_all_configs()
    print(f"\nTotal configurations: {len(configs)}")
    
    # Test each config with unbounded control
    valid_configs = []
    
    print(f"\n{'Configuration':<45} {'Unbounded θ_final':>18} {'Converges?':>12}")
    print("-" * 80)
    
    for config in configs:
        result = simulate(
            None, config['A'], config['lb'], config['ub'],
            config['J'], config['K_p'], config['K_d'],
            config['theta_0'], config['omega_0'],
            dt=0.1, t_end=60.0, bounded=False
        )
        
        converges = result['converged']
        theta_deg = result['theta_mag_deg']
        
        status = "✓ YES" if converges else "✗ NO"
        print(f"{config['name']:<45} {theta_deg:>15.2f}° {status:>12}")
        
        if converges:
            valid_configs.append(config)
    
    print(f"\nValid configurations (unbounded converges): {len(valid_configs)}/{len(configs)}")
    
    # Now test methods only on valid configs
    print("\n" + "=" * 120)
    print("STEP 2: COMPARE METHODS ON VALID CONFIGURATIONS")
    print("=" * 120)
    
    methods = [
        ('LP', solve_lp),
        ('QP uncon', qp_unconstrained),
        ('1a-Power', qp_1a_power_brake_only),
        ('3b-Sign', qp_3b_sign_critical),
        ('Phase', qp_phase_aware),
        ('Proj', qp_projection_hybrid),
    ]
    
    # Header
    header = f"{'Configuration':<45} |"
    for name, _ in methods:
        header += f" {name:>9} |"
    print(f"\n{header}")
    print("-" * len(header))
    
    all_results = {name: [] for name, _ in methods}
    
    for config in valid_configs:
        row = f"{config['name']:<45} |"
        
        for method_name, method_func in methods:
            result = simulate(
                method_func, config['A'], config['lb'], config['ub'],
                config['J'], config['K_p'], config['K_d'],
                config['theta_0'], config['omega_0'],
                dt=0.2, t_end=60.0, bounded=True
            )
            
            theta_deg = result['theta_mag_deg']
            all_results[method_name].append({
                'config': config['name'],
                'theta_deg': theta_deg,
                'converged': result['converged'],
            })
            
            marker = "✓" if result['converged'] else " "
            row += f" {theta_deg:>7.1f}°{marker}|"
        
        print(row)
    
    # Summary
    print("\n" + "=" * 120)
    print("SUMMARY STATISTICS (only valid configurations)")
    print("=" * 120)
    
    print(f"\n{'Method':<12} {'Mean θ':>10} {'Median θ':>10} {'Max θ':>10} {'Converged':>12}")
    print("-" * 60)
    
    for method_name, _ in methods:
        results = all_results[method_name]
        thetas = [r['theta_deg'] for r in results]
        converged = sum(1 for r in results if r['converged'])
        
        print(f"{method_name:<12} {np.mean(thetas):>9.2f}° {np.median(thetas):>9.2f}° {np.max(thetas):>9.1f}° {converged:>8}/{len(results):<3}")
    
    # Head-to-head
    print("\n" + "=" * 120)
    print("HEAD-TO-HEAD WINS")
    print("=" * 120)
    
    method_names = [name for name, _ in methods]
    wins = {name: 0 for name in method_names}
    
    for i in range(len(valid_configs)):
        best_theta = float('inf')
        best_methods = []
        for method_name in method_names:
            theta = all_results[method_name][i]['theta_deg']
            if theta < best_theta - 0.1:  # Clear winner (0.1° margin)
                best_theta = theta
                best_methods = [method_name]
            elif theta < best_theta + 0.1:  # Tie
                best_methods.append(method_name)
        
        for m in best_methods:
            wins[m] += 1.0 / len(best_methods)
    
    print(f"\nWins (best final θ, ties split):")
    for name in sorted(wins, key=wins.get, reverse=True):
        pct = 100 * wins[name] / len(valid_configs)
        bar = "█" * int(pct / 3)
        print(f"  {name:<12}: {wins[name]:>5.1f} ({pct:>5.1f}%) {bar}")
    
    # Where does each method win?
    print("\n" + "=" * 120)
    print("WHERE EACH METHOD WINS (best or tied)")
    print("=" * 120)
    
    for method_name in method_names:
        wins_list = []
        for i, config in enumerate(valid_configs):
            theta = all_results[method_name][i]['theta_deg']
            is_best = True
            for other_name in method_names:
                if other_name != method_name:
                    other_theta = all_results[other_name][i]['theta_deg']
                    if other_theta < theta - 0.1:
                        is_best = False
                        break
            if is_best:
                wins_list.append((config['name'], theta))
        
        print(f"\n{method_name}:")
        if wins_list:
            for config_name, theta in wins_list[:8]:
                print(f"  {config_name}: {theta:.1f}°")
            if len(wins_list) > 8:
                print(f"  ... and {len(wins_list) - 8} more")
        else:
            print("  (none)")


if __name__ == "__main__":
    np.random.seed(42)
    import warnings
    warnings.filterwarnings('ignore')
    
    main()
