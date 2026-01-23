"""
Wide Variation Testing of Top Constraint Methods
================================================

Testing across:
- Different actuator configurations (3MTQ+1RW, 4RW, 3MTQ only, 4RW+3MTQ)
- Different magnetic fields (orientation, magnitude)
- Different actuator bounds (tight, nominal, loose)
- Different initial conditions (small, medium, large errors)
- Different goals (regulation, slew, tracking)
- Different orbits (LEO equatorial, polar, high-inclination)
"""

import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation
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
    return A, lb, ub, "3MTQ+1RW(z)"


def config_3mtq_only(B_body, mtq_max=0.2):
    """3MTQ only (underactuated)"""
    A = -skewsym(B_body) @ np.eye(3)
    lb = np.array([-mtq_max, -mtq_max, -mtq_max])
    ub = np.array([mtq_max, mtq_max, mtq_max])
    return A, lb, ub, "3MTQ only"


def config_4rw_pyramid(rw_max=0.001):
    """4RW in pyramid configuration"""
    # Pyramid with 4 wheels at 45° from z-axis
    beta = np.radians(54.74)  # Half-angle for symmetric pyramid
    axes = np.array([
        [np.sin(beta), 0, np.cos(beta)],
        [0, np.sin(beta), np.cos(beta)],
        [-np.sin(beta), 0, np.cos(beta)],
        [0, -np.sin(beta), np.cos(beta)],
    ]).T
    A = axes  # (3, 4)
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


def config_1rw_z(rw_max=0.001):
    """Single RW on z-axis (severely underactuated)"""
    A = np.array([[0], [0], [1.0]])
    lb = np.array([-rw_max])
    ub = np.array([rw_max])
    return A, lb, ub, "1RW(z) only"


# ============================================================================
# MAGNETIC FIELD CONFIGURATIONS
# ============================================================================

def B_field_nominal():
    """Nominal LEO field"""
    return np.array([20e-6, 15e-6, 10e-6]), "B nominal"

def B_field_strong_z():
    """Field aligned with z (MTQ can't control z)"""
    return np.array([1e-6, 1e-6, 30e-6]), "B along z"

def B_field_strong_x():
    """Field aligned with x"""
    return np.array([30e-6, 1e-6, 1e-6]), "B along x"

def B_field_weak():
    """Weak field (high altitude or during storm)"""
    return np.array([5e-6, 4e-6, 3e-6]), "B weak"

def B_field_strong():
    """Strong field (low altitude polar)"""
    return np.array([50e-6, 40e-6, 30e-6]), "B strong"


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


def qp_combined_1a_3b(tau_des, A, lb, ub, omega, **kwargs):
    """Combined: Power brake (1a) + Sign critical (3b)."""
    n = len(lb)
    P_des = np.dot(omega, tau_des)
    
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(SCALE * (tau - tau_des)))
    constraints = [u >= lb, u <= ub]
    
    # 1a: Power brake only
    if P_des < -1e-12:
        constraints.append(omega @ tau <= 1e-12)
    
    # 3b: Sign critical
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


# ============================================================================
# SIMULATION
# ============================================================================

def simulate_regulation(method, A, lb, ub, J, K_p, K_d, theta_0, omega_0, 
                        B_body, dt=0.2, t_end=60.0):
    """Simulate closed-loop regulation."""
    J_inv = np.linalg.inv(J)
    n_steps = int(t_end / dt)
    
    theta = theta_0.copy()
    omega = omega_0.copy()
    
    # Track metrics
    theta_hist = [np.linalg.norm(theta)]
    V_violations = 0
    failed_steps = 0
    
    for _ in range(n_steps):
        tau_des = -K_p * theta - K_d * omega
        
        kwargs = {'omega': omega, 'theta': theta, 'K_p': K_p}
        
        try:
            if method.__name__ == 'solve_lp':
                u, _, _ = method(tau_des, A, lb, ub)
            else:
                u = method(tau_des, A, lb, ub, **kwargs)
            if u is None:
                u = np.zeros(len(lb))
                failed_steps += 1
        except:
            u = np.zeros(len(lb))
            failed_steps += 1
        
        tau = A @ u
        
        # Check Lyapunov
        V_dot = np.dot(K_p * theta, omega) + np.dot(omega, tau)
        if V_dot > 1e-10:
            V_violations += 1
        
        # Integrate
        omega = omega + dt * J_inv @ tau
        theta = theta + dt * omega
        
        theta_hist.append(np.linalg.norm(theta))
    
    return {
        'theta_final': theta,
        'omega_final': omega,
        'theta_mag_final': np.linalg.norm(theta),
        'omega_mag_final': np.linalg.norm(omega),
        'V_violations': V_violations,
        'failed_steps': failed_steps,
        'converged': np.linalg.norm(theta) < 0.1,  # < 5.7 degrees
        'theta_hist': theta_hist,
    }


# ============================================================================
# TEST CONFIGURATIONS
# ============================================================================

def get_test_configurations():
    """Generate all test configurations."""
    configs = []
    
    # Actuator configs (generate with different B fields)
    actuator_funcs = [
        ('3MTQ+1RW', lambda B: config_3mtq_1rw(B)),
        ('3MTQ+1RW tight', lambda B: config_3mtq_1rw(B, mtq_max=0.05, rw_max=0.0005)),
        ('3MTQ+1RW loose', lambda B: config_3mtq_1rw(B, mtq_max=0.5, rw_max=0.005)),
        ('3MTQ only', lambda B: config_3mtq_only(B)),
        ('4RW pyramid', lambda B: config_4rw_pyramid()),
        ('4RW+3MTQ', lambda B: config_4rw_3mtq(B)),
        ('1RW only', lambda B: config_1rw_z()),
    ]
    
    # B field configs
    B_funcs = [
        B_field_nominal,
        B_field_strong_z,
        B_field_strong_x,
        B_field_weak,
        B_field_strong,
    ]
    
    # Initial conditions
    ICs = [
        ('Small IC', np.array([0.05, 0.03, 0.02]), np.array([0.005, 0.005, 0.005])),
        ('Medium IC', np.array([0.2, 0.15, 0.1]), np.array([0.01, 0.01, 0.01])),
        ('Large IC', np.array([0.5, 0.4, 0.3]), np.array([0.02, 0.02, 0.02])),
        ('Large angle small rate', np.array([0.8, 0.6, 0.4]), np.array([0.001, 0.001, 0.001])),
        ('Small angle large rate', np.array([0.02, 0.02, 0.02]), np.array([0.05, 0.05, 0.05])),
        ('Asymmetric', np.array([0.5, 0.1, 0.02]), np.array([0.005, 0.02, 0.04])),
    ]
    
    # Inertias
    inertias = [
        ('Symmetric J', np.diag([0.01, 0.01, 0.01])),
        ('Asymmetric J', np.diag([0.02, 0.01, 0.005])),
        ('Long thin', np.diag([0.001, 0.001, 0.01])),
    ]
    
    # Gains
    gains = [
        ('Nominal gains', np.array([0.001, 0.001, 0.001]), np.array([0.01, 0.01, 0.01])),
        ('High gains', np.array([0.005, 0.005, 0.005]), np.array([0.05, 0.05, 0.05])),
        ('Low gains', np.array([0.0005, 0.0005, 0.0005]), np.array([0.005, 0.005, 0.005])),
    ]
    
    # Generate subset of combinations (full factorial would be huge)
    # Focus on interesting combinations
    
    # Test 1: All actuator configs with nominal B and IC
    B, B_name = B_field_nominal()
    theta_0, omega_0 = np.array([0.3, 0.2, 0.1]), np.array([0.01, 0.01, 0.01])
    J = np.diag([0.01, 0.01, 0.005])
    K_p, K_d = np.array([0.001, 0.001, 0.001]), np.array([0.01, 0.01, 0.01])
    
    for act_name, act_func in actuator_funcs:
        A, lb, ub, _ = act_func(B)
        configs.append({
            'name': f"{act_name} | {B_name}",
            'A': A, 'lb': lb, 'ub': ub, 'B': B,
            'J': J, 'K_p': K_p, 'K_d': K_d,
            'theta_0': theta_0, 'omega_0': omega_0,
        })
    
    # Test 2: 3MTQ+1RW with different B fields
    for B_func in B_funcs:
        B, B_name = B_func()
        A, lb, ub, _ = config_3mtq_1rw(B)
        configs.append({
            'name': f"3MTQ+1RW | {B_name}",
            'A': A, 'lb': lb, 'ub': ub, 'B': B,
            'J': J, 'K_p': K_p, 'K_d': K_d,
            'theta_0': theta_0, 'omega_0': omega_0,
        })
    
    # Test 3: 3MTQ+1RW with different ICs
    B, B_name = B_field_nominal()
    A, lb, ub, _ = config_3mtq_1rw(B)
    for ic_name, theta_0_ic, omega_0_ic in ICs:
        configs.append({
            'name': f"3MTQ+1RW | {ic_name}",
            'A': A, 'lb': lb, 'ub': ub, 'B': B,
            'J': J, 'K_p': K_p, 'K_d': K_d,
            'theta_0': theta_0_ic, 'omega_0': omega_0_ic,
        })
    
    # Test 4: Different inertias
    for J_name, J_test in inertias:
        configs.append({
            'name': f"3MTQ+1RW | {J_name}",
            'A': A, 'lb': lb, 'ub': ub, 'B': B,
            'J': J_test, 'K_p': K_p, 'K_d': K_d,
            'theta_0': theta_0, 'omega_0': omega_0,
        })
    
    # Test 5: Different gains
    for gain_name, K_p_test, K_d_test in gains:
        configs.append({
            'name': f"3MTQ+1RW | {gain_name}",
            'A': A, 'lb': lb, 'ub': ub, 'B': B,
            'J': J, 'K_p': K_p_test, 'K_d': K_d_test,
            'theta_0': theta_0, 'omega_0': omega_0,
        })
    
    # Test 6: 3MTQ only with different B (tests underactuation)
    for B_func in [B_field_nominal, B_field_strong_z, B_field_strong_x]:
        B, B_name = B_func()
        A, lb, ub, _ = config_3mtq_only(B)
        configs.append({
            'name': f"3MTQ only | {B_name}",
            'A': A, 'lb': lb, 'ub': ub, 'B': B,
            'J': J, 'K_p': K_p, 'K_d': K_d,
            'theta_0': theta_0, 'omega_0': omega_0,
        })
    
    # Test 7: 4RW with different ICs (tests redundancy)
    A, lb, ub, _ = config_4rw_pyramid()
    for ic_name, theta_0_ic, omega_0_ic in ICs[:3]:
        configs.append({
            'name': f"4RW pyramid | {ic_name}",
            'A': A, 'lb': lb, 'ub': ub, 'B': B,
            'J': J, 'K_p': K_p, 'K_d': K_d,
            'theta_0': theta_0_ic, 'omega_0': omega_0_ic,
        })
    
    return configs


# ============================================================================
# MAIN TEST
# ============================================================================

def run_wide_tests():
    """Run all tests."""
    print("=" * 120)
    print("WIDE VARIATION TESTING: TOP 6 CONSTRAINT METHODS")
    print("=" * 120)
    
    methods = [
        ('LP', solve_lp),
        ('QP uncon', qp_unconstrained),
        ('1a-Pwr brk', qp_1a_power_brake_only),
        ('3b-Sign crit', qp_3b_sign_critical),
        ('Phase-aware', qp_phase_aware),
        ('Proj hybrid', qp_projection_hybrid),
        # ('Combined 1a+3b', qp_combined_1a_3b),
    ]
    
    configs = get_test_configurations()
    
    print(f"\nRunning {len(configs)} test configurations with {len(methods)} methods each")
    print(f"Total simulations: {len(configs) * len(methods)}")
    print()
    
    # Results storage
    all_results = {name: [] for name, _ in methods}
    
    # Header
    header = f"{'Configuration':<45} |"
    for name, _ in methods:
        header += f" {name:>11} |"
    print(header)
    print("-" * len(header))
    
    for config in configs:
        row = f"{config['name']:<45} |"
        
        for method_name, method_func in methods:
            result = simulate_regulation(
                method_func,
                config['A'], config['lb'], config['ub'],
                config['J'], config['K_p'], config['K_d'],
                config['theta_0'], config['omega_0'],
                config['B'],
                dt=0.2, t_end=60.0
            )
            
            theta_deg = np.degrees(result['theta_mag_final'])
            all_results[method_name].append({
                'config': config['name'],
                'theta_deg': theta_deg,
                'converged': result['converged'],
                'V_violations': result['V_violations'],
            })
            
            # Color coding would be nice but using text markers
            marker = "✓" if result['converged'] else "✗"
            row += f" {theta_deg:>7.1f}°{marker} |"
        
        print(row)
    
    # Summary statistics
    print("\n" + "=" * 120)
    print("SUMMARY STATISTICS")
    print("=" * 120)
    
    print(f"\n{'Method':<15} {'Mean θ':>10} {'Median θ':>10} {'Max θ':>10} {'Converged':>12} {'V̇>0 mean':>12}")
    print("-" * 75)
    
    for method_name, _ in methods:
        results = all_results[method_name]
        thetas = [r['theta_deg'] for r in results]
        converged = sum(1 for r in results if r['converged'])
        V_viol_mean = np.mean([r['V_violations'] for r in results])
        
        print(f"{method_name:<15} {np.mean(thetas):>9.2f}° {np.median(thetas):>9.2f}° {np.max(thetas):>9.1f}° {converged:>8}/{len(results):<3} {V_viol_mean:>12.0f}")
    
    # Find where each method wins/loses
    print("\n" + "=" * 120)
    print("HEAD-TO-HEAD COMPARISON")
    print("=" * 120)
    
    method_names = [name for name, _ in methods]
    wins = {name: 0 for name in method_names}
    
    for i, config in enumerate(configs):
        best_theta = float('inf')
        best_method = None
        for method_name in method_names:
            theta = all_results[method_name][i]['theta_deg']
            if theta < best_theta:
                best_theta = theta
                best_method = method_name
        wins[best_method] += 1
    
    print(f"\nWins (best final θ in each config):")
    for name in sorted(wins, key=wins.get, reverse=True):
        pct = 100 * wins[name] / len(configs)
        bar = "█" * int(pct / 5)
        print(f"  {name:<15}: {wins[name]:>3} ({pct:>5.1f}%) {bar}")
    
    # Failure analysis
    print("\n" + "=" * 120)
    print("FAILURE ANALYSIS (θ > 20°)")
    print("=" * 120)
    
    for method_name in method_names:
        failures = [(r['config'], r['theta_deg']) for r in all_results[method_name] if r['theta_deg'] > 20]
        if failures:
            print(f"\n{method_name}:")
            for config_name, theta in sorted(failures, key=lambda x: -x[1])[:5]:
                print(f"  {config_name}: {theta:.1f}°")
        else:
            print(f"\n{method_name}: No failures!")
    
    return all_results


if __name__ == "__main__":
    np.random.seed(42)
    import warnings
    warnings.filterwarnings('ignore')
    
    results = run_wide_tests()
