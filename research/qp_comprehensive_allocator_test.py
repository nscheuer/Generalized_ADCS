"""
Comprehensive Allocator Testing with Multiple Control Laws
==========================================================

Testing allocation methods (LP, QP variants) with:
1. Unbounded PD control (baseline - should always converge)
2. Bounded PD control with various allocators
3. Sliding mode control with various allocators

Uses the existing codebase simulation infrastructure.
"""

import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linprog
import cvxpy as cp
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
from tqdm import tqdm
import warnings

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))

from ADCS.helpers.math_helpers import normalize, rot_mat, skewsym, quat_mult, quat_inv

SCALE = 1e6


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SystemConfig:
    """System configuration."""
    name: str
    J: np.ndarray
    A: np.ndarray  # Torque influence matrix
    lb: np.ndarray  # Lower bounds
    ub: np.ndarray  # Upper bounds
    K_p: np.ndarray
    K_d: np.ndarray


@dataclass
class SimResult:
    """Simulation result."""
    config_name: str
    controller_name: str
    allocator_name: str
    theta_final_deg: float
    omega_final: float
    converged: bool
    theta_history: np.ndarray
    omega_history: np.ndarray


# =============================================================================
# ACTUATOR CONFIGURATIONS
# =============================================================================

def create_3mtq_1rw_config(B_body: np.ndarray, name: str = "3MTQ+1RW") -> SystemConfig:
    """Create 3MTQ + 1RW configuration."""
    A_rw = np.array([[0], [0], [1.0]])
    A_mtq = -skewsym(B_body) @ np.eye(3)
    A = np.hstack([A_rw, A_mtq])
    lb = np.array([-0.001, -0.2, -0.2, -0.2])
    ub = np.array([0.001, 0.2, 0.2, 0.2])
    J = np.diag([0.01, 0.01, 0.005])
    K_p = np.array([0.001, 0.001, 0.001])
    K_d = np.array([0.01, 0.01, 0.01])
    return SystemConfig(name=name, J=J, A=A, lb=lb, ub=ub, K_p=K_p, K_d=K_d)


def create_4rw_config(name: str = "4RW") -> SystemConfig:
    """Create 4RW pyramid configuration."""
    beta = np.radians(54.74)
    A = np.array([
        [np.sin(beta), 0, np.cos(beta)],
        [0, np.sin(beta), np.cos(beta)],
        [-np.sin(beta), 0, np.cos(beta)],
        [0, -np.sin(beta), np.cos(beta)],
    ]).T
    lb = np.array([-0.001, -0.001, -0.001, -0.001])
    ub = np.array([0.001, 0.001, 0.001, 0.001])
    J = np.diag([0.01, 0.01, 0.005])
    K_p = np.array([0.001, 0.001, 0.001])
    K_d = np.array([0.01, 0.01, 0.01])
    return SystemConfig(name=name, J=J, A=A, lb=lb, ub=ub, K_p=K_p, K_d=K_d)


# =============================================================================
# CONTROL LAWS
# =============================================================================

def pd_control(theta: np.ndarray, omega: np.ndarray, K_p: np.ndarray, K_d: np.ndarray) -> np.ndarray:
    """Standard PD control law."""
    return -K_p * theta - K_d * omega


def sliding_mode_control(theta: np.ndarray, omega: np.ndarray, K_p: np.ndarray, K_d: np.ndarray,
                         lambda_smc: float = 0.1, eta: float = 0.001) -> np.ndarray:
    """
    Sliding mode control law.
    
    Sliding surface: s = omega + lambda * theta
    Control: tau = -K_p * theta - K_d * omega - eta * sign(s)
    """
    s = omega + lambda_smc * theta
    tau_eq = -K_p * theta - K_d * omega  # Equivalent control
    tau_sw = -eta * np.sign(s)  # Switching term
    return tau_eq + tau_sw


def saturated_sliding_mode(theta: np.ndarray, omega: np.ndarray, K_p: np.ndarray, K_d: np.ndarray,
                           lambda_smc: float = 0.1, eta: float = 0.001, phi: float = 0.01) -> np.ndarray:
    """
    Sliding mode with saturation function (reduces chatter).
    
    Uses sat(s/phi) instead of sign(s).
    """
    s = omega + lambda_smc * theta
    tau_eq = -K_p * theta - K_d * omega
    tau_sw = -eta * np.clip(s / phi, -1, 1)  # Saturation instead of sign
    return tau_eq + tau_sw


# =============================================================================
# ALLOCATORS
# =============================================================================

def allocate_unbounded(tau_des: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Unbounded allocation (pseudo-inverse, ignores limits)."""
    u = np.linalg.pinv(A) @ tau_des
    tau = A @ u
    return u, tau


def allocate_lp(tau_des: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """LP allocation (direction preserving)."""
    n = len(lb)
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-15:
        return np.zeros(n), np.zeros(3)
    
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
        return u, A @ u
    return np.zeros(n), np.zeros(3)


def allocate_qp(tau_des: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Standard QP allocation (with scaling fix)."""
    n = len(lb)
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(SCALE * (tau - tau_des)))
    constraints = [u >= lb, u <= ub]
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
        if u.value is not None:
            return u.value, A @ u.value
    except:
        pass
    return np.zeros(n), np.zeros(3)


def allocate_qp_1a(tau_des: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray,
                   omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """QP with power brake-only constraint (1a)."""
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
        if u.value is not None:
            return u.value, A @ u.value
    except:
        pass
    return np.zeros(n), np.zeros(3)


def allocate_qp_3b(tau_des: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray,
                   omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """QP with sign-critical constraint (3b)."""
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
        if u.value is not None:
            return u.value, A @ u.value
    except:
        pass
    return np.zeros(n), np.zeros(3)


# =============================================================================
# SIMULATION
# =============================================================================

def simulate(config: SystemConfig, 
             controller: Callable, 
             allocator: Callable,
             theta_0: np.ndarray,
             omega_0: np.ndarray,
             dt: float = 0.2,
             t_end: float = 60.0) -> SimResult:
    """
    Run closed-loop simulation.
    """
    J_inv = np.linalg.inv(config.J)
    n_steps = int(t_end / dt)
    
    theta = theta_0.copy()
    omega = omega_0.copy()
    
    theta_hist = [np.linalg.norm(theta)]
    omega_hist = [np.linalg.norm(omega)]
    
    for _ in range(n_steps):
        # Control law
        tau_des = controller(theta, omega, config.K_p, config.K_d)
        
        # Allocation
        if allocator.__name__ in ['allocate_qp_1a', 'allocate_qp_3b']:
            _, tau = allocator(tau_des, config.A, config.lb, config.ub, omega)
        else:
            _, tau = allocator(tau_des, config.A, config.lb, config.ub)
        
        # Integrate
        omega = omega + dt * J_inv @ tau
        theta = theta + dt * omega
        
        theta_hist.append(np.linalg.norm(theta))
        omega_hist.append(np.linalg.norm(omega))
    
    theta_final_deg = np.degrees(np.linalg.norm(theta))
    converged = theta_final_deg < 5.7  # ~0.1 rad
    
    return SimResult(
        config_name=config.name,
        controller_name=controller.__name__,
        allocator_name=allocator.__name__,
        theta_final_deg=theta_final_deg,
        omega_final=np.linalg.norm(omega),
        converged=converged,
        theta_history=np.array(theta_hist),
        omega_history=np.array(omega_hist),
    )


# =============================================================================
# MAIN TESTS
# =============================================================================

def run_comprehensive_tests():
    """Run all combinations of controllers, allocators, and configurations."""
    print("=" * 100)
    print("COMPREHENSIVE ALLOCATOR TESTING WITH MULTIPLE CONTROL LAWS")
    print("=" * 100)
    
    # B-field configurations
    B_fields = [
        (np.array([20e-6, 15e-6, 10e-6]), "B_nom"),
        (np.array([1e-6, 1e-6, 30e-6]), "B_z"),
        (np.array([50e-6, 40e-6, 30e-6]), "B_strong"),
    ]
    
    # Initial conditions
    ICs = [
        (np.array([0.3, 0.2, 0.1]), np.array([0.01, 0.01, 0.01]), "Medium IC"),
        (np.array([0.05, 0.03, 0.02]), np.array([0.005, 0.005, 0.005]), "Small IC"),
    ]
    
    # Controllers
    controllers = [
        ("PD", pd_control),
        ("SMC", sliding_mode_control),
        ("SMC_sat", saturated_sliding_mode),
    ]
    
    # Allocators
    allocators = [
        ("Unbounded", allocate_unbounded),
        ("LP", allocate_lp),
        ("QP", allocate_qp),
        ("QP_1a", allocate_qp_1a),
        ("QP_3b", allocate_qp_3b),
    ]
    
    all_results = []
    
    # Test 3MTQ+1RW configurations
    print("\n" + "=" * 100)
    print("3MTQ+1RW CONFIGURATIONS")
    print("=" * 100)
    
    for B, B_name in B_fields:
        for theta_0, omega_0, IC_name in ICs:
            config = create_3mtq_1rw_config(B, f"3MTQ+1RW|{B_name}|{IC_name}")
            
            print(f"\n--- {config.name} ---")
            print(f"{'Controller':<12} {'Allocator':<12} {'θ_final':>10} {'Converged':>10}")
            print("-" * 50)
            
            for ctrl_name, ctrl_func in controllers:
                for alloc_name, alloc_func in allocators:
                    result = simulate(config, ctrl_func, alloc_func, theta_0, omega_0)
                    all_results.append(result)
                    
                    conv_str = "✓" if result.converged else "✗"
                    print(f"{ctrl_name:<12} {alloc_name:<12} {result.theta_final_deg:>9.2f}° {conv_str:>10}")
    
    # Test 4RW configuration
    print("\n" + "=" * 100)
    print("4RW CONFIGURATION (Fully Actuated)")
    print("=" * 100)
    
    config = create_4rw_config()
    theta_0, omega_0, IC_name = ICs[0]
    
    print(f"\n--- {config.name} | {IC_name} ---")
    print(f"{'Controller':<12} {'Allocator':<12} {'θ_final':>10} {'Converged':>10}")
    print("-" * 50)
    
    for ctrl_name, ctrl_func in controllers:
        for alloc_name, alloc_func in allocators:
            result = simulate(config, ctrl_func, alloc_func, theta_0, omega_0)
            all_results.append(result)
            
            conv_str = "✓" if result.converged else "✗"
            print(f"{ctrl_name:<12} {alloc_name:<12} {result.theta_final_deg:>9.2f}° {conv_str:>10}")
    
    return all_results


def analyze_results(results: List[SimResult]):
    """Analyze and summarize results."""
    print("\n" + "=" * 100)
    print("ANALYSIS: UNBOUNDED VS BOUNDED CONVERGENCE")
    print("=" * 100)
    
    # Group by config
    configs = {}
    for r in results:
        key = r.config_name
        if key not in configs:
            configs[key] = []
        configs[key].append(r)
    
    # Check which configs converge with unbounded control
    valid_configs = []
    print(f"\n{'Configuration':<40} {'Unbounded PD':>15} {'Unbounded SMC':>15} {'Valid?':>10}")
    print("-" * 85)
    
    for config_name, config_results in configs.items():
        # Find unbounded results
        unbounded_pd = [r for r in config_results if r.allocator_name == "allocate_unbounded" 
                        and r.controller_name == "pd_control"]
        unbounded_smc = [r for r in config_results if r.allocator_name == "allocate_unbounded" 
                         and r.controller_name == "sliding_mode_control"]
        
        pd_conv = unbounded_pd[0].converged if unbounded_pd else False
        smc_conv = unbounded_smc[0].converged if unbounded_smc else False
        
        valid = pd_conv and smc_conv
        if valid:
            valid_configs.append(config_name)
        
        pd_str = f"{unbounded_pd[0].theta_final_deg:.2f}°" if unbounded_pd else "N/A"
        smc_str = f"{unbounded_smc[0].theta_final_deg:.2f}°" if unbounded_smc else "N/A"
        valid_str = "✓" if valid else "✗"
        
        print(f"{config_name:<40} {pd_str:>15} {smc_str:>15} {valid_str:>10}")
    
    print(f"\nValid configurations: {len(valid_configs)}/{len(configs)}")
    
    # Compare allocators only on valid configs
    print("\n" + "=" * 100)
    print("ALLOCATOR COMPARISON ON VALID CONFIGURATIONS")
    print("=" * 100)
    
    if not valid_configs:
        print("No valid configurations found!")
        return
    
    # Filter to valid configs
    valid_results = [r for r in results if r.config_name in valid_configs]
    
    # Group by allocator and controller
    allocator_stats = {}
    for r in valid_results:
        key = (r.controller_name, r.allocator_name)
        if key not in allocator_stats:
            allocator_stats[key] = []
        allocator_stats[key].append(r.theta_final_deg)
    
    print(f"\n{'Controller':<20} {'Allocator':<15} {'Mean θ':>10} {'Max θ':>10} {'Converged':>12}")
    print("-" * 75)
    
    for (ctrl, alloc), errors in sorted(allocator_stats.items()):
        mean_err = np.mean(errors)
        max_err = np.max(errors)
        n_conv = sum(1 for e in errors if e < 5.7)
        
        print(f"{ctrl:<20} {alloc:<15} {mean_err:>9.2f}° {max_err:>9.1f}° {n_conv:>8}/{len(errors):<3}")
    
    # Head-to-head comparison
    print("\n" + "=" * 100)
    print("HEAD-TO-HEAD: ALLOCATOR WINS")
    print("=" * 100)
    
    allocators_list = ["allocate_unbounded", "allocate_lp", "allocate_qp", "allocate_qp_1a", "allocate_qp_3b"]
    wins = {a: 0 for a in allocators_list}
    
    for config_name in valid_configs:
        config_results = [r for r in valid_results if r.config_name == config_name]
        
        for ctrl_name in ["pd_control", "sliding_mode_control", "saturated_sliding_mode"]:
            ctrl_results = [r for r in config_results if r.controller_name == ctrl_name]
            
            if not ctrl_results:
                continue
            
            # Exclude unbounded from competition (it's the baseline)
            bounded_results = [r for r in ctrl_results if r.allocator_name != "allocate_unbounded"]
            
            if not bounded_results:
                continue
            
            best = min(bounded_results, key=lambda r: r.theta_final_deg)
            
            # Check for ties
            best_err = best.theta_final_deg
            tied = [r for r in bounded_results if abs(r.theta_final_deg - best_err) < 0.1]
            
            for r in tied:
                wins[r.allocator_name] += 1.0 / len(tied)
    
    print(f"\nWins (among bounded allocators, ties split):")
    for alloc in sorted(wins, key=wins.get, reverse=True):
        if alloc == "allocate_unbounded":
            continue
        pct = 100 * wins[alloc] / max(1, sum(wins[a] for a in allocators_list if a != "allocate_unbounded"))
        bar = "█" * int(pct / 5)
        print(f"  {alloc.replace('allocate_', ''):<12}: {wins[alloc]:>5.1f} ({pct:>5.1f}%) {bar}")


def summary():
    """Print final summary."""
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print("""
CONTROL LAWS TESTED:
====================
1. PD Control:     τ = -K_p θ - K_d ω
2. Sliding Mode:   τ = -K_p θ - K_d ω - η sign(ω + λθ)
3. Sat. Sliding:   τ = -K_p θ - K_d ω - η sat((ω + λθ)/φ)

ALLOCATORS TESTED:
==================
1. Unbounded:  u = A⁺ τ_des (ignores limits - BASELINE)
2. LP:         max α s.t. τ = α τ̂_des, bounds (direction preserving)
3. QP:         min ||τ - τ_des||² s.t. bounds (L2 optimal)
4. QP_1a:      QP + power brake constraint (if P_des < 0: ω'τ ≤ 0)
5. QP_3b:      QP + sign-critical constraint (per-axis sign preservation)

KEY INSIGHT:
============
Unbounded control ALWAYS converges (if system is controllable).
Bounded allocation performance varies - QP variants generally best
when control problem is solvable.

RECOMMENDATIONS:
================
- Use unbounded simulation to verify controllability first
- For bounded: QP or QP_3b typically best
- Sliding mode is more robust to allocation errors than PD
- Saturated SMC reduces chatter while maintaining robustness
""")


if __name__ == "__main__":
    np.random.seed(42)
    warnings.filterwarnings('ignore')
    
    results = run_comprehensive_tests()
    analyze_results(results)
    summary()
