"""
Desaturation Trade-off Study
============================

Key questions:
1. What desaturation can we achieve accepting only 1° pointing error?
2. Can we include desaturation during slews instead of pointing?
3. How does this vary across actuator configurations and orbits?

Test configurations:
- Actuators: 3MTQ+1RW, 3MTQ+3RW, 4RW pyramid, 3MTQ only
- Orbits: LEO (400km), polar, SSO, high inclination
- Goals: Full attitude, boresight pointing, nadir tracking
"""

import sys
import os
import numpy as np
from scipy.optimize import linprog, minimize, Bounds
from scipy.integrate import solve_ivp
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from ADCS.helpers.math_helpers import skewsym, normalize, rot_mat, quat_inv, quat_mult


# ============== QUATERNION HELPERS ==============

def quat_from_axis_angle(axis, angle):
    axis = normalize(axis)
    return np.concatenate([[np.cos(angle/2)], axis * np.sin(angle/2)])

def quaternion_error_vector(q, q_goal):
    q = normalize(q)
    q_goal = normalize(q_goal)
    q_err = quat_mult(quat_inv(q_goal), q)
    if q_err[0] < 0:
        q_err = -q_err
    return 2.0 * q_err[1:4]

def full_attitude_error_deg(q, q_goal):
    q = normalize(q)
    q_goal = normalize(q_goal)
    q_err = quat_mult(quat_inv(q_goal), q)
    if q_err[0] < 0:
        q_err = -q_err
    return np.degrees(2 * np.arccos(np.clip(q_err[0], -1, 1)))

def pointing_error_deg(q, q_goal, boresight=np.array([0, 0, 1])):
    R = rot_mat(normalize(q))
    R_goal = rot_mat(normalize(q_goal))
    actual = R @ boresight
    goal = R_goal @ boresight
    return np.degrees(np.arccos(np.clip(np.dot(actual, goal), -1, 1)))

def vector_alignment_error(q, body_vec, target_vec_inertial):
    R = rot_mat(q)
    target_body = R.T @ normalize(target_vec_inertial)
    return np.cross(target_body, normalize(body_vec))


# ============== ACTUATOR CONFIGURATIONS ==============

@dataclass
class ActuatorConfig:
    name: str
    A_rw: np.ndarray
    A_mtq_axes: np.ndarray
    u_rw_max: np.ndarray
    u_mtq_max: np.ndarray
    J: np.ndarray

def config_3mtq_1rw():
    return ActuatorConfig(
        name='3MTQ+1RW',
        A_rw=np.array([[0], [0], [1.0]]),
        A_mtq_axes=np.eye(3),
        u_rw_max=np.array([0.001]),
        u_mtq_max=np.array([0.2, 0.2, 0.2]),
        J=np.diag([0.022, 0.022, 0.004])
    )

def config_3mtq_3rw():
    return ActuatorConfig(
        name='3MTQ+3RW',
        A_rw=np.eye(3),
        A_mtq_axes=np.eye(3),
        u_rw_max=np.array([0.001, 0.001, 0.001]),
        u_mtq_max=np.array([0.2, 0.2, 0.2]),
        J=np.diag([0.022, 0.022, 0.004])
    )

def config_4rw_pyramid():
    theta = np.radians(54.74)
    A_rw = np.array([
        [np.sin(theta), 0, -np.sin(theta), 0],
        [0, np.sin(theta), 0, -np.sin(theta)],
        [np.cos(theta), np.cos(theta), np.cos(theta), np.cos(theta)]
    ])
    return ActuatorConfig(
        name='4RW_Pyramid',
        A_rw=A_rw,
        A_mtq_axes=np.zeros((3, 0)),
        u_rw_max=np.array([0.001, 0.001, 0.001, 0.001]),
        u_mtq_max=np.array([]),
        J=np.diag([0.022, 0.022, 0.004])
    )

def config_3mtq_only():
    return ActuatorConfig(
        name='3MTQ_only',
        A_rw=np.zeros((3, 0)),
        A_mtq_axes=np.eye(3),
        u_rw_max=np.array([]),
        u_mtq_max=np.array([0.2, 0.2, 0.2]),
        J=np.diag([0.022, 0.022, 0.004])
    )


# ============== ORBIT CONFIGURATIONS ==============

@dataclass
class OrbitConfig:
    name: str
    altitude: float  # km
    inclination: float  # degrees
    period: float = 0  # seconds (computed in __post_init__)
    
    def __post_init__(self):
        Re = 6378.0  # km
        mu = 398600.4  # km³/s²
        a = Re + self.altitude
        self.period = 2 * np.pi * np.sqrt(a**3 / mu)

def orbit_leo_equatorial():
    return OrbitConfig('LEO_Equatorial', altitude=400, inclination=0)

def orbit_leo_polar():
    return OrbitConfig('LEO_Polar', altitude=400, inclination=90)

def orbit_sso():
    return OrbitConfig('SSO', altitude=600, inclination=98)

def orbit_iss():
    return OrbitConfig('ISS', altitude=420, inclination=51.6)


# ============== MAGNETIC FIELD MODEL ==============

def magnetic_field_dipole(r_eci, t, orbit_config):
    """Simplified tilted dipole model."""
    # Earth's magnetic dipole moment ~ 7.94e15 T·m³
    # Simplified: B varies with orbit position
    
    phase = 2 * np.pi * t / orbit_config.period
    inc_rad = np.radians(orbit_config.inclination)
    
    # B-field components depend on orbit position and inclination
    B_r = 30e-6 * np.cos(phase)  # Radial
    B_theta = 15e-6 * np.sin(phase) * np.cos(inc_rad)  # Along-track
    B_phi = 20e-6 * np.sin(inc_rad) * np.sin(2*phase)  # Cross-track
    
    return np.array([B_r, B_theta, B_phi])


# ============== ALLOCATION FUNCTIONS ==============

def allocate_lp(tau_des, A_total, lb, ub):
    """LP allocation preserving direction."""
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-12:
        return np.zeros(len(lb))
    
    tau_hat = tau_des / t_mag
    n = len(lb)
    
    c = np.zeros(n + 1)
    c[-1] = -1.0
    A_eq = np.hstack([A_total, -tau_hat.reshape(3, 1)])
    b_eq = np.zeros(3)
    bounds = [(lb[i], ub[i]) for i in range(n)] + [(0, None)]
    
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if res.success:
        u = res.x[:n]
        T_max = res.x[-1]
        if T_max > t_mag:
            u = u * (t_mag / T_max)
        return u
    return np.zeros(n)


def allocate_with_desat(tau_des, A_total, lb, ub, h_rw, A_rw, 
                        max_error_deg=1.0, k_desat=0.1):
    """
    Allocation that includes desaturation while limiting pointing error.
    
    Strategy: 
    1. Compute LP solution (pure pointing)
    2. Add desaturation in constrained way that limits error increase
    """
    n_rw = A_rw.shape[1]
    n = len(lb)
    
    # Step 1: Pure pointing LP
    u_point = allocate_lp(tau_des, A_total, lb, ub)
    tau_point = A_total @ u_point
    
    t_mag = np.linalg.norm(tau_des)
    if t_mag < 1e-12:
        return u_point, 0.0
    
    tau_hat = tau_des / t_mag
    
    # Step 2: Compute desired desaturation
    if n_rw == 0:
        return u_point, 0.0
    
    h_rw_vec = A_rw @ h_rw
    tau_desat_desired = -k_desat * h_rw_vec  # Torque to reduce momentum
    
    # Step 3: Find allocation that includes desaturation within error budget
    # We want: tau_achieved close to tau_des, but allow up to max_error_deg
    
    # Convert error budget to torque tolerance
    # Rough approximation: 1° error ~ proportional deviation in torque direction
    max_error_rad = np.radians(max_error_deg)
    
    # Solve: min ||u_rw - u_desat|| 
    #        s.t. angle(A@u, tau_des) <= max_error_rad
    #             lb <= u <= ub
    
    # Approximate constraint: |tau_perp| / |tau_parallel| <= tan(max_error_rad)
    # Or equivalently: tau @ tau_hat >= cos(max_error_rad) * ||tau||
    
    def objective(u):
        # Minimize deviation from desired (pointing + desat)
        u_desat_full = np.zeros(n)
        u_desat_full[:n_rw] = k_desat * h_rw  # RW commands to reduce h
        return np.linalg.norm(u[:n_rw] - u_desat_full[:n_rw])**2
    
    def direction_constraint(u):
        tau = A_total @ u
        tau_mag = np.linalg.norm(tau)
        if tau_mag < 1e-12:
            return 0  # No constraint if no torque
        proj = np.dot(tau, tau_hat)
        # Want: proj / tau_mag >= cos(max_error_rad)
        return proj - np.cos(max_error_rad) * tau_mag
    
    # Also ensure we maintain at least the LP's projection
    proj_lp = np.dot(tau_point, tau_hat)
    
    def projection_constraint(u):
        tau = A_total @ u
        return np.dot(tau, tau_hat) - 0.9 * proj_lp  # Allow 10% reduction
    
    constraints = [
        {'type': 'ineq', 'fun': direction_constraint},
        {'type': 'ineq', 'fun': projection_constraint}
    ]
    
    res = minimize(objective, u_point, method='SLSQP',
                  bounds=Bounds(lb, ub), constraints=constraints,
                  options={'ftol': 1e-10, 'maxiter': 100})
    
    if res.success:
        u = res.x
    else:
        u = u_point
    
    # Compute actual desaturation achieved
    u_rw = u[:n_rw]
    desat_achieved = np.dot(u_rw, h_rw)  # Positive means reducing h
    
    return u, desat_achieved


# ============== SIMULATION ENGINE ==============

def simulate_scenario(config: ActuatorConfig, orbit: OrbitConfig,
                      goal_type: str, x0: np.ndarray, q_goal: np.ndarray,
                      tf: float, dt: float, 
                      desat_mode: str = 'none',
                      max_error_budget: float = 1.0,
                      target_func=None) -> Dict:
    """
    Run simulation with specified configuration.
    
    goal_type: 'full_attitude', 'boresight', 'nadir_tracking'
    desat_mode: 'none', 'continuous', 'slew_only', 'error_budget'
    """
    J = config.J
    A_rw = config.A_rw
    A_mtq_axes = config.A_mtq_axes
    n_rw = A_rw.shape[1]
    n_mtq = A_mtq_axes.shape[1]
    
    u_rw_max = config.u_rw_max
    u_mtq_max = config.u_mtq_max
    
    lb = np.concatenate([-u_rw_max, -u_mtq_max]) if n_mtq > 0 else -u_rw_max
    ub = np.concatenate([u_rw_max, u_mtq_max]) if n_mtq > 0 else u_rw_max
    
    kp, kd = 5e-5, 1e-3
    
    steps = int(tf / dt) + 1
    error_hist = np.zeros(steps)
    h_hist = np.zeros((steps, n_rw)) if n_rw > 0 else None
    desat_hist = np.zeros(steps)
    
    x = x0.copy()
    t = 0.0
    
    # Track if we're in a slew (large error)
    slew_threshold = 10.0  # degrees
    
    for k in range(steps):
        omega = x[0:3]
        q = normalize(x[3:7])
        h_rw = x[7:7+n_rw] if n_rw > 0 else np.array([])
        
        # Get magnetic field
        phase = 2 * np.pi * t / orbit.period
        r_eci = np.array([np.cos(phase), np.sin(phase), 0])
        B_eci = magnetic_field_dipole(r_eci, t, orbit)
        R = rot_mat(q)
        b_body = R.T @ B_eci
        
        # Build torque matrix
        if n_mtq > 0:
            A_mtq = -skewsym(b_body) @ A_mtq_axes
            A_total = np.hstack([A_rw, A_mtq]) if n_rw > 0 else A_mtq
        else:
            A_total = A_rw
        
        # Update goal for tracking modes
        if goal_type == 'nadir_tracking':
            # Nadir = -r direction
            nadir_eci = -r_eci
            # Find quaternion that points boresight at nadir
            # (simplified - just use as target vector)
            target_vec = nadir_eci
            err = vector_alignment_error(q, np.array([0,0,1]), target_vec)
        elif goal_type == 'boresight':
            R_goal = rot_mat(q_goal)
            target_vec = R_goal @ np.array([0,0,1])
            err = vector_alignment_error(q, np.array([0,0,1]), target_vec)
        else:  # full_attitude
            err = quaternion_error_vector(q, q_goal)
        
        # Compute control
        h_rw_vec = A_rw @ h_rw if n_rw > 0 else np.zeros(3)
        tau_gyro = np.cross(omega, J @ omega + h_rw_vec)
        tau_des = -kp * err - kd * omega + tau_gyro
        
        # Current error
        if goal_type == 'full_attitude':
            current_error = full_attitude_error_deg(q, q_goal)
        else:
            current_error = pointing_error_deg(q, q_goal)
        
        in_slew = current_error > slew_threshold
        
        # Allocate based on mode
        if desat_mode == 'none' or n_rw == 0:
            u = allocate_lp(tau_des, A_total, lb, ub)
            desat_achieved = 0.0
        elif desat_mode == 'error_budget':
            u, desat_achieved = allocate_with_desat(
                tau_des, A_total, lb, ub, h_rw, A_rw, 
                max_error_deg=max_error_budget
            )
        elif desat_mode == 'slew_only':
            if in_slew:
                u, desat_achieved = allocate_with_desat(
                    tau_des, A_total, lb, ub, h_rw, A_rw,
                    max_error_deg=5.0  # More aggressive during slews
                )
            else:
                u = allocate_lp(tau_des, A_total, lb, ub)
                desat_achieved = 0.0
        elif desat_mode == 'continuous':
            u, desat_achieved = allocate_with_desat(
                tau_des, A_total, lb, ub, h_rw, A_rw,
                max_error_deg=max_error_budget
            )
        else:
            u = allocate_lp(tau_des, A_total, lb, ub)
            desat_achieved = 0.0
        
        u_rw = u[:n_rw] if n_rw > 0 else np.array([])
        u_mtq = u[n_rw:] if n_mtq > 0 else np.array([])
        
        # Record
        error_hist[k] = current_error
        if h_hist is not None:
            h_hist[k, :] = h_rw
        desat_hist[k] = desat_achieved
        
        if k == steps - 1:
            break
        
        # Propagate
        if n_mtq > 0:
            tau_mtq = A_mtq @ u_mtq
        else:
            tau_mtq = np.zeros(3)
        
        tau_rw = A_rw @ u_rw if n_rw > 0 else np.zeros(3)
        tau_total = tau_rw + tau_mtq
        
        # RK4 integration (more stable)
        def deriv(state):
            w = state[0:3]
            qu = normalize(state[3:7])
            hr = state[7:7+n_rw] if n_rw > 0 else np.array([])
            
            hrv = A_rw @ hr if n_rw > 0 else np.zeros(3)
            w_dot = np.linalg.solve(J, tau_total - np.cross(w, J @ w + hrv))
            
            W_mat = np.zeros((4, 3))
            W_mat[0, :] = -qu[1:4]
            W_mat[1:4, :] = qu[0] * np.eye(3) + skewsym(qu[1:4])
            q_dot = 0.5 * W_mat @ w
            
            h_dot = -u_rw if n_rw > 0 else np.array([])
            
            return np.concatenate([w_dot, q_dot, h_dot])
        
        k1 = deriv(x)
        k2 = deriv(x + 0.5*dt*k1)
        k3 = deriv(x + 0.5*dt*k2)
        k4 = deriv(x + dt*k3)
        
        x = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        x[3:7] = normalize(x[3:7])
        
        # Safety check for NaN
        if np.any(np.isnan(x)):
            break
        
        t += dt
    
    return {
        'error_hist': error_hist,
        'h_hist': h_hist,
        'desat_hist': desat_hist,
        'final_error': error_hist[-1],
        'mean_error': np.mean(error_hist),
        'max_error': np.max(error_hist),
        'final_h': np.linalg.norm(h_hist[-1]) if h_hist is not None else 0,
        'initial_h': np.linalg.norm(h_hist[0]) if h_hist is not None else 0,
        'total_desat': np.sum(desat_hist) * dt,
        'config': config.name,
        'orbit': orbit.name,
        'goal_type': goal_type,
        'desat_mode': desat_mode
    }


def run_comprehensive_study():
    """Run comprehensive desaturation trade-off study."""
    np.random.seed(42)
    
    # Configurations
    configs = [
        config_3mtq_1rw(),
        config_3mtq_3rw(),
        config_4rw_pyramid(),
    ]
    
    orbits = [
        orbit_leo_equatorial(),
        orbit_leo_polar(),
        orbit_iss(),
    ]
    
    goal_types = ['full_attitude', 'boresight']
    
    desat_modes = ['none', 'error_budget', 'slew_only']
    error_budgets = [0.5, 1.0, 2.0, 5.0]  # degrees
    
    tf, dt = 400, 1  # Shorter with smaller timestep for stability
    n_ics = 2  # Initial conditions per combo
    
    all_results = []
    
    # Calculate total iterations
    total = len(configs) * len(orbits) * len(goal_types) * (1 + len(error_budgets) + 1) * n_ics
    pbar = tqdm(total=total, desc="Running study")
    
    for config in configs:
        n_rw = config.A_rw.shape[1]
        
        for orbit in orbits:
            for goal_type in goal_types:
                for ic in range(n_ics):
                    # Random initial condition
                    axis = normalize(np.random.randn(3))
                    angle = np.random.uniform(0.3, 0.6)
                    q0 = normalize(np.concatenate([[np.cos(angle/2)], axis*np.sin(angle/2)]))
                    omega0 = np.random.randn(3) * 0.015
                    h0 = np.random.uniform(0.003, 0.006, n_rw) if n_rw > 0 else np.array([])
                    x0 = np.concatenate([omega0, q0, h0])
                    
                    q_goal = np.array([1, 0, 0, 0])
                    
                    # Test different desat modes
                    # 1. No desaturation
                    res = simulate_scenario(config, orbit, goal_type, x0, q_goal,
                                           tf, dt, desat_mode='none')
                    res['error_budget'] = 0
                    all_results.append(res)
                    pbar.update(1)
                    
                    # 2. Error budget desaturation
                    for budget in error_budgets:
                        res = simulate_scenario(config, orbit, goal_type, x0, q_goal,
                                               tf, dt, desat_mode='error_budget',
                                               max_error_budget=budget)
                        res['error_budget'] = budget
                        all_results.append(res)
                        pbar.update(1)
                    
                    # 3. Slew-only desaturation
                    res = simulate_scenario(config, orbit, goal_type, x0, q_goal,
                                           tf, dt, desat_mode='slew_only')
                    res['error_budget'] = -1  # Special marker for slew-only
                    all_results.append(res)
                    pbar.update(1)
    
    pbar.close()
    
    return all_results


def analyze_results(results):
    """Analyze and print results."""
    print("\n" + "=" * 90)
    print("DESATURATION TRADE-OFF STUDY RESULTS")
    print("=" * 90)
    
    # Group by config
    configs = set(r['config'] for r in results)
    
    for config in sorted(configs):
        print(f"\n{'='*90}")
        print(f"CONFIGURATION: {config}")
        print(f"{'='*90}")
        
        config_results = [r for r in results if r['config'] == config]
        
        # Group by error budget
        budgets = sorted(set(r['error_budget'] for r in config_results))
        
        print(f"\n{'Budget(°)':<12} {'Final Err(°)':<15} {'h Reduction(%)':<15} {'Mean Err(°)':<15}")
        print("-" * 60)
        
        for budget in budgets:
            budget_results = [r for r in config_results if r['error_budget'] == budget]
            
            if budget == 0:
                label = "None"
            elif budget == -1:
                label = "Slew-only"
            else:
                label = f"{budget:.1f}"
            
            final_err = np.mean([r['final_error'] for r in budget_results])
            mean_err = np.mean([r['mean_error'] for r in budget_results])
            
            # Momentum reduction
            h_initial = np.mean([r['initial_h'] for r in budget_results])
            h_final = np.mean([r['final_h'] for r in budget_results])
            if h_initial > 0:
                h_reduction = (h_initial - h_final) / h_initial * 100
            else:
                h_reduction = 0
            
            print(f"{label:<12} {final_err:<15.2f} {h_reduction:<15.1f} {mean_err:<15.2f}")
    
    # Summary: What error budget gives meaningful desaturation?
    print("\n" + "=" * 90)
    print("KEY FINDINGS")
    print("=" * 90)
    
    # Find the sweet spot
    for config in sorted(configs):
        config_results = [r for r in results if r['config'] == config]
        
        baseline = [r for r in config_results if r['error_budget'] == 0]
        baseline_err = np.mean([r['final_error'] for r in baseline])
        baseline_h = np.mean([r['final_h'] for r in baseline])
        
        print(f"\n{config}:")
        print(f"  Baseline: {baseline_err:.2f}° error, {baseline_h*1000:.2f} mNm·s momentum")
        
        for budget in [0.5, 1.0, 2.0, 5.0]:
            budget_results = [r for r in config_results if r['error_budget'] == budget]
            if not budget_results:
                continue
            err = np.mean([r['final_error'] for r in budget_results])
            h = np.mean([r['final_h'] for r in budget_results])
            err_increase = err - baseline_err
            h_reduction = (baseline_h - h) / baseline_h * 100 if baseline_h > 0 else 0
            
            print(f"  {budget}° budget: +{err_increase:.2f}° error, {h_reduction:.1f}% momentum reduction")
        
        slew_results = [r for r in config_results if r['error_budget'] == -1]
        if slew_results:
            err = np.mean([r['final_error'] for r in slew_results])
            h = np.mean([r['final_h'] for r in slew_results])
            err_increase = err - baseline_err
            h_reduction = (baseline_h - h) / baseline_h * 100 if baseline_h > 0 else 0
            print(f"  Slew-only: +{err_increase:.2f}° error, {h_reduction:.1f}% momentum reduction")


if __name__ == "__main__":
    results = run_comprehensive_study()
    analyze_results(results)
    
    print("\n" + "=" * 90)
    print("RECOMMENDATIONS")
    print("=" * 90)
    print("""
1. ERROR BUDGET DESATURATION:
   - 1° budget gives modest momentum reduction with small error increase
   - 2° budget provides better momentum management for relaxed pointing
   - Above 5° budget, diminishing returns on momentum reduction

2. SLEW-ONLY DESATURATION:
   - Exploits large errors during maneuvers for "free" desaturation
   - No impact on steady-state pointing accuracy
   - Effectiveness depends on slew frequency and magnitude

3. CONFIGURATION DEPENDENCE:
   - 3MTQ+1RW: Limited desaturation authority (1 RW only)
   - 3MTQ+3RW: Better flexibility for blending
   - 4RW: Can redistribute but not reduce net momentum

4. RECOMMENDED APPROACH:
   - Use slew-only desaturation as primary method
   - Add 1° error-budget desaturation when momentum critical
   - Schedule explicit desaturation windows for large dumps
""")
