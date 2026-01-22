"""
Comprehensive planner tuning tests for ALTRO trajectory optimization.

This script systematically tests different planner settings to find configurations that:
- Produce smooth trajectories (minimal oscillations)
- Respect constraints (actuator bounds, wmax)
- Achieve near-zero final angle and angular velocity error
- Converge quickly (ideally within trajectory)
- Run fast (<20s ALTRO time for 500s trajectory)

Test categories:
1. Cost weight variations (angle, ang_vel, control)
2. Penalty/Augmented Lagrangian settings
3. Initial trajectory (bdot_on modes)
4. Iteration limits
5. Regularization settings
"""
import sys
import os as os_module
import numpy as np
import time

sys.path.append(os_module.path.abspath(os_module.path.join(__file__, "../../../..")))

from ADCS.CONOPS.goals import ECI_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.helpers import PlannerSettings, Trajectory
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.helpers.math_helpers import random_n_unit_vec, normalize


def evaluate_trajectory(traj: Trajectory, goal_vec: np.ndarray, wmax: float, umax: np.ndarray) -> dict:
    """
    Evaluate trajectory quality metrics.

    Returns dict with:
    - final_error_deg: Final pointing error
    - final_w_deg: Final angular velocity magnitude
    - max_w_deg: Maximum angular velocity
    - max_error_deg: Maximum pointing error during trajectory
    - min_error_deg: Minimum pointing error (best achieved)
    - time_to_1deg: Time to reach 1 degree error (or None)
    - time_to_5deg: Time to reach 5 degree error (or None)
    - constraint_violations: Dict of constraint violation counts
    - smoothness: Control rate of change metrics
    - oscillation_score: Number of control sign changes
    """
    # Extract data
    states = traj.states.T if traj._is_row_major else traj.states
    if states.shape[0] != traj.n_steps:
        states = states.T
    controls = traj.controls.T if traj.controls.shape[0] != traj.n_steps - 1 else traj.controls
    if controls.shape[0] == traj.ctrl_dim:
        controls = controls.T

    N = traj.n_steps
    times = traj.times

    # Angular velocity
    w = states[:, :3]
    w_mag_deg = np.rad2deg(np.linalg.norm(w, axis=1))

    # Pointing error
    errors_deg = []
    for i in range(N):
        q = states[i, 3:7]
        w_s, x_q, y_q, z_q = q
        R = np.array([
            [1 - 2*(y_q**2 + z_q**2), 2*(x_q*y_q - z_q*w_s), 2*(x_q*z_q + y_q*w_s)],
            [2*(x_q*y_q + z_q*w_s), 1 - 2*(x_q**2 + z_q**2), 2*(y_q*z_q - x_q*w_s)],
            [2*(x_q*z_q - y_q*w_s), 2*(y_q*z_q + x_q*w_s), 1 - 2*(x_q**2 + y_q**2)]
        ])
        body_boresight = np.array([0, 0, 1])
        eci_boresight = R @ body_boresight
        error_rad = np.arccos(np.clip(np.dot(eci_boresight, goal_vec), -1, 1))
        errors_deg.append(np.rad2deg(error_rad))
    errors_deg = np.array(errors_deg)

    # Time to reach error thresholds
    time_to_1deg = None
    time_to_5deg = None
    dt_sec = (times[1] - times[0]) * TimeConstants.cent2sec if len(times) > 1 else 1.0
    for i, err in enumerate(errors_deg):
        if err <= 5.0 and time_to_5deg is None:
            time_to_5deg = i * dt_sec
        if err <= 1.0 and time_to_1deg is None:
            time_to_1deg = i * dt_sec
            break

    # Constraint violations
    n_ctrl = min(len(controls), N - 1)
    wmax_violations = np.sum(w_mag_deg > np.rad2deg(wmax) * 1.01)  # 1% tolerance

    u_violations = 0
    for i in range(n_ctrl):
        u = controls[i]
        for j, (u_j, u_max_j) in enumerate(zip(u, umax)):
            if abs(u_j) > u_max_j * 1.01:  # 1% tolerance
                u_violations += 1

    # Control smoothness (rate of change)
    if n_ctrl > 1:
        du = np.diff(controls[:n_ctrl], axis=0)
        du_mag = np.linalg.norm(du, axis=1)
        max_du = np.max(du_mag)
        mean_du = np.mean(du_mag)
    else:
        max_du = 0
        mean_du = 0

    # Oscillation score (sign changes in control)
    n_mtq = min(3, controls.shape[1])
    sign_changes = np.sum(np.diff(np.sign(controls[:n_ctrl, :n_mtq]), axis=0) != 0)

    return {
        'final_error_deg': errors_deg[-1],
        'final_w_deg': w_mag_deg[-1],
        'max_w_deg': np.max(w_mag_deg),
        'max_error_deg': np.max(errors_deg),
        'min_error_deg': np.min(errors_deg),
        'time_to_1deg': time_to_1deg,
        'time_to_5deg': time_to_5deg,
        'wmax_violations': wmax_violations,
        'u_violations': u_violations,
        'max_control_rate': max_du,
        'mean_control_rate': mean_du,
        'oscillation_score': sign_changes,
    }


def run_single_test(
    name: str,
    real_sat,
    orb,
    os0_for_traj,
    x0: np.ndarray,
    goals: GoalList,
    goal_vec: np.ndarray,
    tf: float,
    dt_planning: float,
    # ===========================================
    # Settings that affect TRAJECTORY OPTIMIZATION
    # (cost_main, cost_second, pass1, pass2, init_traj, bdot_on)
    # ===========================================
    # Initial trajectory generation
    bdot_on: int = 2,  # 0=zeros, 1=basic bdot, 2=smart bdot, 3=smart+noise
    # Constraints
    wmax: float = 5*np.pi/180,  # Angular velocity limit (hard constraint)
    # Cost weights for TRAJECTORY optimization (pass1 & pass2)
    # NOTE: cost_tvlqr only affects K gains for tracking, NOT the trajectory!
    angle: float = 1e6,      # Running cost on attitude error
    angle_N: float = 1e10,   # Terminal cost on attitude error
    ang_vel: float = 1e4,    # Running cost on angular velocity
    ang_vel_N: float = 1e8,  # Terminal cost on angular velocity
    control_mult: float = 1.0,  # Multiplier for actuator cost
    mtq_control_weight: float = 1e3,   # Individual MTQ cost weight (in C++ satellite)
    rw_control_weight: float = 1e5,    # Individual RW cost weight (in C++ satellite)
    use_raw_control_cost: bool = True,  # True=|u|^2, False=|u-u_prev|^2
    ang_cost_func_type: int = 2,  # 0=(1-dot), 1=0.5*(1-dot)^2, 2=acos (recommended), 3=0.5*acos^2
    use_full_cost_hessian: bool = False,  # True=full Newton, False=Gauss-Newton
    # Pass 1 settings (exploration phase)
    p1_max_outer: int = 20,
    p1_max_inner: int = 150,
    p1_penalty_init: float = 1e-3,  # Low = more exploration
    p1_penalty_scale: float = 10.0,
    p1_grad_tol: float = 1e-4,
    p1_c_max: float = 0.0002,  # Constraint violation tolerance
    # Pass 2 settings (refinement phase)
    p2_max_outer: int = 20,
    p2_max_inner: int = 75,
    p2_penalty_init: float = 1e4,  # High = strict constraint enforcement
    p2_penalty_scale: float = 10.0,
    p2_grad_tol: float = 1e-4,
    p2_c_max: float = 0.0002,
    # Regularization (affects convergence stability)
    reg_init: float = 1e-2,
    use_dynamics_hess: int = 1,  # Include dynamics Hessian terms
    verbose: bool = False,
) -> dict:
    """Run a single planner test with specified settings."""

    planner_settings = PlannerSettings(
        est_sat=real_sat,
        bdot_on=bdot_on,
        dt_tp=dt_planning,
        dt_tvlqr=1,
    )

    # Constraint
    planner_settings.wmax = wmax

    # Control weights
    planner_settings.mtq_control_weight = mtq_control_weight
    planner_settings.rw_control_weight = rw_control_weight

    # Cost weights - main pass
    planner_settings.cost_main.angle = angle
    planner_settings.cost_main.angle_N = angle_N
    planner_settings.cost_main.ang_vel = ang_vel
    planner_settings.cost_main.ang_vel_N = ang_vel_N
    planner_settings.cost_main.control_mult = control_mult
    planner_settings.cost_main.use_raw_control_cost = use_raw_control_cost
    planner_settings.cost_main.ang_cost_func_type = ang_cost_func_type
    planner_settings.cost_main.use_full_cost_hessian = use_full_cost_hessian

    # Cost weights - second pass (often same or slightly different)
    planner_settings.cost_second.angle = angle
    planner_settings.cost_second.angle_N = angle_N
    planner_settings.cost_second.ang_vel = ang_vel
    planner_settings.cost_second.ang_vel_N = ang_vel_N
    planner_settings.cost_second.control_mult = control_mult
    planner_settings.cost_second.use_raw_control_cost = use_raw_control_cost
    planner_settings.cost_second.ang_cost_func_type = ang_cost_func_type
    planner_settings.cost_second.use_full_cost_hessian = use_full_cost_hessian

    # TVLQR costs - only affect gain computation for tracking, NOT trajectory!
    # We still set them to reasonable values for when tracking is tested later
    planner_settings.cost_tvlqr.angle = angle * 100
    planner_settings.cost_tvlqr.angle_N = angle_N * 100
    planner_settings.cost_tvlqr.ang_vel = ang_vel * 100
    planner_settings.cost_tvlqr.ang_vel_N = ang_vel_N * 100
    planner_settings.cost_tvlqr.control_mult = control_mult * 1e8
    planner_settings.cost_tvlqr.use_raw_control_cost = True
    planner_settings.cost_tvlqr.use_full_cost_hessian = use_full_cost_hessian

    # Disturbances
    planner_settings.plan_for_aero = True
    planner_settings.plan_for_srp = True
    planner_settings.plan_for_gg = True

    # Pass 1 solver settings
    planner_settings.pass1.convergence.max_outer_iter = p1_max_outer
    planner_settings.pass1.convergence.max_inner_iter = p1_max_inner
    planner_settings.pass1.convergence.grad_tol = p1_grad_tol
    planner_settings.pass1.convergence.c_max = p1_c_max
    planner_settings.pass1.aug_lag.penalty_init = p1_penalty_init
    planner_settings.pass1.aug_lag.penalty_scale = p1_penalty_scale
    planner_settings.pass1.regularization.reg_init = reg_init
    planner_settings.pass1.regularization.use_dynamics_hess = use_dynamics_hess

    # Pass 2 solver settings
    planner_settings.pass2.convergence.max_outer_iter = p2_max_outer
    planner_settings.pass2.convergence.max_inner_iter = p2_max_inner
    planner_settings.pass2.convergence.grad_tol = p2_grad_tol
    planner_settings.pass2.convergence.c_max = p2_c_max
    planner_settings.pass2.aug_lag.penalty_init = p2_penalty_init
    planner_settings.pass2.aug_lag.penalty_scale = p2_penalty_scale
    planner_settings.pass2.regularization.reg_init = reg_init
    planner_settings.pass2.regularization.use_dynamics_hess = use_dynamics_hess

    controller = Plan_and_Track_LQR(
        est_sat=real_sat,
        planner_settings=planner_settings,
    )

    # Instrument for ALTRO timing only
    altro_time = 0

    def timed_calc(t_start, duration, x_0, os_0, goals_arg, verbose_arg=False):
        nonlocal altro_time
        from ADCS.controller.helpers import reorder_controls_cpp_to_python, reorder_gains_cpp_to_python

        controller.planner.setVerbosity(verbose_arg)
        dt_seconds = controller.planner_settings.dt_tvlqr
        N = int(np.ceil(duration / dt_seconds)) + 1
        t_end = t_start + (duration * TimeConstants.sec2cent)

        vecsPy = controller._propagate_environment(os_0, t_start, t_end, dt_seconds, N, goals_arg)
        x_0_clean = np.copy(x_0.astype(np.float64).flatten(), order='C')
        bdotOn_val = controller.planner_settings.bdot_on

        t_altro_start = time.perf_counter()
        (_, _, _, lqr_opt, _) = controller.planner.trajOpt(vecsPy, N, t_start, t_end, x_0_clean, int(bdotOn_val))
        t_altro_end = time.perf_counter()
        altro_time = t_altro_end - t_altro_start

        (Xset, Uset_cpp, Tset, Kset_cpp, Sset, lqr_times) = lqr_opt
        Uset = reorder_controls_cpp_to_python(Uset_cpp, controller.est_sat.actuators)
        Kset = reorder_gains_cpp_to_python(Kset_cpp, controller.est_sat.actuators)

        return (np.array(lqr_times), Xset, Uset, Kset, Sset)

    controller._calculate_trajectory_common = timed_calc

    try:
        traj = controller.calculate_trajectory(
            t_start=0.22,
            duration=tf,
            x_0=x0,
            os_0=os0_for_traj,
            goals=goals,
            verbose=verbose,
        )
    except Exception as e:
        return {
            'name': name,
            'altro_time': float('inf'),
            'error': str(e),
            'final_error_deg': float('inf'),
            'final_w_deg': float('inf'),
        }

    # Evaluate trajectory
    umax = planner_settings.umax
    metrics = evaluate_trajectory(traj, goal_vec, wmax, umax)
    metrics['name'] = name
    metrics['altro_time'] = altro_time
    metrics['error'] = None

    return metrics


def print_results_table(results: list, title: str):
    """Print results in a formatted table."""
    print(f"\n{'='*100}")
    print(f"{title}")
    print(f"{'='*100}")
    print(f"{'Name':<35} {'ALTRO':>7} {'FinalErr':>8} {'FinalW':>7} {'MaxW':>6} {'MinErr':>7} {'Osc':>4} {'Viol':>5} {'Quality':<8}")
    print(f"{'':<35} {'(s)':>7} {'(deg)':>8} {'(d/s)':>7} {'(d/s)':>6} {'(deg)':>7} {'':<4} {'':<5} {'':<8}")
    print("-" * 100)

    for r in results:
        if r.get('error'):
            print(f"{r['name']:<35} {'FAIL':>7} {'-':>8} {'-':>7} {'-':>6} {'-':>7} {'-':>4} {'-':>5} {'ERROR':<8}")
            continue

        # Quality assessment
        quality = "GOOD" if (r['final_error_deg'] < 5 and
                           r['final_w_deg'] < 5 and
                           r['wmax_violations'] == 0 and
                           r['u_violations'] == 0 and
                           r['altro_time'] < 20) else "BAD"
        if quality == "GOOD" and r['final_error_deg'] < 1 and r['final_w_deg'] < 1:
            quality = "GREAT"

        viol = r['wmax_violations'] + r['u_violations']
        print(f"{r['name']:<35} {r['altro_time']:>7.1f} {r['final_error_deg']:>8.2f} {r['final_w_deg']:>7.2f} "
              f"{r['max_w_deg']:>6.1f} {r['min_error_deg']:>7.2f} {r['oscillation_score']:>4} {viol:>5} {quality:<8}")


def setup_test_environment(seed: int = 37, tf: float = 500):
    """Create shared test environment (orbit, satellite, initial state)."""
    np.random.seed(seed)

    # Create satellite
    real_sat = create_beavercube2_cubesat(estimated=False)
    real_sat.rw_actuators[0].h = 0.0

    # Initial conditions - moderately challenging
    w0 = random_n_unit_vec(3) * np.random.uniform(0.5, 1.0) * np.pi / 180.0
    q0 = normalize(np.random.randn(4))
    h0 = np.array([0.0])
    x0 = np.concatenate([w0, q0, h0])

    # Create orbit
    ephem = Ephemeris()
    start_time = 0.22 - 1 * TimeConstants.sec2cent
    end_time = 0.22 + tf * TimeConstants.sec2cent
    R = 7000 * np.array([0, np.sqrt(2) / 2, np.sqrt(2) / 2])
    V = np.array([8, 0, 0])

    print("Creating orbit...")
    os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
    orb = Orbit(os0=os0, end_time=end_time, dt=1, use_J2=True, fast=False)
    os0_for_traj = orb.get_os(0.22)
    print("Orbit created.")

    # Goal
    goal_vec = normalize(np.array([0, 0, 1]))
    goal = ECI_Goal(goal_vec)
    goals = GoalList({0.22: goal})

    return real_sat, orb, os0_for_traj, x0, goals, goal_vec


def run_cost_weight_tests(real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf=500, dt_planning=30):
    """Test various cost weight configurations."""
    print("\n" + "="*50)
    print("TEST SET 1: Cost Weight Variations")
    print("="*50)

    results = []

    # Baseline
    results.append(run_single_test(
        "Baseline (default)",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        angle=1e6, angle_N=1e10, ang_vel=1e4, ang_vel_N=1e8,
    ))

    # Vary angle costs
    results.append(run_single_test(
        "Higher angle cost",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        angle=1e8, angle_N=1e12, ang_vel=1e4, ang_vel_N=1e8,
    ))

    results.append(run_single_test(
        "Lower angle cost",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        angle=1e4, angle_N=1e8, ang_vel=1e4, ang_vel_N=1e8,
    ))

    # Vary ang_vel costs
    results.append(run_single_test(
        "Higher ang_vel cost",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        angle=1e6, angle_N=1e10, ang_vel=1e6, ang_vel_N=1e10,
    ))

    results.append(run_single_test(
        "Lower ang_vel cost",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        angle=1e6, angle_N=1e10, ang_vel=1e2, ang_vel_N=1e6,
    ))

    # High terminal vs running ratio
    results.append(run_single_test(
        "Very high terminal ratio (1e6x)",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        angle=1e4, angle_N=1e10, ang_vel=1e2, ang_vel_N=1e8,
    ))

    # Control cost variations
    results.append(run_single_test(
        "High control_mult",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        angle=1e6, angle_N=1e10, ang_vel=1e4, ang_vel_N=1e8,
        control_mult=100.0,
    ))

    results.append(run_single_test(
        "Low control_mult",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        angle=1e6, angle_N=1e10, ang_vel=1e4, ang_vel_N=1e8,
        control_mult=0.01,
    ))

    # MTQ vs RW weight ratio
    results.append(run_single_test(
        "Prefer MTQ (low rw_weight)",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        angle=1e6, angle_N=1e10, ang_vel=1e4, ang_vel_N=1e8,
        mtq_control_weight=1e2, rw_control_weight=1e6,
    ))

    results.append(run_single_test(
        "Prefer RW (low mtq_weight)",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        angle=1e6, angle_N=1e10, ang_vel=1e4, ang_vel_N=1e8,
        mtq_control_weight=1e6, rw_control_weight=1e2,
    ))

    print_results_table(results, "Cost Weight Test Results")
    return results


def run_penalty_tests(real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf=500, dt_planning=30):
    """Test various penalty/augmented Lagrangian settings."""
    print("\n" + "="*50)
    print("TEST SET 2: Penalty & Aug Lag Settings")
    print("="*50)

    results = []

    # Baseline
    results.append(run_single_test(
        "Default penalties",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        p1_penalty_init=1e-3, p1_penalty_scale=10, p2_penalty_init=1e4, p2_penalty_scale=10,
    ))

    # Higher pass1 penalty
    results.append(run_single_test(
        "High P1 penalty (1e0)",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        p1_penalty_init=1e0, p1_penalty_scale=10, p2_penalty_init=1e4, p2_penalty_scale=10,
    ))

    results.append(run_single_test(
        "Very high P1 penalty (1e2)",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        p1_penalty_init=1e2, p1_penalty_scale=10, p2_penalty_init=1e4, p2_penalty_scale=10,
    ))

    # Lower pass2 penalty
    results.append(run_single_test(
        "Low P2 penalty (1e2)",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        p1_penalty_init=1e-3, p1_penalty_scale=10, p2_penalty_init=1e2, p2_penalty_scale=10,
    ))

    # Faster penalty scaling
    results.append(run_single_test(
        "Fast penalty scale (50x)",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        p1_penalty_init=1e-3, p1_penalty_scale=50, p2_penalty_init=1e4, p2_penalty_scale=50,
    ))

    # Tighter constraint tolerance
    results.append(run_single_test(
        "Tight c_max (1e-4)",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        p1_c_max=1e-4, p2_c_max=1e-4,
    ))

    # Looser constraint tolerance
    results.append(run_single_test(
        "Loose c_max (1e-2)",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        p1_c_max=1e-2, p2_c_max=1e-2,
    ))

    print_results_table(results, "Penalty Settings Test Results")
    return results


def run_initial_traj_tests(real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf=500, dt_planning=30):
    """Test different initial trajectory modes (bdot_on)."""
    print("\n" + "="*50)
    print("TEST SET 3: Initial Trajectory (bdot_on)")
    print("="*50)

    results = []

    results.append(run_single_test(
        "bdot_on=1 (basic bdot)",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        bdot_on=1,
    ))

    results.append(run_single_test(
        "bdot_on=2 (smart bdot)",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        bdot_on=2,
    ))

    results.append(run_single_test(
        "bdot_on=3 (smart+noise)",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        bdot_on=3,
    ))

    results.append(run_single_test(
        "bdot_on=0 (zeros)",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        bdot_on=0,
    ))

    print_results_table(results, "Initial Trajectory Test Results")
    return results


def run_iteration_tests(real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf=500, dt_planning=30):
    """Test different iteration limits."""
    print("\n" + "="*50)
    print("TEST SET 4: Iteration Limits")
    print("="*50)

    results = []

    # Baseline
    results.append(run_single_test(
        "Default (20/150, 20/75)",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        p1_max_outer=20, p1_max_inner=150, p2_max_outer=20, p2_max_inner=75,
    ))

    # Fewer iterations (faster)
    results.append(run_single_test(
        "Fewer P1 (10/100)",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        p1_max_outer=10, p1_max_inner=100, p2_max_outer=20, p2_max_inner=75,
    ))

    results.append(run_single_test(
        "Minimal P1 (5/50)",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        p1_max_outer=5, p1_max_inner=50, p2_max_outer=20, p2_max_inner=75,
    ))

    # More iterations (higher quality?)
    results.append(run_single_test(
        "More P1 (30/200)",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        p1_max_outer=30, p1_max_inner=200, p2_max_outer=20, p2_max_inner=75,
    ))

    # Focus on P1, minimal P2
    results.append(run_single_test(
        "P1 focus (25/150, 5/30)",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        p1_max_outer=25, p1_max_inner=150, p2_max_outer=5, p2_max_inner=30,
    ))

    # Skip P2 almost entirely
    results.append(run_single_test(
        "P1 only (20/150, 1/10)",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        p1_max_outer=20, p1_max_inner=150, p2_max_outer=1, p2_max_inner=10,
    ))

    print_results_table(results, "Iteration Limit Test Results")
    return results


def run_regularization_tests(real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf=500, dt_planning=30):
    """Test different regularization settings."""
    print("\n" + "="*50)
    print("TEST SET 5: Regularization Settings")
    print("="*50)

    results = []

    # Baseline
    results.append(run_single_test(
        "Default reg (1e-2, dyn_hess=1)",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        reg_init=1e-2, use_dynamics_hess=1,
    ))

    # Lower regularization
    results.append(run_single_test(
        "Low reg (1e-4)",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        reg_init=1e-4, use_dynamics_hess=1,
    ))

    # Higher regularization
    results.append(run_single_test(
        "High reg (1e0)",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        reg_init=1e0, use_dynamics_hess=1,
    ))

    # No dynamics Hessian
    results.append(run_single_test(
        "No dynamics Hessian",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        reg_init=1e-2, use_dynamics_hess=0,
    ))

    # Full Hessian vs Gauss-Newton
    results.append(run_single_test(
        "Full cost Hessian",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        reg_init=1e-2, use_dynamics_hess=1, use_full_cost_hessian=True,
    ))

    results.append(run_single_test(
        "Gauss-Newton (no cost Hess)",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        reg_init=1e-2, use_dynamics_hess=0, use_full_cost_hessian=False,
    ))

    print_results_table(results, "Regularization Test Results")
    return results


def run_wmax_tests(real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf=500, dt_planning=30):
    """Test different wmax (angular velocity constraint) settings."""
    print("\n" + "="*50)
    print("TEST SET 6: Angular Velocity Constraint (wmax)")
    print("="*50)

    results = []

    for wmax_deg in [1, 2, 5, 10, 20]:
        results.append(run_single_test(
            f"wmax = {wmax_deg} deg/s",
            real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
            wmax=wmax_deg * np.pi / 180,
        ))

    print_results_table(results, "wmax Constraint Test Results")
    return results


def run_best_combinations(real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf=500, dt_planning=30):
    """Test promising combinations based on individual test insights."""
    print("\n" + "="*50)
    print("TEST SET 7: Best Combinations")
    print("="*50)

    results = []

    # Combo 1: High terminal, smart bdot, moderate iterations
    results.append(run_single_test(
        "Combo1: High term, smart bdot",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        bdot_on=2,
        angle=1e4, angle_N=1e10, ang_vel=1e2, ang_vel_N=1e8,
        p1_max_outer=15, p1_max_inner=100, p2_max_outer=10, p2_max_inner=50,
        use_full_cost_hessian=False,
    ))

    # Combo 2: Balanced costs, high penalty start
    results.append(run_single_test(
        "Combo2: Balanced, high P1 penalty",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        bdot_on=2,
        angle=1e6, angle_N=1e10, ang_vel=1e4, ang_vel_N=1e8,
        p1_penalty_init=1e0, p1_penalty_scale=20,
        p1_max_outer=15, p1_max_inner=100, p2_max_outer=5, p2_max_inner=30,
    ))

    # Combo 3: Speed-focused
    results.append(run_single_test(
        "Combo3: Speed focus",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        bdot_on=2,
        angle=1e6, angle_N=1e10, ang_vel=1e4, ang_vel_N=1e8,
        p1_max_outer=10, p1_max_inner=80, p2_max_outer=3, p2_max_inner=20,
        p1_penalty_init=1e0, p1_penalty_scale=30,
        use_full_cost_hessian=False, use_dynamics_hess=0,
    ))

    # Combo 4: Quality-focused
    results.append(run_single_test(
        "Combo4: Quality focus",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        bdot_on=2,
        angle=1e6, angle_N=1e12, ang_vel=1e4, ang_vel_N=1e10,
        p1_max_outer=25, p1_max_inner=150, p2_max_outer=15, p2_max_inner=75,
        p1_penalty_init=1e-2, p1_penalty_scale=10,
        p2_penalty_init=1e5, p2_penalty_scale=10,
    ))

    # Combo 5: Smooth trajectory focus
    results.append(run_single_test(
        "Combo5: Smooth traj",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        bdot_on=2,
        angle=1e5, angle_N=1e10, ang_vel=1e5, ang_vel_N=1e9,
        control_mult=10.0,
        wmax=10*np.pi/180,
    ))

    # Combo 6: Aggressive maneuver
    results.append(run_single_test(
        "Combo6: Aggressive",
        real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning,
        bdot_on=2,
        angle=1e8, angle_N=1e12, ang_vel=1e2, ang_vel_N=1e6,
        control_mult=0.1,
        wmax=20*np.pi/180,
    ))

    print_results_table(results, "Best Combinations Test Results")
    return results


def main():
    print("="*80)
    print("COMPREHENSIVE PLANNER TUNING TESTS")
    print("="*80)

    # Setup
    tf = 500
    dt_planning = 30
    real_sat, orb, os0_for_traj, x0, goals, goal_vec = setup_test_environment(seed=37, tf=tf)

    print(f"\nTest parameters:")
    print(f"  Trajectory duration: {tf}s")
    print(f"  Planning timestep: {dt_planning}s")
    print(f"  Initial |w|: {np.rad2deg(np.linalg.norm(x0[:3])):.2f} deg/s")

    all_results = []

    # Run all test sets
    all_results.extend(run_cost_weight_tests(real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning))
    all_results.extend(run_penalty_tests(real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning))
    all_results.extend(run_initial_traj_tests(real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning))
    all_results.extend(run_iteration_tests(real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning))
    all_results.extend(run_regularization_tests(real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning))
    all_results.extend(run_wmax_tests(real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning))
    all_results.extend(run_best_combinations(real_sat, orb, os0_for_traj, x0, goals, goal_vec, tf, dt_planning))

    # Summary: Best results
    print("\n" + "="*100)
    print("SUMMARY: Top 10 Configurations (sorted by final error, filtered by quality)")
    print("="*100)

    # Filter out failures and sort by quality
    good_results = [r for r in all_results if not r.get('error') and
                    r['wmax_violations'] == 0 and r['u_violations'] == 0 and
                    r['altro_time'] < 30]
    good_results.sort(key=lambda r: (r['final_error_deg'] + r['final_w_deg']))

    print_results_table(good_results[:10], "Top 10 Configurations")

    # Also show fastest good configurations
    print("\n" + "="*100)
    print("SUMMARY: Fastest Good Configurations (error < 5deg, w < 5deg/s, no violations)")
    print("="*100)

    fast_good = [r for r in good_results if r['final_error_deg'] < 5 and r['final_w_deg'] < 5]
    fast_good.sort(key=lambda r: r['altro_time'])

    print_results_table(fast_good[:10], "Fastest Good Configurations")


if __name__ == "__main__":
    main()
