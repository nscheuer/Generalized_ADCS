"""Parameter sweep to find fastest settings for bdot_on=2 with good trajectory quality."""
import sys
import os
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))

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


def test_settings(
    seed: int,
    tf: float,
    dt_planning: float,
    max_outer_1: int,
    max_inner_1: int,
    max_outer_2: int,
    max_inner_2: int,
    grad_tol: float,
    c_max: float,
    penalty_init: float,
    orb,  # Pre-computed orbit to save time
    os0_for_traj,
    real_sat,
    x,
    goals,
    # New parameters for cost weights and Hessian
    angle_cost: float = 1e6,
    angle_N_cost: float = 1e10,
    ang_vel_cost: float = 1e3,
    ang_vel_N_cost: float = 1e7,
    use_full_hessian: bool = True,
):
    """Test a specific parameter configuration and return timing + quality metrics."""

    planner_settings = PlannerSettings(
        est_sat=real_sat,
        bdot_on=2,  # Smart bdot
        dt_tp=dt_planning,
        dt_tvlqr=1,
    )

    # Cost weights (configurable)
    planner_settings.cost_main.ang_vel = ang_vel_cost
    planner_settings.cost_second.ang_vel = ang_vel_cost
    planner_settings.cost_tvlqr.ang_vel = ang_vel_cost * 1e3
    planner_settings.cost_main.ang_vel_N = ang_vel_N_cost
    planner_settings.cost_second.ang_vel_N = ang_vel_N_cost
    planner_settings.cost_tvlqr.ang_vel_N = ang_vel_N_cost * 1e2
    planner_settings.cost_main.angle = angle_cost
    planner_settings.cost_second.angle = angle_cost
    planner_settings.cost_tvlqr.angle = angle_cost * 1e4
    planner_settings.cost_main.angle_N = angle_N_cost
    planner_settings.cost_second.angle_N = angle_N_cost
    planner_settings.cost_tvlqr.angle_N = angle_N_cost * 1e3
    planner_settings.cost_main.use_raw_control_cost = False
    planner_settings.cost_second.use_raw_control_cost = False
    planner_settings.cost_tvlqr.use_raw_control_cost = True
    planner_settings.plan_for_aero = True
    planner_settings.plan_for_srp = True
    planner_settings.plan_for_gg = True
    planner_settings.cost_tvlqr.control_mult = 1e8
    planner_settings.cost_second.control_mult = 1e8

    # Hessian settings
    planner_settings.cost_main.use_full_cost_hessian = use_full_hessian
    planner_settings.cost_second.use_full_cost_hessian = use_full_hessian
    planner_settings.cost_tvlqr.use_full_cost_hessian = use_full_hessian

    # Test parameters
    planner_settings.pass1.convergence.max_outer_iter = max_outer_1
    planner_settings.pass1.convergence.max_inner_iter = max_inner_1
    planner_settings.pass2.convergence.max_outer_iter = max_outer_2
    planner_settings.pass2.convergence.max_inner_iter = max_inner_2
    planner_settings.pass1.convergence.grad_tol = grad_tol
    planner_settings.pass1.convergence.ilqr_cost_tol = 0.1
    planner_settings.pass1.convergence.c_max = c_max
    planner_settings.pass2.convergence.grad_tol = grad_tol
    planner_settings.pass2.convergence.ilqr_cost_tol = 0.1
    planner_settings.pass2.convergence.c_max = c_max
    planner_settings.pass1.aug_lag.penalty_init = penalty_init
    planner_settings.pass1.aug_lag.penalty_scale = 20
    planner_settings.pass2.aug_lag.penalty_scale = 20

    controller = Plan_and_Track_LQR(
        est_sat=real_sat,
        planner_settings=planner_settings,
    )

    # Instrument for timing
    altro_times = []
    original_calc_common = controller._calculate_trajectory_common

    def timed_calc(t_start, duration, x_0, os_0, goals_arg, verbose=False):
        from ADCS.controller.helpers import reorder_controls_cpp_to_python, reorder_gains_cpp_to_python
        controller.planner.setVerbosity(verbose)
        dt_seconds = controller.planner_settings.dt_tvlqr
        N = int(np.ceil(duration / dt_seconds)) + 1
        t_end_j2000 = t_start + (duration * TimeConstants.sec2cent)
        vecsPy = controller._propagate_environment(os_0, t_start, t_end_j2000, dt_seconds, N, goals_arg)
        x_0_clean = np.copy(x_0.astype(np.float64).flatten(), order='C')
        bdotOn = controller.planner_settings.bdot_on

        t_altro_start = time.perf_counter()
        (_, _, _, lqr_opt, _) = controller.planner.trajOpt(vecsPy, N, t_start, t_end_j2000, x_0_clean, int(bdotOn))
        t_altro_end = time.perf_counter()
        altro_times.append(t_altro_end - t_altro_start)

        (Xset, Uset_cpp, Tset, Kset_cpp, Sset, lqr_times) = lqr_opt
        Uset = reorder_controls_cpp_to_python(Uset_cpp, controller.est_sat.actuators)
        Kset = reorder_gains_cpp_to_python(Kset_cpp, controller.est_sat.actuators)
        return (np.array(lqr_times), Xset, Uset, Kset, Sset)

    controller._calculate_trajectory_common = timed_calc

    try:
        traj = controller.calculate_trajectory(
            t_start=0.22,
            duration=tf,
            x_0=x,
            os_0=os0_for_traj,
            goals=goals,
            verbose=False,
        )
    except Exception as e:
        return {'altro_time': float('inf'), 'final_ang_vel': float('inf'), 'final_error_deg': float('inf'), 'error': str(e)}

    altro_time = altro_times[0] if altro_times else float('inf')

    # Evaluate quality
    final_state = traj.states[:, -1]
    final_w = final_state[:3]
    final_q = final_state[3:7]

    goal_vec = normalize(np.array([0, 0, 1]))
    w_scalar, x_q, y_q, z_q = final_q
    R_mat = np.array([
        [1 - 2*(y_q**2 + z_q**2), 2*(x_q*y_q - z_q*w_scalar), 2*(x_q*z_q + y_q*w_scalar)],
        [2*(x_q*y_q + z_q*w_scalar), 1 - 2*(x_q**2 + z_q**2), 2*(y_q*z_q - x_q*w_scalar)],
        [2*(x_q*z_q - y_q*w_scalar), 2*(y_q*z_q + x_q*w_scalar), 1 - 2*(x_q**2 + y_q**2)]
    ])
    body_boresight = np.array([0, 0, 1])
    eci_boresight = R_mat @ body_boresight
    error_rad = np.arccos(np.clip(np.dot(eci_boresight, goal_vec), -1, 1))

    return {
        'altro_time': altro_time,
        'final_ang_vel': np.rad2deg(np.linalg.norm(final_w)),
        'final_error_deg': np.rad2deg(error_rad),
    }


def main():
    print("=" * 80)
    print("PARAMETER SWEEP FOR bdot_on=2 OPTIMIZATION")
    print("=" * 80)

    # Fixed parameters
    seed = 37
    tf = 500
    dt_planning = 50

    np.random.seed(seed)

    # Create satellite and initial state
    real_sat = create_beavercube2_cubesat(estimated=False)
    real_sat.rw_actuators[0].h = 0.0
    w0 = random_n_unit_vec(3) * np.random.uniform(0.5, 1.0) * np.pi / 180.0
    q0 = normalize(np.random.randn(4))
    h0 = np.array([0.0])
    x = np.concatenate([w0, q0, h0])

    # Create orbit once
    print("Creating orbit (shared across all tests)...")
    ephem = Ephemeris()
    start_time = 0.22 - 1 * TimeConstants.sec2cent
    end_time = 0.22 + tf * TimeConstants.sec2cent
    R = 7000 * np.array([0, np.sqrt(2) / 2, np.sqrt(2) / 2])
    V = np.array([8, 0, 0])
    os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
    orb = Orbit(os0=os0, end_time=end_time, dt=1, use_J2=True, fast=False)
    os0_for_traj = orb.get_os(0.22)
    print("Orbit created.\n")

    # Goal
    goal_vec = normalize(np.array([0, 0, 1]))
    goal = ECI_Goal(goal_vec)
    goals = GoalList({0.22: goal})

    # Parameter configurations to test
    # Format: (max_outer_1, max_inner_1, max_outer_2, max_inner_2, grad_tol, c_max, penalty_init,
    #          angle_cost, angle_N_cost, ang_vel_cost, ang_vel_N_cost, use_full_hessian, name)
    configs = [
        # Best from previous sweep - "Focus pass1"
        (5, 15, 1, 8, 0.05, 0.05, 10.0, 1e6, 1e10, 1e3, 1e7, True, "Focus pass1 (baseline)"),

        # Try without full Hessian (Gauss-Newton only - should be faster per iter)
        (5, 15, 1, 8, 0.05, 0.05, 10.0, 1e6, 1e10, 1e3, 1e7, False, "Focus pass1 no Hess"),

        # Different cost weight ratios
        (5, 15, 1, 8, 0.05, 0.05, 10.0, 1e5, 1e9, 1e3, 1e7, True, "Lower angle cost"),
        (5, 15, 1, 8, 0.05, 0.05, 10.0, 1e7, 1e11, 1e3, 1e7, True, "Higher angle cost"),
        (5, 15, 1, 8, 0.05, 0.05, 10.0, 1e6, 1e10, 1e2, 1e6, True, "Lower ang_vel cost"),
        (5, 15, 1, 8, 0.05, 0.05, 10.0, 1e6, 1e10, 1e4, 1e8, True, "Higher ang_vel cost"),

        # Very high terminal vs running ratio
        (5, 15, 1, 8, 0.05, 0.05, 10.0, 1e5, 1e11, 1e2, 1e8, True, "High terminal ratio"),

        # No Hessian + cost tuning combos
        (5, 15, 1, 8, 0.05, 0.05, 10.0, 1e5, 1e11, 1e2, 1e8, False, "High term ratio no Hess"),
        (5, 15, 1, 8, 0.05, 0.05, 10.0, 1e5, 1e9, 1e3, 1e7, False, "Low angle no Hess"),

        # Even fewer iterations with no Hessian
        (4, 12, 1, 6, 0.05, 0.05, 10.0, 1e6, 1e10, 1e3, 1e7, False, "Ultra-fast no Hess"),
        (6, 12, 1, 6, 0.05, 0.05, 10.0, 1e6, 1e10, 1e3, 1e7, False, "More outer no Hess"),
    ]

    results = []
    print(f"{'Config':<28} {'ALTRO(s)':>10} {'AngVel':>10} {'Error':>10} {'Quality':>10}")
    print("-" * 80)

    for cfg in configs:
        (max_outer_1, max_inner_1, max_outer_2, max_inner_2, grad_tol, c_max, penalty_init,
         angle_cost, angle_N_cost, ang_vel_cost, ang_vel_N_cost, use_full_hessian, name) = cfg

        result = test_settings(
            seed=seed,
            tf=tf,
            dt_planning=dt_planning,
            max_outer_1=max_outer_1,
            max_inner_1=max_inner_1,
            max_outer_2=max_outer_2,
            max_inner_2=max_inner_2,
            grad_tol=grad_tol,
            c_max=c_max,
            penalty_init=penalty_init,
            orb=orb,
            os0_for_traj=os0_for_traj,
            real_sat=real_sat,
            x=x,
            goals=goals,
            angle_cost=angle_cost,
            angle_N_cost=angle_N_cost,
            ang_vel_cost=ang_vel_cost,
            ang_vel_N_cost=ang_vel_N_cost,
            use_full_hessian=use_full_hessian,
        )

        # Quality check: ang_vel < 5 deg/s and error < 5 deg is "good"
        quality = "GOOD" if result['final_ang_vel'] < 5 and result['final_error_deg'] < 5 else "BAD"

        print(f"{name:<28} {result['altro_time']:>10.2f} {result['final_ang_vel']:>10.2f} {result['final_error_deg']:>10.2f} {quality:>10}")

        result['name'] = name
        result['config'] = cfg
        results.append(result)

    print("\n" + "=" * 80)
    print("BEST CONFIGURATIONS (Quality=GOOD, sorted by ALTRO time)")
    print("=" * 80)

    good_results = [r for r in results if r['final_ang_vel'] < 5 and r['final_error_deg'] < 5]
    good_results.sort(key=lambda r: r['altro_time'])

    for r in good_results[:5]:
        cfg = r['config']
        print(f"{r['name']:<25}: ALTRO={r['altro_time']:.2f}s, vel={r['final_ang_vel']:.2f}°/s, err={r['final_error_deg']:.2f}°")
        print(f"  Settings: outer1={cfg[0]}, inner1={cfg[1]}, outer2={cfg[2]}, inner2={cfg[3]}, grad_tol={cfg[4]}, c_max={cfg[5]}, penalty={cfg[6]}")


if __name__ == "__main__":
    main()
