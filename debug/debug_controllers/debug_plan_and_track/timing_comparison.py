"""Quick timing comparison of bdot_on=1 vs bdot_on=2 for ALTRO only."""
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


def time_altro_only(bdot_on: int, seed: int = 37, tf: float = 500, dt_planning: float = 50):
    """Time just the ALTRO optimization, reusing orbit data."""
    np.random.seed(seed)

    # Create satellite
    real_sat = create_beavercube2_cubesat(estimated=False)
    real_sat.rw_actuators[0].h = 0.0

    # Initial state
    w0 = random_n_unit_vec(3) * np.random.uniform(0.5, 1.0) * np.pi / 180.0
    q0 = normalize(np.random.randn(4))
    h0 = np.array([0.0])
    x = np.concatenate([w0, q0, h0])

    # Create orbit (do this once outside timing)
    ephem = Ephemeris()
    start_time = 0.22 - 1 * TimeConstants.sec2cent
    end_time = 0.22 + tf * TimeConstants.sec2cent
    R = 7000 * np.array([0, np.sqrt(2) / 2, np.sqrt(2) / 2])
    V = np.array([8, 0, 0])

    print(f"Creating orbit for bdot_on={bdot_on}...")
    t_orbit_start = time.perf_counter()
    os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
    orb = Orbit(os0=os0, end_time=end_time, dt=1, use_J2=True, fast=False)
    t_orbit_end = time.perf_counter()
    orbit_time = t_orbit_end - t_orbit_start
    print(f"  Orbit creation: {orbit_time:.2f}s")

    # Setup planner with fast convergence settings
    planner_settings = PlannerSettings(
        est_sat=real_sat,
        bdot_on=bdot_on,
        dt_tp=dt_planning,
        dt_tvlqr=1,
    )

    # Fast convergence settings
    planner_settings.cost_main.ang_vel = 1e3
    planner_settings.cost_second.ang_vel = 1e3
    planner_settings.cost_tvlqr.ang_vel = 1e6
    planner_settings.cost_main.ang_vel_N = 1e7
    planner_settings.cost_second.ang_vel_N = 1e7
    planner_settings.cost_tvlqr.ang_vel_N = 1e9
    planner_settings.cost_main.angle = 1e6
    planner_settings.cost_second.angle = 1e6
    planner_settings.cost_tvlqr.angle = 1e10
    planner_settings.cost_main.angle_N = 1e10
    planner_settings.cost_second.angle_N = 1e10
    planner_settings.cost_tvlqr.angle_N = 1e13
    planner_settings.cost_main.use_raw_control_cost = False
    planner_settings.cost_second.use_raw_control_cost = False
    planner_settings.cost_tvlqr.use_raw_control_cost = True
    planner_settings.plan_for_aero = True
    planner_settings.plan_for_srp = True
    planner_settings.plan_for_gg = True
    planner_settings.cost_tvlqr.control_mult = 1e8
    planner_settings.cost_second.control_mult = 1e8

    # Fast iteration settings
    planner_settings.pass1.convergence.max_outer_iter = 3
    planner_settings.pass1.convergence.max_inner_iter = 15
    planner_settings.pass2.convergence.max_outer_iter = 2
    planner_settings.pass2.convergence.max_inner_iter = 10
    planner_settings.pass1.convergence.grad_tol = 0.05
    planner_settings.pass1.convergence.ilqr_cost_tol = 0.1
    planner_settings.pass1.convergence.c_max = 0.05
    planner_settings.pass2.convergence.grad_tol = 0.05
    planner_settings.pass2.convergence.ilqr_cost_tol = 0.1
    planner_settings.pass2.convergence.c_max = 0.05
    planner_settings.pass1.aug_lag.penalty_init = 10.0
    planner_settings.pass1.aug_lag.penalty_scale = 20
    planner_settings.pass2.aug_lag.penalty_scale = 20

    controller = Plan_and_Track_LQR(
        est_sat=real_sat,
        planner_settings=planner_settings,
    )

    # We'll use a custom calculate_trajectory that times the components
    altro_times = []
    env_times = []

    # Save reference to original method
    original_calc_common = controller._calculate_trajectory_common

    def timed_calculate_common(t_start, duration, x_0, os_0, goals, verbose=False):
        from ADCS.orbits.universal_constants import TimeConstants
        from ADCS.controller.helpers import reorder_controls_cpp_to_python, reorder_gains_cpp_to_python

        controller.planner.setVerbosity(verbose)
        dt_seconds = controller.planner_settings.dt_tvlqr
        N = int(np.ceil(duration / dt_seconds)) + 1
        t_end_j2000 = t_start + (duration * TimeConstants.sec2cent)

        # Time environment propagation
        t_env_start = time.perf_counter()
        vecsPy = controller._propagate_environment(os_0, t_start, t_end_j2000, dt_seconds, N, goals)
        t_env_end = time.perf_counter()
        env_times.append(t_env_end - t_env_start)

        x_0_clean = np.copy(x_0.astype(np.float64).flatten(), order='C')
        bdotOn = controller.planner_settings.bdot_on

        # Time ALTRO trajOpt
        t_altro_start = time.perf_counter()
        (_, _, _, lqr_opt, _) = controller.planner.trajOpt(vecsPy, N, t_start, t_end_j2000, x_0_clean, int(bdotOn))
        t_altro_end = time.perf_counter()
        altro_times.append(t_altro_end - t_altro_start)

        (Xset, Uset_cpp, Tset, Kset_cpp, Sset, lqr_times) = lqr_opt
        Uset = reorder_controls_cpp_to_python(Uset_cpp, controller.est_sat.actuators)
        Kset = reorder_gains_cpp_to_python(Kset_cpp, controller.est_sat.actuators)

        return (np.array(lqr_times), Xset, Uset, Kset, Sset)

    controller._calculate_trajectory_common = timed_calculate_common

    # Goal setup
    goal_vec = normalize(np.array([0, 0, 1]))
    goal = ECI_Goal(goal_vec)
    goals = GoalList({0.22: goal})

    # Calculate trajectory and time it
    print(f"Calculating trajectory with bdot_on={bdot_on}...")
    os0_for_traj = orb.get_os(0.22)

    t_total_start = time.perf_counter()
    traj = controller.calculate_trajectory(
        t_start=0.22,
        duration=tf,
        x_0=x,
        os_0=os0_for_traj,
        goals=goals,
        verbose=False,
    )
    t_total_end = time.perf_counter()
    total_calc_time = t_total_end - t_total_start

    altro_time = altro_times[0] if altro_times else 0
    env_prop_time = env_times[0] if env_times else 0

    # Evaluate trajectory quality
    final_state = traj.states[:, -1]
    final_w = final_state[:3]
    final_q = final_state[3:7]

    # Calculate final tracking error
    w_scalar, x_q, y_q, z_q = final_q
    R_mat = np.array([
        [1 - 2*(y_q**2 + z_q**2), 2*(x_q*y_q - z_q*w_scalar), 2*(x_q*z_q + y_q*w_scalar)],
        [2*(x_q*y_q + z_q*w_scalar), 1 - 2*(x_q**2 + z_q**2), 2*(y_q*z_q - x_q*w_scalar)],
        [2*(x_q*z_q - y_q*w_scalar), 2*(y_q*z_q + x_q*w_scalar), 1 - 2*(x_q**2 + y_q**2)]
    ])
    body_boresight = np.array([0, 0, 1])
    eci_boresight = R_mat @ body_boresight
    error_rad = np.arccos(np.clip(np.dot(eci_boresight, goal_vec), -1, 1))
    final_ang_vel = np.rad2deg(np.linalg.norm(final_w))
    final_error_deg = np.rad2deg(error_rad)

    print(f"\n=== bdot_on={bdot_on} RESULTS ===")
    print(f"  Orbit creation time:     {orbit_time:.2f}s")
    print(f"  Environment propagation: {env_prop_time:.2f}s")
    print(f"  ALTRO trajOpt time:      {altro_time:.2f}s")
    print(f"  Total calculate_traj:    {total_calc_time:.2f}s")
    print(f"  ---")
    print(f"  Final angular velocity:  {final_ang_vel:.4f} deg/s")
    print(f"  Final tracking error:    {final_error_deg:.4f} deg")
    print(f"  Trajectory N steps:      {traj.n_steps}")

    return {
        'bdot_on': bdot_on,
        'orbit_time': orbit_time,
        'env_prop_time': env_prop_time,
        'altro_time': altro_time,
        'total_calc_time': total_calc_time,
        'final_ang_vel': final_ang_vel,
        'final_error_deg': final_error_deg,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("TIMING COMPARISON: bdot_on=2 vs bdot_on=3 (with smart bdot)")
    print("=" * 70)

    results_2 = time_altro_only(bdot_on=2)
    print()
    results_3 = time_altro_only(bdot_on=3)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<30} {'bdot=2':>12} {'bdot=3':>12} {'Diff':>12}")
    print("-" * 70)
    print(f"{'ALTRO trajOpt time (s)':<30} {results_2['altro_time']:>12.2f} {results_3['altro_time']:>12.2f} {results_3['altro_time'] - results_2['altro_time']:>+12.2f}")
    print(f"{'Env propagation time (s)':<30} {results_2['env_prop_time']:>12.2f} {results_3['env_prop_time']:>12.2f} {results_3['env_prop_time'] - results_2['env_prop_time']:>+12.2f}")
    print(f"{'Total calc_traj time (s)':<30} {results_2['total_calc_time']:>12.2f} {results_3['total_calc_time']:>12.2f} {results_3['total_calc_time'] - results_2['total_calc_time']:>+12.2f}")
    print("-" * 70)
    print(f"{'Final ang velocity (deg/s)':<30} {results_2['final_ang_vel']:>12.4f} {results_3['final_ang_vel']:>12.4f}")
    print(f"{'Final tracking error (deg)':<30} {results_2['final_error_deg']:>12.4f} {results_3['final_error_deg']:>12.4f}")
    print("\nNote: bdot_on=3 adds random noise to smart bdot initial trajectory")
