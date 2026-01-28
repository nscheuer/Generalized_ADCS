import sys
import os as os_pack
import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Union
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

sys.path.append(os_pack.path.abspath(os_pack.path.join(__file__, "../../..")))
from ADCS.CONOPS.goals import Fixed_Attitude_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.helpers import PlannerSettings, Trajectory, planner_settings
from ADCS.controller.helpers.planner_subsettings import CostWeights
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import random_n_unit_vec, normalize
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit

from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat

from ADCS.helpers.plotting.animate_estimator import animate_attitude
from ADCS.helpers.plotting.plot_estimator import plot_state_comparison
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window
from ADCS.helpers.plotting.plot_controller import plot_control, plot_rw_momentum, plot_target_tracking
from ADCS.helpers.plotting.animate_orbit import animate_orbit
from ADCS.helpers.plotting.animate_orbit_pyvista import animate_orbit_pyvista

def debug_altro(verbose: bool = False, tf: float = 1000, dt: float = 1, real_orbit: bool = False) -> Union[np.ndarray, np.ndarray, List[Orbital_State], np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(1)
    t0 = 0
    N = int((tf-t0)/dt)

    real_sat = create_beavercube2_cubesat(estimated=False)
    rw_h0 = 0.0001

    rng = np.random.default_rng(seed=2333)
    w0 = normalize(rng.standard_normal(3)) * (rng.uniform(0.1, 1.0) * np.pi / 180.0)
    q0 = normalize(rng.standard_normal(4))
    h0 = np.array([rw_h0])
    x = np.concatenate([w0, q0, h0])

    start_time = 0.22 - 1*TimeConstants.sec2cent
    orb = create_random_circular_orbit(7000, dt=1, tf=1000, use_J2=True, fast=False)
    os0 = orb.get_os(J2000=start_time)

    # Build Planner
    planner_settings = PlannerSettings(
        est_sat=real_sat,
        bdot_on=0,  # Skip bdot initial guess (faster, more reliable)
        dt_tp=50,
        dt_tvlqr=1,
    )

    planner_settings.verbosity = False
    planner_settings.cost_main.use_full_cost_hessian = True
    planner_settings.pass1.regularization.use_dynamics_hess = 1
    planner_settings.init_traj.bdot_gain = 500
    planner_settings.pass1.aug_lag.penalty_init = 1e-3
    planner_settings.pass1.aug_lag.penalty_scale = 10
    planner_settings.pass1.convergence.max_outer_iter = 15
    planner_settings.pass1.convergence.max_inner_iter = 40
    planner_settings.pass2.aug_lag.penalty_init = 1e5
    planner_settings.pass2.aug_lag.penalty_scale = 10
    planner_settings.pass2.convergence.max_outer_iter = 8
    planner_settings.pass2.convergence.max_inner_iter = 20

    planner_settings.cost_main = CostWeights(
        angle=1e1,
        angle_N=1e1,   # 10x running cost
        ang_vel=1e5,
        ang_vel_N=1e5, # 10x running cost
        ang_vel_err_dir=1e2,
        ang_vel_err_dir_N=0.0,
        ang_vel_mag=0.0,
        ang_vel_mag_N=0.0,
        control_mult=1.0,
        ang_cost_func_type=2,
    )
        
    planner_settings.cost_second = planner_settings.cost_main
        
    planner_settings.cost_tvlqr = CostWeights(
        angle=1e5,
        angle_N=1e6,
        ang_vel=1e6,
        ang_vel_N=1e8,
        ang_vel_mag=0.0,
        ang_vel_mag_N=0.0,
        control_mult=1.0,
        ang_cost_func_type=2,
    )

    controller = Plan_and_Track_LQR(est_sat=real_sat, planner_settings=planner_settings)

    time_hist = np.nan*np.zeros(N)
    state_hist = np.nan*np.zeros((N, len(x)))
    os_hist: List[Orbital_State] = list()
    sensor_hist: np.ndarray = np.nan*np.zeros((N, len(real_sat.sensors + real_sat.rw_actuators)))
    u_hist = np.nan*np.zeros((N, len(real_sat.actuators)))
    q_goal_hist = np.nan*np.zeros((N, 4))

    t = t0
    ind = 0
    steps = int((tf - t0)/dt)

    goals = GoalList({0.22: Fixed_Attitude_Goal(np.array([0, 0, -1, 0]))})

    traj_duration = tf - t0  # [s]

    print("\n========== ALTRO TRAJECTORY PLANNING ==========")
    print(f"Requested trajectory duration : {traj_duration:.2f} s")

    t_plan_start = time.perf_counter()

    traj: Trajectory = controller.calculate_trajectory(
        t_start=0.22,
        duration=traj_duration,
        x_0=x,
        os_0=os0,
        goals=goals,
        verbose=verbose 
    )

    t_plan_end = time.perf_counter()
    plan_wall_time = t_plan_end - t_plan_start

    print(f"Trajectory planning wall time : {plan_wall_time:.3f} s")

    if traj_duration > 0:
        rtf = plan_wall_time / traj_duration
        print(f"Real-time factor (RTF)        : {rtf:.3f} x")
        print(f"Equivalent speed             : {1/rtf:.2f} x real-time")

    print("==============================================\n")
    controller.set_active_trajectory(traj)
    time_hist_traj = (traj.times-start_time)*TimeConstants.cent2sec
    state_hist_traj = traj.states.T
    u_hist_traj = traj.controls.T

    plot_state_comparison(time=time_hist_traj, state_hist=state_hist_traj)
    plot_control(time=time_hist_traj, u_hist=u_hist_traj)

    q_goal_traj_hist = np.vstack([goals.to_ref(t=J2000, os0=orb.get_os(J2000))[0] for J2000 in traj.times])
    plot_target_tracking(state_hist=state_hist_traj, boresight_hist=q_goal_traj_hist, body_boresight=np.array([0, 1, 0]))
    plot_rw_momentum(time=time_hist_traj, state_hist=state_hist_traj)
    create_close_all_button_window()
    
    for step in tqdm(range(steps), desc="Simulating ALTRO"):
        J2000 = 0.22 + t*TimeConstants.sec2cent
        os = orb.get_os(J2000=J2000)

        sens = real_sat.sensor_readings(x=x, os=os)
        u = controller.find_u(x_hat=x, sens=sens, est_sat=real_sat, os_hat=os)

        if verbose:
            print("u: ", u)

        time_hist[ind] = t
        state_hist[ind,:] = x
        os_hist += [os]
        sensor_hist[ind,:] = sens
        u_hist[ind,:] = u

        
        # Updated reference logging: Query GoalList for the reference at this time
        # Note: to_ref now returns (eci, omega), we take [0] for the ECI vector
        q_goal, w_goal = goals.to_ref(t=J2000, os0=os)
        q_goal_hist[ind, :] = q_goal

        ind += 1
        t += dt
        prev_os = os.copy()
        os = orb.get_os(0.22+(t-t0)*TimeConstants.sec2cent)

        out = solve_ivp(fun=real_sat.dynamics_for_solver, t_span=(0, dt), y0=x, method="RK45", args=(u, prev_os, os), rtol=1e-7, atol=1e-7)
        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])

    return time_hist, state_hist, os_hist, sensor_hist, u_hist, q_goal_hist


def plot_mtq_w_rw_align_to_eci(verbose: bool = False, tf: float = 1000, dt: float = 10, real_orbit: bool = False) -> None:
    (time_hist, state_hist, os_hist, sensor_hist, u_hist, q_goal_hist) = debug_altro(verbose=verbose, tf=tf, dt=dt, real_orbit=real_orbit)

    animate_attitude(time=time_hist, state_hist=state_hist, os_hist=os_hist, boresight_goal_hist=q_goal_hist)
    plot_state_comparison(time=time_hist, state_hist=state_hist)
    plot_control(time=time_hist, u_hist=u_hist)
    plot_rw_momentum(time=time_hist, state_hist=state_hist)
    # animate_orbit_pyvista(time_hist=time_hist, state_hist=state_hist, os_hist=os_hist, boresight_goal_hist=boresight_hist, coord_goal=goal)
    plot_target_tracking(state_hist=state_hist, boresight_hist=q_goal_hist, body_boresight=np.array([0, 1, 0]))
    #animate_orbit(time_hist=time_hist, state_hist=state_hist, os_hist=os_hist, boresight_goal_hist=boresight_hist, coord_goal=goal)
    create_close_all_button_window()
    print("Yay!")

if __name__ == "__main__":
    plot_mtq_w_rw_align_to_eci(verbose=True, tf = 1000, dt = 1, real_orbit=True)