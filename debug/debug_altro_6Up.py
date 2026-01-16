import sys
import os as os_pack
import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Union
from tqdm import tqdm
import matplotlib.pyplot as plt
import pytest

sys.path.append(os_pack.path.abspath(os_pack.path.join(__file__, "../..")))
from ADCS.CONOPS.goals import Goal, ECI_Goal, Coordinate_Goal, No_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_exact import Plan_and_Track_Exact
from ADCS.controller.helpers import PlannerSettings, Trajectory, planner_settings
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import random_n_unit_vec, normalize

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

    mtq_max_torque = 5.0
    mtqs = [MTQ(axis=j, max_torque=mtq_max_torque) for j in MathConstants.unitvecs]

    rw_max_torque = 0.005
    rw_J = 0.0014
    rw_h0 = 0.001
    rw_hmax = 0.015
    rws = [RW(axis=j, max_torque=rw_max_torque, J=rw_J, h=rw_h0, h_max=rw_hmax) for j in MathConstants.unitvecs]

    acts = rws + mtqs
    rwN = sum([isinstance(act,RW) for act in acts])

    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]

    real_sat = Satellite(mass=10.165, J_0=np.diagflat([0.0969,0.1235,0.1918]), actuators=acts, sensors=mtms, boresight=np.array([0, 0, 1]))

    w0 = random_n_unit_vec(3)*np.random.uniform(1, 2)*np.pi/180.0
    w0 = np.array([0.001, 0.002, -0.001])
    q0 = random_n_unit_vec(4)
    q0 = normalize(np.array([1, 0, 0, 0]))
    h0 = np.array([rw_h0]*rwN)
    x = np.concatenate([w0, q0, h0])

    ephem = Ephemeris()
    start_time = 0.22 - 1*TimeConstants.sec2cent
    end_time = 0.22 + (tf-t0)*TimeConstants.sec2cent
    R = 7000*np.array([0, np.sqrt(2)/2, np.sqrt(2)/2])
    V = np.array([8, 0, 0])
    if real_orbit:
        # Real Orbit Generation
        os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
        orb = Orbit(os0=os0, end_time=end_time, dt=dt, use_J2=True, fast=False)
    else:
        os0 = Orbital_State(ephem=ephem, J2000=0.22-1*TimeConstants.sec2cent, R=R, V=V, B=np.array([0, 0.1, 0]), S=np.array([1e5+1, 0, 0]), rho=5e-12)
        dur = int((tf-t0)/dt)+10
        orbs = [os0]*(dur+10)
        for j in range(dur):
            orbs[j] = os0.copy()
            orbs[j].J2000 = os0.J2000 + j*dt*TimeConstants.sec2cent
        orb = Orbit(orbs)

    # Build Planner
    planner_settings = PlannerSettings(est_sat=real_sat, bdot_on=0,dt_tp = 1.0)
    # planner_settings.rw_AM_weight = 0  # Disable AM cost
    # planner_settings.rw_stic_weight = 0  # Disable stiction cost - causes non-convex Hessian!
    planner_settings.verbosity = verbose
    planner_settings.rw_control_weight = 1e8  # Default value
    planner_settings.mtq_control_weight = 1e0
    planner_settings.cost_main.ang_vel = 1e4  # Default value
    planner_settings.cost_second.ang_vel = 1e4
    planner_settings.cost_main.use_raw_control_cost = True  # Use control rate cost to penalize oscillation

    # Try higher initial penalty to enforce constraints earlier
    planner_settings.pass1.aug_lag.penalty_init = 1e-3

    controller = Plan_and_Track_Exact(est_sat=real_sat, planner_settings=planner_settings)

    time_hist = np.nan*np.zeros(N)
    state_hist = np.nan*np.zeros((N, len(x)))
    os_hist: List[Orbital_State] = list()
    sensor_hist: np.ndarray = np.nan*np.zeros((N, len(real_sat.sensors + real_sat.rw_actuators)))
    u_hist = np.nan*np.zeros((N, len(acts)))
    boresight_hist = np.nan*np.zeros((N, 3))

    t = t0
    ind = 0
    steps = int((tf - t0)/dt)

    # Simplified goal - just ECI_Goal from start, no transition
    goals = GoalList({0.22: ECI_Goal(np.array([1, 1, 1]))})

    print("Computing Trajectory (One-Shot)")
    traj: Trajectory = controller.calculate_trajectory(
        t_start=0.22,
        duration=tf-t0,
        x_0=x,
        os_0=os0,
        goals=goals,
        verbose=verbose 
    )
    # traj.plot_eci_trajectory()
    controller.set_active_trajectory(traj)
    time_hist_traj = (traj.times-start_time)*TimeConstants.cent2sec
    state_hist_traj = traj.states.T
    u_hist_traj = traj.controls.T

    # Save trajectory data for analysis
    save_path = os_pack.path.join(os_pack.path.dirname(__file__), "debug_altro_trajectory.npz")
    np.savez(save_path,
        times=time_hist_traj,
        states=state_hist_traj,
        controls=u_hist_traj,
        rw_max_torque=rw_max_torque,
        rw_hmax=rw_hmax,
        J=real_sat.J_0,
    )
    print(f"\nTrajectory saved to: {save_path}")

    # # Analyze control pattern
    # print("\n" + "="*60)
    # print("CONTROL PATTERN ANALYSIS")
    # print("="*60)
    # print(f"RW max torque: {rw_max_torque} Nm")
    # print(f"Number of timesteps: {len(time_hist_traj)}")
    # print(f"dt: {dt} s")

    # # Check for sign changes in each control channel
    # for ch in range(u_hist_traj.shape[1]):
    #     u_ch = u_hist_traj[:, ch]
    #     sign_changes = np.sum(np.diff(np.sign(u_ch)) != 0)
    #     at_max = np.sum(np.abs(u_ch) > 0.95 * rw_max_torque)
    #     print(f"\nChannel {ch}:")
    #     print(f"  Sign changes: {sign_changes} / {len(u_ch)-1} timesteps")
    #     print(f"  At >95% saturation: {at_max} / {len(u_ch)} timesteps")
    #     print(f"  Range: [{u_ch.min():.6f}, {u_ch.max():.6f}]")

    #     # Check for rapid oscillation (sign change every 1-2 steps)
    #     if len(u_ch) > 3:
    #         signs = np.sign(u_ch)
    #         rapid_osc = 0
    #         for i in range(len(signs) - 2):
    #             if signs[i] != signs[i+1] and signs[i+1] != signs[i+2]:
    #                 rapid_osc += 1
    #         if rapid_osc > 0:
    #             print(f"  WARNING: {rapid_osc} instances of rapid oscillation (+-+ or -+- pattern)")

    # # Print first 20 control values to see the pattern
    # print("\n" + "-"*60)
    # print("First 20 control values:")
    # print("-"*60)
    # print("   k |    u[0]    |    u[1]    |    u[2]    |")
    # print("-"*60)
    # for k in range(min(20, len(u_hist_traj))):
    #     u = u_hist_traj[k]
    #     print(f" {k:3d} | {u[0]:10.6f} | {u[1]:10.6f} | {u[2]:10.6f} |")

    # # Check control differences (should be smooth for bang-bang, spiky for oscillation)
    # print("\n" + "-"*60)
    # print("Control differences (du = u[k+1] - u[k]):")
    # print("-"*60)
    # print("   k |   du[0]    |   du[1]    |   du[2]    |")
    # print("-"*60)
    # for k in range(min(19, len(u_hist_traj)-1)):
    #     du = u_hist_traj[k+1] - u_hist_traj[k]
    #     print(f" {k:3d} | {du[0]:10.6f} | {du[1]:10.6f} | {du[2]:10.6f} |")

    plot_state_comparison(time=time_hist_traj, state_hist=state_hist_traj)
    plot_control(time=time_hist_traj, u_hist=u_hist_traj)

    boresight_traj_hist = np.vstack([goals.to_ref(t=J2000, os0=orb.get_os(J2000))[0] for J2000 in traj.times])
    plot_target_tracking(state_hist=state_hist_traj, boresight_hist=boresight_traj_hist, body_boresight=np.array([0, 0, 1]))

    if rwN>0:
        plot_rw_momentum(time=time_hist_traj, state_hist=state_hist_traj)
    plt.show()
    plt.close("all")
    # create_close_all_button_window()
    
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
        eci_goal, w_goal = goals.to_ref(t=J2000, os0=os)
        boresight_hist[ind, :] = eci_goal

        ind += 1
        t += dt
        prev_os = os.copy()
        os = orb.get_os(0.22+(t-t0)*TimeConstants.sec2cent)

        out = solve_ivp(fun=real_sat.dynamics_for_solver, t_span=(0, dt), y0=x, method="RK45", args=(u, prev_os, os), rtol=1e-7, atol=1e-7)
        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])

    return time_hist, state_hist, os_hist, sensor_hist, u_hist, boresight_hist


def plot_mtq_w_rw_align_to_eci(verbose: bool = False, tf: float = 1000, dt: float = 10, real_orbit: bool = False) -> None:
    (time_hist, state_hist, os_hist, sensor_hist, u_hist, boresight_hist) = debug_altro(verbose=verbose, tf=tf, dt=dt, real_orbit=real_orbit)

    # animate_attitude(time=time_hist, state_hist=state_hist, os_hist=os_hist, boresight_goal_hist=boresight_hist)
    plot_state_comparison(time=time_hist, state_hist=state_hist)
    plot_control(time=time_hist, u_hist=u_hist)
    plot_rw_momentum(time=time_hist, state_hist=state_hist)
    goal = Coordinate_Goal(lat=38.7223, lon=-10, alt=0)
    # animate_orbit_pyvista(time_hist=time_hist, state_hist=state_hist, os_hist=os_hist, boresight_goal_hist=boresight_hist, coord_goal=goal)
    plot_target_tracking(state_hist=state_hist, boresight_hist=boresight_hist, body_boresight=np.array([0, 0, 1]))
    #animate_orbit(time_hist=time_hist, state_hist=state_hist, os_hist=os_hist, boresight_goal_hist=boresight_hist, coord_goal=goal)
    plt.show()
    create_close_all_button_window()
    print("Yay!")

if __name__ == "__main__":
    plot_mtq_w_rw_align_to_eci(verbose=False, tf = 100, dt = 1, real_orbit=True)