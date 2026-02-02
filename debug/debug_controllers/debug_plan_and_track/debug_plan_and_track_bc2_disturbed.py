"""
Debug script for Plan and Track LQR controller with disturbance estimation on BC2 satellite.

This script tests trajectory planning and TVLQR tracking using the ALTRO planner
with the KwDist formulation that includes disturbance estimation/compensation.
Similar to debug_plan_and_track_bc2.py but uses the Plan_and_Track_LQR_Disturbed controller.
"""
import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Union, Tuple
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))

from ADCS.CONOPS.goals import ECI_Goal, Coordinate_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr_disturbed import Plan_and_Track_LQR_Disturbed
from ADCS.controller.helpers import PlannerSettings, Trajectory
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.helpers.math_helpers import random_n_unit_vec, normalize

from ADCS.helpers.plotting.animate_estimator import animate_attitude
from ADCS.helpers.plotting.plot_estimator import plot_state_comparison
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window
from ADCS.helpers.plotting.plot_controller import plot_control, plot_rw_momentum, plot_target_tracking

import matplotlib.pyplot as plt


def test_plan_and_track_lqr_disturbed(
    verbose: bool = False,
    tf: float = 1000,
    dt: float = 1,
    dt_planning: float = 1,
    real_orbit: bool = True,
    seed: int = 37,
) -> Tuple[np.ndarray, np.ndarray, List[Orbital_State], np.ndarray, np.ndarray, np.ndarray, Trajectory, np.ndarray]:
    """
    Test the Plan and Track LQR controller with disturbance compensation on BC2 satellite.

    Args:
        verbose: Print debug information
        tf: Final time in seconds
        dt: Simulation timestep in seconds
        dt_planning: Trajectory planner timestep in seconds
        real_orbit: Use real orbit propagation (True) or simplified orbit (False)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (time_hist, state_hist, os_hist, sensor_hist, u_hist, boresight_hist, trajectory, dist_torque_hist)
    """
    np.random.seed(seed)
    t0 = 0
    N = int((tf - t0) / dt)

    # Create BC2 satellite
    rw_h0 = 0.0
    real_sat = create_beavercube2_cubesat(estimated=False)
    real_sat.rw_actuators[0].h = rw_h0

    # Initial conditions
    w0 = random_n_unit_vec(3) * np.random.uniform(0.5, 1.0) * np.pi / 180.0
    q0 = normalize(np.random.randn(4))
    h0 = np.array([rw_h0])
    x = np.concatenate([w0, q0, h0])

    print(f"Initial angular velocity: {np.rad2deg(np.linalg.norm(w0)):.2f} deg/s")
    print(f"Initial quaternion: {q0}")

    # Create orbit
    ephem = Ephemeris()
    start_time = 0.22 - 1 * TimeConstants.sec2cent
    end_time = 0.22 + (tf - t0) * TimeConstants.sec2cent
    R = 7000 * np.array([0, np.sqrt(2) / 2, np.sqrt(2) / 2])
    V = np.array([8, 0, 0])

    if real_orbit:
        print("Creating real orbit (this may take a moment)...")
        os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
        orb = Orbit(os0=os0, end_time=end_time, dt=dt, use_J2=True, fast=False)
    else:
        os0 = Orbital_State(
            ephem=ephem,
            J2000=0.22 - 1 * TimeConstants.sec2cent,
            R=R,
            V=V,
            B=np.array([0, 0.1, 0]),
            S=np.array([1e5 + 1, 0, 0]),
            rho=5e-12,
        )
        dur = int((tf - t0) / dt) + 10
        orbs = [os0] * (dur + 10)
        for j in range(dur):
            orbs[j] = os0.copy()
            orbs[j].J2000 = os0.J2000 + j * dt * TimeConstants.sec2cent
        orb = Orbit(orbs)

    # Setup planner with disturbance estimation
    print("Setting up trajectory planner with disturbance estimation...")
    planner_settings = PlannerSettings(
        est_sat=real_sat,
        bdot_on=1,  # Enable bdot for initial detumble
        dt_tp=dt_planning,  # Coarse trajectory planner timestep
        dt_tvlqr=dt,  # Fine TVLQR timestep
    )
    planner_settings.verbosity = verbose

    # Control weights - make RW much cheaper so planner strongly prefers it
    # Lower weight = cheaper actuator in the cost function
    # Matched to bc2.py which uses rw=1e-8, mtq=1e4 (RW is 10^12x cheaper)
    planner_settings.rw_control_weight = 1e-8  # Essentially free RW usage
    planner_settings.mtq_control_weight = 1e4  # MTQ more expensive

    # Angle costs - prioritize pointing accuracy (matched to bc2.py)
    planner_settings.cost_main.angle = 1e8
    planner_settings.cost_second.angle = 1e8
    planner_settings.cost_tvlqr.angle = 1e10

    # Angular velocity costs
    planner_settings.cost_main.ang_vel = 1e4
    planner_settings.cost_second.ang_vel = 1e4
    planner_settings.cost_tvlqr.ang_vel = 1e6

    # Terminal angular velocity costs
    planner_settings.cost_main.ang_vel_N = 1e6
    planner_settings.cost_second.ang_vel_N = 1e6
    planner_settings.cost_tvlqr.ang_vel_N = 1e9

    # Terminal angle costs - very high for strong goal reaching (matched to bc2.py)
    planner_settings.cost_main.angle_N = 1e10
    planner_settings.cost_second.angle_N = 1e10
    planner_settings.cost_tvlqr.angle_N = 1e13

    # Control cost settings
    # use_raw_control_cost=False: penalize control RATE (smoother trajectories)
    # use_raw_control_cost=True: penalize control magnitude (for TVLQR tracking)
    planner_settings.cost_main.use_raw_control_cost = False  # Smooth trajectory
    planner_settings.cost_second.use_raw_control_cost = False
    planner_settings.cost_tvlqr.use_raw_control_cost = True  # Direct control for tracking

    # Higher control_mult for TVLQR prevents aggressive feedback oscillations
    planner_settings.cost_main.control_mult = 1.0
    planner_settings.cost_second.control_mult = 1e8  # Matched to bc2.py
    planner_settings.cost_tvlqr.control_mult = 1e8   # Prevents TVLQR oscillations

    # Enable disturbance planning
    planner_settings.plan_for_aero = True
    planner_settings.plan_for_srp = True
    planner_settings.plan_for_gg = True

    # Try higher initial penalty to enforce constraints earlier
    planner_settings.pass1.aug_lag.penalty_init = 1e-3

    # Create controller with disturbance compensation
    controller = Plan_and_Track_LQR_Disturbed(
        est_sat=real_sat,
        planner_settings=planner_settings,
    )

    # Goal setup
    goal_vec = normalize(np.array([0, 0, 1]))
    goal = ECI_Goal(goal_vec)
    goals = GoalList({0.22: goal})

    # Calculate trajectory
    print("Calculating trajectory with KwDist gains...")
    os0_for_traj = orb.get_os(0.22)
    try:
        traj: Trajectory = controller.calculate_trajectory(
            t_start=0.22,
            duration=tf - t0,
            x_0=x,
            os_0=os0_for_traj,
            goals=goals,
            verbose=verbose,
        )
        controller.set_active_trajectory(traj)
        traj_duration_centuries = traj.end_time - traj.start_time
        traj_duration_seconds = traj_duration_centuries / TimeConstants.sec2cent
        print(f"Trajectory calculated successfully!")
        print(f"  Start: {traj.start_time:.6f}, End: {traj.end_time:.6f} (J2000 centuries)")
        print(f"  Duration: {traj_duration_seconds:.1f}s")
        print(f"  N steps: {traj.n_steps}, Gains shape: {traj.gains.shape}")
        print(f"  Disturbance compensation: ENABLED (using est_sat.dist_torques)")
    except Exception as e:
        print(f"Trajectory calculation failed: {e}")
        raise

    time_hist_traj = (traj.times - start_time) * TimeConstants.cent2sec
    state_hist_traj = traj.states.T
    u_hist_traj = traj.controls.T

    plot_state_comparison(time=time_hist_traj, state_hist=state_hist_traj)
    plot_control(time=time_hist_traj, u_hist=u_hist_traj)

    boresight_traj_hist = np.vstack([goals.to_ref(t=J2000, os0=orb.get_os(J2000))[0] for J2000 in traj.times])
    plot_target_tracking(state_hist=state_hist_traj, boresight_hist=boresight_traj_hist, body_boresight=np.array([0, 0, 1]))

    plot_rw_momentum(time=time_hist_traj, state_hist=state_hist_traj)
    create_close_all_button_window()

    # Initialize history arrays
    time_hist = np.nan * np.zeros(N)
    state_hist = np.nan * np.zeros((N, len(x)))
    os_hist: List[Orbital_State] = []
    sensor_hist = np.nan * np.zeros((N, len(real_sat.sensors + real_sat.rw_actuators)))
    u_hist = np.nan * np.zeros((N, len(real_sat.actuators)))
    boresight_hist = np.nan * np.zeros((N, 4))
    dist_torque_hist = np.nan * np.zeros((N, 3))  # Track disturbance torques from satellite model

    # Simulation loop
    t = t0
    ind = 0
    steps = int((tf - t0) / dt)

    print(f"Running simulation for {tf}s with dt={dt}s...")
    for step in tqdm(range(steps), desc="Simulating Plan & Track LQR Disturbed"):
        J2000 = 0.22 + t * TimeConstants.sec2cent
        os = orb.get_os(J2000=J2000)

        sens = real_sat.sensor_readings(x=x, os=os)

        # Get control from TVLQR tracking with disturbance compensation
        try:
            u = controller.find_u(x_hat=x, sens=sens, est_sat=real_sat, os_hat=os)
        except RuntimeError as e:
            print(f"Controller error at t={t}: {e}")
            break

        # Store history
        time_hist[ind] = t
        state_hist[ind, :] = x
        os_hist.append(os)
        sensor_hist[ind, :] = sens
        u_hist[ind, :] = u
        eci_goal, w_goal = goal.to_ref(os0=os)
        boresight_hist[ind, :] = eci_goal
        # Get disturbance torque from satellite model for logging
        dist_torque_hist[ind, :] = real_sat.dist_torques(x=x, os=os)

        # Propagate dynamics
        ind += 1
        t += dt
        prev_os = os.copy()
        os_next = orb.get_os(0.22 + (t - t0) * TimeConstants.sec2cent)

        out = solve_ivp(
            fun=real_sat.dynamics_for_solver,
            t_span=(0, dt),
            y0=x,
            method="RK45",
            args=(u, prev_os, os_next),
            rtol=1e-7,
            atol=1e-7,
        )
        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])

    return time_hist, state_hist, os_hist, sensor_hist, u_hist, boresight_hist, traj, dist_torque_hist


def plot_disturbance_torques(time: np.ndarray, dist_torque_hist: np.ndarray) -> None:
    """Plot the disturbance torque history from satellite model."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, dist_torque_hist[:, 0], label="τ_x")
    ax.plot(time, dist_torque_hist[:, 1], label="τ_y")
    ax.plot(time, dist_torque_hist[:, 2], label="τ_z")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Disturbance Torque [Nm]")
    ax.set_title("Disturbance Torques (from satellite model)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()


def plot_plan_and_track_lqr_disturbed(
    verbose: bool = False,
    tf: float = 1000,
    dt: float = 1,
    dt_planning: float = 1,
    real_orbit: bool = True,
    seed: int = 37,
) -> None:
    """
    Run and plot the Plan and Track LQR controller with disturbance compensation test.
    """
    results = test_plan_and_track_lqr_disturbed(
        verbose=verbose,
        tf=tf,
        dt=dt,
        dt_planning=dt_planning,
        real_orbit=real_orbit,
        seed=seed,
    )
    time_hist, state_hist, os_hist, sensor_hist, u_hist, boresight_hist, traj, dist_torque_hist = results

    # Trim NaN values
    valid_idx = ~np.isnan(time_hist)
    time_hist = time_hist[valid_idx]
    state_hist = state_hist[valid_idx]
    u_hist = u_hist[valid_idx]
    boresight_hist = boresight_hist[valid_idx]
    dist_torque_hist = dist_torque_hist[valid_idx]

    print(f"\n--- Simulation Complete ---")
    print(f"Final angular velocity: {np.rad2deg(np.linalg.norm(state_hist[-1, :3])):.4f} deg/s")
    print(f"Final quaternion: {state_hist[-1, 3:7]}")
    print(f"Final disturbance torque: {dist_torque_hist[-1]} Nm")

    # Calculate final tracking error
    q_final = state_hist[-1, 3:7]
    # Rotation matrix from body to inertial
    w, x_q, y_q, z_q = q_final
    R = np.array([
        [1 - 2*(y_q**2 + z_q**2), 2*(x_q*y_q - z_q*w), 2*(x_q*z_q + y_q*w)],
        [2*(x_q*y_q + z_q*w), 1 - 2*(x_q**2 + z_q**2), 2*(y_q*z_q - x_q*w)],
        [2*(x_q*z_q - y_q*w), 2*(y_q*z_q + x_q*w), 1 - 2*(x_q**2 + y_q**2)]
    ])
    body_boresight = np.array([0, 0, 1])
    eci_boresight = R @ body_boresight
    goal_eci = boresight_hist[-1]
    error_rad = np.arccos(np.clip(np.dot(eci_boresight, goal_eci), -1, 1))
    print(f"Final tracking error: {np.rad2deg(error_rad):.4f} deg")

    # Plot results
    animate_attitude(time=time_hist, state_hist=state_hist, os_hist=os_hist, boresight_goal_hist=boresight_hist)
    plot_control(time=time_hist, u_hist=u_hist)
    plot_state_comparison(time=time_hist, state_hist=state_hist)
    plot_target_tracking(state_hist=state_hist, boresight_hist=boresight_hist, body_boresight=np.array([0, 0, 1]))
    plot_disturbance_torques(time=time_hist, dist_torque_hist=dist_torque_hist)

    create_close_all_button_window()


if __name__ == "__main__":
    plot_plan_and_track_lqr_disturbed(
        verbose=False,
        tf=500,  # Reduced for faster debugging
        dt=1,
        dt_planning=30,  # Coarser planning timestep
        real_orbit=True,
        seed=37,
    )
