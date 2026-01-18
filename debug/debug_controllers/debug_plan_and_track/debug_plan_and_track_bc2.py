"""
Debug script for Plan and Track LQR controller with BC2 satellite.

This script tests trajectory planning and TVLQR tracking using the ALTRO planner.
Similar to debug_mtq_w_rw_lp_bc2.py but uses trajectory-based control.
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
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
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


def test_plan_and_track_lqr(
    verbose: bool = False,
    tf: float = 1000,
    dt: float = 2,
    dt_planning: float = 1,
    real_orbit: bool = True,
    seed: int = 37,
) -> Tuple[np.ndarray, np.ndarray, List[Orbital_State], np.ndarray, np.ndarray, np.ndarray, Trajectory]:
    """
    Test the Plan and Track LQR controller with BC2 satellite.

    Args:
        verbose: Print debug information
        tf: Final time in seconds
        dt: Simulation timestep in seconds
        dt_planning: Trajectory planner timestep in seconds
        real_orbit: Use real orbit propagation (True) or simplified orbit (False)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (time_hist, state_hist, os_hist, sensor_hist, u_hist, boresight_hist, trajectory)
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
        orb = Orbit(os0=os0, end_time=end_time, dt=dt_planning, use_J2=True, fast=True)
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

    # Setup planner
    print("Setting up trajectory planner...")
    # dt_tp is the coarse trajectory planner timestep (ALTRO optimization step)
    # dt_tvlqr is the finer TVLQR feedback controller timestep
    planner_settings = PlannerSettings(
        est_sat=real_sat,
        bdot_on=0,
        dt_tp=10.0,  # Coarse trajectory planner timestep (10s)
        dt_tvlqr=dt_planning,  # Fine TVLQR timestep
    )
    planner_settings.verbosity = verbose

    controller = Plan_and_Track_LQR(
        est_sat=real_sat,
        planner_settings=planner_settings,
    )

    # Goal setup
    goal_vec = normalize(np.array([0, 0, 1]))
    goal = ECI_Goal(goal_vec)
    goals = GoalList({0.22: goal})

    # Calculate trajectory
    print("Calculating trajectory...")
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
    except Exception as e:
        print(f"Trajectory calculation failed: {e}")
        raise

    # Initialize history arrays
    time_hist = np.nan * np.zeros(N)
    state_hist = np.nan * np.zeros((N, len(x)))
    os_hist: List[Orbital_State] = []
    sensor_hist = np.nan * np.zeros((N, len(real_sat.sensors + real_sat.rw_actuators)))
    u_hist = np.nan * np.zeros((N, len(real_sat.actuators)))
    boresight_hist = np.nan * np.zeros((N, 3))

    # Simulation loop
    t = t0
    ind = 0
    steps = int((tf - t0) / dt)

    print(f"Running simulation for {tf}s with dt={dt}s...")
    for step in tqdm(range(steps), desc="Simulating Plan & Track LQR"):
        J2000 = 0.22 + t * TimeConstants.sec2cent
        os = orb.get_os(J2000=J2000)

        sens = real_sat.sensor_readings(x=x, os=os)

        # Get control from TVLQR tracking
        try:
            u = controller.find_u(x_hat=x, sens=sens, est_sat=real_sat, os_hat=os)
        except RuntimeError as e:
            print(f"Controller error at t={t}: {e}")
            break

        if verbose:
            print(f"t={t:.1f}s, u={u}")

        # Store history
        time_hist[ind] = t
        state_hist[ind, :] = x
        os_hist.append(os)
        sensor_hist[ind, :] = sens
        u_hist[ind, :] = u
        eci_goal, w_goal = goal.to_ref(os0=os)
        boresight_hist[ind, :] = eci_goal

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

    return time_hist, state_hist, os_hist, sensor_hist, u_hist, boresight_hist, traj


def plot_plan_and_track_lqr(
    verbose: bool = False,
    tf: float = 1000,
    dt: float = 2,
    dt_planning: float = 1,
    real_orbit: bool = True,
    seed: int = 37,
) -> None:
    """
    Run and plot the Plan and Track LQR controller test.
    """
    results = test_plan_and_track_lqr(
        verbose=verbose,
        tf=tf,
        dt=dt,
        dt_planning=dt_planning,
        real_orbit=real_orbit,
        seed=seed,
    )
    time_hist, state_hist, os_hist, sensor_hist, u_hist, boresight_hist, traj = results

    # Trim NaN values
    valid_idx = ~np.isnan(time_hist)
    time_hist = time_hist[valid_idx]
    state_hist = state_hist[valid_idx]
    u_hist = u_hist[valid_idx]
    boresight_hist = boresight_hist[valid_idx]

    print(f"\n--- Simulation Complete ---")
    print(f"Final angular velocity: {np.rad2deg(np.linalg.norm(state_hist[-1, :3])):.4f} deg/s")
    print(f"Final quaternion: {state_hist[-1, 3:7]}")

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

    create_close_all_button_window()


if __name__ == "__main__":
    plot_plan_and_track_lqr(
        verbose=False,
        tf=300,  # Reduced for faster debugging
        dt=2,
        dt_planning=2,  # Coarser planning timestep
        real_orbit=True,
        seed=37,
    )
