import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Union, Tuple
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))

from ADCS.CONOPS.goallist import GoalList
from ADCS.CONOPS.goals import Goal, ECI_Goal, No_Goal
from ADCS.controller import MTQ_w_RW_LP
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.helpers.math_helpers import random_n_unit_vec, normalize

from ADCS.helpers.plotting.animate_estimator import animate_attitude
from ADCS.helpers.plotting.plot_estimator import plot_state_comparison
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window
from ADCS.helpers.plotting.plot_controller import plot_control, plot_target_tracking
from ADCS.helpers.plotting.animate_orbit_pyvista import animate_orbit_pyvista


GOAL1_ECI = np.array([1.0, 0.2, 0.1])
GOAL2_ECI = np.array([-0.3, 1.0, 0.15])
GOAL3_ECI = np.array([0.05, -0.25, 1.0])


def test_MTQ_w_RW_LP_multi_goal(
    verbose: bool = False,
    tf: float = 1000,
    dt: float = 10,
    real_orbit: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, List[Orbital_State], np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, List[Orbital_State], np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Multi-goal tracking with gaps (No_Goal) in between goals.

    Schedule (seconds since sim start):
      [0, 300)   -> Goal 1 (ECI)
      [300, 350) -> No_Goal
      [350, 650) -> Goal 2 (ECI)
      [650, 700) -> No_Goal
      [700, end) -> Goal 3 (ECI)
    """
    np.random.seed(37)
    t0 = 0.0
    N = int((tf - t0) / dt)

    # Satellite setup
    rw_h0 = 0.0
    real_sat = create_beavercube2_cubesat(estimated=False)
    real_sat.rw_actuators[0].h = rw_h0

    # Initial conditions
    w0 = random_n_unit_vec(3) * np.random.uniform(1, 2) * np.pi / 180.0
    w0 = np.array([0.0, 0.0, 0.0])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    h0 = np.array([rw_h0])
    x = np.concatenate([w0, q0, h0])

    # Orbit setup (same structure as your original file)
    ephem = Ephemeris()
    t_start = 0.22  # J2000 reference you were using
    start_time = t_start - 1 * TimeConstants.sec2cent
    end_time = t_start + (tf - t0) * TimeConstants.sec2cent

    R = 7000 * np.array([0.0, np.sqrt(2) / 2, np.sqrt(2) / 2])
    V = np.array([8.0, 0.0, 0.0])

    if real_orbit:
        os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
        orb = Orbit(os0=os0, end_time=end_time, dt=dt, use_J2=True, fast=False)
    else:
        os0 = Orbital_State(
            ephem=ephem,
            J2000=start_time,
            R=R,
            V=V,
            B=np.array([0.0, 0.1, 0.0]),
            S=np.array([1e5 + 1.0, 0.0, 0.0]),
            rho=5e-12,
        )
        dur = int((tf - t0) / dt) + 10
        orbs = [os0] * (dur + 10)
        for j in range(dur):
            orbs[j] = os0.copy()
            orbs[j].J2000 = os0.J2000 + j * dt * TimeConstants.sec2cent
        orb = Orbit(orbs)

    # Controller
    controller = MTQ_w_RW_LP(
        est_sat=real_sat,
        p_gain=0.00005,
        d_gain=0.002,
        c_gain=0.001,
        h_target=np.array([0.0, 0.0, 0.0]),
    )

    # Goals (hardcoded ECI vectors)
    g1 = ECI_Goal(normalize(GOAL1_ECI))
    g2 = ECI_Goal(normalize(GOAL2_ECI))
    g3 = ECI_Goal(normalize(GOAL3_ECI))

    sec2cent = TimeConstants.sec2cent

    goals = GoalList(
        {
            t_start: g1,
            t_start + 300 * sec2cent: No_Goal(),
            t_start + 350 * sec2cent: g2,
            t_start + 650 * sec2cent: No_Goal(),
            t_start + 700 * sec2cent: g3,
        }
    )

    # Histories
    time_hist = np.nan * np.zeros(N)
    state_hist = np.nan * np.zeros((N, len(x)))
    os_hist: List[Orbital_State] = []
    sensor_hist = np.nan * np.zeros((N, len(real_sat.sensors + real_sat.rw_actuators)))
    u_hist = np.nan * np.zeros((N, len(real_sat.actuators)))
    boresight_hist = np.nan * np.zeros((N, 3))
    goal_index_hist = np.nan * np.zeros(N)  # 0 = No_Goal, 1/2/3 correspond to goals

    t = t0
    steps = int((tf - t0) / dt)

    for i in tqdm(range(steps), desc="Simulating MTQ_w_RW (multi-goal)"):
        J2000 = t_start + t * sec2cent
        os = orb.get_os(J2000=J2000)

        active_goal: Goal = goals.get_active_goal(J2000)

        sens = real_sat.sensor_readings(x=x, os=os)
        u = controller.find_u(x_hat=x, sens=sens, est_sat=real_sat, os_hat=os, goal=active_goal)

        if verbose:
            print("u:", u)

        time_hist[i] = t
        state_hist[i, :] = x
        os_hist.append(os)
        sensor_hist[i, :] = sens
        u_hist[i, :] = u

        # Reference boresight from the GoalList (so it reflects No_Goal segments too)
        eci_goal_ref, _ = goals.to_ref(t=J2000, os0=os)
        boresight_hist[i, :] = eci_goal_ref

        # Goal index tracking (matches schedule in seconds)
        if t < 300:
            goal_index_hist[i] = 1
        elif t < 350:
            goal_index_hist[i] = 0
        elif t < 650:
            goal_index_hist[i] = 2
        elif t < 700:
            goal_index_hist[i] = 0
        else:
            goal_index_hist[i] = 3

        # Propagate
        t_next = t + dt
        os_next = orb.get_os(t_start + t_next * sec2cent)

        out = solve_ivp(
            fun=real_sat.dynamics_for_solver,
            t_span=(0, dt),
            y0=x,
            method="RK45",
            args=(u, os.copy(), os_next),
            rtol=1e-7,
            atol=1e-7,
        )
        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])

        t = t_next

    return time_hist, state_hist, os_hist, sensor_hist, u_hist, boresight_hist, goal_index_hist


def plot_MTQ_w_RW_LP_multi_goal(
    verbose: bool = False,
    tf: float = 1000,
    dt: float = 10,
    real_orbit: bool = False,
) -> None:
    (
        time_hist,
        state_hist,
        os_hist,
        sensor_hist,
        u_hist,
        boresight_hist,
        goal_index_hist,
    ) = test_MTQ_w_RW_LP_multi_goal(verbose=verbose, tf=tf, dt=dt, real_orbit=real_orbit)

    # Standard plots/animations (unchanged style)
    animate_attitude(time=time_hist, state_hist=state_hist, os_hist=os_hist, boresight_goal_hist=boresight_hist)
    plot_control(time=time_hist, u_hist=u_hist)
    plot_state_comparison(time=time_hist, state_hist=state_hist)

    # Orbit animation: we’re tracking ECI goals (not a Coordinate_Goal), so pass coord_goal=None
    animate_orbit_pyvista(
        time_hist=time_hist,
        state_hist=state_hist,
        os_hist=os_hist,
        boresight_goal_hist=boresight_hist,
        coord_goal=None,
    )

    plot_target_tracking(state_hist=state_hist, boresight_hist=boresight_hist, body_boresight=np.array([0, 1, 0]))

    create_close_all_button_window()


if __name__ == "__main__":
    plot_MTQ_w_RW_LP_multi_goal(verbose=False, tf=1000, dt=2, real_orbit=True)
