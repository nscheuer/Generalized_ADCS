import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Union
from tqdm import tqdm
import pytest

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from ADCS.CONOPS.goals import Goal, ECI_Goal, Coordinate_Goal
from ADCS.controller import MTQ_w_RW
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import random_n_unit_vec, normalize

from plotting.animate_estimator import animate_attitude
from plotting.plot_estimator import plot_state_comparison
from plotting.close_all_plots import create_close_all_button_window
from plotting.plot_controller import plot_control, plot_rw_momentum
from plotting.animate_orbit import animate_orbit

def test_mtq_w_rw_align_to_eci(verbose: bool = False, tf: float = 1000, dt: float = 10, real_orbit: bool = False) -> Union[np.ndarray, np.ndarray, List[Orbital_State], np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(1)
    t0 = 0
    N = int((tf-t0)/dt)

    mtm_max_torque = 0.1
    mtqs = [MTQ(axis=j, max_torque=mtm_max_torque) for j in MathConstants.unitvecs]

    rw_max_torque = 4.51
    rw_J = 0.22
    rw_h0 = 1
    rw_hmax = 3.8
    rws = [RW(axis=j, max_torque=rw_max_torque, J=rw_J, h=rw_h0, h_max=rw_hmax) for j in MathConstants.unitvecs]

    acts = mtqs+rws

    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]

    real_sat = Satellite(mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]), actuators=acts, sensors=mtms, boresight=np.array([0, 0, 1]))

    w0 = random_n_unit_vec(3)*np.random.uniform(1, 2)*np.pi/180.0
    w0 = np.array([0, 0, 0])
    q0 = random_n_unit_vec(4)
    q0 = normalize(np.array([1, 0, 0, 0]))
    h0 = np.array([rw_h0, rw_h0, rw_h0])
    x = np.concatenate([w0, q0, h0])

    ephem = Ephemeris()
    start_time = 0.22 - 1*TimeConstants.sec2cent
    end_time = 0.22 + (tf-t0)*TimeConstants.sec2cent
    R = 7000*np.array([0, -np.sqrt(2)/2, np.sqrt(2)/2])
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

    # Controller
    controller = MTQ_w_RW(est_sat=real_sat, p_gain=0.1, d_gain=0.7, c_gain=0.1, h_target=np.array([0, 0, 0]))

    time_hist = np.nan*np.zeros(N)
    state_hist = np.nan*np.zeros((N, 10))
    os_hist: List[Orbital_State] = list()
    sensor_hist: np.ndarray = np.nan*np.zeros((N, 3))
    u_hist = np.nan*np.zeros((N, len(acts)))
    boresight_hist = np.nan*np.zeros((N, 3))

    t = t0
    ind = 0
    steps = int((tf - t0)/dt)

    goal = ECI_Goal(np.array([1, 0, 0]))
    goal = Coordinate_Goal(lat=42.3555, lon=71.0565, alt=0)

    for step in tqdm(range(steps), desc="Simulating MTQ_w_RW"):
        J2000 = 0.22 + t*TimeConstants.sec2cent
        os = orb.get_os(J2000=J2000)

        sens = real_sat.sensor_readings(x=x, os=os)
        u = controller.find_u(x_hat=x, sens=sens, est_sat=real_sat, os_hat=os, goal=goal)

        if verbose:
            print("u: ", u)

        time_hist[ind] = t
        state_hist[ind,:] = x
        os_hist += [os]
        sensor_hist[ind,:] = sens
        u_hist[ind,:] = u
        eci_goal, w_goal = goal.to_ref(x_hat=x, os0=os)
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
    (time_hist, state_hist, os_hist, sensor_hist, u_hist, boresight_hist) = test_mtq_w_rw_align_to_eci(verbose=verbose, tf=tf, dt=dt, real_orbit=real_orbit)

    animate_attitude(time=time_hist, state_hist=state_hist, os_hist=os_hist, boresight_goal_hist=boresight_hist)
    plot_control(time=time_hist, u_hist=u_hist)
    plot_state_comparison(time=time_hist, state_hist=state_hist)
    plot_rw_momentum(time=time_hist, state_hist=state_hist)
    # animate_orbit(time_hist=time_hist, state_hist=state_hist, os_hist=os_hist)
    create_close_all_button_window()

if __name__ == "__main__":
    plot_mtq_w_rw_align_to_eci(verbose=False, tf = 100, dt = 1, real_orbit=True)