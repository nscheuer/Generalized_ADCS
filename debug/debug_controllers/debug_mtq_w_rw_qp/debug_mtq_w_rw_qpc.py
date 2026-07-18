import sys
import os
import numpy as np
from ADCS.state import State
from scipy.integrate import solve_ivp
from typing import List, Union
from tqdm import tqdm
import pytest

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from ADCS.CONOPS.goals import Goal, ECI_Goal, Coordinate_Goal
from ADCS.controller.mtq_w_rw_QPC import MTQ_w_RW_QPC
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

def test_MTQ_w_RW_QP_align(verbose: bool = False, tf: float = 1000, dt: float = 10, real_orbit: bool = False) -> Union[np.ndarray, np.ndarray, List[Orbital_State], np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(1)
    t0 = 0
    N = int((tf-t0)/dt)

    mtq_max_torque = 0.4
    mtqs = [MTQ(axis=j, max_torque=mtq_max_torque) for j in MathConstants.unitvecs]

    rw_max_torque = 7*0.001
    rw_J = 0.001
    rw_h0 = 5*0.001
    rw_hmax = 16.2*0.001
    rws = [RW(axis=j, max_torque=rw_max_torque, J=rw_J, h=rw_h0, h_max=rw_hmax) for j in MathConstants.unitvecs]
    rws.pop()
    rws.pop()

    acts = mtqs+rws

    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]

    real_sat = Satellite(mass=1.2, J_0=np.diagflat([0.022, 0.022, 0.004]), actuators=acts, sensors=mtms, boresight=np.array([0, 0, 1]))

    w0 = random_n_unit_vec(3)*np.random.uniform(1, 2)*np.pi/180.0
    w0 = np.array([0, 0, 0])
    q0 = random_n_unit_vec(4)
    q0 = normalize(np.array([1, 0, 0, 0]))
    h0 = np.array([rw_h0]*len(rws))
    x = State(w=w0, q=q0, h=h0)

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

    # Controller
    controller = MTQ_w_RW_QPC(est_sat=real_sat, p_gain=0.00005, d_gain=0.001, c_gain=0.001, h_target=np.zeros(3))

    time_hist = np.nan*np.zeros(N)
    state_hist = np.nan*np.zeros((N, len(x)))
    os_hist: List[Orbital_State] = list()
    sensor_hist: np.ndarray = np.nan*np.zeros((N, len(real_sat.sensors + real_sat.rw_actuators)))
    u_hist = np.nan*np.zeros((N, len(acts)))
    boresight_hist = np.nan*np.zeros((N, 4))

    t = t0
    ind = 0
    steps = int((tf - t0)/dt)

    # goal = ECI_Goal(np.array([1, 0, 0]))
    goal = Coordinate_Goal(lat=9, lon=-70, alt=0)

    for step in tqdm(range(steps), desc="Simulating MTQ_w_RW"):
        J2000 = 0.22 + t*TimeConstants.sec2cent
        os = orb.get_os(J2000=J2000)

        sens = real_sat.sensor_readings(x=x, os=os)
        u,_ = controller.find_u(x_hat=x, sens=sens, est_sat=real_sat, os_hat=os, goal=goal)

        if verbose:
            print("u: ", u)

        time_hist[ind] = t
        state_hist[ind,:] = x.as_array()
        os_hist += [os]
        sensor_hist[ind,:] = sens
        u_hist[ind,:] = u
        eci_goal, w_goal = goal.to_ref(os0=os)
        boresight_hist[ind, :] = eci_goal

        ind += 1
        t += dt
        prev_os = os.copy()
        os = orb.get_os(0.22+(t-t0)*TimeConstants.sec2cent)

        out = solve_ivp(fun=real_sat.dynamics_for_solver, t_span=(0, dt), y0=x.as_array(), method="RK45", args=(u, prev_os, os), rtol=1e-7, atol=1e-7)
        x = State.from_array(out.y[:, -1]).normalized()

    return time_hist, state_hist, os_hist, sensor_hist, u_hist, boresight_hist


def plot_MTQ_w_RW_QP_align(verbose: bool = False, tf: float = 1000, dt: float = 10, real_orbit: bool = False) -> None:
    (time_hist, state_hist, os_hist, sensor_hist, u_hist, boresight_hist) = test_MTQ_w_RW_QP_align(verbose=verbose, tf=tf, dt=dt, real_orbit=real_orbit)

    animate_attitude(time=time_hist, state_hist=state_hist, os_hist=os_hist, boresight_goal_hist=boresight_hist)
    plot_control(time=time_hist, u_hist=u_hist)
    plot_state_comparison(time=time_hist, state_hist=state_hist)
    plot_rw_momentum(time=time_hist, state_hist=state_hist)
    goal = Coordinate_Goal(lat=9, lon=-70, alt=0)
    #animate_orbit_pyvista(time_hist=time_hist, state_hist=state_hist, os_hist=os_hist, boresight_goal_hist=boresight_hist, coord_goal=goal)
    plot_target_tracking(state_hist=state_hist, boresight_hist=boresight_hist, body_boresight=np.array([0, 0, 1]))
    #animate_orbit(time_hist=time_hist, state_hist=state_hist, os_hist=os_hist, boresight_goal_hist=boresight_hist, coord_goal=goal)
    create_close_all_button_window()

if __name__ == "__main__":
    plot_MTQ_w_RW_QP_align(verbose=False, tf = 500, dt = 2, real_orbit=True)