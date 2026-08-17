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

from ADCS.helpers.plotting.animate_estimator import animate_attitude
from ADCS.helpers.plotting.plot_estimator import plot_state_comparison
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window
from ADCS.helpers.plotting.plot_controller import plot_control, plot_rw_momentum
from ADCS.helpers.plotting.animate_orbit import animate_orbit
from ADCS.helpers.plotting.animate_orbit_pyvista import animate_orbit_pyvista

def test_goal_coordinate_fixed_os(dt, tf, t0):
    N = int((tf-t0)/dt)

    R = 10000*np.array([0, -np.sqrt(2)/2, np.sqrt(2)/2])
    V = np.array([8, 0, 0])
    ephem = Ephemeris()
    os0 = Orbital_State(ephem=ephem, J2000=0.22-1*TimeConstants.sec2cent, R=R, V=V, B=np.array([0, 0.1, 0]), S=np.array([1e5+1, 0, 0]), rho=5e-12)
    dur = int((tf-t0)/dt)+10
    orbs = [os0]*(dur+10)
    for j in range(dur):
        orbs[j] = os0.copy()
        orbs[j].J2000 = os0.J2000 + j*dt*TimeConstants.sec2cent
    orb = Orbit(orbs)

    real_sat = Satellite(mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]), boresight=np.array([0, 0, 1]))
    
    w0 = np.array([0, 0, 0])
    q0 = normalize(np.array([1, 0, 0, 0]))
    x = State(w=w0, q=q0)

    time_hist = np.nan*np.zeros(N)
    state_hist: List[State] = []
    os_hist: List[Orbital_State] = list()
    boresight_goal_hist = np.nan*np.zeros((N, 3))

    t = t0
    ind = 0
    steps = int((tf - t0)/dt)

    goal = Coordinate_Goal(lat=42.3555, lon=71.0565, alt=0)

    for step in tqdm(range(steps), desc="Simulating rotation of Earth"):
        J2000 = 0.22 + t*TimeConstants.sec2cent
        os = orb.get_os(J2000=J2000)

        sens = real_sat.sensor_readings(x=x, os=os)
        u = np.array([])        

        time_hist[ind] = t
        state_hist.append(x.copy())
        os_hist += [os]
        eci_goal, w_goal = goal.to_ref(os0=os)
        boresight_goal_hist[ind, :] = eci_goal

        ind += 1
        t += dt
        prev_os = os.copy()
        os = orb.get_os(0.22+(t-t0)*TimeConstants.sec2cent)

        out = solve_ivp(fun=real_sat.dynamics_for_solver, t_span=(0, dt), y0=x.as_array(), method="RK45", args=(u, prev_os, os), rtol=1e-7, atol=1e-7)
        x = State.from_array(out.y[:, -1])
        x = x.normalized()

    return time_hist, state_hist, os_hist, boresight_goal_hist


def test_goal_coordinate_real_os(dt, tf, t0):
    N = int((tf-t0)/dt)

    R = 7000*np.array([0, np.sqrt(2)/2, np.sqrt(2)/2])
    V = np.array([8, 0, 0])
    ephem = Ephemeris()
    start_time = 0.22 - 1*TimeConstants.sec2cent
    end_time = 0.22 + (tf-t0)*TimeConstants.sec2cent
    os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
    orb = Orbit(os0=os0, end_time=end_time, dt=dt, zonal_J=2, fast=False)

    real_sat = Satellite(mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]), boresight=np.array([0, 0, 1]))
    
    w0 = np.array([0, 0, 0])
    q0 = normalize(np.array([1, 0, 0, 0]))
    x = State(w=w0, q=q0)

    time_hist = np.nan*np.zeros(N)
    state_hist: List[State] = []
    os_hist: List[Orbital_State] = list()
    boresight_goal_hist = np.nan*np.zeros((N, 3))

    t = t0
    ind = 0
    steps = int((tf - t0)/dt)

    goal = Coordinate_Goal(lat=47.3769, lon=8.5417, alt=0)

    for step in tqdm(range(steps), desc="Simulating rotation of Earth"):
        J2000 = 0.22 + t*TimeConstants.sec2cent
        os = orb.get_os(J2000=J2000)

        sens = real_sat.sensor_readings(x=x, os=os)
        u = np.array([])        

        time_hist[ind] = t
        state_hist.append(x.copy())
        os_hist += [os]
        eci_goal, w_goal = goal.to_ref(os0=os)
        boresight_goal_hist[ind, :] = eci_goal

        ind += 1
        t += dt
        prev_os = os.copy()
        os = orb.get_os(0.22+(t-t0)*TimeConstants.sec2cent)

        out = solve_ivp(fun=real_sat.dynamics_for_solver, t_span=(0, dt), y0=x.as_array(), method="RK45", args=(u, prev_os, os), rtol=1e-7, atol=1e-7)
        x = State.from_array(out.y[:, -1])
        x = x.normalized()

    return time_hist, state_hist, os_hist, boresight_goal_hist


def plot_test_goal_coordinate_fixed_os(dt, tf, t0):
    (time_hist, state_hist, os_hist, boresight_goal_hist) = test_goal_coordinate_fixed_os(dt, tf, t0)

    animate_attitude(time=time_hist, state_hist=state_hist, os_hist=os_hist, boresight_goal_hist=boresight_goal_hist)
    animate_orbit(time_hist=time_hist, state_hist=state_hist, os_hist=os_hist)
    create_close_all_button_window()


def plot_test_goal_coordinate_real_os(dt, tf, t0):
    (time_hist, state_hist, os_hist, boresight_goal_hist) = test_goal_coordinate_real_os(dt, tf, t0)

    animate_attitude(time=time_hist, state_hist=state_hist, os_hist=os_hist, boresight_goal_hist=boresight_goal_hist)
    goal = Coordinate_Goal(lat=47.3769, lon=8.5417, alt=0)
    animate_orbit_pyvista(time_hist=time_hist, state_hist=state_hist, os_hist=os_hist, boresight_goal_hist=boresight_goal_hist, coord_goal=goal)
    animate_orbit(time_hist=time_hist, state_hist=state_hist, os_hist=os_hist, boresight_goal_hist=boresight_goal_hist, coord_goal=goal)
    create_close_all_button_window()

if __name__ == "__main__":
    plot_test_goal_coordinate_real_os(dt=100, tf=9000, t0=0)
