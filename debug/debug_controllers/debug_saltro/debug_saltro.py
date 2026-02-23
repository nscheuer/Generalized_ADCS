import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Union
from tqdm import tqdm
import pytest

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
from ADCS.CONOPS.goals import Goal, ECI_Goal, Coordinate_Goal, No_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.neo_planner.NEO_planner_settings import PlannerSettings
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

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../../../.."))
saltro_path = os.path.join(parent_dir, "SALTRO", "build")
sys.path.append(saltro_path)

import saltro_py

if __name__ == "__main__":
    np.random.seed(1)
    t0 = 0
    tf = 1000
    dt = 1.0
    N = int((tf-t0)/dt)

    mtm_max_torque = 0.1
    mtqs = [MTQ(axis=j, max_torque=mtm_max_torque) for j in MathConstants.unitvecs]
    rw_max_torque = 7*0.001
    rw_J = 0.001
    rw_h0 = 5*0.001
    rw_hmax = 16.2*0.001
    rws = [RW(axis=j, max_torque=rw_max_torque, J=rw_J, h=rw_h0, h_max=rw_hmax) for j in MathConstants.unitvecs]

    acts = mtqs+rws

    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]

    real_sat = Satellite(mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]), actuators=acts, sensors=mtms, boresight=np.array([0, 0, 1]))

    w0 = random_n_unit_vec(3)*np.random.uniform(1, 2)*np.pi/180.0
    w0 = np.array([0.01, 0, 0])
    q0 = random_n_unit_vec(4)
    q0 = normalize(np.array([1, 0, 0, 0]))
    h0 = np.array([rw_h0, rw_h0, rw_h0])
    x = np.concatenate([w0, q0, h0])

    ephem = Ephemeris()
    start_time = 0.22 - 1*TimeConstants.sec2cent
    end_time = 0.22 + (tf-t0)*TimeConstants.sec2cent
    R = 7000*np.array([0, np.sqrt(2)/2, np.sqrt(2)/2])
    V = np.array([8, 0, 0])
    os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
    orb = Orbit(os0=os0, end_time=end_time, dt=dt, use_J2=True, fast=False)
    
    # Planner Settings
    planner_settings = PlannerSettings(est_sat=real_sat)
    print("Created Planner Settings")
    
    # Convert to C++ object
    cpp_settings = planner_settings.to_cpp()
    print("Created C++ Planner Settings")
    
    # Convert Satellite to C++ object
    cpp_satellite = 
    
    