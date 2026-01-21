import sys
import os as os_pack
import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Union
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

sys.path.append(os_pack.path.abspath(os_pack.path.join(__file__, "../../..")))
from ADCS.CONOPS.goals import Goal, ECI_Goal, Coordinate_Goal, No_Goal
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
from ADCS.satellite_hardware.actuators import RW, Noise, Bias

from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat

from ADCS.helpers.plotting.animate_estimator import animate_attitude
from ADCS.helpers.plotting.plot_estimator import plot_state_comparison
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window
from ADCS.helpers.plotting.plot_controller import plot_control, plot_rw_momentum, plot_target_tracking
from ADCS.helpers.plotting.animate_orbit import animate_orbit
from ADCS.helpers.plotting.animate_orbit_pyvista import animate_orbit_pyvista

if __name__ == "__main__":
    real_sat = create_beavercube2_cubesat()
    real_sat.disturbances = []
    real_sat.rw_actuators[0].noise = Noise()
    real_sat.rw_actuators[0].bias = Bias()
    real_sat.rw_actuators[0].h_meas_noise = Noise()

    u = np.array([-1.3742516980563680e-04, 1.5796918982339877e-04, 4.0225230972646745e-05, 1.6363576550741210e-08])
    w = np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    h = np.array([1.0e-04])

    R = 7000*np.array([0, np.sqrt(2)/2, np.sqrt(2)/2])
    V = np.array([8, 0, 0])
    ephem = Ephemeris()
    os = Orbital_State(ephem=ephem, J2000=0.22-1*TimeConstants.sec2cent, R=R, V=V, B=np.array([1.0184463954847115e-06, -3.3784390385208494e-05, -9.1243334918349954e-06]), S=np.array([1e5+1, 0, 0]), rho=5e-12)

    xdot = real_sat.dynamics_core(x=np.concatenate([w, q, h]), u=u, orbital_state=os)


