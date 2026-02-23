__all__ = ["SALTRO"]

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import itertools
import warnings

from ADCS.CONOPS.goallist import GoalList
from ADCS.controller import Controller
from ADCS.controller.helpers.quaternion_math import vector_alignment_error
from ADCS.controller.helpers import Trajectory
from ADCS.controller.neo_planner.NEO_planner_settings import PlannerSettings
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.sensors import MTM
from ADCS.helpers.math_helpers import rot_mat, skewsym, limit

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
saltro_path = os.path.join(parent_dir, "SALTRO", "build")
sys.path.append(saltro_path)

import saltro_py

class SALTRO(Controller):
    def __init__(self, est_sat: EstimatedSatellite, planner_settings: PlannerSettings):
        self.est_sat = est_sat
        self.planner_settings = planner_settings

    def calculate_trajectory(self, goallist: GoalList, x0: np.ndarray) -> Trajectory:
        lqr_times, Xset, Uset, Kset, Sset = saltro_py.trajOpt(self.est_sat, self.planner_settings, goallist, x0)
        return Trajectory(lqr_times, Xset, Uset, Kset, Sset)