__all__ = ["Plan_and_Track_LQR"]

import numpy as np
from typing import List, Optional, Dict, Tuple

from ADCS.controller import Controller
from ADCS.controller.helpers import PlannerSettings
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.orbit import Orbit
from ADCS.controller.helpers.build_csat import build_cpp_satellite
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.actuators import Actuator

import trajectory_planner.build.tplaunch as tplaunch
import trajectory_planner.build.pysat as pysat

class Plan_and_Track_LQR(Controller):
    def __init__(self, est_sat: EstimatedSatellite, planner_settings: PlannerSettings) -> None:
        self.est_sat = est_sat
        self.planner_settings = planner_settings

        csat: pysat.Satellite = build_cpp_satellite(est_sat=est_sat, planner_settings=planner_settings)
        self.planner = tplaunch.Planner(
            self.csat,
            planner_settings.systemSettings(),
            planner_settings.mainAlilqrSettings(),
            planner_settings.secondAlilqrSettings(),
            planner_settings.initTrajSettings(),
            planner_settings.optMainCostSettings(),
            planner_settings.optSecondCostSettings(),
            # 0 implies standard LQR tracking formulation
            planner_settings.optTVLQRCostSettings(tracking_LQR_formulation=0) 
        )

        self.active_trajectory: Optional[Dict] = None
        self.traj_end_time = -1.0

        self.state_dim = est_sat.state_len
        self.ctrl_dim = est_sat.control_len

    def find_u(self, x_hat: np.ndarray, sens: np.ndarray, est_sat: EstimatedSatellite, os_hat: Orbital_State, goal_vector_eci: np.ndarray | None = None, w_ref: np.ndarray | None = None) -> np.ndarray:
        current_time = os_hat.J2000

        # Replanning Logic in case no plan or near end of plan:
        if self.active_trajectory is None or current_time >= (self.traj_end_time - self.planner_settings.traj_overlap * self.planner_settings.dt_tvlqr):
            self._replan(x_hat, os_hat, current_time)

        times = self.active_trajectory['t']

    def _replan(self, x_0: np.ndarray, os_0: Orbital_State, t_start: float) -> None:
        dt = self.planner_settings.dt_tvlqr
        N = self.planner_settings.tvlqr_len
        t_end = t_start + N * dt

        # Precalculate environment
        vecs_tuple = self._propagate_environment(os_0, t_start, t_end, dt, N)

    def _propagate_environment(self, os_0: Orbital_State, t_start: float, t_end: float, dt: float, N: int) -> Tuple:
        times = np.linspace(t_start, t_end, N)
        R_eci = np.zeros((3, N))
        V_eci = np.zeros((3, N))
        B_eci = np.zeros((3, N))
        S_eci = np.zeros((3, N))
        Rho = np.zeros((1, N))

        current_os = os_0
        current_orbit = Orbit(os0=current_os, end_time=t_end, dt=dt, use_J2=True)
        
        for i, t in enumerate(times):
            if i > 0:
                current_os = current_orbit.get_os(J2000=t)
            
            R_eci[:, i] = current_os.R
            V_eci[:, i] = current_os.V
            B_eci[:, i] = current_os.B
            S_eci[:, i] = current_os.S
            Rho[0, i] = current_os.rho


            

