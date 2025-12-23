__all__ = ["Plan_and_Track_LQR"]

import numpy as np
from typing import List, Optional, Dict, Tuple

from ADCS.CONOPS.goallist import GoalList
from ADCS.controller import Controller
from ADCS.controller.helpers import PlannerSettings, Trajectory
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.orbit import Orbit
from ADCS.controller.helpers.build_csat import build_cpp_satellite
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.actuators import Actuator
from ADCS.orbits.universal_constants import TimeConstants

import trajectory_planner.build.tplaunch as tplaunch
import trajectory_planner.build.pysat as pysat

class Plan_and_Track_LQR(Controller):
    def __init__(self, est_sat: EstimatedSatellite, planner_settings: PlannerSettings) -> None:
        self.est_sat = est_sat
        self.planner_settings = planner_settings

        self.csat: pysat.Satellite = build_cpp_satellite(est_sat=est_sat, planner_settings=planner_settings)
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
        self.planner.setquaternionTo3VecMode(0)

        self.active_trajectory: Trajectory = None

        self.state_dim = est_sat.state_len
        self.ctrl_dim = est_sat.control_len

    def find_u(self, x_hat: np.ndarray, sens: np.ndarray, est_sat: EstimatedSatellite, os_hat: Orbital_State, goal_vector_eci: np.ndarray | None = None, w_ref: np.ndarray | None = None) -> np.ndarray:
        current_time = os_hat.J2000

        if self.active_trajectory is None:
            raise RuntimeError(f"Plan_and_Track_LQR: No active trajectory set at t={current_time}")
        
        if not self.active_trajectory.is_valid_time(current_time):
            raise RuntimeError(f"Plan_and_Track_LQR: Active trajectory expired or not started. "
                                f"Current: {current_time}, Traj: [{self.active_trajectory.start_time}, {self.active_trajectory.end_time}]")

        return self.active_trajectory.compute_tracking_control(current_time, x_hat)

    def calculate_trajectory(self, 
                             t_start: float, 
                             duration: float, 
                             x_0: np.ndarray, 
                             os_0: Orbital_State, 
                             goals: GoalList, 
                             verbose: bool = False) -> Trajectory:
        """
        Calculates a trajectory.
        
        Args:
            t_start: Absolute J2000 Century timestamp.
            duration: Duration in SECONDS. (Intuitive for users)
            x_0: Initial state vector.
            os_0: Initial orbital state.
            goals: GoalList object.
        """
        if verbose: print(f"Planning traj: Start={t_start:.5f}, Dur={duration}s")
        
        # 1. Standardize Time Steps in Seconds
        dt_seconds = self.planner_settings.dt_tvlqr
        
        # 2. Calculate N using Seconds (Safe from floating point underflow)
        # We ceil to ensure we cover the full duration if it's not a perfect multiple
        N = int(np.ceil(duration / dt_seconds)) + 1
        
        # 3. Calculate End Time in Centuries
        t_end = t_start + (duration * TimeConstants.sec2cent)

        # 4. Propagate Environment
        vecsPy = self._propagate_environment(os_0, t_start, t_end, dt_seconds, N, goals)

        # 5. Run Optimizer
        bdotOn = self.planner_settings.bdot_on
        (_, _, _, lqr_opt, _) = self.planner.trajOpt(vecsPy, N, t_start, t_end, x_0.astype(np.float64), bdotOn)
        (Xset, Uset, Tset, Kset, Sset, lqr_times) = lqr_opt

        return Trajectory(np.array(lqr_times), Xset, Uset, Kset, Sset)

    def set_active_trajectory(self, traj: Trajectory) -> None:
        self.active_trajectory = traj

    def _propagate_environment(self, os_0: Orbital_State, t_start: float, t_end: float, dt_seconds: float, N: int, goals: GoalList) -> Tuple:
        """
        Generates environment vectors.
        
        Args:
            t_start, t_end: J2000 Centuries.
            dt_seconds: Step size in SECONDS.
        """
        # Buffer end time slightly (in centuries) to prevent rounding errors in Orbit class
        buffer_centuries = 10 * dt_seconds * TimeConstants.sec2cent
        t_end_buffered = t_end + buffer_centuries

        # Orbit Class expects dt in Seconds (based on your class definition)
        sim_orbit = Orbit(os0=os_0, end_time=t_end_buffered, dt=dt_seconds, use_J2=True, fast=False)
        
        # get_range also expects dt in Seconds
        tp_orbit = sim_orbit.get_range(t_start, t_end, dt_seconds)

        # --- FIX: ROBUST ARRAY HANDLING ---
        # 1. Get raw list-of-lists
        orbit_data_lists = tp_orbit.get_vecs() # [R, V, B, S, rho]
        
        # 2. Convert to (D, M) numpy arrays safely
        raw_vecs = []
        for data_list in orbit_data_lists:
            arr = np.array(data_list)
            # Ensure shape is (Dimensions, Steps)
            if arr.ndim == 2: arr = arr.T  
            elif arr.ndim == 1: arr = arr.reshape(1, -1)
            
            # Slice to exactly N steps to match C++ requirement
            raw_vecs.append(arr[:, :N])
            
        # Extract the Time array and slice to N
        times_arr = np.array(tp_orbit.times)[:N]
        
        # Verify we have enough data points (Edge case safety)
        actual_N = len(times_arr)
        if actual_N < N:
             # Pad if the orbit propagator came up short (rare but possible with floats)
             pad_len = N - actual_N
             times_arr = np.pad(times_arr, (0, pad_len), 'edge')
             raw_vecs = [np.pad(v, ((0,0), (0, pad_len)), 'edge') for v in raw_vecs]

        # 3. Calculate Goals
        goal_vecs_eci = np.zeros((3, N))
        sat_body_vecs = np.zeros((3, N))
        prop_vals = np.zeros(N)

        for i, t in enumerate(times_arr):
            os_at_t = sim_orbit.get_os(t)
            (g_vec_eci, w_ref) = goals.to_ref(t, os_at_t)
            goal_vecs_eci[:, i] = g_vec_eci
            sat_body_vecs[:, i] = self.est_sat.boresight

        # 4. Package for C++
        t_c = np.copy(times_arr, order='C')
        vecs_c = [np.copy(np.squeeze(k).T, order='C') for k in raw_vecs]
        goal_c = np.copy(goal_vecs_eci.T, order='C')
        sat_body_c = np.copy(sat_body_vecs.T, order='C')
        prop_c = np.copy(prop_vals, order='C')

        return tuple([t_c] + vecs_c + [sat_body_c, goal_c, prop_c])



            

