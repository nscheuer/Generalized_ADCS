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
print(f"DEBUG: Loading library from: {tplaunch.__file__}")
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

    def set_active_trajectory(self, traj: Trajectory) -> None:
        self.active_trajectory = traj

    def calculate_trajectory(self, 
                             t_start: float, 
                             duration: float, 
                             x_0: np.ndarray, 
                             os_0: Orbital_State, 
                             goals: GoalList, 
                             verbose: bool = False) -> Trajectory:
        
        if verbose: print(f"Planning traj: Start={t_start:.5f}, Dur={duration}s")
        
        dt_seconds = self.planner_settings.dt_tvlqr
        
        # Calculate N
        N = int(np.ceil(duration / dt_seconds)) + 1
        
        t_end = t_start + (duration * TimeConstants.sec2cent)

        # Propagate Environment
        vecsPy = self._propagate_environment(os_0, t_start, t_end, dt_seconds, N, goals)

        # SANITIZE x_0: Force Float64, Copy, and C-Order
        x_0_clean = np.copy(x_0.astype(np.float64).flatten(), order='C')
        
        bdotOn = self.planner_settings.bdot_on
        (_, _, _, lqr_opt, _) = self.planner.trajOpt(vecsPy, N, t_start, t_end, x_0_clean, bdotOn)
        (Xset, Uset, Tset, Kset, Sset, lqr_times) = lqr_opt

        return Trajectory(np.array(lqr_times), Xset, Uset, Kset, Sset)

    def _propagate_environment(self, os_0: Orbital_State, t_start: float, t_end: float, dt_seconds: float, N: int, goals: GoalList) -> Tuple:
        """
        Generates environment vectors and sanitizes them for C++.
        """
        buffer_centuries = 10 * dt_seconds * TimeConstants.sec2cent
        t_end_buffered = t_end + buffer_centuries

        # Orbit Propagation
        sim_orbit = Orbit(os0=os_0, end_time=t_end_buffered, dt=dt_seconds, use_J2=True, fast=False)
        tp_orbit = sim_orbit.get_range(t_start, t_end, dt_seconds)

        # ---------------------------------------------------------
        # 1. EXTRACT RAW DATA
        # ---------------------------------------------------------
        orbit_data_lists = tp_orbit.get_vecs() 
        times_arr = np.array(tp_orbit.times)

        # ---------------------------------------------------------
        # 2. ALIGN LENGTHS (Clip or Pad to N)
        # ---------------------------------------------------------
        current_len = len(times_arr)

        if current_len > N:
            # Slice
            times_arr = times_arr[:N]
            orbit_data_arrays = []
            for d in orbit_data_lists:
                arr = np.array(d)
                if arr.ndim == 2: orbit_data_arrays.append(arr[:, :N])
                else: orbit_data_arrays.append(arr[:N])
                    
        elif current_len < N:
            # Pad
            pad_amt = N - current_len
            times_arr = np.pad(times_arr, (0, pad_amt), 'edge')
            orbit_data_arrays = []
            for d in orbit_data_lists:
                arr = np.array(d)
                if arr.ndim == 2: orbit_data_arrays.append(np.pad(arr, ((0,0), (0, pad_amt)), 'edge'))
                else: orbit_data_arrays.append(np.pad(arr, (0, pad_amt), 'edge'))
        else:
            orbit_data_arrays = [np.array(d) for d in orbit_data_lists]

        # ---------------------------------------------------------
        # 3. CALCULATE GOALS
        # ---------------------------------------------------------
        goal_vecs_eci = np.zeros((3, N))
        sat_body_vecs = np.zeros((3, N))
        prop_vals = np.zeros(N)

        for i in range(N):
            t = times_arr[i]
            os_at_t = sim_orbit.get_os(t)
            (g_vec_eci, w_ref) = goals.to_ref(t, os_at_t)
            goal_vecs_eci[:, i] = g_vec_eci
            sat_body_vecs[:, i] = self.est_sat.boresight

        # ---------------------------------------------------------
        # 4. FINAL SANITIZATION & DIAGNOSTICS
        # ---------------------------------------------------------
        print(f"--- DEBUG: Preparing C++ Data (N={N}) ---")

        def clean_matrix(name, arr):
            arr = np.squeeze(arr)
            # Transpose if (3, N) -> (N, 3)
            if arr.ndim == 2 and arr.shape[0] == 3 and arr.shape[1] == N:
                arr = arr.T
            
            clean_arr = np.copy(arr.astype(np.float64), order='C')
            print(f"  Matrix '{name}': Shape {clean_arr.shape}, First {clean_arr.flatten()[0]:.8f}")
            return clean_arr

        def clean_vector(name, arr):
            # Flatten ensures (N, 1) becomes (N,)
            clean_arr = np.copy(arr.astype(np.float64).flatten(), order='C')
            print(f"  Vector '{name}': Shape {clean_arr.shape}, First {clean_arr[0]:.8f}")
            return clean_arr

        # Time (Vector)
        t_c = clean_vector("Time", times_arr)

        # Orbit Vectors [R, V, B, S, rho]
        names = ["R", "V", "B", "S", "Rho"]
        vecs_c = []
        for i, vec in enumerate(orbit_data_arrays):
            # If after squeeze it is 1D, it's a vector (like Rho)
            if np.squeeze(vec).ndim == 1:
                vecs_c.append(clean_vector(names[i], vec))
            else:
                vecs_c.append(clean_matrix(names[i], vec))

        # Goal Vectors
        sat_body_c = clean_matrix("SatBody", sat_body_vecs)
        goal_c = clean_matrix("Goal", goal_vecs_eci)
        prop_c = clean_vector("Prop", prop_vals)
        
        print("--- End Debug ---")

        return tuple([t_c] + vecs_c + [sat_body_c, goal_c, prop_c])