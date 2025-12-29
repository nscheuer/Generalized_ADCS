__all__ = ["Plan_and_Track_LQR"]

import numpy as np
from typing import List, Optional, Dict, Tuple
from scipy.integrate import solve_ivp

from ADCS.CONOPS.goallist import GoalList
from ADCS.controller import Controller
from ADCS.controller.helpers import PlannerSettings, Trajectory
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.orbit import Orbit
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.actuators import Actuator, RW, MTQ
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.helpers.size_helpers import check_numpy_size

# import trajectory_planner.build.tplaunch as tplaunch
# print(f"DEBUG: Loading library from: {tplaunch.__file__}")
# import trajectory_planner.build.pysat as pysat

class Plan_and_Track_LQR(Controller):
    def __init__(self, est_sat: EstimatedSatellite, planner_settings: PlannerSettings) -> None:
        self.est_sat = est_sat
        self.planner_settings = planner_settings

        # self.csat: pysat.Satellite = build_cpp_satellite(est_sat=est_sat, planner_settings=planner_settings)
        # self.planner = tplaunch.Planner(
        #     self.csat,
        #     planner_settings.systemSettings(),
        #     planner_settings.mainAlilqrSettings(),
        #     planner_settings.secondAlilqrSettings(),
        #     planner_settings.initTrajSettings(),
        #     planner_settings.optMainCostSettings(),
        #     planner_settings.optSecondCostSettings(),
        #     # 0 implies standard LQR tracking formulation
        #     planner_settings.optTVLQRCostSettings(tracking_LQR_formulation=0) 
        # )
        # self.planner.setquaternionTo3VecMode(0)

        self.active_trajectory: Trajectory = None

        self.dt = planner_settings.systemSettings()[1]
        self.dt_tvlqr = planner_settings.systemSettings()[2]

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
        (_, _, _, lqr_opt, _) = self._trajOpt(vecsPy, N, t_start, t_end, x_0_clean, bdotOn)
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

        return tuple([t_c] + vecs_c + [sat_body_c, goal_c, prop_c])
    

    def _trajOpt(self, vecsPy: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], N: int, t_start: float, t_end: float, x_0_clean: np.ndarray, bdotOn: bool) -> Tuple:
        initial_guess = self._trajOptBefore(vecsPy, self.dt, t_start, t_end, x_0_clean, bdotOn)

    def _trajOptBefore(self, vecsPy: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], dt: float, t_start: float, t_end: float, x_0_clean: np.ndarray, bdotOn: bool) -> Tuple:
        resampled_vecs = self._resample_vecs(vecsPy, dt, t_start, t_end)
        times = resampled_vecs[0]
        traj_length = len(times)

        goal_vecs = resampled_vecs[7] # Goal Vector
        sat_vecs = resampled_vecs[6] # Boresight

        x0 = np.copy(x_0_clean)
        q_norm = np.linalg.norm(x0[3:7])
        if q_norm > 1e-9:
            x0[3:7] /= q_norm

        num_mtq = len([a for a in self.est_sat.actuators if isinstance(a, MTQ)])
        num_rw = len([a for a in self.est_sat.actuators if isinstance(a, RW)])

        traj_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] = None # (X, U, t, TQ)

        if bdotOn == False or num_mtq < 3:
            # Initialize with random trajectory
            if self.planner_settings.verbosity:
                print("bdotOn is false (or invalid), generating random initial trajectory!")

            u_max_arr = self.planner_settings.umax.reshape(-1, 1)
            rand_scale = 1000.0

            U_rand = (u_max_arr * np.random.randn(self.ctrl_dim, traj_length)) / rand_scale

            traj_data = self._generateInitialTrajectory(dt, x0, U_rand, resampled_vecs)
        else:
            if self.planner_settings.verbosity:
                print("Generating initial trajectory using bdot")
            
            traj_data, _ = self._bdot(x0, dt, traj_length, resampled_vecs)

        X_final = traj_data[0]
        U_final = traj_data[1]

        if np.any(np.isnan(X_final)) or np.any(np.isnan(U_final)):
             print(f"WARNING: NaNs detected in initial trajectory (bdotOn={bdotOn})")
             # Fallback or debug print could go here

        return (traj_data, resampled_vecs, self.planner_settings.optMainCostSettings())   
    
    def _generateInitialTrajectory(self, dt: float, x0: np.ndarray, Uset: np.ndarray, vecs: Tuple) -> Tuple:
        """
        Propagates the satellite state given an initial state x0 and a sequence of controls Uset.
        Mimics C++ OldPlanner::generateInitialTrajectory.
        
        Args:
            dt: Time step in seconds.
            x0: Initial state vector (dim_state,).
            Uset: Control inputs (dim_ctrl, N).
            vecs: Resampled environment vectors tuple.
            
        Returns:
            Tuple (Xset, Uset, t, TQset) where:
              - Xset: State history (dim_state, N)
              - Uset: The input control history
              - t: Time vector
              - TQset: Torque history (3, N) - Placeholder for now as py dynamics might not return it directly
        """
        # Unpack Environment Vectors (already resampled)
        # 0:t, 1:R, 2:V, 3:B, 4:S, 5:Rho, 6:SatBody, 7:Goal, 8:Prop
        times = vecs[0]
        R_set = vecs[1]
        V_set = vecs[2]
        B_set = vecs[3]
        S_set = vecs[4]
        # Prop_set = vecs[8] # Not strictly used in simple dynamics usually, but avail if needed
        
        N = Uset.shape[1]
        state_dim = x0.shape[0]
        
        # Initialize output arrays
        Xset = np.zeros((state_dim, N))
        TQset = np.zeros((3, N)) # Placeholder: C++ rk4z returns torque, solve_ivp doesn't usually
        
        # Copy and normalize initial state
        xk = np.copy(x0)
        # Normalize quaternion (indices 3:7)
        q_norm = np.linalg.norm(xk[3:7])
        if q_norm > 1e-9:
            xk[3:7] /= q_norm
            
        Xset[:, 0] = xk

        for k in range(1, N):
            # 1. Get Control for previous step
            uk = Uset[:, k-1]
            
            class MiniOS:
                def __init__(self, R, V, B, S):
                    self.R = R
                    self.V = V
                    self.B_eci = B
                    self.Sun_eci = S
            
            # k-1 index for start of interval
            prev_os = MiniOS(
                R=R_set[k-1, :], 
                V=V_set[k-1, :], 
                B=B_set[k-1, :], 
                S=S_set[k-1, :]
            )
            
            # k index for end of interval
            next_os = MiniOS(
                R=R_set[k, :], 
                V=V_set[k, :], 
                B=B_set[k, :], 
                S=S_set[k, :]
            )
            
            sol = solve_ivp(
                fun=self.est_sat.dynamics_for_solver, 
                t_span=(0, dt), 
                y0=xk, 
                method='RK45', # or 'RK45'
                args=(uk, prev_os, next_os),
                rtol=1e-7, 
                atol=1e-7
            )
            
            # Extract result
            xk = sol.y[:, -1]
            
            # Normalize quaternion
            norm_q = np.linalg.norm(xk[3:7])
            if norm_q > 1e-9:
                xk[3:7] /= norm_q
                
            Xset[:, k] = xk
            
        return (Xset, Uset, times, TQset)

    def _resample_vecs(self, vecsPy: Tuple, dt_seconds: float, t_start: float, t_end: float) -> Tuple:
        t_old = vecsPy[0]

        dt_cent = dt_seconds * TimeConstants.sec2cent

        t_new = np.arange(t_start, t_end, dt_cent)

        if len(t_new) == 0 or (t_end - t_new[-1]) > 1e-12:
            t_new = np.append(t_new, t_end)

        def interp_linear(y):
            """Linear interpolation for vectors (Nx3) or scalars (N,)"""
            # Ensure input is (N, D) or (N,)
            if y.ndim == 2 and y.shape[0] != len(t_old) and y.shape[1] == len(t_old):
                y = y.T  # Transpose if (3, N) to (N, 3) for interpolation
            
            if y.ndim == 1:
                return np.interp(t_new, t_old, y)
            else:
                # Interpolate each column (dimension) separately
                y_new = np.zeros((len(t_new), y.shape[1]))
                for i in range(y.shape[1]):
                    y_new[:, i] = np.interp(t_new, t_old, y[:, i])
                return y_new
            
        def interp_nearest(y):
            """Nearest neighbor interpolation (for propagation status)"""
            # Find indices in t_old closest to t_new
            idx = np.searchsorted(t_old, t_new, side="left")
            idx = np.clip(idx, 0, len(t_old) - 1)
            
            # Check previous index to see if it's closer
            prev_idx = np.clip(idx - 1, 0, len(t_old) - 1)
            mask = np.abs(t_new - t_old[idx]) > np.abs(t_new - t_old[prev_idx])
            idx[mask] = prev_idx[mask]
            
            return y[idx]
        
        R_new = interp_linear(vecsPy[1])
        V_new = interp_linear(vecsPy[2])
        B_new = interp_linear(vecsPy[3])
        S_new = interp_linear(vecsPy[4])
        Rho_new = interp_linear(vecsPy[5])
        
        SatBody_new = interp_linear(vecsPy[6])
        Goal_new = interp_linear(vecsPy[7])
        
        Prop_new = interp_nearest(vecsPy[8])
        
        return (t_new, R_new, V_new, B_new, S_new, Rho_new, SatBody_new, Goal_new, Prop_new)