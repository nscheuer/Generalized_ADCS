__all__ = ["Plan_and_Track_Exact"]

import numpy as np
from typing import Tuple

from ADCS.CONOPS.goallist import GoalList
from ADCS.controller import Controller
from ADCS.controller.helpers import PlannerSettings, Trajectory, reorder_controls_cpp_to_python, reorder_gains_cpp_to_python
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.orbit import Orbit
from ADCS.controller.helpers.build_csat import build_cpp_satellite
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.orbits.universal_constants import TimeConstants

import trajectory_planner.build.tplaunch as tplaunch
import trajectory_planner.build.pysat as pysat

class Plan_and_Track_Exact(Controller):
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


        self.planner.setquaternionTo3VecMode(2)

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

        return self.active_trajectory.get_control_at(current_time)

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

        self.planner.setVerbosity(verbose)
        # self.planner.setquaternionTo3VecMode(2)
        
        dt_seconds = self.planner_settings.dt_tvlqr
        
        # Calculate N
        N = int(np.ceil(duration / dt_seconds)) + 1
        
        t_end = t_start + (duration * TimeConstants.sec2cent)

        # Propagate Environment
        vecsPy = self._propagate_environment(os_0, t_start, t_end, dt_seconds, N, goals)

        # SANITIZE x_0: Force Float64, Copy, and C-Order
        x_0_clean = np.copy(x_0.astype(np.float64).flatten(), order='C')
        
        bdotOn = self.planner_settings.bdot_on
        if verbose:
            print("=== PYTHON VECTOR_INFO_FORM DEBUG (before trajOpt) ===")
            labels = [
                "t", "R", "V", "B", "S",
                "A (satvec)", "E (ECIvec)", "p (prop)", "rho",
            ]
            for i, (lbl, x) in enumerate(zip(labels, vecsPy)):
                if isinstance(x, np.ndarray):
                    print(
                        f"{i}: {lbl:<12} "
                        f"ndim={x.ndim} "
                        f"shape={x.shape} "
                        f"dtype={x.dtype} "
                        f"C={x.flags['C_CONTIGUOUS']} "
                        f"F={x.flags['F_CONTIGUOUS']}"
                    )
                else:
                    print(f"{i}: {lbl:<12} type={type(x)}")
            print("==============================================")
        # (traj_initial, vecs_dt, costset_initial) = self.planner.prepareForAlilqr(vecsPy,self.planner_settings.dt_tp,t_start, t_end, x_0_clean, bdotOn)
        # (Xset_initial, Uset_initial, Tset, unsure) =  traj_initial


        # traj_initial_py = Trajectory(np.array(Tset), Xset_initial, Uset_initial, [], [])
        # time_hist_initial = (traj_initial_py.times-t_start)*TimeConstants.cent2sec
        # state_hist_initial = traj_initial_py.states.T
        # u_hist_initial = traj_initial_py.controls.T

        # plot_state_comparison(time=time_hist_initial, state_hist=state_hist_initial)
        # plot_control(time=time_hist_initial, u_hist=u_hist_initial)

        # boresight_traj_hist = np.vstack([goals.to_ref(t=J2000, os0=orb.get_os(J2000))[0] for J2000 in traj.times])
        # plot_target_tracking(state_hist=state_hist_traj, boresight_hist=boresight_traj_hist, body_boresight=np.array([0, 0, 1]))
        

        (_, _, _, lqr_opt, _) = self.planner.trajOpt(vecsPy, N, t_start, t_end, x_0_clean, int(bdotOn))
        (Xset, Uset_cpp, Tset, Kset_cpp, Sset, lqr_times) = lqr_opt

        # Reorder controls and gains from C++ ordering (MTQ, RW) to Python actuator ordering
        Uset = reorder_controls_cpp_to_python(Uset_cpp, self.est_sat.actuators)
        Kset = reorder_gains_cpp_to_python(Kset_cpp, self.est_sat.actuators)

        return Trajectory(np.array(lqr_times), Xset, Uset, Kset, Sset)

    def _propagate_environment(self, os_0: Orbital_State, t_start: float, t_end: float,
                           dt_seconds: float, N: int, goals: GoalList) -> Tuple:
        """
        Generates environment vectors in the exact format C++ expects:
        t: (N,)
        r,v,b,s,a,e: (3,N)  (FORTRAN contiguous)
        p,rho: (N,)
        """
        buffer_centuries = 10 * dt_seconds * TimeConstants.sec2cent
        t_end_buffered = t_end + buffer_centuries

        sim_orbit = Orbit(os0=os_0, end_time=t_end_buffered, dt=dt_seconds, use_J2=True, fast=False)
        tp_orbit = sim_orbit.get_range(t_start, t_end, dt_seconds)

        orbit_data_lists = tp_orbit.get_vecs()
        times_arr = np.asarray(tp_orbit.times, dtype=np.float64)

        # -------------------------
        # Clip/pad times to N
        # -------------------------
        curN = times_arr.shape[0]
        if curN > N:
            times_arr = times_arr[:N]
        elif curN < N:
            times_arr = np.pad(times_arr, (0, N - curN), mode="edge")

        # Helper: force (3,N) float64 Fortran-contiguous
        def to_mat3xN(name: str, x) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64)
            x = np.squeeze(x)

            if x.ndim != 2:
                raise ValueError(f"{name} must be 2D, got ndim={x.ndim}, shape={x.shape}")

            # Accept either (3,N) or (N,3)
            if x.shape == (3, N):
                y = x
            elif x.shape == (N, 3):
                y = x.T
            else:
                raise ValueError(f"{name} has unexpected shape {x.shape}; expected (3,{N}) or ({N},3)")

            # Armadillo-friendly memory layout
            return np.asfortranarray(y, dtype=np.float64)

        # Helper: force (N,) float64 contiguous
        def to_vecN(name: str, x) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64).reshape(-1)
            if x.shape[0] == N:
                return np.ascontiguousarray(x, dtype=np.float64)
            if x.shape[0] > N:
                return np.ascontiguousarray(x[:N], dtype=np.float64)
            return np.ascontiguousarray(np.pad(x, (0, N - x.shape[0]), mode="edge"), dtype=np.float64)

        # -------------------------
        # Orbit vectors: expect [R,V,B,S,Rho] from get_vecs()
        # -------------------------
        # Convert lists->arrays before shape logic
        R_raw, V_raw, B_raw, S_raw, Rho_raw = [np.asarray(d) for d in orbit_data_lists]

        # Clip/pad each (handles both (3,curN) and (curN,3))
        def clip_pad_mat(x):
            x = np.asarray(x, dtype=np.float64)
            if x.ndim != 2:
                return x
            if x.shape[0] == 3:
                # (3,curN)
                if x.shape[1] > N:  return x[:, :N]
                if x.shape[1] < N:  return np.pad(x, ((0,0),(0, N-x.shape[1])), mode="edge")
                return x
            if x.shape[1] == 3:
                # (curN,3)
                if x.shape[0] > N:  return x[:N, :]
                if x.shape[0] < N:  return np.pad(x, ((0, N-x.shape[0]), (0,0)), mode="edge")
                return x
            return x

        R = to_mat3xN("R", clip_pad_mat(R_raw))
        V = to_mat3xN("V", clip_pad_mat(V_raw))
        B = to_mat3xN("B", clip_pad_mat(B_raw))
        S = to_mat3xN("S", clip_pad_mat(S_raw))
        rho = to_vecN("Rho", Rho_raw)

        # -------------------------
        # Goals / attitude vectors
        # -------------------------
        goal_vecs_eci = np.zeros((3, N), dtype=np.float64, order="F")
        sat_body_vecs = np.zeros((3, N), dtype=np.float64, order="F")
        prop_vals     = np.zeros(N, dtype=np.float64)

        for i in range(N):
            t = float(times_arr[i])
            os_at_t = sim_orbit.get_os(t)
            g_vec_eci, _w_ref = goals.to_ref(t, os_at_t)
            goal_vecs_eci[:, i] = np.asarray(g_vec_eci, dtype=np.float64).reshape(3)
            sat_body_vecs[:, i] = np.asarray(self.est_sat.boresight, dtype=np.float64).reshape(3)

        A = np.asfortranarray(sat_body_vecs, dtype=np.float64)      # a in C++
        E = np.asfortranarray(goal_vecs_eci, dtype=np.float64)      # e in C++
        p = np.ascontiguousarray(prop_vals.reshape(-1), dtype=np.float64)

        t_c = np.ascontiguousarray(times_arr.reshape(-1), dtype=np.float64)

        return (t_c, R, V, B, S, A, E, p, rho)