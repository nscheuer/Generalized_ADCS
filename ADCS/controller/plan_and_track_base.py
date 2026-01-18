"""
Base class for Plan and Track controllers.

This module provides the common functionality shared by all Plan and Track
controller variants (LQR, LQR with disturbance estimation, Exact).
"""
from __future__ import annotations

__all__ = ["PlanAndTrackBase"]

import numpy as np
from typing import Tuple, Optional
from numpy.typing import NDArray

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


class PlanAndTrackBase(Controller):
    """
    Base class for Plan and Track controllers.

    Provides common functionality for trajectory-based controllers including:
    - Planner initialization
    - Environment propagation
    - Trajectory management

    Subclasses must implement:
    - find_u(): Control computation method
    - calculate_trajectory(): Trajectory generation method

    Attributes:
        est_sat: Estimated satellite model
        planner_settings: Configuration for the trajectory planner
        csat: C++ satellite model for the planner
        planner: C++ ALTRO planner instance
        active_trajectory: Currently active trajectory for tracking
        state_dim: Dimension of state vector
        ctrl_dim: Dimension of control vector
    """

    est_sat: EstimatedSatellite
    planner_settings: PlannerSettings
    csat: pysat.Satellite
    planner: tplaunch.Planner
    active_trajectory: Optional[Trajectory]
    state_dim: int
    ctrl_dim: int

    def _init_planner(
        self,
        est_sat: EstimatedSatellite,
        planner_settings: PlannerSettings,
        tracking_lqr_formulation: int,
        quat_to_3vec_mode: int = 0
    ) -> None:
        """
        Common planner initialization.

        Args:
            est_sat: Estimated satellite model with actuators and sensors
            planner_settings: Configuration for the ALTRO trajectory planner
            tracking_lqr_formulation: LQR tracking formulation type:
                0 = Standard TVLQR
                2 = findKwDist (with disturbance estimation)
            quat_to_3vec_mode: Quaternion to 3-vector conversion mode:
                0 = Use reduced 3-parameter attitude representation
                2 = Use exact 4-parameter quaternion
        """
        self.est_sat = est_sat
        self.planner_settings = planner_settings

        self.csat = build_cpp_satellite(est_sat=est_sat, planner_settings=planner_settings)
        self.planner = tplaunch.Planner(
            self.csat,
            planner_settings.systemSettings(),
            planner_settings.mainAlilqrSettings(),
            planner_settings.secondAlilqrSettings(),
            planner_settings.initTrajSettings(),
            planner_settings.optMainCostSettings(),
            planner_settings.optSecondCostSettings(),
            planner_settings.optTVLQRCostSettings(tracking_LQR_formulation=tracking_lqr_formulation)
        )
        self.planner.setquaternionTo3VecMode(quat_to_3vec_mode)

        self.active_trajectory = None

        self.state_dim = est_sat.state_len
        self.ctrl_dim = est_sat.control_len

    def set_active_trajectory(self, traj: Trajectory) -> None:
        """Set the active trajectory for tracking."""
        self.active_trajectory = traj

    def _propagate_environment(
        self,
        os_0: Orbital_State,
        t_start: float,
        t_end: float,
        dt_seconds: float,
        N: int,
        goals: GoalList
    ) -> Tuple[NDArray[np.float64], ...]:
        """
        Propagate environment vectors for the C++ planner.

        Generates environment vectors in the exact format C++ expects:
        - t: (N,) time array
        - r,v,b,s,a,e: (3,N) position, velocity, B-field, sun, attitude, goal vectors (FORTRAN contiguous)
        - p,rho: (N,) propellant and density arrays

        Args:
            os_0: Initial orbital state
            t_start: Start time in J2000 centuries
            t_end: End time in J2000 centuries
            dt_seconds: Time step in seconds
            N: Number of time points
            goals: Goal list for attitude reference

        Returns:
            Tuple of (t, R, V, B, S, A, E, p, rho) arrays
        """
        buffer_centuries = 10 * dt_seconds * TimeConstants.sec2cent
        t_end_buffered = t_end + buffer_centuries

        sim_orbit = Orbit(os0=os_0, end_time=t_end_buffered, dt=dt_seconds, use_J2=True, fast=True)
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

    def _calculate_trajectory_common(
        self,
        t_start: float,
        duration: float,
        x_0: np.ndarray,
        os_0: Orbital_State,
        goals: GoalList,
        verbose: bool = False
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Common trajectory calculation logic.

        Handles environment propagation, planner call, and control/gain reordering.

        Args:
            t_start: Start time in J2000 centuries
            duration: Duration in seconds
            x_0: Initial state vector
            os_0: Initial orbital state
            goals: Goal list for attitude reference
            verbose: Whether to print debug information

        Returns:
            Tuple of (lqr_times, Xset, Uset, Kset, Sset)
        """
        if verbose:
            print(f"Planning traj: Start={t_start:.5f}, Dur={duration}s")

        self.planner.setVerbosity(verbose)

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

        (_, _, _, lqr_opt, _) = self.planner.trajOpt(vecsPy, N, t_start, t_end, x_0_clean, int(bdotOn))
        (Xset, Uset_cpp, Tset, Kset_cpp, Sset, lqr_times) = lqr_opt

        # Reorder controls and gains from C++ ordering (MTQ, RW) to Python actuator ordering
        Uset = reorder_controls_cpp_to_python(Uset_cpp, self.est_sat.actuators)
        Kset = reorder_gains_cpp_to_python(Kset_cpp, self.est_sat.actuators)

        return (np.array(lqr_times), Xset, Uset, Kset, Sset)
