"""
Plan and Track TinyMPC Controller (C++ Implementation).

Uses the C++ TinyMPC ADMM solver via Python bindings for real-time
trajectory tracking with explicit constraint handling.

This controller provides:
- Fast tracking via C++ ADMM solver (~1-10ms per solve)
- Explicit handling of actuator constraints
- Warm-start support for efficient consecutive solves
- Proper MPC formulation (not just saturated LQR)
"""
from __future__ import annotations

__all__ = ["Plan_and_Track_TinyMPC_Cpp", "Plan_and_Track_TinyMPC_Cpp_Python"]

import numpy as np
from typing import Optional, Tuple, Callable
from numpy.typing import NDArray

from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_base import PlanAndTrackBase
from ADCS.controller.helpers import (
    PlannerSettings, Trajectory, PythonALILQRv2,
    OptimizationResult, IterationData, LivePlannerViz
)
from ADCS.controller.helpers.tinympc_settings import TinyMPCSettings
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.helpers.math_helpers import rot_mat

# Import C++ TinyMPC bindings
try:
    import trajectory_planner.build.pytinympc as pytinympc
    _HAS_CPP_TINYMPC = True
except ImportError:
    _HAS_CPP_TINYMPC = False
    pytinympc = None


class Plan_and_Track_TinyMPC_Cpp(PlanAndTrackBase):
    """
    Trajectory-following controller using ALTRO planning and C++ TinyMPC tracking.

    This controller uses:
    - ALTRO (via C++) for trajectory planning (Plan phase)
    - TinyMPC (via C++) for real-time tracking with constraints (Track phase)

    The C++ implementation provides:
    - Fast ADMM solver (~1-10ms per solve vs ~50ms+ for Python scipy)
    - Efficient memory management via Armadillo
    - Warm-start support for consecutive solves

    Compared to TVLQR tracking:
    - TinyMPC respects actuator bounds explicitly
    - Better disturbance rejection via receding horizon optimization
    - More computation but still real-time capable

    Note:
        Requires building the C++ pytinympc module. If not available,
        use Plan_and_Track_MPC as a fallback (Python ADMM).
    """

    def __init__(
        self,
        est_sat: EstimatedSatellite,
        planner_settings: PlannerSettings,
        tinympc_settings: Optional[TinyMPCSettings] = None
    ) -> None:
        """
        Initialize Plan and Track TinyMPC controller (C++ version).

        Args:
            est_sat: Estimated satellite model with actuators and sensors
            planner_settings: Configuration for the ALTRO trajectory planner
            tinympc_settings: Configuration for TinyMPC solver (uses defaults if None)

        Raises:
            ImportError: If C++ pytinympc module is not available
        """
        if not _HAS_CPP_TINYMPC:
            raise ImportError(
                "C++ TinyMPC bindings not available. "
                "Build the trajectory_planner with pytinympc module or use Plan_and_Track_MPC instead."
            )

        # Initialize ALTRO planner
        self._init_planner(est_sat, planner_settings, tracking_lqr_formulation=0)

        # Store settings
        self.tinympc_settings = tinympc_settings if tinympc_settings is not None else TinyMPCSettings()
        self.sat = est_sat

        # Create C++ TinyMPC controller
        self._mpc = pytinympc.TinyMPCController(
            self.csat,
            self.tinympc_settings.to_cpp_tuple()
        )

        # Set cost matrices from planner settings
        Q, R, Qf = self._build_cost_matrices()
        self._mpc.setCostMatrices(
            np.asfortranarray(Q, dtype=np.float64),
            np.asfortranarray(R, dtype=np.float64),
            np.asfortranarray(Qf, dtype=np.float64)
        )

        # Track state
        self._last_solve_time_ms = 0.0

    def _build_cost_matrices(self) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Build Q, R, Qf matrices from planner cost settings.

        Maps the CostWeights from planner_settings to REDUCED ERROR STATE cost
        matrices suitable for MPC tracking.

        Error state dimension: n_err = 6 + n_rw
        - Angular velocity error: indices 0:3
        - Attitude error (3D linearized): indices 3:6
        - RW momentum error: indices 6:6+n_rw

        Returns:
            (Q, R, Qf): Cost matrices for error state
        """
        cost = self.planner_settings.cost_tvlqr
        n = self.state_dim
        n_err = n - 1  # Reduced dimension for quaternion linearization
        n_rw = n - 7
        m = self.ctrl_dim

        # State tracking cost Q (n_err x n_err)
        Q = np.zeros((n_err, n_err), dtype=np.float64)
        # Angular velocity weights (indices 0:3)
        for i in range(3):
            Q[i, i] = cost.ang_vel
        # Attitude error weights (indices 3:6)
        for i in range(3, 6):
            Q[i, i] = cost.angle
        # RW momentum weights (indices 6:6+n_rw)
        for i in range(6, n_err):
            Q[i, i] = cost.ang_vel * 0.1

        # Control cost R (m x m)
        R = cost.control_mult * np.eye(m, dtype=np.float64)

        # Terminal cost Qf - larger for stability
        Qf = Q.copy()
        for i in range(3, 6):
            Qf[i, i] = cost.angle_N

        return Q, R, Qf

    def _build_dynamics_info(self, os: Orbital_State) -> tuple:
        """Build dynamics info tuple for C++ TinyMPC."""
        # (B_field, R_eci, prop_torq_on, V_eci, sun_vec, dist_on)
        return (
            np.asfortranarray(os.B, dtype=np.float64),       # B_field
            np.asfortranarray(os.R, dtype=np.float64),       # R (ECI position)
            0,                                                # prop_torq_on
            np.asfortranarray(os.V, dtype=np.float64),       # V (ECI velocity)
            np.asfortranarray(os.S, dtype=np.float64),       # sun_vec (S, not sun)
            0                                                 # dist_on
        )

    def find_u(
        self,
        x_hat: NDArray[np.float64],
        sens: NDArray[np.float64],
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        B_body: Optional[NDArray[np.float64]] = None,
        **kwargs
    ) -> NDArray[np.float64]:
        """
        Compute control using C++ TinyMPC solver.

        Args:
            x_hat: Estimated state [w, q, h_rw]
            sens: Sensor measurements (unused)
            est_sat: Estimated satellite model
            os_hat: Estimated orbital state
            B_body: B-field in body frame (computed if not provided)

        Returns:
            Control vector [m_mtq, tau_rw]
        """
        if self.active_trajectory is None:
            return np.zeros(self.ctrl_dim)

        current_time = os_hat.J2000
        if not self.active_trajectory.is_valid_time(current_time):
            raise RuntimeError(
                f"Trajectory expired. Current: {current_time}, "
                f"Traj: [{self.active_trajectory.start_time}, {self.active_trajectory.end_time}]"
            )

        # Get B-field in body frame
        if B_body is None:
            R = rot_mat(x_hat[3:7])
            B_body = R.T @ os_hat.B

        # Build dynamics info
        dynamics_info = self._build_dynamics_info(os_hat)

        # Solve TinyMPC
        result = self._mpc.solve(
            np.asfortranarray(x_hat, dtype=np.float64),
            current_time,
            np.asfortranarray(B_body, dtype=np.float64),
            np.asfortranarray(os_hat.S, dtype=np.float64),
            dynamics_info
        )

        # Unpack result: (u_opt, X_pred, U_pred, iterations, solve_time_ms, converged, tracking_error)
        u_opt = result[0]
        self._last_solve_time_ms = result[4]

        return u_opt

    def calculate_trajectory(
        self,
        t_start: float,
        duration: float,
        x_0: np.ndarray,
        os_0: Orbital_State,
        goals: GoalList,
        verbose: bool = False,
    ) -> Trajectory:
        """
        Calculate trajectory using C++ ALTRO planner.

        After planning, loads the trajectory into the C++ TinyMPC for tracking.
        """
        lqr_times, Xset, Uset, Kset, Sset = self._calculate_trajectory_common(
            t_start, duration, x_0, os_0, goals, verbose
        )

        trajectory = Trajectory(lqr_times, Xset, Uset, Kset, Sset)

        # Load into C++ TinyMPC for tracking
        dt = self.planner_settings.dt_tvlqr
        
        # If use_trajectory_gains is enabled, pass the K gains from ALTRO
        # K format: (m * n_err, N) where n_err = state_dim - 1 (7 for 1 RW)
        if self.tinympc_settings.use_altro_gains and Kset is not None and Kset.size > 0:
            K_flat = np.asfortranarray(Kset, dtype=np.float64)
        else:
            K_flat = np.array([])  # TinyMPC will compute its own Riccati gains
            
        self._mpc.loadReferenceFromALTRO(
            np.asfortranarray(Xset, dtype=np.float64),
            np.asfortranarray(Uset, dtype=np.float64),
            K_flat,
            np.asfortranarray(lqr_times, dtype=np.float64),
            dt
        )

        return trajectory

    @property
    def last_solve_time_ms(self) -> float:
        """Last TinyMPC solve time in milliseconds."""
        return self._last_solve_time_ms


class Plan_and_Track_TinyMPC_Cpp_Python(PlanAndTrackBase):
    """
    Plan-and-Track with Python ALILQR planning and C++ TinyMPC tracking.

    Same tracking as Plan_and_Track_TinyMPC_Cpp but uses Python planner
    for live visualization of trajectory optimization.
    """

    def __init__(
        self,
        est_sat: EstimatedSatellite,
        planner_settings: PlannerSettings,
        tinympc_settings: Optional[TinyMPCSettings] = None
    ) -> None:
        """Initialize with Python planner and C++ TinyMPC tracker."""
        if not _HAS_CPP_TINYMPC:
            raise ImportError(
                "C++ TinyMPC bindings not available. "
                "Build the trajectory_planner with pytinympc module."
            )

        self._init_planner(est_sat, planner_settings, tracking_lqr_formulation=0)
        self.tinympc_settings = tinympc_settings if tinympc_settings is not None else TinyMPCSettings()
        self.sat = est_sat

        # Create C++ TinyMPC controller
        self._mpc = pytinympc.TinyMPCController(
            self.csat,
            self.tinympc_settings.to_cpp_tuple()
        )

        # Set cost matrices
        Q, R, Qf = self._build_cost_matrices_internal()
        self._mpc.setCostMatrices(
            np.asfortranarray(Q, dtype=np.float64),
            np.asfortranarray(R, dtype=np.float64),
            np.asfortranarray(Qf, dtype=np.float64)
        )

        self._last_solve_time_ms = 0.0

    def _build_cost_matrices_internal(self) -> Tuple[NDArray, NDArray, NDArray]:
        """Build Q, R, Qf matrices."""
        cost = self.planner_settings.cost_tvlqr
        n = self.state_dim
        n_err = n - 1
        n_rw = n - 7
        m = self.ctrl_dim

        Q = np.zeros((n_err, n_err), dtype=np.float64)
        for i in range(3):
            Q[i, i] = cost.ang_vel
        for i in range(3, 6):
            Q[i, i] = cost.angle
        for i in range(6, n_err):
            Q[i, i] = cost.ang_vel * 0.1

        R = cost.control_mult * np.eye(m, dtype=np.float64)

        Qf = Q.copy()
        for i in range(3, 6):
            Qf[i, i] = cost.angle_N

        return Q, R, Qf

    def _build_dynamics_info(self, os: Orbital_State) -> tuple:
        """Build dynamics info tuple for C++ TinyMPC."""
        return (
            np.asfortranarray(os.B, dtype=np.float64),
            np.asfortranarray(os.R, dtype=np.float64),
            0,
            np.asfortranarray(os.V, dtype=np.float64),
            np.asfortranarray(os.S, dtype=np.float64),
            0
        )

    def find_u(
        self,
        x_hat: NDArray[np.float64],
        sens: NDArray[np.float64],
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        B_body: Optional[NDArray[np.float64]] = None,
        **kwargs
    ) -> NDArray[np.float64]:
        """Compute control using C++ TinyMPC solver."""
        if self.active_trajectory is None:
            return np.zeros(self.ctrl_dim)

        current_time = os_hat.J2000
        if not self.active_trajectory.is_valid_time(current_time):
            raise RuntimeError("Trajectory expired")

        if B_body is None:
            R = rot_mat(x_hat[3:7])
            B_body = R.T @ os_hat.B

        dynamics_info = self._build_dynamics_info(os_hat)

        result = self._mpc.solve(
            np.asfortranarray(x_hat, dtype=np.float64),
            current_time,
            np.asfortranarray(B_body, dtype=np.float64),
            np.asfortranarray(os_hat.S, dtype=np.float64),
            dynamics_info
        )

        u_opt = result[0]
        self._last_solve_time_ms = result[4]
        return u_opt

    def calculate_trajectory(
        self,
        t_start: float,
        duration: float,
        x_0: np.ndarray,
        os_0: Orbital_State,
        goals: GoalList,
        verbose: bool = False,
        visualize: bool = False,
        viz_save_path: Optional[str] = None,
        iteration_callback: Optional[Callable[[IterationData], None]] = None,
    ) -> Optional[Trajectory]:
        """
        Calculate trajectory using Python ALILQR with optional visualization.
        Then loads into C++ TinyMPC for tracking.
        """
        py_planner = PythonALILQRv2(self.sat, os_0.orb, self.planner_settings)

        # Set up visualization
        viz = None
        if visualize:
            viz = LivePlannerViz(self.sat, os_0.orb, goals, t_start, duration)

        def combined_callback(data: IterationData):
            if viz:
                viz.update(data)
            if iteration_callback:
                iteration_callback(data)

        # Two-pass optimization
        result1 = py_planner.optimize(
            x_0, t_start, duration, goals,
            pass_num=1, verbose=verbose,
            iteration_callback=combined_callback if (viz or iteration_callback) else None
        )

        if result1 is None or not result1.success:
            if verbose:
                print("Pass 1 failed")
            return None

        result2 = py_planner.optimize(
            x_0, t_start, duration, goals,
            pass_num=2, verbose=verbose,
            X_warm=result1.X, U_warm=result1.U,
            iteration_callback=combined_callback if (viz or iteration_callback) else None
        )

        result = result2 if (result2 and result2.success) else result1

        if viz and viz_save_path:
            viz.save(viz_save_path)

        N = result.U.shape[1]
        dt = self.planner_settings.dt_tvlqr
        times = t_start + np.arange(N + 1) * dt * TimeConstants.sec2cent

        trajectory = Trajectory(times, result.X, result.U, result.K, result.S)

        # Load into C++ TinyMPC for tracking
        # Note: K_ref is optional, TinyMPC computes its own Riccati gains
        self._mpc.loadReferenceFromALTRO(
            np.asfortranarray(result.X, dtype=np.float64),
            np.asfortranarray(result.U, dtype=np.float64),
            np.array([]),  # Skip K gains
            np.asfortranarray(times, dtype=np.float64),
            dt
        )

        return trajectory

    @property
    def last_solve_time_ms(self) -> float:
        """Last TinyMPC solve time in milliseconds."""
        return self._last_solve_time_ms
