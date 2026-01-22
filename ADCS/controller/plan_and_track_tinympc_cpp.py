"""
Plan and Track TinyMPC Controller (C++ Implementation).

Uses the C++ TinyMPC ADMM solver via Python bindings for real-time
trajectory tracking with automatic re-planning support.

This controller provides:
- Fast tracking via C++ ADMM solver (~1-10ms per solve)
- Explicit handling of actuator constraints
- Warm-start support for efficient consecutive solves
- Threshold-based re-planning triggers
"""
from __future__ import annotations

__all__ = ["Plan_and_Track_TinyMPC_Cpp"]

import numpy as np
from typing import Optional, Tuple
from numpy.typing import NDArray

from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_base import PlanAndTrackBase
from ADCS.controller.helpers import PlannerSettings, Trajectory
from ADCS.controller.helpers.tinympc_settings import TinyMPCSettings
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.helpers.math_helpers import quat_diff, quat_to_vec3, normalize

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
    - Fast ADMM solver (~1-10ms per solve vs ~10-50ms for Python)
    - Efficient memory management via Armadillo
    - Warm-start support for consecutive solves

    Compared to TVLQR tracking:
    - TinyMPC respects actuator bounds explicitly
    - Better disturbance rejection via receding horizon optimization
    - More computation but still real-time capable

    Attributes:
        tinympc_settings: Configuration for TinyMPC solver
        active_trajectory: Current ALTRO trajectory being tracked

    Note:
        Requires building the C++ pytinympc module. If not available,
        use Plan_and_Track_TinyMPC_Py as a fallback.
    """

    tinympc_settings: TinyMPCSettings
    _mpc: "pytinympc.TinyMPCController"
    _last_replan_time: float

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
                "Build the trajectory_planner with pytinympc module or use Plan_and_Track_TinyMPC_Py instead."
            )

        # Initialize ALTRO planner (standard TVLQR formulation=0)
        self._init_planner(est_sat, planner_settings, tracking_lqr_formulation=0)

        # Store TinyMPC settings
        self.tinympc_settings = tinympc_settings if tinympc_settings is not None else TinyMPCSettings()

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

        # Re-planning state
        self._last_replan_time = -np.inf

    def _build_cost_matrices(self) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Build Q, R, Qf matrices from planner cost settings.

        Maps the CostWeights from planner_settings to REDUCED ERROR STATE cost
        matrices suitable for MPC tracking.

        Error state dimension: n_err = 6 + n_rw = n - 1
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
        # Attitude error weights (indices 3:6) - 3D linearized, not 4D quaternion
        for i in range(3, 6):
            Q[i, i] = cost.angle
        # RW momentum weights (indices 6:6+n_rw) - typically smaller
        for i in range(6, n_err):
            Q[i, i] = cost.ang_vel * 0.1

        # Control cost R (m x m)
        R = cost.control_mult * np.eye(m, dtype=np.float64)

        # Terminal cost Qf (n_err x n_err)
        Qf = np.zeros((n_err, n_err), dtype=np.float64)
        for i in range(3):
            Qf[i, i] = cost.ang_vel_N
        for i in range(3, 6):
            Qf[i, i] = cost.angle_N
        for i in range(6, n_err):
            Qf[i, i] = cost.ang_vel_N * 0.1

        return Q, R, Qf

    def _check_replan_needed(
        self,
        x_current: NDArray[np.float64],
        current_time: float
    ) -> bool:
        """
        Check if re-planning should be triggered based on tracking error.

        Args:
            x_current: Current state
            current_time: Current time (J2000 centuries)

        Returns:
            True if re-planning is recommended
        """
        if not self.tinympc_settings.replan_enabled:
            return False

        # Check minimum interval since last replan
        # Convert J2000 centuries to seconds for comparison
        time_since_replan = (current_time - self._last_replan_time) * 3155760000  # centuries to seconds
        if time_since_replan < self.tinympc_settings.replan_min_interval:
            return False

        if self.active_trajectory is None:
            return True

        if not self.active_trajectory.is_valid_time(current_time):
            return True

        # Get reference state
        x_ref = self.active_trajectory.get_state_at(current_time)

        # Compute attitude error (geodesic distance)
        q_curr = normalize(x_current[3:7])
        q_ref = normalize(x_ref[3:7])
        q_err = quat_diff(q_curr, q_ref)
        # Geodesic angle = 2 * arcsin(|vec(q_err)|)
        vec_err = quat_to_vec3(q_err)
        vec_norm = np.linalg.norm(vec_err)
        attitude_error = 2.0 * np.arcsin(min(vec_norm, 1.0))

        # Compute angular velocity error
        w_err = np.linalg.norm(x_current[0:3] - x_ref[0:3])

        # Check thresholds
        if attitude_error > self.tinympc_settings.replan_attitude_threshold:
            return True
        if w_err > self.tinympc_settings.replan_angvel_threshold:
            return True

        return False

    def find_u(
        self,
        x_hat: NDArray[np.float64],
        sens: NDArray[np.float64],
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        goal_vector_eci: Optional[NDArray[np.float64]] = None,
        w_ref: Optional[NDArray[np.float64]] = None,
        clip: bool = True
    ) -> NDArray[np.float64]:
        """
        Compute control using C++ TinyMPC tracking.

        Args:
            x_hat: Estimated state vector [w(3), q(4), h(n_rw)]
            sens: Sensor measurements (unused)
            est_sat: Estimated satellite model (unused, dynamics from C++)
            os_hat: Estimated orbital state
            goal_vector_eci: Goal vector in ECI (unused, from trajectory)
            w_ref: Reference angular velocity (unused, from trajectory)
            clip: Clip control to hardware limits (default True)

        Returns:
            Optimal control vector (m,)
        """
        current_time = os_hat.J2000

        # Check trajectory validity
        if self.active_trajectory is None:
            raise RuntimeError(f"TinyMPC_Cpp: No active trajectory at t={current_time}")

        if not self.active_trajectory.is_valid_time(current_time):
            raise RuntimeError(
                f"TinyMPC_Cpp: Trajectory expired at t={current_time}. "
                f"Valid range: [{self.active_trajectory.start_time}, {self.active_trajectory.end_time}]"
            )

        # Build dynamics info tuple (same format as ALTRO)
        B_field = np.ascontiguousarray(os_hat.B, dtype=np.float64)
        sun_vec = np.ascontiguousarray(os_hat.S, dtype=np.float64)
        R_eci = np.ascontiguousarray(os_hat.R, dtype=np.float64)
        V_eci = np.ascontiguousarray(os_hat.V, dtype=np.float64)
        rho = float(os_hat.rho) if hasattr(os_hat, 'rho') and os_hat.rho is not None else 0.0

        dynamics_info = (
            B_field,                                    # B-field
            R_eci,                                      # Position
            int(self.planner_settings.plan_for_prop),  # Propellant torque on
            V_eci,                                      # Velocity
            sun_vec,                                    # Sun vector
            int(self.planner_settings.plan_for_aero or self.planner_settings.plan_for_srp),  # Disturbances on
            rho                                         # Atmospheric density
        )

        # Solve TinyMPC
        x_current = np.ascontiguousarray(x_hat, dtype=np.float64)

        result = self._mpc.solve(
            x_current,
            current_time,
            B_field,
            sun_vec,
            dynamics_info
        )

        # Unpack result: (u_opt, X_pred, U_pred, iterations, solve_time_ms, converged, tracking_error)
        u_opt, X_pred, U_pred, iterations, solve_time, converged, tracking_error = result

        if self.tinympc_settings.verbose >= 1:
            status = "converged" if converged else "max_iter"
            print(f"TinyMPC_Cpp: {iterations} iters, {solve_time:.2f}ms, "
                  f"{status}, err={tracking_error:.4f}")

        return self.clip_control(np.asarray(u_opt), clip=clip)

    def calculate_trajectory(
        self,
        t_start: float,
        duration: float,
        x_0: np.ndarray,
        os_0: Orbital_State,
        goals: GoalList,
        verbose: bool = False,
        vecsPy_precomputed: tuple = None,
        N_precomputed: int = None,
        t_end_precomputed: float = None
    ) -> Trajectory:
        """
        Calculate an optimal trajectory using ALTRO and load into C++ TinyMPC.

        This calls the ALTRO planner to generate an optimal reference trajectory,
        then loads it into the C++ TinyMPC solver for tracking.

        Args:
            t_start: Start time in J2000 centuries
            duration: Duration in seconds
            x_0: Initial state vector
            os_0: Initial orbital state
            goals: Goal list for attitude reference
            verbose: Whether to print debug information
            vecsPy_precomputed: Optional pre-computed environment vectors
            N_precomputed: Number of timesteps (with precomputed vecs)
            t_end_precomputed: End time (with precomputed vecs)

        Returns:
            Trajectory object (also loaded into internal C++ TinyMPC solver)
        """
        # Call base class trajectory computation (ALTRO)
        lqr_times, Xset, Uset, Kset, Sset = self._calculate_trajectory_common(
            t_start, duration, x_0, os_0, goals, verbose,
            vecsPy_precomputed, N_precomputed, t_end_precomputed
        )

        traj = Trajectory(lqr_times, Xset, Uset, Kset, Sset)

        # Load reference into C++ TinyMPC solver
        dt = self.planner_settings.dt_tvlqr
        self._mpc.loadReferenceFromALTRO(
            np.asfortranarray(Xset, dtype=np.float64),
            np.asfortranarray(Uset, dtype=np.float64),
            np.asfortranarray(Kset, dtype=np.float64),
            np.ascontiguousarray(lqr_times, dtype=np.float64),
            dt
        )

        # Reset internal state and update replan tracking
        self._mpc.reset()
        self._last_replan_time = t_start

        return traj

    def needs_replan(self, x_current: NDArray[np.float64], os: Orbital_State) -> bool:
        """
        Check if re-planning is recommended.

        This is a convenience method that can be called by the user to decide
        whether to trigger a new trajectory calculation.

        Args:
            x_current: Current state
            os: Current orbital state

        Returns:
            True if re-planning is recommended
        """
        return self._check_replan_needed(x_current, os.J2000)
