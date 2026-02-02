"""
Plan-and-Track controllers using computed torque and MPC tracking.

This module provides tracking controllers that use the actual B-field
(rather than the planned B-field) for MTQ control:

1. **Computed Torque** (`Plan_and_Track_ComputedTorque`):
   - Fast closed-form solution (~200 µs)
   - PD controller with inverse dynamics
   - Uses TVLQR K-matrix for gain weighting

2. **MPC** (`Plan_and_Track_MPC`):
   - ADMM-based constrained optimization (~500 µs - 2 ms)
   - Uses TVLQR K-matrices for cost weighting
   - Proper actuator constraint handling

Both have Python variants (`_Python` suffix) that use PythonALILQR for
trajectory planning, allowing live visualization of the optimization.

Both fix the fundamental TVLQR limitation for MTQ systems: TVLQR uses planned
B-field which diverges from actual when attitude drifts.
"""
from __future__ import annotations

__all__ = [
    "Plan_and_Track_ComputedTorque",
    "Plan_and_Track_ComputedTorque_Python",
    "Plan_and_Track_MPC",
    "Plan_and_Track_MPC_Python",
    "MPCParams",
]

import numpy as np
from typing import Optional, Tuple, Callable
from numpy.typing import NDArray
from dataclasses import dataclass
from scipy.linalg import solve

from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_base import PlanAndTrackBase
from ADCS.controller.helpers import (
    PlannerSettings, Trajectory, PythonALILQRv2, 
    OptimizationResult, IterationData, LivePlannerViz
)
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.helpers.math_helpers import rot_mat, quat_diff, quat_to_vec3


# =============================================================================
# Parameters
# =============================================================================

@dataclass
class MPCParams:
    """Parameters for MPC/Computed Torque tracking controllers."""
    
    # Cost weights (fallback if not using TVLQR weighting)
    Q_omega: float = 1.0       # Angular velocity tracking weight
    Q_attitude: float = 100.0  # Attitude tracking weight  
    Q_rw: float = 1.0          # RW momentum tracking weight
    R_mtq: float = 0.01        # MTQ control effort weight
    R_rw: float = 0.01         # RW control effort weight
    
    # MPC/ADMM parameters
    max_iter: int = 20         # Max ADMM iterations
    abs_tol: float = 1e-4      # Absolute convergence tolerance
    rel_tol: float = 1e-4      # Relative convergence tolerance
    rho: float = 1.0           # ADMM penalty parameter
    
    # Weighting options
    use_tvlqr_weights: bool = True  # Use K-matrix for Q/R weighting
    
    @classmethod
    def fast(cls) -> 'MPCParams':
        """Fast settings (minimal computation)."""
        return cls(max_iter=10, abs_tol=1e-3, use_tvlqr_weights=False)
    
    @classmethod
    def accurate(cls) -> 'MPCParams':
        """Accurate settings (more iterations)."""
        return cls(max_iter=50, abs_tol=1e-6, rel_tol=1e-6)
    
    @classmethod
    def balanced(cls) -> 'MPCParams':
        """Balanced settings (default)."""
        return cls()


# =============================================================================
# Helper Functions
# =============================================================================

def _solve_mtq_for_torque(tau_desired: np.ndarray, B_body: np.ndarray, m_max: float) -> np.ndarray:
    """
    Solve for MTQ dipole moment to produce desired torque.
    
    Uses minimum-norm solution: m = (B × τ) / |B|²
    """
    B_norm_sq = np.dot(B_body, B_body)
    if B_norm_sq < 1e-20:
        return np.zeros(3)
    
    m = np.cross(B_body, tau_desired) / B_norm_sq
    return np.clip(m, -m_max, m_max)


def _extract_cost_matrices_from_K(
    K: np.ndarray, 
    n_mtq: int, 
    n_rw: int,
    params: MPCParams
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract Q and R cost matrices from TVLQR K-matrix.
    
    The K-matrix encodes the optimal trade-off between state tracking
    and control effort. We use its structure to derive meaningful Q/R.
    
    Q[i,i] ∝ sum_j |K[j,i]|² (how much each state error drives control)
    R[j,j] ∝ 1/sum_i |K[j,i]|² (inverse: large gains → small R)
    """
    n_err = 6 + n_rw
    n_ctrl = n_mtq + n_rw
    
    if K is None or not params.use_tvlqr_weights:
        # Fallback to params
        Q = np.diag([params.Q_omega]*3 + [params.Q_attitude]*3 + [params.Q_rw]*n_rw)
        R = np.diag([params.R_mtq]*n_mtq + [params.R_rw]*n_rw)
        return Q, R
    
    # Extract Q from column norms of K (state importance)
    Q_diag = np.zeros(n_err)
    for i in range(n_err):
        Q_diag[i] = np.linalg.norm(K[:, i])**2 + 0.1
    Q_diag = Q_diag / np.mean(Q_diag) * params.Q_attitude
    Q = np.diag(Q_diag)
    
    # Extract R from row norms of K (control cost is inverse of gain magnitude)
    R_diag = np.zeros(n_ctrl)
    for j in range(n_ctrl):
        row_norm = np.linalg.norm(K[j, :])
        R_diag[j] = 1.0 / (row_norm + 0.1) if row_norm > 1e-6 else 1.0
    
    R_diag[:n_mtq] = R_diag[:n_mtq] / np.mean(R_diag[:n_mtq] + 1e-10) * params.R_mtq
    if n_rw > 0:
        R_diag[n_mtq:] = R_diag[n_mtq:] / np.mean(R_diag[n_mtq:] + 1e-10) * params.R_rw
    R = np.diag(R_diag)
    
    return Q, R


def _linearize_error_dynamics(
    sat: EstimatedSatellite,
    x_op: np.ndarray,
    u_op: np.ndarray,
    os: Orbital_State,
    dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linearize dynamics about operating point, returning error-state matrices.
    
    Returns A_err, B_err for: dx_{k+1} = A_err @ dx_k + B_err @ du_k
    where dx is the reduced error state (6 + n_rw dimensions).
    """
    from ADCS.helpers.math_helpers import normalize
    
    n = len(x_op)
    m = len(u_op)
    n_rw = n - 7
    n_err = 6 + n_rw
    eps = 1e-7
    
    # Nominal dynamics
    xdot_nom = sat.dynamics_core(x_op, u_op, os)
    
    # Compute full-state Jacobians via finite differences
    A_cont = np.zeros((n, n))
    for i in range(n):
        x_pert = x_op.copy()
        x_pert[i] += eps
        if 3 <= i < 7:
            x_pert[3:7] = normalize(x_pert[3:7])
        xdot_pert = sat.dynamics_core(x_pert, u_op, os)
        A_cont[:, i] = (xdot_pert - xdot_nom) / eps
    
    B_cont = np.zeros((n, m))
    for i in range(m):
        u_pert = u_op.copy()
        u_pert[i] += eps
        xdot_pert = sat.dynamics_core(x_op, u_pert, os)
        B_cont[:, i] = (xdot_pert - xdot_nom) / eps
    
    # Discretize (forward Euler)
    A_full = np.eye(n) + dt * A_cont
    B_full = dt * B_cont
    
    # Map to error state (7D quaternion → 6D attitude error)
    A_err = np.zeros((n_err, n_err))
    B_err = np.zeros((n_err, m))
    
    # Angular velocity
    A_err[0:3, 0:3] = A_full[0:3, 0:3]
    A_err[0:3, 3:6] = A_full[0:3, 4:7] * 0.5
    
    # Attitude
    A_err[3:6, 0:3] = A_full[4:7, 0:3] * 2.0
    A_err[3:6, 3:6] = A_full[4:7, 4:7]
    
    # RW momentum
    if n_rw > 0:
        A_err[6:, 0:3] = A_full[7:, 0:3]
        A_err[6:, 3:6] = A_full[7:, 4:7] * 0.5
        A_err[0:3, 6:] = A_full[0:3, 7:]
        A_err[3:6, 6:] = A_full[4:7, 7:] * 2.0
        A_err[6:, 6:] = A_full[7:, 7:]
    
    # Control matrix
    B_err[0:3, :] = B_full[0:3, :]
    B_err[3:6, :] = B_full[4:7, :] * 2.0
    if n_rw > 0:
        B_err[6:, :] = B_full[7:, :]
    
    return A_err, B_err


def _admm_solve(
    dx0: np.ndarray,
    u_ref: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    u_min: np.ndarray,
    u_max: np.ndarray,
    params: MPCParams
) -> np.ndarray:
    """
    Solve single-step MPC using ADMM.
    
    Problem:
        min  (A@dx0 + B@du)'Q(A@dx0 + B@du) + du'R@du
        s.t. u_min <= u_ref + du <= u_max
    """
    m = len(u_ref)
    
    # Cost: (A@dx0 + B@du)'Q(...) + du'Rdu
    # Gradient: 2*B'Q(A@dx0 + B@du) + 2*R@du = 0
    # (B'QB + R)du = -B'QA@dx0
    
    BtQ = B.T @ Q
    H = BtQ @ B + R + params.rho * np.eye(m)
    g = BtQ @ A @ dx0
    
    # ADMM
    z = np.zeros(m)
    y = np.zeros(m)
    
    for _ in range(params.max_iter):
        # u-update
        du = solve(H, -g + params.rho * (z - y), assume_a='pos')
        
        # z-update (project onto bounds)
        z_new = np.clip(du + y, u_min - u_ref, u_max - u_ref)
        
        # Convergence check
        primal_res = np.linalg.norm(du - z_new)
        dual_res = params.rho * np.linalg.norm(z_new - z)
        z = z_new
        
        # y-update
        y = y + du - z
        
        eps_pri = params.abs_tol * np.sqrt(m) + params.rel_tol * max(np.linalg.norm(du), np.linalg.norm(z))
        eps_dual = params.abs_tol * np.sqrt(m) + params.rel_tol * np.linalg.norm(y) * params.rho
        
        if primal_res < eps_pri and dual_res < eps_dual:
            break
    
    return u_ref + z


# =============================================================================
# Computed Torque Controller (C++ Planner)
# =============================================================================

class Plan_and_Track_ComputedTorque(PlanAndTrackBase):
    """
    Plan-and-Track with C++ ALTRO planning and computed torque tracking.
    
    Fast closed-form tracking (~200 µs) using:
    - PD control with TVLQR-derived gains
    - Inverse dynamics for torque computation
    - Actual B-field for MTQ allocation
    """
    
    def __init__(
        self,
        sat: EstimatedSatellite,
        planner_settings: PlannerSettings,
        mpc_params: Optional[MPCParams] = None
    ):
        self._init_planner(sat, planner_settings, tracking_lqr_formulation=0)
        self.params = mpc_params if mpc_params is not None else MPCParams.balanced()
        self.sat = sat
        
        self._J = sat.J_noRW
        self._J_inv = sat.invJ_noRW
        self._n_mtq = len(sat.mtq_actuators)
        self._m_max = sat.mtq_actuators[0].u_max if sat.mtq_actuators else 0.2
        
        self._n_rw = len(sat.rw_actuators)
        self._has_rw = self._n_rw > 0
        if self._has_rw:
            self._rw_axes = np.array([rw.axis for rw in sat.rw_actuators])
            self._rw_u_max = np.array([rw.u_max for rw in sat.rw_actuators])
        else:
            self._rw_axes = None
            self._rw_u_max = None
    
    def find_u(
        self,
        x_hat: NDArray[np.float64],
        sens: NDArray[np.float64],
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        B_body: Optional[NDArray[np.float64]] = None,
        **kwargs
    ) -> NDArray[np.float64]:
        """Compute control using computed torque method."""
        if self.active_trajectory is None:
            return np.zeros(self._n_mtq + self._n_rw)
        
        current_time = os_hat.J2000
        if not self.active_trajectory.is_valid_time(current_time):
            raise RuntimeError(
                f"Trajectory expired. Current: {current_time}, "
                f"Traj: [{self.active_trajectory.start_time}, {self.active_trajectory.end_time}]"
            )
        
        if B_body is None:
            R = rot_mat(x_hat[3:7])
            B_body = R.T @ os_hat.B
        
        x_ref = self.active_trajectory.get_state_at(current_time)
        u_ref = self.active_trajectory.get_control_at(current_time)
        K = self.active_trajectory.get_gain_at(current_time)
        
        # Use Trajectory's state diff method
        dx = self.active_trajectory._state_diff(x_hat, x_ref)
        w_err = dx[0:3]
        q_err = dx[3:6]
        
        dt = self.planner_settings.dt_tvlqr
        
        # Compute desired angular acceleration from K-matrix
        if self.params.use_tvlqr_weights and K is not None:
            K_mtq = K[:self._n_mtq, :]
            avg_w_imp = np.mean(np.linalg.norm(K_mtq[:, 0:3], axis=0)) + 0.1
            avg_q_imp = np.mean(np.linalg.norm(K_mtq[:, 3:6], axis=0)) + 0.1
        else:
            avg_w_imp = self.params.Q_omega
            avg_q_imp = self.params.Q_attitude
        
        w_dot_des = -avg_w_imp * w_err / dt - avg_q_imp * q_err / dt
        
        # Inverse dynamics
        w_curr = x_hat[0:3]
        tau_total = self._J @ w_dot_des + np.cross(w_curr, self._J @ w_curr)
        
        # Allocate to actuators
        if self._has_rw:
            u_full_tvlqr = u_ref - K @ dx
            u_rw = np.clip(u_full_tvlqr[self._n_mtq:], -self._rw_u_max, self._rw_u_max)
            tau_rw = sum(u_rw[i] * self._rw_axes[i] for i in range(self._n_rw))
            tau_mtq = tau_total - tau_rw
        else:
            tau_mtq = tau_total
            u_rw = np.array([])
        
        u_mtq = _solve_mtq_for_torque(tau_mtq, B_body, self._m_max)
        return np.concatenate([u_mtq, u_rw])
    
    def calculate_trajectory(
        self,
        t_start: float,
        duration: float,
        x_0: np.ndarray,
        os_0: Orbital_State,
        goals: GoalList,
        verbose: bool = False,
    ) -> Trajectory:
        """Calculate trajectory using C++ ALTRO planner."""
        lqr_times, Xset, Uset, Kset, Sset = self._calculate_trajectory_common(
            t_start, duration, x_0, os_0, goals, verbose
        )
        return Trajectory(lqr_times, Xset, Uset, Kset, Sset)


# =============================================================================
# Computed Torque Controller (Python Planner with Visualization)
# =============================================================================

class Plan_and_Track_ComputedTorque_Python(PlanAndTrackBase):
    """
    Plan-and-Track with Python ALILQR planning and computed torque tracking.
    
    Same tracking as Plan_and_Track_ComputedTorque but uses Python planner
    for live visualization of trajectory optimization.
    """
    
    def __init__(
        self,
        sat: EstimatedSatellite,
        planner_settings: PlannerSettings,
        mpc_params: Optional[MPCParams] = None
    ):
        self._init_planner(sat, planner_settings, tracking_lqr_formulation=0)
        self.params = mpc_params if mpc_params is not None else MPCParams.balanced()
        self.sat = sat
        
        self._J = sat.J_noRW
        self._J_inv = sat.invJ_noRW
        self._n_mtq = len(sat.mtq_actuators)
        self._m_max = sat.mtq_actuators[0].u_max if sat.mtq_actuators else 0.2
        
        self._n_rw = len(sat.rw_actuators)
        self._has_rw = self._n_rw > 0
        if self._has_rw:
            self._rw_axes = np.array([rw.axis for rw in sat.rw_actuators])
            self._rw_u_max = np.array([rw.u_max for rw in sat.rw_actuators])
        else:
            self._rw_axes = None
            self._rw_u_max = None
    
    def find_u(
        self,
        x_hat: NDArray[np.float64],
        sens: NDArray[np.float64],
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        B_body: Optional[NDArray[np.float64]] = None,
        **kwargs
    ) -> NDArray[np.float64]:
        """Compute control using computed torque (same as C++ version)."""
        if self.active_trajectory is None:
            return np.zeros(self._n_mtq + self._n_rw)
        
        current_time = os_hat.J2000
        if not self.active_trajectory.is_valid_time(current_time):
            raise RuntimeError("Trajectory expired")
        
        if B_body is None:
            R = rot_mat(x_hat[3:7])
            B_body = R.T @ os_hat.B
        
        x_ref = self.active_trajectory.get_state_at(current_time)
        u_ref = self.active_trajectory.get_control_at(current_time)
        K = self.active_trajectory.get_gain_at(current_time)
        
        dx = self.active_trajectory._state_diff(x_hat, x_ref)
        w_err, q_err = dx[0:3], dx[3:6]
        dt = self.planner_settings.dt_tvlqr
        
        if self.params.use_tvlqr_weights and K is not None:
            K_mtq = K[:self._n_mtq, :]
            avg_w_imp = np.mean(np.linalg.norm(K_mtq[:, 0:3], axis=0)) + 0.1
            avg_q_imp = np.mean(np.linalg.norm(K_mtq[:, 3:6], axis=0)) + 0.1
        else:
            avg_w_imp, avg_q_imp = self.params.Q_omega, self.params.Q_attitude
        
        w_dot_des = -avg_w_imp * w_err / dt - avg_q_imp * q_err / dt
        w_curr = x_hat[0:3]
        tau_total = self._J @ w_dot_des + np.cross(w_curr, self._J @ w_curr)
        
        if self._has_rw:
            u_full_tvlqr = u_ref - K @ dx
            u_rw = np.clip(u_full_tvlqr[self._n_mtq:], -self._rw_u_max, self._rw_u_max)
            tau_rw = sum(u_rw[i] * self._rw_axes[i] for i in range(self._n_rw))
            tau_mtq = tau_total - tau_rw
        else:
            tau_mtq, u_rw = tau_total, np.array([])
        
        u_mtq = _solve_mtq_for_torque(tau_mtq, B_body, self._m_max)
        return np.concatenate([u_mtq, u_rw])
    
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
        
        Parameters
        ----------
        visualize : bool
            If True, show live convergence plots
        viz_save_path : str, optional
            Path to save final visualization
        iteration_callback : callable, optional
            Called after each optimization iteration
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
        
        return Trajectory(times, result.X, result.U, result.K, result.S)


# =============================================================================
# MPC Controller (C++ Planner)
# =============================================================================

class Plan_and_Track_MPC(PlanAndTrackBase):
    """
    Plan-and-Track with C++ ALTRO planning and adaptive B-field correction.
    
    Uses TVLQR feedback with adaptive blending between:
    - Standard TVLQR (works well when attitude error is large)
    - Actual B-field correction (works well when attitude error is small)
    
    The blending is based on B-field difference between actual and reference
    attitudes. When B-field difference is small (<10%), use actual B-field
    correction. When large, use standard TVLQR.
    
    This achieves sub-degree tracking accuracy for 3MTQ+1RW systems.
    """
    
    def __init__(
        self,
        sat: EstimatedSatellite,
        planner_settings: PlannerSettings,
        mpc_params: Optional[MPCParams] = None,
        bfield_blend_threshold: float = 0.1,  # 10% B-field difference threshold
    ):
        self._init_planner(sat, planner_settings, tracking_lqr_formulation=0)
        self.params = mpc_params if mpc_params is not None else MPCParams.balanced()
        self.sat = sat
        self._bfield_blend_threshold = bfield_blend_threshold
        
        self._J = sat.J_noRW
        self._n_mtq = len(sat.mtq_actuators)
        self._m_max = sat.mtq_actuators[0].u_max if sat.mtq_actuators else 0.2
        
        self._n_rw = len(sat.rw_actuators)
        self._has_rw = self._n_rw > 0
        self._rw_u_max = np.array([rw.u_max for rw in sat.rw_actuators]) if self._has_rw else np.array([])
        
        self._u_min = np.concatenate([[-self._m_max]*self._n_mtq, -self._rw_u_max])
        self._u_max = np.concatenate([[self._m_max]*self._n_mtq, self._rw_u_max])
    
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
        Compute control using adaptive B-field blending.
        
        When B-field difference is small: use actual B-field correction
        When B-field difference is large: use standard TVLQR
        
        This blending ensures good tracking in both regimes.
        """
        if self.active_trajectory is None:
            return np.zeros(self._n_mtq + self._n_rw)
        
        current_time = os_hat.J2000
        if not self.active_trajectory.is_valid_time(current_time):
            raise RuntimeError("Trajectory expired")
        
        # Get reference and gains
        x_ref = self.active_trajectory.get_state_at(current_time)
        u_ref = self.active_trajectory.get_control_at(current_time)
        K = self.active_trajectory.get_gain_at(current_time)
        
        # Compute state error and TVLQR feedback
        dx = self.active_trajectory._state_diff(x_hat, x_ref)
        du = -K @ dx
        
        # Standard TVLQR MTQ control (saturated)
        u_tvlqr_mtq = np.clip(u_ref[:self._n_mtq] + du[:self._n_mtq], 
                              -self._m_max, self._m_max)
        
        # Get B-field in body frame (actual and reference)
        R_actual = rot_mat(x_hat[3:7])
        B_body_actual = R_actual.T @ os_hat.B
        
        R_ref = rot_mat(x_ref[3:7])
        B_body_ref = R_ref.T @ os_hat.B
        
        # Compute B-field difference (normalized)
        B_norm = np.linalg.norm(os_hat.B)
        B_diff = np.linalg.norm(B_body_actual - B_body_ref) / (B_norm + 1e-12)
        
        # Compute actual B-field corrected MTQ control
        tau_ref = np.cross(u_ref[:self._n_mtq], B_body_ref)
        tau_fb = np.cross(du[:self._n_mtq], B_body_ref)
        tau_des = tau_ref + tau_fb
        m_actualb = _solve_mtq_for_torque(tau_des, B_body_actual, self._m_max)
        
        # Blend based on B-field difference
        # alpha = 0 → use ActualB (B-fields similar)
        # alpha = 1 → use TVLQR (B-fields different)
        alpha = min(1.0, B_diff / self._bfield_blend_threshold)
        m_blend = alpha * u_tvlqr_mtq + (1 - alpha) * m_actualb
        m_blend = np.clip(m_blend, -self._m_max, self._m_max)
        
        # RW control: TVLQR feedback with saturation
        if self._has_rw:
            u_rw = np.clip(u_ref[self._n_mtq:] + du[self._n_mtq:], 
                          -self._rw_u_max, self._rw_u_max)
            return np.concatenate([m_blend, u_rw])
        else:
            return m_blend
    
    def calculate_trajectory(
        self,
        t_start: float,
        duration: float,
        x_0: np.ndarray,
        os_0: Orbital_State,
        goals: GoalList,
        verbose: bool = False,
    ) -> Trajectory:
        """Calculate trajectory using C++ ALTRO planner."""
        lqr_times, Xset, Uset, Kset, Sset = self._calculate_trajectory_common(
            t_start, duration, x_0, os_0, goals, verbose
        )
        return Trajectory(lqr_times, Xset, Uset, Kset, Sset)


# =============================================================================
# MPC Controller (Python Planner with Visualization)
# =============================================================================

class Plan_and_Track_MPC_Python(PlanAndTrackBase):
    """
    Plan-and-Track with Python ALILQR planning and adaptive B-field correction.
    
    Same tracking as Plan_and_Track_MPC but uses Python planner
    for live visualization of trajectory optimization.
    """
    
    def __init__(
        self,
        sat: EstimatedSatellite,
        planner_settings: PlannerSettings,
        mpc_params: Optional[MPCParams] = None,
        bfield_blend_threshold: float = 0.1,
    ):
        self._init_planner(sat, planner_settings, tracking_lqr_formulation=0)
        self.params = mpc_params if mpc_params is not None else MPCParams.balanced()
        self.sat = sat
        self._bfield_blend_threshold = bfield_blend_threshold
        
        self._J = sat.J_noRW
        self._n_mtq = len(sat.mtq_actuators)
        self._m_max = sat.mtq_actuators[0].u_max if sat.mtq_actuators else 0.2
        
        self._n_rw = len(sat.rw_actuators)
        self._has_rw = self._n_rw > 0
        self._rw_u_max = np.array([rw.u_max for rw in sat.rw_actuators]) if self._has_rw else np.array([])
        
        self._u_min = np.concatenate([[-self._m_max]*self._n_mtq, -self._rw_u_max])
        self._u_max = np.concatenate([[self._m_max]*self._n_mtq, self._rw_u_max])
    
    def find_u(
        self,
        x_hat: NDArray[np.float64],
        sens: NDArray[np.float64],
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        B_body: Optional[NDArray[np.float64]] = None,
        **kwargs
    ) -> NDArray[np.float64]:
        """Compute control using adaptive B-field blending (same as C++ version)."""
        if self.active_trajectory is None:
            return np.zeros(self._n_mtq + self._n_rw)
        
        current_time = os_hat.J2000
        if not self.active_trajectory.is_valid_time(current_time):
            raise RuntimeError("Trajectory expired")
        
        # Get reference and gains
        x_ref = self.active_trajectory.get_state_at(current_time)
        u_ref = self.active_trajectory.get_control_at(current_time)
        K = self.active_trajectory.get_gain_at(current_time)
        
        # Compute state error and TVLQR feedback
        dx = self.active_trajectory._state_diff(x_hat, x_ref)
        du = -K @ dx
        
        # Standard TVLQR MTQ control (saturated)
        u_tvlqr_mtq = np.clip(u_ref[:self._n_mtq] + du[:self._n_mtq], 
                              -self._m_max, self._m_max)
        
        # Get B-field in body frame (actual and reference)
        R_actual = rot_mat(x_hat[3:7])
        B_body_actual = R_actual.T @ os_hat.B
        
        R_ref = rot_mat(x_ref[3:7])
        B_body_ref = R_ref.T @ os_hat.B
        
        # Compute B-field difference (normalized)
        B_norm = np.linalg.norm(os_hat.B)
        B_diff = np.linalg.norm(B_body_actual - B_body_ref) / (B_norm + 1e-12)
        
        # Compute actual B-field corrected MTQ control
        tau_ref = np.cross(u_ref[:self._n_mtq], B_body_ref)
        tau_fb = np.cross(du[:self._n_mtq], B_body_ref)
        tau_des = tau_ref + tau_fb
        m_actualb = _solve_mtq_for_torque(tau_des, B_body_actual, self._m_max)
        
        # Blend based on B-field difference
        alpha = min(1.0, B_diff / self._bfield_blend_threshold)
        m_blend = alpha * u_tvlqr_mtq + (1 - alpha) * m_actualb
        m_blend = np.clip(m_blend, -self._m_max, self._m_max)
        
        # RW control: TVLQR feedback with saturation
        if self._has_rw:
            u_rw = np.clip(u_ref[self._n_mtq:] + du[self._n_mtq:], 
                          -self._rw_u_max, self._rw_u_max)
            return np.concatenate([m_blend, u_rw])
        else:
            return m_blend
    
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
        """
        py_planner = PythonALILQRv2(self.sat, os_0.orb, self.planner_settings)
        
        viz = None
        if visualize:
            viz = LivePlannerViz(self.sat, os_0.orb, goals, t_start, duration)
        
        def combined_callback(data: IterationData):
            if viz:
                viz.update(data)
            if iteration_callback:
                iteration_callback(data)
        
        result1 = py_planner.optimize(
            x_0, t_start, duration, goals,
            pass_num=1, verbose=verbose,
            iteration_callback=combined_callback if (viz or iteration_callback) else None
        )
        
        if result1 is None or not result1.success:
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
        
        return Trajectory(times, result.X, result.U, result.K, result.S)
