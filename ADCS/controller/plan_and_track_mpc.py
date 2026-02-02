"""
Plan-and-Track controllers using computed torque and MPC tracking.

This module provides tracking controllers that use the actual B-field
(rather than the planned B-field) for MTQ control:

1. **Computed Torque** (`Plan_and_Track_ComputedTorque`):
   - Fast closed-form solution (~200 µs)
   - PD controller with inverse dynamics
   - Uses TVLQR K-matrix norms for gain weighting

2. **True MPC** (`Plan_and_Track_MPC`):
   - ADMM-based single-step MPC (~500 µs - 2 ms)
   - Uses TVLQR K-matrices for cost weighting
   - Proper constraint handling via ADMM
   - Uses satellite dynamics for accurate prediction

Both fix the fundamental TVLQR limitation for MTQ systems: TVLQR uses planned
B-field which diverges from actual when attitude drifts.
"""
from __future__ import annotations

__all__ = [
    "Plan_and_Track_ComputedTorque",
    "Plan_and_Track_MPC", 
    "MPCParams",
]

import numpy as np
from typing import Optional, Tuple
from numpy.typing import NDArray
from dataclasses import dataclass
from scipy.linalg import solve

from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_base import PlanAndTrackBase
from ADCS.controller.helpers import PlannerSettings, Trajectory
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.helpers.math_helpers import rot_mat, quat_diff, quat_to_vec3, normalize


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
    
    # MPC-specific options
    horizon: int = 1           # Prediction horizon (1 = single-step)
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

def _compute_state_error(x_curr: np.ndarray, x_ref: np.ndarray, n_rw: int) -> np.ndarray:
    """
    Compute reduced state error (6 + n_rw dimensions).
    
    Converts 4D quaternion error to 3D rotation vector using quat_to_vec3.
    """
    dx = np.zeros(6 + n_rw)
    
    # Angular velocity error
    dx[0:3] = x_curr[0:3] - x_ref[0:3]
    
    # Attitude error: q_err = q_ref^{-1} ⊗ q_curr, then to 3-vec
    q_err = quat_diff(x_ref[3:7], x_curr[3:7])
    dx[3:6] = quat_to_vec3(q_err, mode=0)  # MRP representation
    
    # RW momentum error
    if n_rw > 0:
        dx[6:6+n_rw] = x_curr[7:7+n_rw] - x_ref[7:7+n_rw]
    
    return dx


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
    
    Parameters
    ----------
    K : ndarray, shape (n_ctrl, n_err)
        TVLQR gain matrix
    n_mtq : int
        Number of MTQ actuators
    n_rw : int
        Number of RW actuators
    params : MPCParams
        Parameters with fallback weights
        
    Returns
    -------
    Q : ndarray, shape (n_err, n_err)
        State cost matrix (diagonal)
    R : ndarray, shape (n_ctrl, n_ctrl)
        Control cost matrix (diagonal)
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
        Q_diag[i] = np.linalg.norm(K[:, i])**2 + 0.1  # Add small offset
    
    # Normalize to reasonable scale
    Q_diag = Q_diag / np.mean(Q_diag) * params.Q_attitude
    Q = np.diag(Q_diag)
    
    # Extract R from row norms of K (control cost is inverse of gain magnitude)
    R_diag = np.zeros(n_ctrl)
    for j in range(n_ctrl):
        row_norm = np.linalg.norm(K[j, :])
        if row_norm > 1e-6:
            R_diag[j] = 1.0 / (row_norm + 0.1)
        else:
            R_diag[j] = 1.0
    
    # Normalize and scale
    R_diag[:n_mtq] = R_diag[:n_mtq] / np.mean(R_diag[:n_mtq] + 1e-10) * params.R_mtq
    if n_rw > 0:
        R_diag[n_mtq:] = R_diag[n_mtq:] / np.mean(R_diag[n_mtq:] + 1e-10) * params.R_rw
    R = np.diag(R_diag)
    
    return Q, R


def _linearize_dynamics(
    sat: EstimatedSatellite,
    x_op: np.ndarray,
    u_op: np.ndarray,
    os: Orbital_State,
    dt: float,
    n_err: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linearize dynamics about operating point using finite differences.
    
    Returns error-state dynamics: dx_{k+1} = A_err @ dx_k + B_err @ du_k
    
    Parameters
    ----------
    sat : EstimatedSatellite
        Satellite model
    x_op : ndarray
        Operating point state
    u_op : ndarray
        Operating point control
    os : Orbital_State
        Orbital state for environment
    dt : float
        Timestep
    n_err : int
        Error state dimension (6 + n_rw)
        
    Returns
    -------
    A_err : ndarray, shape (n_err, n_err)
        Error state transition matrix
    B_err : ndarray, shape (n_err, n_ctrl)
        Error control matrix
    """
    n = len(x_op)
    m = len(u_op)
    n_rw = n - 7
    eps = 1e-7
    
    # Get nominal dynamics
    xdot_nom = sat.dynamics_core(x_op, u_op, os)
    
    # Compute A = df/dx via finite differences
    A_cont = np.zeros((n, n))
    for i in range(n):
        x_pert = x_op.copy()
        x_pert[i] += eps
        if 3 <= i < 7:  # Normalize quaternion
            x_pert[3:7] = normalize(x_pert[3:7])
        xdot_pert = sat.dynamics_core(x_pert, u_op, os)
        A_cont[:, i] = (xdot_pert - xdot_nom) / eps
    
    # Compute B = df/du via finite differences
    B_cont = np.zeros((n, m))
    for i in range(m):
        u_pert = u_op.copy()
        u_pert[i] += eps
        xdot_pert = sat.dynamics_core(x_op, u_pert, os)
        B_cont[:, i] = (xdot_pert - xdot_nom) / eps
    
    # Discretize (forward Euler)
    A_full = np.eye(n) + dt * A_cont
    B_full = dt * B_cont
    
    # Map to error state (7D → 6D for quaternion)
    # Error state: [w_err(3), theta_err(3), h_err(n_rw)]
    A_err = np.zeros((n_err, n_err))
    B_err = np.zeros((n_err, m))
    
    # Angular velocity blocks
    A_err[0:3, 0:3] = A_full[0:3, 0:3]
    A_err[0:3, 3:6] = A_full[0:3, 4:7] * 0.5  # quat → angle scale
    
    # Attitude blocks (quat → 3D angle)
    A_err[3:6, 0:3] = A_full[4:7, 0:3] * 2.0
    A_err[3:6, 3:6] = A_full[4:7, 4:7]
    
    # RW momentum blocks
    if n_rw > 0:
        A_err[6:, 0:3] = A_full[7:, 0:3]
        A_err[6:, 3:6] = A_full[7:, 4:7] * 0.5
        A_err[0:3, 6:] = A_full[0:3, 7:]
        A_err[3:6, 6:] = A_full[4:7, 7:] * 2.0
        A_err[6:, 6:] = A_full[7:, 7:]
    
    # Control Jacobian
    B_err[0:3, :] = B_full[0:3, :]
    B_err[3:6, :] = B_full[4:7, :] * 2.0
    if n_rw > 0:
        B_err[6:, :] = B_full[7:, :]
    
    return A_err, B_err


def _admm_solve(
    x0: np.ndarray,
    x_ref: np.ndarray,
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
        min  dx'Qdx + du'Rdu
        s.t. dx = A @ dx0 + B @ du
             u_min <= u_ref + du <= u_max
    
    ADMM splits: du = z, where z satisfies bounds
    
    Parameters
    ----------
    x0 : ndarray
        Current state
    x_ref : ndarray
        Reference state at next timestep
    u_ref : ndarray
        Reference control
    A : ndarray
        Error state matrix
    B : ndarray
        Error control matrix
    Q : ndarray
        State cost
    R : ndarray
        Control cost
    u_min, u_max : ndarray
        Control bounds
    params : MPCParams
        ADMM parameters
        
    Returns
    -------
    u_opt : ndarray
        Optimal control
    """
    n_err = len(Q)
    m = len(u_ref)
    
    # Current error state
    n_rw = n_err - 6
    dx0 = _compute_state_error(x0, x_ref, n_rw)
    
    # Compute unconstrained optimal du via normal equations
    # min dx'Qdx + du'Rdu where dx = A@dx0 + B@du
    # ∂/∂du = 2B'Q(A@dx0 + B@du) + 2R@du = 0
    # (B'QB + R)du = -B'QA@dx0
    
    BtQ = B.T @ Q
    H = BtQ @ B + R + params.rho * np.eye(m)  # Add ADMM penalty
    g = BtQ @ A @ dx0
    
    # ADMM variables
    z = np.zeros(m)  # Projected control deviation
    y = np.zeros(m)  # Dual variable
    
    for _ in range(params.max_iter):
        # u-update: solve (H + rho*I) @ du = -g + rho*(z - y)
        du = solve(H, -g + params.rho * (z - y), assume_a='pos')
        
        # z-update: project du + y onto [u_min - u_ref, u_max - u_ref]
        z_new = np.clip(du + y, u_min - u_ref, u_max - u_ref)
        
        # Check convergence
        primal_res = np.linalg.norm(du - z_new)
        dual_res = params.rho * np.linalg.norm(z_new - z)
        
        z = z_new
        
        # y-update: dual variable
        y = y + du - z
        
        # Convergence check
        eps_pri = params.abs_tol * np.sqrt(m) + params.rel_tol * max(np.linalg.norm(du), np.linalg.norm(z))
        eps_dual = params.abs_tol * np.sqrt(m) + params.rel_tol * np.linalg.norm(y) * params.rho
        
        if primal_res < eps_pri and dual_res < eps_dual:
            break
    
    return u_ref + z


# =============================================================================
# Computed Torque Controller
# =============================================================================

class Plan_and_Track_ComputedTorque(PlanAndTrackBase):
    """
    Plan-and-Track controller using ALTRO planning with computed torque tracking.
    
    Computed torque control is a fast closed-form method that:
    1. Computes state error
    2. Derives desired angular acceleration from PD gains
    3. Computes required torque via inverse dynamics
    4. Solves for MTQ dipole using actual B-field
    
    For RW, uses standard TVLQR since RW torque is attitude-independent.
    """
    
    def __init__(
        self,
        sat: EstimatedSatellite,
        planner_settings: PlannerSettings,
        mpc_params: Optional[MPCParams] = None
    ):
        # Initialize base class planner
        self._init_planner(sat, planner_settings, tracking_lqr_formulation=0)
        self.params = mpc_params if mpc_params is not None else MPCParams.balanced()
        self.sat = sat
        
        # Cache satellite properties
        self._J = sat.J_noRW
        self._J_inv = sat.invJ_noRW
        self._n_mtq = len(sat.mtq_actuators)
        self._m_max = sat.mtq_actuators[0].u_max if sat.mtq_actuators else 0.2
        
        # RW properties
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
        
        # Get B-field in body frame
        if B_body is None:
            R = rot_mat(x_hat[3:7])  # body → ECI
            B_body = R.T @ os_hat.B  # ECI → body
        
        # Get reference trajectory
        x_ref = self.active_trajectory.get_state_at(current_time)
        u_ref = self.active_trajectory.get_control_at(current_time)
        K = self.active_trajectory.get_gain_at(current_time)
        
        dt = self.planner_settings.dt_tvlqr
        
        # Compute state error
        dx = _compute_state_error(x_hat, x_ref, self._n_rw)
        w_err = dx[0:3]
        q_err = dx[3:6]
        
        # --- Compute desired angular acceleration ---
        if self.params.use_tvlqr_weights and K is not None:
            # Extract importance weights from K-matrix column norms
            K_mtq = K[:self._n_mtq, :]
            K_w = K_mtq[:, 0:3]
            K_q = K_mtq[:, 3:6]
            avg_w_imp = np.mean(np.linalg.norm(K_w, axis=0)) + 0.1
            avg_q_imp = np.mean(np.linalg.norm(K_q, axis=0)) + 0.1
        else:
            avg_w_imp = self.params.Q_omega
            avg_q_imp = self.params.Q_attitude
        
        # PD-like control: ω̇_des = -K_ω·ω_err - K_θ·θ_err
        w_dot_des = -avg_w_imp * w_err / dt - avg_q_imp * q_err / dt
        
        # --- Inverse dynamics ---
        w_curr = x_hat[0:3]
        tau_total = self._J @ w_dot_des + np.cross(w_curr, self._J @ w_curr)
        
        # --- Allocate to actuators ---
        if self._has_rw:
            # RW: use TVLQR (attitude-independent torque)
            u_full_tvlqr = u_ref - K @ dx
            u_rw = np.clip(u_full_tvlqr[self._n_mtq:], -self._rw_u_max, self._rw_u_max)
            
            # Compute torque provided by RW
            tau_rw = sum(u_rw[i] * self._rw_axes[i] for i in range(self._n_rw))
            tau_mtq = tau_total - tau_rw
        else:
            tau_mtq = tau_total
            u_rw = np.array([])
        
        # Solve for MTQ dipole using actual B-field
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
        """Calculate trajectory using C++ ALTRO planner (via base class)."""
        lqr_times, Xset, Uset, Kset, Sset = self._calculate_trajectory_common(
            t_start, duration, x_0, os_0, goals, verbose
        )
        return Trajectory(lqr_times, Xset, Uset, Kset, Sset)


# =============================================================================
# True MPC Controller (ADMM-based)
# =============================================================================

class Plan_and_Track_MPC(PlanAndTrackBase):
    """
    Plan-and-Track controller using ALTRO planning with ADMM-based MPC tracking.
    
    Uses the TinyMPC algorithm (ADMM) for efficient constrained MPC:
    1. Linearize dynamics about reference
    2. Extract Q/R from TVLQR K-matrix
    3. Solve box-constrained QP via ADMM
    
    This properly handles actuator constraints while using the trajectory
    optimizer's understanding of error importance (encoded in K-matrices).
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
        self._n_mtq = len(sat.mtq_actuators)
        self._m_max = sat.mtq_actuators[0].u_max if sat.mtq_actuators else 0.2
        
        self._n_rw = len(sat.rw_actuators)
        self._has_rw = self._n_rw > 0
        if self._has_rw:
            self._rw_u_max = np.array([rw.u_max for rw in sat.rw_actuators])
        else:
            self._rw_u_max = np.array([])
        
        # Build control bounds
        self._u_min = np.concatenate([[-self._m_max]*self._n_mtq, -self._rw_u_max])
        self._u_max = np.concatenate([[self._m_max]*self._n_mtq, self._rw_u_max])
        
        # Cache for linearized dynamics
        self._A_cache = None
        self._B_cache = None
        self._cache_time = None
    
    def find_u(
        self,
        x_hat: NDArray[np.float64],
        sens: NDArray[np.float64],
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        B_body: Optional[NDArray[np.float64]] = None,
        **kwargs
    ) -> NDArray[np.float64]:
        """Compute control using ADMM-based MPC."""
        if self.active_trajectory is None:
            return np.zeros(self._n_mtq + self._n_rw)
        
        current_time = os_hat.J2000
        if not self.active_trajectory.is_valid_time(current_time):
            raise RuntimeError("Trajectory expired")
        
        # Get reference at NEXT timestep (MPC target)
        dt = self.planner_settings.dt_tvlqr
        from ADCS.orbits.universal_constants import TimeConstants
        t_next = current_time + dt * TimeConstants.sec2cent
        
        if self.active_trajectory.is_valid_time(t_next):
            x_ref = self.active_trajectory.get_state_at(t_next)
        else:
            x_ref = self.active_trajectory.get_state_at(current_time)
        
        u_ref = self.active_trajectory.get_control_at(current_time)
        K = self.active_trajectory.get_gain_at(current_time)
        
        # Extract Q/R from K-matrix
        Q, R = _extract_cost_matrices_from_K(K, self._n_mtq, self._n_rw, self.params)
        
        # Linearize dynamics (with caching)
        n_err = 6 + self._n_rw
        if self._cache_time != current_time or self._A_cache is None:
            x_ref_curr = self.active_trajectory.get_state_at(current_time)
            A, B = _linearize_dynamics(est_sat, x_ref_curr, u_ref, os_hat, dt, n_err)
            self._A_cache = A
            self._B_cache = B
            self._cache_time = current_time
        else:
            A, B = self._A_cache, self._B_cache
        
        # Solve via ADMM
        u_opt = _admm_solve(
            x_hat, x_ref, u_ref, A, B, Q, R,
            self._u_min, self._u_max, self.params
        )
        
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
        """Calculate trajectory using C++ ALTRO planner (via base class)."""
        lqr_times, Xset, Uset, Kset, Sset = self._calculate_trajectory_common(
            t_start, duration, x_0, os_0, goals, verbose
        )
        return Trajectory(lqr_times, Xset, Uset, Kset, Sset)
