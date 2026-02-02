"""
Plan-and-Track controllers using computed torque and MPC tracking.

This module provides two tracking approaches that use the actual B-field
(rather than the planned B-field) for MTQ control:

1. **Computed Torque** (`Plan_and_Track_ComputedTorque`):
   - Fast closed-form solution (~150 µs)
   - PD controller with inverse dynamics
   - Uses TVLQR K-matrix norms for gain weighting

2. **True MPC** (`Plan_and_Track_MPC`):
   - Single-step optimal control problem
   - Solves QP with actuator constraints
   - More accurate but slower (~1-5 ms)

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
from typing import Optional
from numpy.typing import NDArray
from dataclasses import dataclass
from scipy.optimize import minimize

from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_base import PlanAndTrackBase
from ADCS.controller.helpers import PlannerSettings, Trajectory
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.helpers.math_helpers import rot_mat, quat_diff, quat_to_vec3


# =============================================================================
# Parameters
# =============================================================================

@dataclass
class MPCParams:
    """Parameters for MPC/Computed Torque tracking controllers."""
    
    # Cost weights
    Q_omega: float = 1.0       # Angular velocity tracking weight
    Q_attitude: float = 100.0  # Attitude tracking weight  
    Q_rw: float = 1.0          # RW momentum tracking weight
    R_mtq: float = 0.01        # MTQ control effort weight
    R_rw: float = 0.01         # RW control effort weight
    
    # MPC-specific options
    max_iter: int = 50         # Max optimization iterations
    tolerance: float = 1e-6    # Convergence tolerance
    
    # Computed torque options
    use_tvlqr_weights: bool = True  # Use K-matrix column norms for weighting
    
    @classmethod
    def fast(cls) -> 'MPCParams':
        """Fast settings (minimal computation)."""
        return cls(max_iter=20, use_tvlqr_weights=False)
    
    @classmethod
    def accurate(cls) -> 'MPCParams':
        """Accurate settings (more iterations)."""
        return cls(max_iter=100, tolerance=1e-8)
    
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
# True MPC Controller
# =============================================================================

class Plan_and_Track_MPC(PlanAndTrackBase):
    """
    Plan-and-Track controller using ALTRO planning with single-step MPC tracking.
    
    True MPC solves an optimization problem at each timestep:
        min  (x₁ - x_ref)ᵀ Q (x₁ - x_ref) + uᵀ R u
        s.t. x₁ = f(x₀, u)
             u_min ≤ u ≤ u_max
    
    Uses the satellite's dynamics_core for accurate dynamics.
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
        """Compute control using single-step MPC optimization."""
        if self.active_trajectory is None:
            return np.zeros(self._n_mtq + self._n_rw)
        
        current_time = os_hat.J2000
        if not self.active_trajectory.is_valid_time(current_time):
            raise RuntimeError("Trajectory expired")
        
        # Get reference at NEXT timestep for MPC target
        dt = self.planner_settings.dt_tvlqr
        from ADCS.orbits.universal_constants import TimeConstants
        t_next = current_time + dt * TimeConstants.sec2cent
        
        if self.active_trajectory.is_valid_time(t_next):
            x_ref = self.active_trajectory.get_state_at(t_next)
        else:
            x_ref = self.active_trajectory.get_state_at(current_time)
        
        u_ref = self.active_trajectory.get_control_at(current_time)
        
        # Build cost matrices
        n_u = self._n_mtq + self._n_rw
        Q = np.diag([self.params.Q_omega]*3 + [self.params.Q_attitude]*3 + 
                    [self.params.Q_rw]*self._n_rw)
        R = np.diag([self.params.R_mtq]*self._n_mtq + [self.params.R_rw]*self._n_rw)
        
        def cost(u):
            """MPC cost: tracking error + control effort."""
            # Propagate one step using satellite dynamics
            xdot = est_sat.dynamics_core(x_hat, u, os_hat)
            x_next = x_hat + xdot * dt
            x_next[3:7] = x_next[3:7] / np.linalg.norm(x_next[3:7])  # Normalize quat
            
            # State error
            dx = _compute_state_error(x_next, x_ref, self._n_rw)
            du = u - u_ref
            
            return dx @ Q @ dx + du @ R @ du
        
        # Bounds
        bounds = [(-self._m_max, self._m_max)] * self._n_mtq
        if self._has_rw:
            for i in range(self._n_rw):
                bounds.append((-self._rw_u_max[i], self._rw_u_max[i]))
        
        # Optimize
        result = minimize(
            cost, u_ref,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': self.params.max_iter, 'ftol': self.params.tolerance}
        )
        
        return result.x
    
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
