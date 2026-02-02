"""
Model Predictive Control (MPC) tracker for MTQ-only systems.

This module provides simple MPC tracking that re-computes optimal controls
at each timestep using the ACTUAL B-field, fixing the fundamental limitation
of TVLQR for MTQ-only systems.

Key insight: MTQ torque = m × B depends on B-field which depends on attitude.
TVLQR uses planned B-field which diverges from actual. MPC uses actual B-field.
"""
import numpy as np
from scipy.optimize import minimize
from typing import Callable, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MPCParams:
    """Parameters for MPC tracking controller."""
    
    # Cost weights
    Q_omega: float = 1.0       # Angular velocity tracking weight
    Q_attitude: float = 100.0  # Attitude tracking weight  
    R_control: float = 0.01    # Control effort weight
    
    # MPC horizon
    horizon: int = 1           # Number of lookahead steps (1 = simple, 3+ = full MPC)
    
    # Solver options
    max_iter: int = 50         # Max optimization iterations
    tolerance: float = 1e-6    # Convergence tolerance
    
    @classmethod
    def fast(cls) -> 'MPCParams':
        """Fast MPC settings (1-step, minimal computation)."""
        return cls(horizon=1, max_iter=20)
    
    @classmethod
    def accurate(cls) -> 'MPCParams':
        """Accurate MPC settings (multi-step horizon)."""
        return cls(horizon=3, max_iter=50)
    
    @classmethod
    def balanced(cls) -> 'MPCParams':
        """Balanced settings for good tracking without excessive computation."""
        return cls(horizon=1, Q_omega=1.0, Q_attitude=100.0, R_control=0.001)


def quat_error_vec(q_curr: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """
    Compute quaternion error as 3-vector.
    
    Parameters
    ----------
    q_curr : ndarray, shape (4,)
        Current quaternion [w, x, y, z]
    q_ref : ndarray, shape (4,)
        Reference quaternion [w, x, y, z]
        
    Returns
    -------
    q_err_vec : ndarray, shape (3,)
        Error vector (≈ rotation axis × angle for small errors)
    """
    # q_err = q_ref^{-1} * q_curr
    q_ref_inv = np.array([q_ref[0], -q_ref[1], -q_ref[2], -q_ref[3]])
    
    # Quaternion multiplication
    w1, x1, y1, z1 = q_ref_inv
    w2, x2, y2, z2 = q_curr
    q_err = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])
    
    # Ensure positive scalar part for consistent error direction
    if q_err[0] < 0:
        q_err = -q_err
    
    # Return 2 * vector part (small angle approximation: 2*vec ≈ rotation vector)
    return 2 * q_err[1:4]


def propagate_attitude(
    x: np.ndarray, 
    u_mtq: np.ndarray, 
    B_body: np.ndarray, 
    J: np.ndarray, 
    dt: float
) -> np.ndarray:
    """
    Propagate attitude dynamics one timestep.
    
    Parameters
    ----------
    x : ndarray, shape (7,)
        State [omega_x, omega_y, omega_z, q_w, q_x, q_y, q_z]
    u_mtq : ndarray, shape (3,)
        MTQ dipole moment command [Am²]
    B_body : ndarray, shape (3,)
        Magnetic field in body frame [T]
    J : ndarray, shape (3, 3)
        Inertia tensor [kg⋅m²]
    dt : float
        Timestep [s]
        
    Returns
    -------
    x_next : ndarray, shape (7,)
        Next state
    """
    w = x[0:3]
    q = x[3:7]
    
    # MTQ torque
    tau = np.cross(u_mtq, B_body)
    
    # Angular acceleration
    w_dot = np.linalg.solve(J, tau - np.cross(w, J @ w))
    
    # Quaternion kinematics
    q_w, q_x, q_y, q_z = q
    wx, wy, wz = w
    q_dot = 0.5 * np.array([
        -q_x*wx - q_y*wy - q_z*wz,
         q_w*wx + q_y*wz - q_z*wy,
         q_w*wy + q_z*wx - q_x*wz,
         q_w*wz + q_x*wy - q_y*wx
    ])
    
    # Euler integration
    w_next = w + w_dot * dt
    q_next = q + q_dot * dt
    q_next = q_next / np.linalg.norm(q_next)  # Normalize
    
    return np.concatenate([w_next, q_next])


def solve_mtq_for_torque(
    tau_desired: np.ndarray, 
    B_body: np.ndarray, 
    m_max: float
) -> np.ndarray:
    """
    Solve for MTQ dipole moment to produce desired torque.
    
    Uses minimum-norm solution: m = (B × τ) / |B|²
    Then clamps to actuator limits.
    
    Parameters
    ----------
    tau_desired : ndarray, shape (3,)
        Desired torque [Nm]
    B_body : ndarray, shape (3,)
        Magnetic field in body frame [T]
    m_max : float
        Maximum dipole moment [Am²]
        
    Returns
    -------
    m : ndarray, shape (3,)
        MTQ dipole command [Am²]
    """
    B_norm_sq = np.dot(B_body, B_body)
    if B_norm_sq < 1e-20:
        return np.zeros(3)
    
    m = np.cross(B_body, tau_desired) / B_norm_sq
    return np.clip(m, -m_max, m_max)


def mpc_one_step(
    x_curr: np.ndarray,
    x_ref: np.ndarray,
    B_body: np.ndarray,
    J: np.ndarray,
    m_max: float,
    dt: float,
    params: MPCParams = None
) -> np.ndarray:
    """
    One-step MPC: Compute optimal MTQ command for trajectory tracking.
    
    This is the fast version using analytical solution:
    1. Compute state error
    2. Compute desired angular acceleration (PD-like)
    3. Compute required torque
    4. Solve for MTQ dipole
    
    Parameters
    ----------
    x_curr : ndarray, shape (7,)
        Current state [omega, quaternion]
    x_ref : ndarray, shape (7,)
        Reference state at current time
    B_body : ndarray, shape (3,)
        Magnetic field in body frame [T]
    J : ndarray, shape (3, 3)
        Inertia tensor
    m_max : float
        Maximum MTQ dipole moment [Am²]
    dt : float
        Control timestep [s]
    params : MPCParams, optional
        MPC parameters
        
    Returns
    -------
    u_mtq : ndarray, shape (3,)
        Optimal MTQ dipole command [Am²]
    """
    if params is None:
        params = MPCParams.balanced()
    
    w_curr = x_curr[0:3]
    w_ref = x_ref[0:3]
    
    # State errors
    w_err = w_curr - w_ref
    q_err = quat_error_vec(x_curr[3:7], x_ref[3:7])
    
    # Desired angular acceleration (PD-like feedback)
    # Gains scaled by error weights
    K_w = params.Q_omega / dt
    K_q = params.Q_attitude / dt
    
    w_dot_des = -K_w * w_err - K_q * q_err
    
    # Required torque: τ = J⋅ω̇ + ω × (J⋅ω)
    tau_des = J @ w_dot_des + np.cross(w_curr, J @ w_curr)
    
    # Solve for MTQ
    return solve_mtq_for_torque(tau_des, B_body, m_max)


def mpc_multi_step(
    x_curr: np.ndarray,
    x_ref_func: Callable[[float], np.ndarray],
    B_body_func: Callable[[float], np.ndarray],
    t_curr: float,
    J: np.ndarray,
    m_max: float,
    dt: float,
    params: MPCParams = None
) -> np.ndarray:
    """
    Multi-step MPC: Optimize control sequence over horizon.
    
    More accurate than one-step but slower. Uses numerical optimization.
    
    Parameters
    ----------
    x_curr : ndarray, shape (7,)
        Current state
    x_ref_func : callable
        Function that returns reference state at time t
    B_body_func : callable
        Function that returns B-field in body frame at time t
    t_curr : float
        Current time [s]
    J : ndarray, shape (3, 3)
        Inertia tensor
    m_max : float
        Maximum MTQ dipole moment [Am²]
    dt : float
        Control timestep [s]
    params : MPCParams, optional
        MPC parameters
        
    Returns
    -------
    u_mtq : ndarray, shape (3,)
        First control in optimal sequence [Am²]
    """
    if params is None:
        params = MPCParams.accurate()
    
    horizon = params.horizon
    n_ctrl = 3 * horizon
    
    def cost_func(m_flat):
        m_seq = m_flat.reshape(horizon, 3)
        
        cost = 0.0
        x = x_curr.copy()
        t = t_curr
        
        for k in range(horizon):
            # Get B-field at current time
            B_body = B_body_func(t)
            
            # Propagate
            x = propagate_attitude(x, m_seq[k], B_body, J, dt)
            t += dt
            
            # Get reference at next time
            x_ref = x_ref_func(t)
            
            # Tracking cost
            w_err = x[0:3] - x_ref[0:3]
            q_err = quat_error_vec(x[3:7], x_ref[3:7])
            
            cost += params.Q_omega * np.dot(w_err, w_err)
            cost += params.Q_attitude * np.dot(q_err, q_err)
            cost += params.R_control * np.dot(m_seq[k], m_seq[k])
        
        return cost
    
    # Initial guess: zeros
    m0 = np.zeros(n_ctrl)
    
    # Bounds
    bounds = [(-m_max, m_max)] * n_ctrl
    
    # Optimize
    result = minimize(
        cost_func, m0, 
        method='L-BFGS-B', 
        bounds=bounds,
        options={'maxiter': params.max_iter, 'ftol': params.tolerance}
    )
    
    return result.x[:3]


def mpc_tvlqr_hybrid(
    x_curr: np.ndarray,
    x_ref: np.ndarray,
    x_ref_next: np.ndarray,
    K_next: np.ndarray,
    B_body: np.ndarray,
    J: np.ndarray,
    m_max: float,
    dt: float,
) -> np.ndarray:
    """
    Hybrid MPC-TVLQR: Use K-matrix weights to prioritize error reduction.
    
    This combines:
    - MPC: Uses actual B-field (not planned) for control computation
    - TVLQR: Uses K-matrix to know which errors matter most
    
    The K-matrix encodes the full trajectory optimization's understanding
    of error importance at each timestep.
    
    Parameters
    ----------
    x_curr : ndarray, shape (7,)
        Current state [omega, quaternion]
    x_ref : ndarray, shape (7,)
        Reference state at current time
    x_ref_next : ndarray, shape (7,)
        Reference state at next time (t + dt)
    K_next : ndarray, shape (3, 6)
        TVLQR gain matrix at next timestep
    B_body : ndarray, shape (3,)
        Magnetic field in body frame [T]
    J : ndarray, shape (3, 3)
        Inertia tensor
    m_max : float
        Maximum MTQ dipole moment [Am²]
    dt : float
        Timestep [s]
        
    Returns
    -------
    u_mtq : ndarray, shape (3,)
        Optimal MTQ command [Am²]
    """
    # Current state error
    w_err = x_curr[0:3] - x_ref[0:3]
    q_err = quat_error_vec(x_curr[3:7], x_ref[3:7])
    dx_curr = np.concatenate([w_err, q_err])
    
    # Extract K-matrix column norms to get error importance weights
    # K shape is (3, 6): 3 controls x 6 state errors
    K_w = K_next[:, 0:3]  # Gains on omega error
    K_q = K_next[:, 3:6]  # Gains on attitude error
    
    # Importance = how much each error affects control
    w_importance = np.linalg.norm(K_w, axis=0) + 0.1  # Add small offset
    q_importance = np.linalg.norm(K_q, axis=0) + 0.1
    
    # Average importance for angular velocity vs attitude
    avg_w_importance = np.mean(w_importance)
    avg_q_importance = np.mean(q_importance)
    
    # Desired angular acceleration: weighted by K-derived importance
    w_dot_des = -avg_w_importance * w_err / dt - avg_q_importance * q_err / dt
    
    # Required torque: τ = J⋅ω̇ + ω × (J⋅ω)
    w_curr = x_curr[0:3]
    tau_des = J @ w_dot_des + np.cross(w_curr, J @ w_curr)
    
    # Solve for MTQ using ACTUAL B-field
    return solve_mtq_for_torque(tau_des, B_body, m_max)


class MPCTracker:
    """
    MPC trajectory tracker for MTQ-only systems.
    
    This class wraps the MPC functions and provides a convenient interface
    for tracking a pre-planned trajectory.
    
    Example
    -------
    >>> # Create tracker
    >>> tracker = MPCTracker(J, m_max, dt=0.5, params=MPCParams.balanced())
    >>> 
    >>> # Set reference trajectory
    >>> tracker.set_trajectory(t_ref, X_ref, U_ref)
    >>> 
    >>> # At each timestep
    >>> u = tracker.compute_control(x_curr, B_body, t_curr)
    """
    
    def __init__(
        self,
        J: np.ndarray,
        m_max: float,
        dt: float,
        params: MPCParams = None
    ):
        """
        Initialize MPC tracker.
        
        Parameters
        ----------
        J : ndarray, shape (3, 3)
            Inertia tensor
        m_max : float
            Maximum MTQ dipole moment [Am²]
        dt : float
            Control timestep [s]
        params : MPCParams, optional
            MPC parameters (default: balanced)
        """
        self.J = J
        self.m_max = m_max
        self.dt = dt
        self.params = params if params is not None else MPCParams.balanced()
        
        self._x_ref_interp = None
        self._u_ref_interp = None
        
    def set_trajectory(
        self,
        t: np.ndarray,
        X: np.ndarray,
        U: np.ndarray = None
    ) -> None:
        """
        Set the reference trajectory.
        
        Parameters
        ----------
        t : ndarray, shape (N,)
            Time points [s]
        X : ndarray, shape (7, N)
            State trajectory
        U : ndarray, shape (3, N), optional
            Control trajectory (for feedforward term)
        """
        from scipy.interpolate import interp1d
        
        self._t = t
        self._X = X
        self._U = U
        
        # Create interpolators
        self._x_ref_interp = [
            interp1d(t, X[i, :], kind='linear', fill_value='extrapolate') 
            for i in range(X.shape[0])
        ]
        
        if U is not None:
            self._u_ref_interp = [
                interp1d(t, U[i, :], kind='linear', fill_value='extrapolate')
                for i in range(U.shape[0])
            ]
    
    def get_ref_state(self, t: float) -> np.ndarray:
        """Get reference state at time t."""
        if self._x_ref_interp is None:
            raise ValueError("No trajectory set. Call set_trajectory first.")
        return np.array([interp(t) for interp in self._x_ref_interp])
    
    def get_ref_control(self, t: float) -> np.ndarray:
        """Get reference control at time t."""
        if self._u_ref_interp is None:
            return np.zeros(3)
        return np.array([interp(t) for interp in self._u_ref_interp])
    
    def compute_control(
        self,
        x_curr: np.ndarray,
        B_body: np.ndarray,
        t_curr: float,
        use_feedforward: bool = True
    ) -> np.ndarray:
        """
        Compute optimal control for trajectory tracking.
        
        Parameters
        ----------
        x_curr : ndarray, shape (7,)
            Current state
        B_body : ndarray, shape (3,)
            Current B-field in body frame [T]
        t_curr : float
            Current time [s]
        use_feedforward : bool
            If True, add feedforward term from reference trajectory
            
        Returns
        -------
        u_mtq : ndarray, shape (3,)
            Optimal MTQ dipole command [Am²]
        """
        x_ref = self.get_ref_state(t_curr)
        
        if self.params.horizon <= 1:
            # Fast one-step MPC
            u = mpc_one_step(
                x_curr, x_ref, B_body, self.J, self.m_max, self.dt, self.params
            )
        else:
            # Multi-step MPC (slower)
            u = mpc_multi_step(
                x_curr, self.get_ref_state, 
                lambda t: B_body,  # Assume constant B for simplicity
                t_curr, self.J, self.m_max, self.dt, self.params
            )
        
        # Optionally add feedforward
        if use_feedforward and self._u_ref_interp is not None:
            u_ff = self.get_ref_control(t_curr)
            # Blend: more feedforward when tracking well, more feedback when off
            # Simple: just add small feedforward term
            u = 0.7 * u + 0.3 * u_ff
        
        return np.clip(u, -self.m_max, self.m_max)
    
    def set_tvlqr_gains(self, K: np.ndarray) -> None:
        """
        Set TVLQR gain matrices for hybrid MPC-TVLQR control.
        
        Parameters
        ----------
        K : ndarray, shape (3, 6, N) or (18, N)
            TVLQR gain matrices. If 2D, will be reshaped to (3, 6, N).
        """
        if K.ndim == 2:
            # Reshape from (18, N) to (3, 6, N)
            ctrl_dim, state_dim = 3, 6
            T = K.shape[1]
            self._K = K.reshape(ctrl_dim, state_dim, T)
        else:
            self._K = K
    
    def get_K_at(self, t: float) -> np.ndarray:
        """Get gain matrix at time t."""
        if not hasattr(self, '_K') or self._K is None:
            return np.eye(3, 6)  # Default: identity-like
        
        # Find nearest index
        idx = np.searchsorted(self._t, t)
        idx = np.clip(idx, 0, self._K.shape[2] - 1)
        return self._K[:, :, idx]
    
    def compute_control_hybrid(
        self,
        x_curr: np.ndarray,
        B_body: np.ndarray,
        t_curr: float,
    ) -> np.ndarray:
        """
        Compute control using hybrid MPC-TVLQR approach.
        
        Uses MPC with actual B-field but weights errors using TVLQR K-matrices.
        This is the best approach for MTQ-only systems.
        
        Parameters
        ----------
        x_curr : ndarray, shape (7,)
            Current state
        B_body : ndarray, shape (3,)
            Current B-field in body frame [T]
        t_curr : float
            Current time [s]
            
        Returns
        -------
        u_mtq : ndarray, shape (3,)
            Optimal MTQ dipole command [Am²]
        """
        x_ref = self.get_ref_state(t_curr)
        x_ref_next = self.get_ref_state(t_curr + self.dt)
        K_next = self.get_K_at(t_curr + self.dt)
        
        u = mpc_tvlqr_hybrid(
            x_curr, x_ref, x_ref_next, K_next,
            B_body, self.J, self.m_max, self.dt
        )
        
        return np.clip(u, -self.m_max, self.m_max)
