"""
MTQ-only warm start transition utilities.

For MTQ-only systems, the standard control interpolation between coarse and fine
grids fails because MTQ torque depends on instantaneous B-field direction.
This module provides utilities to properly transition from coarse to fine grid
by solving for controls in the body frame.

Key algorithm:
1. Interpolate states from coarse to fine grid
2. Solve for MTQ controls: m = (B × τ) / |B|²
3. Clamp controls and forward propagate

See papers/Planner/MTQ_DISCRETIZATION_FINDINGS.md for full analysis.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from ADCS.helpers.math_helpers import rot_mat


def solve_mtq_controls_body_frame(Xset_interp, B_eci, dt, J, m_max=None):
    """
    Solve for MTQ controls given interpolated states and B-field in ECI.
    
    The MTQ torque equation is: τ = m × B
    We solve for m using: m = (B × τ) / |B|²
    (This gives the component of m perpendicular to B that produces τ)
    
    Parameters
    ----------
    Xset_interp : ndarray, shape (n_states, N)
        Interpolated states. First 3 are angular velocity, next 4 are quaternion.
    B_eci : ndarray, shape (3, N)
        B-field in ECI frame at each timestep
    dt : float
        Timestep (seconds)
    J : ndarray, shape (3, 3)
        Inertia tensor
    m_max : float, optional
        Maximum MTQ dipole moment. If provided, returns clamped controls.
        
    Returns
    -------
    Uset : ndarray, shape (3, N-1)
        MTQ control inputs (dipole moments)
    mtq_pct : float
        Maximum MTQ usage as percentage of limit (if m_max provided), else None
    """
    N = Xset_interp.shape[1]
    Uset = np.zeros((3, N-1))
    
    for k in range(N-1):
        # Angular velocities
        w_curr = Xset_interp[0:3, k]
        w_next = Xset_interp[0:3, k+1]
        
        # Quaternion to rotation matrix (body -> ECI)
        q = Xset_interp[3:7, k]
        R = rot_mat(q)
        
        # Transform B to body frame
        B_body = R.T @ B_eci[:, k]
        B_sq = np.dot(B_body, B_body)
        
        if B_sq > 1e-20:
            # Required angular acceleration
            w_dot = (w_next - w_curr) / dt
            
            # Required torque from Euler equation: J @ w_dot = τ - w × (J @ w)
            tau = J @ w_dot + np.cross(w_curr, J @ w_curr)
            
            # Solve for m: τ = m × B → m = (B × τ) / |B|²
            Uset[:, k] = np.cross(B_body, tau) / B_sq
    
    mtq_pct = np.abs(Uset).max() / m_max * 100 if m_max else None
    
    if m_max:
        Uset_clamped = np.clip(Uset, -m_max, m_max)
        return Uset_clamped, mtq_pct
    
    return Uset, mtq_pct


def solve_controls_from_trajectory(Xset_interp, B_eci, dt, J, rw_axes, 
                                    m_max=None, rw_torq_max=None):
    """
    Solve for MTQ + RW controls given interpolated states and B-field.
    
    For the desired angular acceleration, we split torque between MTQ and RW:
    - RW provides torque along its axis: τ_rw = -h_dot (reaction)
    - MTQ provides remaining torque perpendicular to B
    
    Parameters
    ----------
    Xset_interp : ndarray, shape (n_states, N)
        Interpolated states. [0:3]=ω, [3:7]=q, [7:7+n_rw]=RW momentum
    B_eci : ndarray, shape (3, N)
        B-field in ECI frame at each timestep
    dt : float
        Timestep (seconds)
    J : ndarray, shape (3, 3)
        Spacecraft inertia tensor (without RW)
    rw_axes : ndarray, shape (n_rw, 3)
        RW spin axes in body frame (unit vectors)
    m_max : float, optional
        Maximum MTQ dipole moment per axis
    rw_torq_max : float, optional
        Maximum RW torque
        
    Returns
    -------
    Uset : ndarray, shape (n_mtq + n_rw, N)
        Control inputs [MTQ moments (3), RW torques (n_rw)]
    """
    N = Xset_interp.shape[1]
    n_rw = rw_axes.shape[0] if rw_axes is not None and len(rw_axes) > 0 else 0
    n_mtq = 3
    n_u = n_mtq + n_rw
    
    Uset = np.zeros((n_u, N))
    
    for k in range(N-1):
        # Angular velocities
        w_curr = Xset_interp[0:3, k]
        w_next = Xset_interp[0:3, k+1]
        
        # Quaternion to rotation matrix (body -> ECI)
        q = Xset_interp[3:7, k]
        R = rot_mat(q)
        
        # Required angular acceleration
        w_dot = (w_next - w_curr) / dt
        
        # Required torque from Euler equation: J @ w_dot = τ_ext - w × (J @ w)
        # τ_ext = τ_mtq + τ_rw
        tau_needed = J @ w_dot + np.cross(w_curr, J @ w_curr)
        
        # RW contribution: τ_rw = -h_dot (along RW axes)
        tau_rw = np.zeros(3)
        if n_rw > 0:
            for i in range(n_rw):
                h_curr = Xset_interp[7+i, k]
                h_next = Xset_interp[7+i, k+1]
                h_dot = (h_next - h_curr) / dt
                
                # RW torque on spacecraft is -h_dot along axis
                rw_torque = -h_dot
                if rw_torq_max is not None:
                    rw_torque = np.clip(rw_torque, -rw_torq_max, rw_torq_max)
                
                Uset[n_mtq + i, k] = rw_torque
                tau_rw += rw_torque * rw_axes[i]
        
        # Remaining torque for MTQ
        tau_mtq_needed = tau_needed - tau_rw
        
        # Transform B to body frame
        B_body = R.T @ B_eci[:, k]
        B_sq = np.dot(B_body, B_body)
        
        if B_sq > 1e-20:
            # Solve for m: τ = m × B → m = (B × τ) / |B|²
            m = np.cross(B_body, tau_mtq_needed) / B_sq
            if m_max is not None:
                m = np.clip(m, -m_max, m_max)
            Uset[0:3, k] = m
    
    # Copy last control to final column
    Uset[:, -1] = Uset[:, -2] if N > 1 else 0
    
    return Uset


def _skew(vec):
    """Return skew-symmetric matrix such that skew(v) @ a = v x a."""
    vx, vy, vz = vec
    return np.array([
        [0.0, -vz,  vy],
        [vz,  0.0, -vx],
        [-vy, vx,  0.0],
    ])


def solve_controls_from_trajectory_regularized(
    Xset_interp,
    B_eci,
    dt,
    J,
    rw_axes,
    u_prior=None,
    reg_lambda: float = 1e-2,
    m_max=None,
    rw_torq_max=None
):
    """
    Solve for MTQ + RW controls with Tikhonov regularization toward a prior control.

    Minimizes: ||A u - tau_needed||^2 + reg_lambda * ||u - u_prior||^2

    This provides a best-effort torque match while staying close to a warm-start
    control (e.g., FOH-interpolated controls). Useful when direct inversion is
    ill-posed due to MTQ null space.
    """
    N = Xset_interp.shape[1]
    n_rw = rw_axes.shape[0] if rw_axes is not None and len(rw_axes) > 0 else 0
    n_mtq = 3
    n_u = n_mtq + n_rw

    if u_prior is None:
        u_prior = np.zeros((n_u, N))

    Uset = np.zeros((n_u, N))
    Iu = np.eye(n_u)

    for k in range(N - 1):
        # Angular velocities
        w_curr = Xset_interp[0:3, k]
        w_next = Xset_interp[0:3, k + 1]

        # Quaternion to rotation matrix (body -> ECI)
        q = Xset_interp[3:7, k]
        R = rot_mat(q)

        # Required angular acceleration and torque
        w_dot = (w_next - w_curr) / dt
        tau_needed = J @ w_dot + np.cross(w_curr, J @ w_curr)

        # Build linear map tau = A u
        B_body = R.T @ B_eci[:, k]
        A_mtq = -_skew(B_body)  # m x B = -skew(B) m
        if n_rw > 0:
            A_rw = rw_axes.T  # columns are axes
            A = np.hstack([A_mtq, A_rw])
        else:
            A = A_mtq

        u0 = u_prior[:, k] if u_prior is not None else np.zeros(n_u)
        lhs = A.T @ A + reg_lambda * Iu
        rhs = A.T @ tau_needed + reg_lambda * u0

        try:
            u = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            u = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

        # Clamp to actuator limits
        if m_max is not None:
            u[0:3] = np.clip(u[0:3], -m_max, m_max)
        if n_rw > 0 and rw_torq_max is not None:
            u[3:] = np.clip(u[3:], -rw_torq_max, rw_torq_max)

        Uset[:, k] = u

    # Copy last control to final column
    Uset[:, -1] = Uset[:, -2] if N > 1 else 0

    return Uset


def interpolate_trajectory_to_finer_grid(Xset_coarse, dt_coarse, dt_fine, tf, use_slerp=True):
    """
    Interpolate trajectory states from coarse to fine grid.
    
    Parameters
    ----------
    Xset_coarse : ndarray, shape (n_states, N_coarse)
        States on coarse grid
    dt_coarse : float
        Coarse timestep (seconds)
    dt_fine : float
        Fine timestep (seconds)
    tf : float
        Total time (seconds)
    use_slerp : bool, optional
        If True, use SLERP for quaternion interpolation (preserves shortest path).
        If False, use cubic interpolation with normalization (legacy behavior).
        Default True.
        
    Returns
    -------
    Xset_fine : ndarray, shape (n_states, N_fine)
        Interpolated states on fine grid (quaternions normalized)
    """
    N_coarse = Xset_coarse.shape[1]
    N_fine = int(tf / dt_fine) + 1
    
    t_coarse = np.linspace(0, tf, N_coarse)
    t_fine = np.linspace(0, tf, N_fine)
    
    n_states = Xset_coarse.shape[0]
    Xset_fine = np.zeros((n_states, N_fine))
    
    if use_slerp:
        # SLERP for quaternions, cubic for other states
        
        # First, ensure quaternion continuity (no sign flips)
        # This is critical for SLERP to take the short path
        quats_coarse = Xset_coarse[3:7, :].T.copy()  # (N_coarse, 4)
        for k in range(1, N_coarse):
            if np.dot(quats_coarse[k], quats_coarse[k-1]) < 0:
                quats_coarse[k] *= -1
        
        # Convert to scipy Rotation objects
        # scipy expects [x, y, z, w] but we use [w, x, y, z], so reorder
        quats_scipy = quats_coarse[:, [1, 2, 3, 0]]  # [w,x,y,z] -> [x,y,z,w]
        
        try:
            rotations = Rotation.from_quat(quats_scipy)
            slerp_interp = Slerp(t_coarse, rotations)
            rotations_fine = slerp_interp(t_fine)
            quats_fine_scipy = rotations_fine.as_quat()  # (N_fine, 4) in [x,y,z,w]
            # Convert back to [w, x, y, z]
            Xset_fine[3, :] = quats_fine_scipy[:, 3]  # w
            Xset_fine[4, :] = quats_fine_scipy[:, 0]  # x
            Xset_fine[5, :] = quats_fine_scipy[:, 1]  # y
            Xset_fine[6, :] = quats_fine_scipy[:, 2]  # z
        except Exception as e:
            # Fallback to cubic if SLERP fails
            print(f"SLERP failed ({e}), falling back to cubic interpolation")
            for i in range(3, 7):
                Xset_fine[i, :] = interp1d(t_coarse, Xset_coarse[i, :], 
                                           kind='cubic', fill_value='extrapolate')(t_fine)
            # Normalize
            for k in range(N_fine):
                q_norm = np.linalg.norm(Xset_fine[3:7, k])
                if q_norm > 1e-10:
                    Xset_fine[3:7, k] /= q_norm
        
        # Cubic interpolation for non-quaternion states (angular velocity, RW momentum, etc.)
        for i in list(range(0, 3)) + list(range(7, n_states)):
            Xset_fine[i, :] = interp1d(t_coarse, Xset_coarse[i, :], 
                                       kind='cubic', fill_value='extrapolate')(t_fine)
    else:
        # Legacy: cubic interpolation for all states
        for i in range(n_states):
            Xset_fine[i, :] = interp1d(t_coarse, Xset_coarse[i, :], 
                                       kind='cubic', fill_value='extrapolate')(t_fine)
        
        # Normalize quaternions (indices 3:7)
        for k in range(N_fine):
            q_norm = np.linalg.norm(Xset_fine[3:7, k])
            if q_norm > 1e-10:
                Xset_fine[3:7, k] /= q_norm
    
    return Xset_fine


def mtq_only_warm_start_transition(result_coarse, dt_coarse, dt_fine, tf, 
                                    vecsPy_fine, J, m_max, planner, t_start, t_end, x0):
    """
    Complete warm start transition for MTQ-only systems.
    
    This function handles the transition from coarse to fine grid optimization
    for MTQ-only systems, where simple control interpolation fails.
    
    Parameters
    ----------
    result_coarse : OptimizationResult
        Result from coarse grid optimization (must have .Xset attribute)
    dt_coarse : float
        Coarse timestep (seconds)
    dt_fine : float  
        Fine timestep (seconds)
    tf : float
        Total time (seconds)
    vecsPy_fine : tuple
        Environment vectors on fine grid. vecsPy_fine[3] is B-field in ECI.
    J : ndarray, shape (3, 3)
        Inertia tensor (without RW contributions for MTQ-only)
    m_max : float
        Maximum MTQ dipole moment
    planner : Planner
        Planner object with prepareForAlilqr and generateInitialTrajectory methods
    t_start, t_end : float
        Time bounds (in Julian centuries)
    x0 : ndarray
        Initial state
        
    Returns
    -------
    traj_warm : tuple
        (Xset, Uset) warm start trajectory for fine grid optimization
    vecs_fine : ndarray
        Environment vectors prepared for ALILQR
    info : dict
        Diagnostic info including:
        - mtq_pct: Maximum MTQ usage as percentage before clamping
        - warm_error_deg: Pointing error after forward propagation (if goal available)
        - N_fine: Number of timesteps on fine grid
    """
    # Step 1: Interpolate states to fine grid
    Xset_interp = interpolate_trajectory_to_finer_grid(
        result_coarse.Xset, dt_coarse, dt_fine, tf
    )
    
    # Step 2: Solve for controls in body frame
    B_eci = vecsPy_fine[3]
    Uset_clamped, mtq_pct = solve_mtq_controls_body_frame(
        Xset_interp, B_eci, dt_fine, J, m_max
    )
    
    # Step 3: Forward propagate with clamped controls
    _, vecs_fine, _ = planner.prepareForAlilqr(vecsPy_fine, dt_fine, t_start, t_end, x0, 0)
    traj_warm = planner.generateInitialTrajectory(dt_fine, x0, Uset_clamped, vecs_fine)
    
    # Compute warm start error for diagnostics
    goal_vec = vecsPy_fine[6][:, -1]  # Goal vector at final time
    goal_norm = np.linalg.norm(goal_vec)
    
    if goal_norm > 0.1:  # Has a valid goal
        goal_vec = goal_vec / goal_norm
        q_final = traj_warm[0][3:7, -1]
        R = Rotation.from_quat([q_final[1], q_final[2], q_final[3], q_final[0]]).as_matrix()
        boresight = np.array([0, 1, 0])  # Default boresight
        warm_error = np.degrees(np.arccos(np.clip(np.dot(R @ boresight, goal_vec), -1, 1)))
    else:
        warm_error = None
    
    info = {
        'mtq_pct': mtq_pct,
        'warm_error_deg': warm_error,
        'N_fine': Xset_interp.shape[1],
    }
    
    return traj_warm, vecs_fine, info


def get_mtq_only_pass2_cost_mods(base_cost, angle_scale=0.1):
    """
    Get cost weight modifications for MTQ-only Pass 2 optimization.
    
    Based on extensive testing, reducing the angle weight by ~10x for Pass 2
    allows constraints to drive the solution and achieves 90% success rate.
    
    Parameters
    ----------
    base_cost : list or tuple
        Base cost settings tuple from optSecondCostSettings()
    angle_scale : float, default=0.1
        Scale factor for angle weights (0.1 = reduce by 10x)
        
    Returns
    -------
    cost_mods : dict
        Dictionary of {index: new_value} modifications to apply to cost tuple
    """
    return {
        0: base_cost[0] * angle_scale,  # angle_weight
        5: base_cost[5] * angle_scale,  # angle_weight_N (terminal)
    }
