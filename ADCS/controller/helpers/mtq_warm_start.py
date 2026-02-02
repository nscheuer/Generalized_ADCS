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


def interpolate_trajectory_to_finer_grid(Xset_coarse, dt_coarse, dt_fine, tf):
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
        
    Returns
    -------
    Xset_fine : ndarray, shape (n_states, N_fine)
        Interpolated states on fine grid (quaternions normalized)
    """
    N_coarse = Xset_coarse.shape[1]
    N_fine = int(tf / dt_fine) + 1
    
    t_coarse = np.linspace(0, tf, N_coarse)
    t_fine = np.linspace(0, tf, N_fine)
    
    # Cubic interpolation for each state
    Xset_fine = np.array([
        interp1d(t_coarse, Xset_coarse[i, :], kind='cubic', fill_value='extrapolate')(t_fine)
        for i in range(Xset_coarse.shape[0])
    ])
    
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
