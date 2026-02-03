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


def create_slerp_initial_trajectory(x0, q_goal, N, dt, w_max=0.01):
    """
    Create initial trajectory using SLERP (spherical linear interpolation).
    
    This guarantees the shortest path on SO(3) and avoids 180° spikes.
    
    Parameters
    ----------
    x0 : array (7,)
        Initial state [w0, q0] where w0 is angular velocity and q0 is quaternion
    q_goal : array (4,)
        Goal quaternion
    N : int
        Number of trajectory points
    dt : float
        Timestep in seconds
    w_max : float
        Maximum angular velocity magnitude (rad/s)
        
    Returns
    -------
    Xset : array (7, N)
        State trajectory following SLERP path
    Uset : array (3, N-1)
        Zero controls (to be computed by optimizer)
    """
    w0 = x0[:3]
    q0 = x0[3:7]
    
    # Ensure q_goal is in same hemisphere as q0 (shortest path)
    if np.dot(q0, q_goal) < 0:
        q_goal = -q_goal
    
    # Create SLERP interpolator
    t_keyframes = np.array([0.0, (N-1) * dt])
    r0 = Rotation.from_quat([q0[1], q0[2], q0[3], q0[0]])  # scipy uses [x,y,z,w]
    r_goal = Rotation.from_quat([q_goal[1], q_goal[2], q_goal[3], q_goal[0]])
    key_rotations = Rotation.concatenate([r0, r_goal])
    slerp = Slerp(t_keyframes, key_rotations)
    
    # Sample along SLERP path
    t_samples = np.linspace(0, (N-1) * dt, N)
    rotations = slerp(t_samples)
    
    # Convert back to [w, x, y, z] quaternions
    quats_scipy = rotations.as_quat()  # [x, y, z, w]
    quats = np.zeros((N, 4))
    quats[:, 0] = quats_scipy[:, 3]  # w
    quats[:, 1:4] = quats_scipy[:, 0:3]  # x, y, z
    
    # Ensure quaternion continuity (no sign flips)
    for k in range(1, N):
        if np.dot(quats[k], quats[k-1]) < 0:
            quats[k] = -quats[k]
    
    # Compute angular velocities from quaternion differences
    # Simple finite difference: w = 2 * q_dot * q_conj
    Xset = np.zeros((7, N))
    Xset[3:7, :] = quats.T
    
    for k in range(N):
        if k == 0:
            Xset[0:3, k] = w0  # Use initial angular velocity
        else:
            # Approximate angular velocity from quaternion change
            q_prev = quats[k-1]
            q_curr = quats[k]
            # dq = q_curr * conj(q_prev)
            q_prev_conj = np.array([q_prev[0], -q_prev[1], -q_prev[2], -q_prev[3]])
            dq_w = q_curr[0]*q_prev_conj[0] - np.dot(q_curr[1:], q_prev_conj[1:])
            dq_v = q_curr[0]*q_prev_conj[1:] + q_prev_conj[0]*q_curr[1:] + np.cross(q_curr[1:], q_prev_conj[1:])
            # w = 2 * dq_v / dt (for small angles)
            w_approx = 2 * dq_v / dt
            # Clamp to reasonable magnitude
            w_mag = np.linalg.norm(w_approx)
            if w_mag > w_max:
                w_approx = w_approx * (w_max / w_mag)
            Xset[0:3, k] = w_approx
    
    # Zero controls (optimizer will compute them)
    Uset = np.zeros((3, N-1))
    
    return Xset, Uset


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


def kgain_warm_start_controls(
    Xset_coarse: np.ndarray,
    Uset_coarse: np.ndarray,
    Kset_coarse: np.ndarray,
    dt_coarse: float,
    dt_fine: float,
    tf: float,
    dynamics_func=None,
    use_slerp: bool = True,
    quat_to_3vec_mode: int = 2,
    verbose: bool = True,
) -> tuple:
    """
    Generate warm-start controls for fine grid using Pass1 K-gains with dynamics propagation.

    This propagates the dynamics at the fine timestep while using the coarse K-gains
    to provide feedback corrections:

        u_fine[k] = u_coarse[i] + K[i] @ (x_sim[k] - x_ref[k])
        x_sim[k+1] = dynamics(x_sim[k], u_fine[k])

    where:
        - i is the coarse index corresponding to fine index k
        - K[i] is the K-gain at coarse timestep i (NOT interpolated)
        - x_ref[k] is the SLERP-interpolated reference state from coarse trajectory
        - x_sim[k] is the actual simulated state

    This preserves the local optimality of Pass1 while adapting to the finer
    discretization through closed-loop simulation.

    Parameters
    ----------
    Xset_coarse : ndarray, shape (n_states, N_coarse)
        States from Pass1 optimization. Format: [ω(3), q(4), h(n_rw)]
    Uset_coarse : ndarray, shape (n_controls, N_coarse)
        Controls from Pass1 optimization
    Kset_coarse : ndarray, shape (N_coarse-1, n_controls, n_err) or (n_controls * n_err, N)
        Feedback gains from Pass1. n_err = 6 + n_rw (reduced error state dimension).
    dt_coarse : float
        Coarse timestep (seconds)
    dt_fine : float
        Fine timestep (seconds)
    tf : float
        Total trajectory time (seconds)
    dynamics_func : callable, optional
        Function dynamics_func(x, u, dt) -> x_next. If None, uses simple Euler integration
        with assumed rigid body + RW dynamics.
    use_slerp : bool, default=True
        Use SLERP for quaternion reference state interpolation
    quat_to_3vec_mode : int, default=2
        Quaternion to 3-vector conversion mode. Must match C++ planner setting.
        - 0: MRP with positive scalar (2*qv/(1+q0))
        - 1: MRP
        - 2: Cayley parameters (qv/q0) - default, matches planner
        - 3: Vector part with positive scalar
        - 4: Vector part
    verbose : bool, default=True
        Print diagnostic information during propagation

    Returns
    -------
    Xset_fine : ndarray, shape (n_states, N_fine)
        Simulated states on fine grid
    Uset_fine : ndarray, shape (n_controls, N_fine)
        K-gain corrected controls on fine grid
    """
    from ADCS.helpers.math_helpers import quat_diff, quat_to_vec3, normalize
    
    N_coarse = Xset_coarse.shape[1]
    N_fine = int(tf / dt_fine) + 1
    n_full_state = Xset_coarse.shape[0]  # 7 + n_rw
    n_controls = Uset_coarse.shape[0]
    n_rw = n_full_state - 7
    n_err = 6 + n_rw  # Reduced error state dimension
    
    # Get SLERP-interpolated reference trajectory
    Xset_ref = interpolate_trajectory_to_finer_grid(
        Xset_coarse, dt_coarse, dt_fine, tf, use_slerp=use_slerp
    )
    # Ensure correct length
    if Xset_ref.shape[1] < N_fine:
        pad = np.tile(Xset_ref[:, -1:], (1, N_fine - Xset_ref.shape[1]))
        Xset_ref = np.hstack([Xset_ref, pad])
    elif Xset_ref.shape[1] > N_fine:
        Xset_ref = Xset_ref[:, :N_fine]
    
    # Handle Kset dimensions
    # Expected: (N-1, n_controls, n_err) but may come in as (n_controls * n_err, N)
    if Kset_coarse.ndim == 2 and Kset_coarse.shape[0] == n_controls * n_err:
        # Flattened format: (n_controls * n_err, N) -> (N, n_controls, n_err)
        # Make a copy to avoid modifying the original array (reshape creates a view)
        N_K = Kset_coarse.shape[1]
        Kset_3d = Kset_coarse.T.reshape(N_K, n_controls, n_err).copy()
        if Kset_3d.shape[0] > N_coarse - 1:
            Kset_3d = Kset_3d[:N_coarse - 1]
    elif Kset_coarse.ndim == 2 and Kset_coarse.shape == (n_controls, n_err):
        # Single gain - tile
        Kset_3d = np.tile(Kset_coarse[np.newaxis, :, :], (N_coarse - 1, 1, 1))
    elif Kset_coarse.ndim == 3:
        Kset_3d = Kset_coarse
    else:
        print(f"Warning: Unknown Kset format {Kset_coarse.shape}, using zero gains")
        Kset_3d = np.zeros((N_coarse - 1, n_controls, n_err))
    
    # Pad Kset if needed
    if Kset_3d.shape[0] < N_coarse - 1:
        pad_count = (N_coarse - 1) - Kset_3d.shape[0]
        Kset_3d = np.concatenate([Kset_3d, np.tile(Kset_3d[-1:], (pad_count, 1, 1))], axis=0)
    
    # Scale RW/non-MTQ controls and K-gains from optimizer units to physical units
    # C++ optimizer uses scaled controls: u_opt = u_physical / NONMTQ_TORQ_SCALE
    # The flattened K format bypasses C++ output scaling (condition K_lqr.n_rows == sat.control_N() fails)
    # So both K and U from Pass1 are in optimizer units for RW - we scale to physical here
    NONMTQ_TORQ_SCALE = 3e-5
    n_mtq = n_controls - n_rw

    # Scale Uset_coarse RW rows to physical units (make a copy to avoid modifying original)
    Uset_coarse_phys = Uset_coarse.copy()
    if n_rw > 0:
        Uset_coarse_phys[n_mtq:, :] *= NONMTQ_TORQ_SCALE
        Kset_3d[:, n_mtq:, :] *= NONMTQ_TORQ_SCALE
    
    # Initialize output arrays
    Xset_fine = np.zeros((n_full_state, N_fine))
    Uset_fine = np.zeros((n_controls, N_fine))
    
    # Start from initial state
    x_sim = Xset_coarse[:, 0].copy()
    Xset_fine[:, 0] = x_sim
    
    # Simple Euler dynamics if none provided
    # Assumes state = [ω(3), q(4), h(n_rw)] and control = [m_mtq(3), τ_rw(n_rw)]
    def default_dynamics(x, u, dt):
        # Very simple integration - real dynamics would need J, B-field, etc.
        # For warm-start purposes, we mainly want the quaternion kinematics right
        w = x[0:3]
        q = x[3:7]
        h = x[7:7+n_rw] if n_rw > 0 else np.array([])
        
        # Quaternion kinematics: q_dot = 0.5 * q ⊗ [0, ω]
        w_quat = np.array([0, w[0], w[1], w[2]])
        q_dot = 0.5 * quat_mult_simple(q, w_quat)
        
        # Simple Euler integration
        x_next = x.copy()
        x_next[3:7] = normalize(q + q_dot * dt)
        # Angular velocity and momentum stay roughly constant (no torque model)
        
        return x_next
    
    def quat_mult_simple(q1, q2):
        """Simple quaternion multiplication [w, x, y, z]."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    # Check if dynamics_func takes timestep index
    import inspect
    if dynamics_func is not None:
        sig = inspect.signature(dynamics_func)
        dyn_takes_k = len(sig.parameters) >= 4
    else:
        dyn_takes_k = False
    
    def dyn(x, u, dt, k=0):
        if dynamics_func is not None:
            if dyn_takes_k:
                return dynamics_func(x, u, dt, k)
            else:
                return dynamics_func(x, u, dt)
        else:
            return default_dynamics(x, u, dt)
    
    # Propagate with K-gain feedback
    for k in range(N_fine - 1):
        # Find coarse index (ZOH - use the K at start of interval)
        i = min(int(k * dt_fine / dt_coarse), N_coarse - 2)
        
        # Get coarse control (ZOH) - use physical units
        u_nom = Uset_coarse_phys[:, i]
        
        # Get K-gain at this coarse index (no interpolation)
        K = Kset_3d[i]
        
        # Compute reduced error state: dx = x_sim - x_ref
        x_ref = Xset_ref[:, k]
        dx = np.zeros(n_err)
        
        # 1. Angular velocity error
        dx[0:3] = x_sim[0:3] - x_ref[0:3]
        
        # 2. Quaternion error (use same mode as C++ planner)
        q_err = quat_diff(x_ref[3:7], x_sim[3:7])
        dx[3:6] = quat_to_vec3(q_err, mode=quat_to_3vec_mode)
        
        # 3. RW momentum error
        if n_rw > 0:
            dx[6:6+n_rw] = x_sim[7:7+n_rw] - x_ref[7:7+n_rw]
        
        # Apply K-gain correction
        du = K @ dx

        # Scale K-gain correction by timestep ratio
        # K-gains were computed for dt_coarse, applying at dt_fine needs scaling
        dt_ratio = dt_fine / dt_coarse
        du = du * dt_ratio  # Smaller correction for smaller timestep

        u = u_nom + du

        # No clamping - let optimizer handle constraint violations
        Uset_fine[:, k] = u
        
        # Propagate dynamics
        x_sim = dyn(x_sim, u, dt_fine, k)
        x_sim[3:7] = normalize(x_sim[3:7])  # Ensure quaternion normalization
        Xset_fine[:, k + 1] = x_sim
        
        # Early exit on NaN
        if np.any(np.isnan(x_sim)):
            print(f"  kgain: NaN at k={k}, u_norm={np.linalg.norm(u):.2e}, dx_norm={np.linalg.norm(dx):.2e}", flush=True)
            break
        

    
    # Final control
    Uset_fine[:, -1] = Uset_fine[:, -2] if N_fine > 1 else Uset_coarse[:, -1]
    


    return Xset_fine, Uset_fine
