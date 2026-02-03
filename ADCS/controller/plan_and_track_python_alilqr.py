"""
Plan-and-Track controller using Python-driven ALILQR optimization.

This controller is identical to Plan_and_Track_Exact but uses the Python ALILQR
implementation that allows step-by-step inspection of the optimization process.

This is useful for:
- Debugging trajectory optimization issues
- Visualizing convergence in real-time
- Analyzing constraint handling behavior
- Research into optimization algorithm modifications
"""
from __future__ import annotations

__all__ = ["Plan_and_Track_PythonALILQR"]

import os
import numpy as np
from typing import Optional, Callable, List
from numpy.typing import NDArray

from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_base import PlanAndTrackBase
from ADCS.controller.helpers import (
    PlannerSettings, Trajectory, reorder_controls_cpp_to_python, 
    reorder_gains_cpp_to_python, PythonALILQR, PythonALILQRv2, 
    IterationData, OptimizationResult, LivePlannerViz
)
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite


class Plan_and_Track_PythonALILQR(PlanAndTrackBase):
    r"""
    Plan-and-Track controller using Python-driven ALILQR for transparent optimization.
    
    This controller extends PlanAndTrackBase to use the PythonALILQR optimizer
    instead of the C++ alilqr() method. This provides full visibility into
    the optimization process at each iteration.
    
    The controller can be configured with callbacks that are invoked after each
    inner iteration, allowing real-time visualization and analysis of:
    
    - Trajectory evolution (Xset, Uset)
    - Cost function convergence
    - Constraint satisfaction
    - Augmented Lagrangian penalty evolution
    - Regularization parameter changes
    
    Mathematical overview
    ---------------------
    The ALILQR algorithm solves:
    
    .. math::
    
        \min_{\mathbf{u}} \sum_{k=0}^{N-1} \ell(\mathbf{x}_k, \mathbf{u}_k) + \ell_N(\mathbf{x}_N)
        
    subject to dynamics and constraints, using an augmented Lagrangian formulation:
    
    .. math::
    
        \mathcal{L}(\mathbf{x}, \mathbf{u}, \boldsymbol{\lambda}, \mu) = 
        J(\mathbf{x}, \mathbf{u}) + \boldsymbol{\lambda}^T \mathbf{c}(\mathbf{x}, \mathbf{u}) 
        + \frac{\mu}{2} \|\mathbf{c}(\mathbf{x}, \mathbf{u})\|^2
    
    The outer loop updates :math:`\boldsymbol{\lambda}` and :math:`\mu`, while
    the inner loop (iLQR) solves unconstrained subproblems via backward/forward passes.
    
    Usage
    -----
    >>> controller = Plan_and_Track_PythonALILQR(est_sat, planner_settings)
    >>> 
    >>> # Set a callback for real-time analysis
    >>> def my_callback(iter_data):
    ...     print(f"Iteration {iter_data.inner_iter}: cost={iter_data.LA:.4e}")
    >>> controller.set_iteration_callback(my_callback)
    >>> 
    >>> # Plan trajectory
    >>> traj = controller.calculate_trajectory(t_start, duration, x_0, os_0, goals)
    >>> 
    >>> # Access optimization history
    >>> for iter_data in controller.last_optimization_result.iterations:
    ...     print(f"Outer {iter_data.outer_iter}, Inner {iter_data.inner_iter}: {iter_data.LA}")
    
    Parameters
    ----------
    est_sat : EstimatedSatellite
        Estimated satellite model
    planner_settings : PlannerSettings
        ALTRO planner configuration
    verbose : bool
        Enable verbose output from Python ALILQR
        
    Attributes
    ----------
    py_alilqr : PythonALILQR
        The Python ALILQR optimizer instance
    last_optimization_result : OptimizationResult
        Result from the most recent trajectory optimization
    iteration_callback : callable
        Optional callback invoked after each iteration
    """
    
    def __init__(
        self,
        est_sat: EstimatedSatellite,
        planner_settings: PlannerSettings,
        verbose: bool = False,
        use_v2: bool = True  # Use v2 implementation by default
    ) -> None:
        # Initialize base planner (creates self.planner)
        self._init_planner(est_sat, planner_settings, tracking_lqr_formulation=0, quat_to_3vec_mode=2)
        
        # Create Python ALILQR wrapper (v2 matches C++ exactly)
        if use_v2:
            self.py_alilqr = PythonALILQRv2(self.planner, verbose=verbose)
        else:
            self.py_alilqr = PythonALILQR(self.planner, verbose=verbose)
        
        # Storage for results and callbacks
        self.last_optimization_result: Optional[OptimizationResult] = None
        self.iteration_callback: Optional[Callable[[IterationData], None]] = None
        self._verbose = verbose
        
    def set_iteration_callback(self, callback: Optional[Callable[[IterationData], None]]) -> None:
        """
        Set a callback function to be invoked after each optimization iteration.
        
        Parameters
        ----------
        callback : callable
            Function with signature callback(iter_data: IterationData) -> None
            Called after each inner iteration with full iteration state
        """
        self.iteration_callback = callback
        # Works with both v1 and v2
        if hasattr(self.py_alilqr, 'set_callback'):
            self.py_alilqr.set_callback(callback)
        else:
            self.py_alilqr.debug_callback = callback
    
    def _scale_rw_gains(self, Kset: np.ndarray) -> np.ndarray:
        """
        Scale RW/magic rows of K-gains from optimizer units to physical units.
        
        The optimizer uses scaled controls, so K-gains need the same scaling:
            K_physical[rw_rows, :] = K_scaled[rw_rows, :] * NONMTQ_TORQ_SCALE
        """
        scale = self.planner.get_nonmtq_torq_scale()
        if scale == 1.0:
            return Kset
        
        n_mtq = self.planner.get_number_MTQ()
        n_rw = self.planner.get_number_RW()
        n_magic = self.planner.get_number_magic()
        
        if n_rw + n_magic == 0:
            return Kset
        
        Kset_scaled = Kset.copy()
        rw_magic_start = n_mtq
        rw_magic_end = n_mtq + n_rw + n_magic
        
        if Kset.ndim == 2:
            # Flattened format: (n_ctrl * n_err, N)
            n_ctrl = n_mtq + n_rw + n_magic
            n_err = Kset.shape[0] // n_ctrl
            for ctrl_idx in range(rw_magic_start, rw_magic_end):
                row_start = ctrl_idx * n_err
                row_end = (ctrl_idx + 1) * n_err
                Kset_scaled[row_start:row_end, :] *= scale
        elif Kset.ndim == 3:
            if Kset.shape[0] > Kset.shape[2]:
                # (N, n_ctrl, n_err)
                Kset_scaled[:, rw_magic_start:rw_magic_end, :] *= scale
            else:
                # (n_ctrl, n_err, N)
                Kset_scaled[rw_magic_start:rw_magic_end, :, :] *= scale
        
        return Kset_scaled
        
    def find_u(
        self,
        x_hat: NDArray[np.float64],
        sens: NDArray[np.float64],
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        goal_vector_eci: Optional[NDArray[np.float64]] = None,
        w_ref: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        """Compute TVLQR tracking control: u = u_ref - K @ dx."""
        current_time = os_hat.J2000
        
        if self.active_trajectory is None:
            raise RuntimeError("No active trajectory set")
            
        if not self.active_trajectory.is_valid_time(current_time):
            raise RuntimeError(f"Active trajectory expired at t={current_time}")
        
        # Get trajectory data
        x_ref = self.active_trajectory.get_state_at(current_time)
        u_ref = self.active_trajectory.get_control_at(current_time)
        K = self.active_trajectory.get_gain_at(current_time)
        
        # Compute state error using trajectory's state_diff (handles quaternion error)
        dx = self.active_trajectory._state_diff(x_hat, x_ref)
        
        # Apply TVLQR feedback: u = u_ref + K @ dx
        # Note: Sign is + because Python ALILQR K-gains have opposite convention from standard LQR
        u = u_ref + K @ dx
        
        # Saturate control to actuator limits
        for i, act in enumerate(self.est_sat.actuators):
            u[i] = np.clip(u[i], -act.u_max, act.u_max)
        
        return u
    
    def calculate_trajectory(
        self,
        t_start: float,
        duration: float,
        x_0: np.ndarray,
        os_0: Orbital_State,
        goals: GoalList,
        verbose: bool = False,
        collect_all_iterations: bool = True,
        visualize: bool = False,
        viz_save_path: Optional[str] = None,
        skip_pass2: bool = False
    ) -> Trajectory:
        """
        Plan a trajectory using Python-driven ALILQR.
        
        This method is similar to the base class but uses PythonALILQR instead
        of the C++ alilqr() method, providing full iteration-level visibility.
        
        Parameters
        ----------
        t_start : float
            Start time in J2000 centuries
        duration : float
            Planning horizon in seconds
        x_0 : np.ndarray
            Initial state
        os_0 : Orbital_State
            Initial orbital state
        goals : GoalList
            Pointing goals
        verbose : bool
            Override verbosity for this call
        collect_all_iterations : bool
            If True, store all iteration data (uses more memory)
        visualize : bool
            If True, show live visualization of optimization convergence
        viz_save_path : str, optional
            If provided, save final visualization to this path
            
        Returns
        -------
        Trajectory
            Planned trajectory with times, states, controls, gains
        """
        if verbose:
            print(f"Planning trajectory: t_start={t_start:.5f}, duration={duration}s")
        
        self.planner.setVerbosity(verbose or self._verbose)
        
        # Get timesteps (like C++ trajOpt)
        dt_coarse = self.planner_settings.dt_tp      # Pass 1: coarse planning
        dt_fine = self.planner_settings.dt_tvlqr    # Pass 2: fine refinement
        
        t_end = t_start + (duration * TimeConstants.sec2cent)
        x_0_clean = np.copy(x_0.astype(np.float64).flatten(), order='C')
        
        # =====================================================================
        # Setup visualization if requested
        # =====================================================================
        live_viz = None
        original_callback = self.iteration_callback
        
        if visualize:
            # Propagate environment to get goal vectors for visualization
            N_viz = int(np.ceil(duration / dt_coarse)) + 1
            vecsPy_viz = self._propagate_environment(os_0, t_start, t_end, dt_coarse, N_viz, goals)
            # vecsPy returns: (t, R, V, B, S, A, E, p, rho)
            #                  0  1  2  3  4  5  6  7  8
            goal_vectors = vecsPy_viz[6]  # E vectors shape (3, N) or (4, N) for quaternion
            
            # Generate actuator names for plot labels
            actuator_names = []
            for act in self.est_sat.mtq_actuators:
                actuator_names.append(f'MTQ_{["x","y","z"][np.argmax(np.abs(act.axis))]}')
            for i, act in enumerate(self.est_sat.rw_actuators):
                actuator_names.append(f'RW{i+1}')
            
            # Create live visualization
            # goal_vectors is (3,N) for pointing or (4,N) for quaternion goal
            live_viz = LivePlannerViz(
                goal_vector_eci=goal_vectors,  # Works for both 3D vectors and 4D quaternions
                body_vector=np.array([0, 1, 0]),  # Boresight (used only for vector goals)
                dt=dt_coarse,
                update_interval=1,
                figsize=(14, 10),
                actuator_names=actuator_names,
                umax=self.planner_settings.umax
            )
            live_viz.start()
            
            # Create callback that updates visualization
            def viz_callback(iter_data: IterationData):
                if original_callback is not None:
                    original_callback(iter_data)
                live_viz.update(iter_data)
            
            self.set_iteration_callback(viz_callback)
        
        # =====================================================================
        # PASS 1: Coarse trajectory optimization (exploration)
        # =====================================================================
        if verbose:
            print(f"=== PASS 1: Exploration (dt={dt_coarse}s) ===")
        
        # Propagate environment at coarse timestep
        N_coarse = int(np.ceil(duration / dt_coarse)) + 1
        vecsPy_coarse = self._propagate_environment(os_0, t_start, t_end, dt_coarse, N_coarse, goals)
        
        # Get pass1 settings
        cost_settings_1 = self.planner_settings.optMainCostSettings()
        alilqr_settings_1 = self.planner_settings.mainAlilqrSettings()
        
        # Create initial trajectory at coarse timestep
        # Multi-start: try multiple bdot modes and pick best initial trajectory
        import os
        multistart_modes = [0, 1, 4, 5] if os.environ.get("PY_ALILQR_MULTISTART", "0") == "1" else [self.planner_settings.bdot_on]
        
        best_traj = None
        best_vecs = None
        best_max_spike = float('inf')
        best_mode = None
        
        for bdotOn in multistart_modes:
            initial_result = self.planner.prepareForAlilqr(
                vecsPy_coarse, dt_coarse, t_start, t_end, x_0_clean, int(bdotOn)
            )
            traj_candidate, vecs_candidate, _ = initial_result
            
            # Evaluate spike for this candidate
            E_goals_candidate = vecs_candidate[6]
            if E_goals_candidate.shape[0] == 4:
                Xset_cand, _, _, _ = traj_candidate
                q_goal_cand = E_goals_candidate[:, -1]
                q_goal_inv = np.array([q_goal_cand[0], -q_goal_cand[1], -q_goal_cand[2], -q_goal_cand[3]])
                max_spike = 0.0
                for k in range(Xset_cand.shape[1]):
                    qk = Xset_cand[3:7, k]
                    qerr_w = q_goal_inv[0]*qk[0] - np.dot(q_goal_inv[1:], qk[1:])
                    angle = np.degrees(2 * np.arccos(np.clip(np.abs(qerr_w), 0, 1)))
                    if angle > max_spike:
                        max_spike = angle
            else:
                max_spike = 0.0  # Non-quaternion goal, can't evaluate spike
            
            if len(multistart_modes) > 1:
                print(f"  [Multistart] bdot={bdotOn}: max_spike={max_spike:.0f}°")
            
            if max_spike < best_max_spike:
                best_max_spike = max_spike
                best_traj = traj_candidate
                best_vecs = vecs_candidate
                best_mode = bdotOn
        
        initial_traj_1 = best_traj
        vecs_dt_coarse = best_vecs
        
        if len(multistart_modes) > 1:
            print(f"  [Multistart] Selected bdot={best_mode} with spike={best_max_spike:.0f}°")
        
        # SLERP fallback: DISABLED - zero controls cause dynamics to diverge
        # if best_max_spike > 170 and os.environ.get("PY_ALILQR_SLERP_FALLBACK", "1") == "1":
        #     ...
        
        # Check for zero-control initialization (env var: PY_ALILQR_ZERO_INIT=1)
        if os.environ.get("PY_ALILQR_ZERO_INIT", "0") == "1":
            Xset, Uset, Tset, TQset = initial_traj_1
            Uset = np.zeros_like(Uset)
            initial_traj_1 = (Xset, Uset, Tset, TQset)
            print("  [Zero-init] Zeroed out initial controls")
        
        # Quaternion hemisphere check: flip goal quaternion if on wrong side
        # This prevents the optimizer from going "the long way" (>180°)
        # vecs_dt_coarse[6] is E (goal quaternion), shape (4, N) for quat goals
        E_goals = vecs_dt_coarse[6]
        if E_goals.shape[0] == 4:  # Quaternion goal
            q0 = x_0_clean[3:7]  # Initial quaternion from state
            q_goal = E_goals[:, 0]  # Goal quaternion at t=0
            
            # Check dot product - if negative, they're on opposite hemispheres
            dot = np.dot(q0, q_goal)
            if dot < 0:
                # Flip the goal quaternion for the entire trajectory
                vecs_dt_coarse = list(vecs_dt_coarse)
                vecs_dt_coarse[6] = -E_goals  # Flip sign
                vecs_dt_coarse = tuple(vecs_dt_coarse)
                print(f"  [Quat-flip] Flipped goal quaternion (dot={dot:.3f} < 0)")
        
        # Check initial trajectory for 180° spike and report
        Xset_init, _, _, _ = initial_traj_1
        if E_goals.shape[0] == 4:
            q_goal_final = vecs_dt_coarse[6][:, -1]  # Use potentially flipped goal
            q_goal_inv = np.array([q_goal_final[0], -q_goal_final[1], -q_goal_final[2], -q_goal_final[3]])
            angles_init = np.zeros(Xset_init.shape[1])
            for k in range(Xset_init.shape[1]):
                qk = Xset_init[3:7, k]
                qerr_w = q_goal_inv[0]*qk[0] - np.dot(q_goal_inv[1:], qk[1:])
                angles_init[k] = np.degrees(2 * np.arccos(np.clip(np.abs(qerr_w), 0, 1)))
            max_angle = np.max(angles_init)
            max_idx = np.argmax(angles_init)
            print(f"  [Init-traj] Angles: start={angles_init[0]:.0f}° max={max_angle:.0f}°@{max_idx} end={angles_init[-1]:.0f}°")
        
        result1 = self.py_alilqr.optimize(
            dt=dt_coarse,
            initial_traj=initial_traj_1,
            vecs=vecs_dt_coarse,
            cost_settings=cost_settings_1,
            alilqr_settings=alilqr_settings_1,
            is_first_search=True,
            collect_all=collect_all_iterations,
            pass_label="Pass1"
        )
        
        if verbose:
            print(f"Pass 1 complete: {result1.total_inner_iters} iterations")
            print(f"  Final cost: {result1.final_cost:.6e}")
            print(f"  Final cmax: {result1.final_cmax:.6e}")
        
        # Store pass1 result
        self.pass1_result = result1
        
        # =====================================================================
        # PASS 2: Fine trajectory refinement (strict constraint enforcement)
        # Interpolate to fine timestep and run with high penalty
        # =====================================================================
        if skip_pass2:
            if verbose:
                print(f"\n=== SKIPPING PASS 2 (using Pass1 result directly) ===")
            # Use Pass1 result directly - just compute gains
            result = result1
            self.pass2_result = None
            self.last_optimization_result = result
            
            # Reorder controls and create trajectory
            Uset = reorder_controls_cpp_to_python(result.Uset, self.est_sat.actuators)
            Kset = reorder_gains_cpp_to_python(result.Kset, self.est_sat.actuators)
            
            N_result = result.Xset.shape[1]
            Sset = np.zeros((1, N_result))
            
            if live_viz is not None:
                if viz_save_path:
                    live_viz.save(viz_save_path)
                live_viz.finish(block=False)
                self.set_iteration_callback(original_callback)
            
            return Trajectory(result.times, result.Xset, Uset, Kset, Sset)
        
        if verbose:
            print(f"\n=== PASS 2: Refinement (dt={dt_fine}s, penalty_init={self.planner_settings.pass2.aug_lag.penalty_init}) ===", flush=True)
        
        # Propagate environment at fine timestep
        N_fine = int(np.ceil(duration / dt_fine)) + 1
        vecsPy_fine = self._propagate_environment(os_0, t_start, t_end, dt_fine, N_fine, goals)
        
        # Get pass2 settings (higher penalty for strict constraints)
        cost_settings_2 = self.planner_settings.optSecondCostSettings()
        alilqr_settings_2 = self.planner_settings.secondAlilqrSettings()
        
        # Get TVLQR-specific cost settings for gain computation
        # These typically have higher control costs to produce smaller, more practical gains
        tvlqr_cost_settings = self.planner_settings.optTVLQRCostSettings(tracking_LQR_formulation=0)
        
        # Interpolate coarse trajectory to fine timestep using ZOH (zero-order hold)
        # FOH caused discretization issues leading to divergence in Pass2
        interp_ratio = int(dt_coarse / dt_fine)
        if interp_ratio > 1:
            Uset_coarse = result1.Uset
            # ZOH: repeat each control for interp_ratio timesteps
            if Uset_coarse.shape[1] >= 3:
                Uset_main = np.repeat(Uset_coarse[:, :-2], interp_ratio, axis=1)
            else:
                Uset_main = np.repeat(Uset_coarse[:, :-1], interp_ratio, axis=1)
            cols_needed = N_fine - Uset_main.shape[1] - 1
            if cols_needed > 0:
                Uset_pad = np.tile(Uset_coarse[:, -2:-1], (1, cols_needed))
                Uset_fine = np.hstack([Uset_main, Uset_pad, Uset_coarse[:, -1:]])
            else:
                Uset_fine = np.hstack([Uset_main[:, :N_fine-1], Uset_coarse[:, -1:]])
            if verbose:
                print(f"  ZOH interpolated controls: {Uset_coarse.shape[1]} -> {Uset_fine.shape[1]}")
        else:
            Uset_fine = result1.Uset
        
        # Prepare vecs at fine timestep
        initial_result_2 = self.planner.prepareForAlilqr(
            vecsPy_fine, dt_fine, t_start, t_end, x_0_clean, 0  # bdot=0
        )
        _, vecs_dt_fine, _ = initial_result_2
        
        # K-gain warm-start: use Pass1 feedback gains to propagate trajectory to fine grid
        # This preserves the Pass1 solution quality by using closed-loop dynamics

        traj_fine = None
        if interp_ratio > 1:
            from ADCS.controller.helpers.mtq_warm_start import kgain_warm_start_controls

            try:
                Kset_coarse = result1.Kset
                

                
                # vecs_dt_fine indices: 0=t, 1=R, 2=V, 3=B, 4=S, 5=satvec, 6=ECIvec, 7=p, 8=rho
                B_eci_fine = vecs_dt_fine[3]  # B-field in ECI: (3, N_fine)
                R_eci_fine = vecs_dt_fine[1]  # Position in ECI: (3, N_fine)
                V_eci_fine = vecs_dt_fine[2]  # Velocity in ECI: (3, N_fine)
                S_eci_fine = vecs_dt_fine[4]  # Sun vector: (3, N_fine)
                rho_fine = vecs_dt_fine[8]    # Density: (N_fine,)
                
                def sat_dynamics(x, u, dt, k):
                    """RK4 integration of satellite dynamics for one timestep."""
                    k_idx = min(k, B_eci_fine.shape[1] - 1)
                    
                    # Create minimal orbital state with required fields
                    class MinimalOS:
                        def __init__(self, B, R, V, S, rho=0.0):
                            self.B = B
                            self.R = R
                            self.V = V
                            self.sun_vec = S
                            self.S = S  # alias
                            self.rho = rho
                        def get_state_vector(self, x):
                            from ADCS.helpers.math_helpers import rot_mat
                            q = x[3:7]
                            R_mat_T = rot_mat(q).T
                            r_body = R_mat_T @ self.R
                            v_body = R_mat_T @ self.V
                            b_body = R_mat_T @ self.B
                            s_body = R_mat_T @ self.sun_vec
                            return {
                                "r": r_body, "v": v_body, 
                                "b": b_body, "s": s_body,
                                "rho": self.rho
                            }
                        def is_sunlit(self):
                            return np.linalg.norm(self.sun_vec) > 0.1
                    
                    rho_k = rho_fine[k_idx] if k_idx < len(rho_fine) else rho_fine[-1]
                    os_k = MinimalOS(
                        B_eci_fine[:, k_idx],
                        R_eci_fine[:, k_idx],
                        V_eci_fine[:, k_idx],
                        S_eci_fine[:, k_idx] if k_idx < S_eci_fine.shape[1] else S_eci_fine[:, -1],
                        rho=rho_k
                    )
                    
                    # RK4 integration (no bias/noise for planner)
                    from ADCS.satellite_hardware.errors import ErrorMode
                    dmode_clean = ErrorMode(add_bias=False, add_noise=False, 
                                            update_bias=False, update_noise=False)
                    
                    def f(x_):
                        return self.est_sat.dynamics_core(x_, u, os_k, dmode=dmode_clean)
                    
                    k1 = f(x)
                    k2 = f(x + 0.5 * dt * k1)
                    k3 = f(x + 0.5 * dt * k2)
                    k4 = f(x + dt * k3)
                    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                    
                    return x_next
                
                Xset_kgain, Uset_kgain = kgain_warm_start_controls(
                    result1.Xset, result1.Uset, Kset_coarse,
                    dt_coarse, dt_fine, duration,
                    dynamics_func=sat_dynamics,
                    use_slerp=True,
                    quat_to_3vec_mode=self.quat_to_3vec_mode,
                    verbose=verbose
                )
                
                # Ensure correct length
                if Uset_kgain.shape[1] != N_fine:
                    if Uset_kgain.shape[1] < N_fine:
                        pad = np.tile(Uset_kgain[:, -1:], (1, N_fine - Uset_kgain.shape[1]))
                        Uset_kgain = np.hstack([Uset_kgain, pad])
                    else:
                        Uset_kgain = Uset_kgain[:, :N_fine]
                if Xset_kgain.shape[1] != N_fine:
                    if Xset_kgain.shape[1] < N_fine:
                        pad = np.tile(Xset_kgain[:, -1:], (1, N_fine - Xset_kgain.shape[1]))
                        Xset_kgain = np.hstack([Xset_kgain, pad])
                    else:
                        Xset_kgain = Xset_kgain[:, :N_fine]
                

                
                # Scale RW controls BACK to optimizer units for C++
                NONMTQ_TORQ_SCALE = 3e-5
                n_rw_acts = len(self.est_sat.rw_actuators)
                n_mtq_acts = len(self.est_sat.mtq_actuators)
                Uset_kgain_opt = Uset_kgain.copy()
                if n_rw_acts > 0:
                    Uset_kgain_opt[n_mtq_acts:n_mtq_acts+n_rw_acts, :] /= NONMTQ_TORQ_SCALE
                
                # Build trajectory tuple for Pass2
                # CRITICAL: times must be in J2000 centuries to match trajectory.get_control_at() queries
                times_fine = np.linspace(t_start, t_end, N_fine)
                TQset_fine = np.zeros((3, N_fine))
                # CRITICAL: C++ expects Fortran-order arrays (column-major)
                traj_fine = (
                    np.asfortranarray(Xset_kgain, dtype=np.float64),
                    np.asfortranarray(Uset_kgain_opt, dtype=np.float64),
                    np.asfortranarray(times_fine, dtype=np.float64),
                    np.asfortranarray(TQset_fine, dtype=np.float64)
                )

            except Exception as e:
                import traceback
                if verbose:
                    print(f"  ERROR in K-gain warm start: {e}")
                    traceback.print_exc()
                traj_fine = None  # Fall back to ZOH below

        # Fallback: ZOH interpolation (also used when interp_ratio==1)
        if traj_fine is None:
            if verbose:
                print(f"  Using ZOH warm start (fallback)")
            try:
                traj_fine = self.planner.generateInitialTrajectory(
                    dt_fine, result1.Xset[:, 0].copy(), Uset_fine, vecs_dt_fine
                )
                # Check for NaN
                Xset_check, _, _, _ = traj_fine
                if np.any(np.isnan(Xset_check)) or np.any(np.isinf(Xset_check)):
                    if verbose:
                        print("  WARNING: Warm start produced NaN, using fresh init")
                    traj_fine, _, _ = initial_result_2
            except Exception as e:
                if verbose:
                    print(f"  WARNING: generateInitialTrajectory failed ({e}), using fresh init")
                traj_fine, _, _ = initial_result_2
        
        result2 = self.py_alilqr.optimize(
            dt=dt_fine,
            initial_traj=traj_fine,
            vecs=vecs_dt_fine,
            cost_settings=cost_settings_2,
            alilqr_settings=alilqr_settings_2,
            is_first_search=False,
            collect_all=collect_all_iterations,
            pass_label="Pass2",
            tvlqr_cost_settings=tvlqr_cost_settings
        )
        
        # Combine results
        result = result2
        result.iterations = result1.iterations + result2.iterations
        
        # Store results for analysis
        self.last_optimization_result = result
        self.pass2_result = result2
        
        if verbose:
            print(f"Pass 2 complete: {result2.total_inner_iters} iterations")
            print(f"  Final cost: {result2.final_cost:.6e}")
            print(f"  Final cmax: {result2.final_cmax:.6e}")
            print(f"\nTotal iterations: {result1.total_inner_iters + result2.total_inner_iters}")
        
        # Reorder controls and create trajectory
        Uset = reorder_controls_cpp_to_python(result.Uset, self.est_sat.actuators)
        Kset = reorder_gains_cpp_to_python(result.Kset, self.est_sat.actuators)
        
        # Scale RW K-gains from optimizer units to physical units
        # (same scaling as controls: K_physical = K_opt * NONMTQ_TORQ_SCALE)
        Kset = self._scale_rw_gains(Kset)
        
        # Create dummy Sset (not computed by Python ALILQR directly)
        N_result = result.Xset.shape[1]
        Sset = np.zeros((1, N_result))
        
        # =====================================================================
        # Cleanup visualization
        # =====================================================================
        if live_viz is not None:
            if viz_save_path:
                live_viz.save(viz_save_path)
                if verbose:
                    print(f"Saved visualization to: {viz_save_path}")
            live_viz.finish(block=False)
            live_viz.close()  # Close the figure window
            # Restore original callback
            self.set_iteration_callback(original_callback)
        
        return Trajectory(result.times, result.Xset, Uset, Kset, Sset)
    
    def calculate_trajectory_step_by_step(
        self,
        t_start: float,
        duration: float,
        x_0: np.ndarray,
        os_0: Orbital_State,
        goals: GoalList
    ):
        """
        Generator that yields iteration data during optimization.
        
        This allows external control of the optimization loop and real-time
        visualization. Iteration can be stopped early by breaking from the loop.
        
        Yields
        ------
        IterationData
            Complete state after each inner iteration
            
        Example
        -------
        >>> for iter_data in controller.calculate_trajectory_step_by_step(...):
        ...     plot_trajectory(iter_data.Xset)
        ...     if iter_data.cmax < 1e-6:
        ...         break  # Stop when constraints satisfied
        """
        self.planner.setVerbosity(False)
        
        dt_seconds = self.planner_settings.dt_tvlqr
        N = int(np.ceil(duration / dt_seconds)) + 1
        t_end = t_start + (duration * TimeConstants.sec2cent)
        
        vecsPy = self._propagate_environment(os_0, t_start, t_end, dt_seconds, N, goals)
        
        cost_settings = self.planner_settings.optMainCostSettings()
        alilqr_settings = self.planner_settings.mainAlilqrSettings()
        
        x_0_clean = np.copy(x_0.astype(np.float64).flatten(), order='C')
        
        bdotOn = self.planner_settings.bdot_on
        initial_result = self.planner.prepareForAlilqr(
            vecsPy, dt_seconds, t_start, t_end, x_0_clean, int(bdotOn)
        )
        initial_traj, vecs_dt, _ = initial_result
        
        yield from self.py_alilqr.optimize_step_by_step(
            dt=dt_seconds,
            initial_traj=initial_traj,
            vecs=vecs_dt,
            cost_settings=cost_settings,
            alilqr_settings=alilqr_settings,
            is_first_search=True
        )
    
    def get_iteration_history(self) -> List[IterationData]:
        """
        Get the iteration history from the last optimization.
        
        Returns
        -------
        list of IterationData
            All recorded iterations from last trajectory calculation
        """
        if self.last_optimization_result is None:
            return []
        return self.last_optimization_result.iterations
    
    def analyze_convergence(self) -> dict:
        """
        Analyze convergence of the last optimization.
        
        Returns
        -------
        dict
            Dictionary with convergence statistics:
            - 'costs': array of cost values
            - 'cmaxes': array of max constraint violations
            - 'grads': array of gradient norms
            - 'outer_iters': number of outer iterations
            - 'inner_iters': number of inner iterations per outer
        """
        if self.last_optimization_result is None:
            return {}
        
        iters = self.last_optimization_result.iterations
        if not iters:
            return {}
        
        costs = np.array([it.LA for it in iters])
        cmaxes = np.array([it.cmax for it in iters])
        grads = np.array([it.grad for it in iters])
        
        # Count inner iterations per outer
        outer_iters = {}
        for it in iters:
            if it.outer_iter not in outer_iters:
                outer_iters[it.outer_iter] = 0
            outer_iters[it.outer_iter] += 1
        
        return {
            'costs': costs,
            'cmaxes': cmaxes,
            'grads': grads,
            'outer_iters': len(outer_iters),
            'inner_iters_per_outer': list(outer_iters.values()),
            'total_iterations': len(iters),
            'final_cost': self.last_optimization_result.final_cost,
            'final_cmax': self.last_optimization_result.final_cmax,
            'success': self.last_optimization_result.success
        }
