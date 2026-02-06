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
    
    @staticmethod
    def _generate_slerp_trajectory(dt, x0, q_goal, N, times):
        """Generate SLERP trajectory from x0 toward q_goal."""
        from ADCS.helpers.math_helpers import normalize
        n_state = len(x0)
        n_ctrl = 3 + (n_state - 7)  # MTQ + RW
        Xset = np.zeros((n_state, N))
        Uset = np.zeros((n_ctrl, N))
        TQset = np.zeros((3, N))
        
        q0 = normalize(x0[3:7])
        qg = normalize(q_goal)
        if np.dot(q0, qg) < 0:
            qg = -qg
        dot_val = np.clip(np.abs(np.dot(q0, qg)), 0, 1)
        theta = np.arccos(dot_val)
        sin_theta = np.sin(theta)
        
        for k in range(N):
            frac = k / (N - 1) if N > 1 else 0
            if theta < 1e-10:
                qk = q0.copy()
            else:
                qk = normalize((np.sin((1-frac)*theta) * q0 + np.sin(frac*theta) * qg) / sin_theta)
            Xset[3:7, k] = qk
            if n_state > 7:
                Xset[7:, k] = x0[7:]  # RW momentum constant
        
        # Angular velocity from finite differences
        for k in range(N - 1):
            qk = normalize(Xset[3:7, k])
            qkp1 = normalize(Xset[3:7, k+1])
            # q_err = qk^-1 * qkp1
            qe_w = qk[0]*qkp1[0] + np.dot(qk[1:], qkp1[1:])
            qe_v = qk[0]*qkp1[1:] - qkp1[0]*qk[1:] - np.cross(qk[1:], qkp1[1:])
            if qe_w < 0:
                qe_w, qe_v = -qe_w, -qe_v
            half_angle = np.arccos(np.clip(np.abs(qe_w), 0, 1))
            if half_angle > 1e-12:
                axis = qe_v / np.linalg.norm(qe_v)
                Xset[0:3, k] = (2 * half_angle / dt) * axis
        
        return (Xset, Uset, times[:N] if len(times) >= N else times, TQset)
    
    @staticmethod
    def _save_trajectory_snapshot(Xset, Uset, dt, duration, filepath, title):
        """Save a diagnostic 4-panel figure of the trajectory state.

        Uses the Agg canvas directly to avoid calling matplotlib.use('Agg'),
        which would permanently switch the backend and break interactive plots.
        """
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        N = Xset.shape[1]
        times = np.arange(N) * dt
        if times[-1] > 0:
            times = times * (duration / times[-1])

        fig = Figure(figsize=(12, 8))
        FigureCanvasAgg(fig)
        axes = fig.subplots(2, 2)

        # Quaternion
        for i, lbl in enumerate(['q0', 'q1', 'q2', 'q3']):
            axes[0, 0].plot(times, Xset[3+i, :], label=lbl)
        axes[0, 0].set_title(f'{title} - Quaternion')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Angular velocity
        for i, lbl in enumerate(['ωx', 'ωy', 'ωz']):
            axes[0, 1].plot(times, np.degrees(Xset[i, :]), label=lbl)
        axes[0, 1].set_title('Angular Velocity (deg/s)')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Controls
        ctrl_times = np.arange(Uset.shape[1]) * dt
        if ctrl_times[-1] > 0:
            ctrl_times = ctrl_times * (duration / ctrl_times[-1])
        for i in range(Uset.shape[0]):
            axes[1, 0].plot(ctrl_times, Uset[i, :], label=f'u{i}')
        axes[1, 0].set_title('Controls')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # RW momentum (if present)
        if Xset.shape[0] > 7:
            for i in range(Xset.shape[0] - 7):
                axes[1, 1].plot(times, Xset[7+i, :], label=f'h_rw{i}')
            axes[1, 1].set_title('RW Momentum')
            axes[1, 1].legend()
        else:
            axes[1, 1].set_title('(no RW)')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].grid(True)

        fig.suptitle(title, fontsize=14)
        fig.tight_layout()
        fig.savefig(filepath, dpi=100)
        print(f"  -> Saved {filepath}")

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
        
        # Initialize trajectory using prepareForAlilqr
        bdotOn = int(self.planner_settings.bdot_on)
        initial_result = self.planner.prepareForAlilqr(
            vecsPy_coarse, dt_coarse, t_start, t_end, x_0_clean, bdotOn
        )
        initial_traj_1, vecs_dt_coarse, _ = initial_result
        
        # if len(multistart_modes) > 1:
        #     print(f"  [Multistart] Selected bdot={best_mode} with spike={best_max_spike:.0f}°")
        
        # SLERP fallback: DISABLED - zero controls cause dynamics to diverge
        # if best_max_spike > 170 and os.environ.get("PY_ALILQR_SLERP_FALLBACK", "1") == "1":
        #     ...
        
        # # Check for zero-control initialization (env var: PY_ALILQR_ZERO_INIT=1)
        # if os.environ.get("PY_ALILQR_ZERO_INIT", "0") == "1":
        #     Xset, Uset, Tset, TQset = initial_traj_1
        #     Uset = np.zeros_like(Uset)
        #     initial_traj_1 = (Xset, Uset, Tset, TQset)
        #     print("  [Zero-init] Zeroed out initial controls")
        
        # Quaternion hemisphere check: flip goal quaternion if on wrong side
        # This prevents the optimizer from going "the long way" (>180°)
        # vecs_dt_coarse[6] is E (goal quaternion), shape (4, N) for quat goals
        E_goals = vecs_dt_coarse[6]
        # if E_goals.shape[0] == 4:  # Quaternion goal
        #     q0 = x_0_clean[3:7]  # Initial quaternion from state
        #     q_goal = E_goals[:, 0]  # Goal quaternion at t=0
            
        #     # Check dot product - if negative, they're on opposite hemispheres
        #     dot = np.dot(q0, q_goal)
        #     if dot < 0:
        #         # Flip the goal quaternion for the entire trajectory
        #         vecs_dt_coarse = list(vecs_dt_coarse)
        #         vecs_dt_coarse[6] = -E_goals  # Flip sign
        #         vecs_dt_coarse = tuple(vecs_dt_coarse)
        #         print(f"  [Quat-flip] Flipped goal quaternion (dot={dot:.3f} < 0)")
        
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
        
        # Infeasible start: replace init with SLERP trajectory for Pass 1
        use_infeasible = getattr(self.planner_settings, 'use_infeasible_start', False)
        if use_infeasible and hasattr(self.planner, 'setUseInfeasibleStart'):
            self.planner.setUseInfeasibleStart(True)
            # Check if quaternion goal (4-vector, no NaN)
            E_goals_check = vecs_dt_coarse[6]
            if E_goals_check.shape[0] == 4 and not np.isnan(E_goals_check[0, 0]):
                q_goal_slerp = E_goals_check[:, 0]
                N_coarse = initial_traj_1[0].shape[1]
                # Generate SLERP trajectory via C++
                try:
                    initial_traj_1 = self.planner.generateSlerpTrajectory(
                        dt_coarse, x_0_clean, q_goal_slerp, N_coarse, vecs_dt_coarse)
                except Exception:
                    # Fallback to Python SLERP
                    initial_traj_1 = self._generate_slerp_trajectory(
                        dt_coarse, x_0_clean, q_goal_slerp, N_coarse, initial_traj_1[2])
                if verbose:
                    print(f"  [Infeasible start] Replaced init with SLERP trajectory")
            self.py_alilqr._use_infeasible_start = True
        
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
        
        # Save Pass 1 final trajectory diagnostic
        try:
            self._save_trajectory_snapshot(
                result1.Xset, result1.Uset, dt_coarse, duration,
                "/tmp/planner_pass1_final.png", "Pass 1 Final"
            )
        except Exception:
            pass
        
        # Store pass1 result
        self.pass1_result = result1
        
        # =====================================================================
        # PASS 2: Fine trajectory refinement (strict constraint enforcement)
        # Interpolate to fine timestep and run with high penalty
        # =====================================================================
        # Check skip_pass2 from settings if not explicitly set
        if not skip_pass2:
            skip_pass2 = getattr(self.planner_settings, 'skip_pass2_optimization', False)
        
        # Enforce: infeasible start requires Pass 2 for dynamic feasibility
        use_infeasible = getattr(self.planner_settings, 'use_infeasible_start', False)
        if use_infeasible and skip_pass2:
            import warnings
            warnings.warn(
                "use_infeasible_start=True requires Pass 2 for dynamic feasibility. "
                "Overriding skip_pass2 to False.", stacklevel=2)
            skip_pass2 = False
        
        if skip_pass2:
            if verbose:
                print(f"\n=== SKIP PASS 2: ZOH forward sim + K-gains (dt={dt_fine}s) ===")
            
            # Match C++: ZOH-interpolate controls to fine grid, forward simulate,
            # compute K-gains at fine dt on the feasible trajectory.
            interp_ratio = int(dt_coarse / dt_fine)
            N_fine = int(np.ceil(duration / dt_fine)) + 1
            N_ctrl_fine = N_fine - 1
            
            # result1.Uset has RW controls in PHYSICAL units (scaled by py_alilqr).
            # generateInitialTrajectory expects OPTIMIZER units, so we need to unscale.
            Uset_coarse = result1.Uset.copy()
            scale = self.planner.get_nonmtq_torq_scale()
            if scale != 1.0:
                n_mtq = self.planner.get_number_MTQ()
                n_rw = self.planner.get_number_RW()
                n_magic = self.planner.get_number_magic()
                if n_rw + n_magic > 0:
                    rw_magic_start = n_mtq
                    rw_magic_end = n_mtq + n_rw + n_magic
                    Uset_coarse[rw_magic_start:rw_magic_end, :] /= scale
            
            if interp_ratio > 1:
                # ZOH: repeat each control except last for interp_ratio steps
                if Uset_coarse.shape[1] >= 2:
                    Uset_main = np.repeat(Uset_coarse[:, :-1], interp_ratio, axis=1)
                else:
                    Uset_main = np.repeat(Uset_coarse, interp_ratio, axis=1)
                # Pad/trim to exact N_ctrl_fine
                if Uset_main.shape[1] < N_ctrl_fine:
                    pad_cols = N_ctrl_fine - Uset_main.shape[1]
                    Uset_main = np.hstack([Uset_main, np.tile(Uset_coarse[:, -1:], (1, pad_cols))])
                Uset_fine = Uset_main[:, :N_ctrl_fine]
            else:
                Uset_fine = Uset_coarse
            
            # Propagate environment at fine timestep
            vecsPy_fine = self._propagate_environment(os_0, t_start, t_end, dt_fine, N_fine, goals)
            initial_result_2 = self.planner.prepareForAlilqr(
                vecsPy_fine, dt_fine, t_start, t_end, x_0_clean, 0
            )
            _, vecs_dt_fine, _ = initial_result_2
            
            # Forward-simulate using C++ rk4z (dynamically feasible trajectory)
            traj_fine = self.planner.generateInitialTrajectory(
                dt_fine, result1.Xset[:, 0].copy(), Uset_fine, vecs_dt_fine
            )
            Xset_fine, Uset_fs, times_fs, TQset_fine = traj_fine
            
            if verbose:
                print(f"  ZOH forward sim: X=({Xset_fine.shape[0]},{Xset_fine.shape[1]}), "
                      f"U=({Uset_fs.shape[0]},{Uset_fs.shape[1]})")
            
            # Save diagnostic
            try:
                self._save_trajectory_snapshot(
                    Xset_fine, Uset_fs, dt_fine, duration,
                    "/tmp/planner_skip_pass2_fwd_sim.png", "Skip Pass2 Forward Sim"
                )
            except Exception:
                pass
            
            # Compute K-gains at fine dt via C++ alilqr with minimal iterations.
            # Uses Pass 2 costs (not TVLQR — those have huge terminal costs that
            # cause K-gain divergence without findK's special handling).
            cost_settings_2 = self.planner_settings.optSecondCostSettings()
            
            # Create minimal alilqr settings: 1 outer, 1 inner iteration
            from ADCS.controller.helpers.planner_subsettings import (
                SolverPassConfig, ConvergenceConfig, AugLagConfig,
                RegularizationConfig, LineSearchConfig
            )
            minimal_pass = SolverPassConfig(
                convergence=ConvergenceConfig(max_outer_iter=1, max_inner_iter=1),
                aug_lag=AugLagConfig(penalty_init=1e-6),
                regularization=RegularizationConfig(),
                line_search=LineSearchConfig()
            )
            minimal_alilqr = (
                minimal_pass.line_search.to_tuple(),
                minimal_pass.aug_lag.to_tuple(),
                minimal_pass.convergence.to_tuple(state_len=self.est_sat.state_len),
                minimal_pass.regularization.to_tuple()
            )
            
            # Suppress verbose output for this minimal run
            prev_verbose = self.planner_settings.verbosity
            self.planner.setVerbosity(False)
            
            # IMPORTANT: Make copies because alilqr modifies arrays in-place!
            # We want to keep the original forward-sim trajectory.
            traj_fine_copy = (
                Xset_fine.copy(),
                Uset_fs.copy(),
                times_fs.copy() if times_fs is not None else None,
                TQset_fine.copy()
            )
            alilqr_result = self.planner.alilqr(
                dt_fine, traj_fine_copy, vecs_dt_fine,
                cost_settings_2, minimal_alilqr, False
            )
            self.planner.setVerbosity(prev_verbose)
            
            opt_result, mu_out, grad_out = alilqr_result
            Xset_kgain, Uset_kgain, TQset_kgain, Kset_kgain, Sset_kgain, times_kgain = opt_result
            
            Kset_final = Kset_kgain
            
            times_fine = np.linspace(t_start, t_end, Xset_fine.shape[1])
            
            # Build a minimal OptimizationResult for analysis
            from ADCS.controller.helpers import OptimizationResult
            TQset_fs = np.zeros((3, Xset_fine.shape[1]))
            result = OptimizationResult(
                success=True,
                Xset=Xset_fine, Uset=Uset_fs, TQset=TQset_fs,
                Kset=Kset_final,
                times=times_fine,
                final_cost=result1.final_cost,
                final_cmax=result1.final_cmax,
                final_grad=0.0,
                iterations=result1.iterations,
                total_inner_iters=result1.total_inner_iters,
                total_outer_iters=result1.total_outer_iters,
            )
            self.pass2_result = None
            self.last_optimization_result = result
            
            # Scale controls to physical units (Uset_fs is in optimizer units)
            # This matches what py_alilqr.optimize() does internally
            Uset_physical = Uset_fs.copy()
            if scale != 1.0 and n_rw + n_magic > 0:
                Uset_physical[rw_magic_start:rw_magic_end, :] *= scale
            
            # Reorder controls and gains to Python convention
            Uset = reorder_controls_cpp_to_python(Uset_physical, self.est_sat.actuators)
            Kset = reorder_gains_cpp_to_python(Kset_final, self.est_sat.actuators)
            Kset = self._scale_rw_gains(Kset)
            
            N_result = Xset_fine.shape[1]
            Sset = np.zeros((1, N_result))
            
            if verbose:
                print(f"  K-gains computed: ({Kset.shape[0]},{Kset.shape[1]})")
            
            if live_viz is not None:
                if viz_save_path:
                    live_viz.save(viz_save_path)
                live_viz.finish(block=False)
                live_viz.close()
                self.set_iteration_callback(original_callback)
            
            return Trajectory(times_fine, Xset_fine, Uset, Kset, Sset)
        
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
        
        # Physics-based warm-start: SLERP states + solve for controls at each fine timestep
        # This computes the MTQ dipole needed to produce the correct angular acceleration
        # given the actual B-field at each fine timestep, avoiding the ZOH/K-gain problems.
        traj_fine = None
        if interp_ratio > 1:
            try:
                from ADCS.controller.helpers.mtq_warm_start import (
                    interpolate_trajectory_to_finer_grid,
                )
                from ADCS.helpers.math_helpers import rot_mat
                
                if verbose:
                    print(f"  Using closed-loop inverse dynamics warm start")
                
                # 1. SLERP-interpolate coarse states to fine grid (reference only)
                Xset_ref = interpolate_trajectory_to_finer_grid(
                    result1.Xset, dt_coarse, dt_fine, duration, use_slerp=True
                )
                if Xset_ref.shape[1] < N_fine:
                    pad = np.tile(Xset_ref[:, -1:], (1, N_fine - Xset_ref.shape[1]))
                    Xset_ref = np.hstack([Xset_ref, pad])
                elif Xset_ref.shape[1] > N_fine:
                    Xset_ref = Xset_ref[:, :N_fine]
                
                # 2. Setup
                B_eci_fine = vecs_dt_fine[3]  # (3, N_fine)
                J = self.est_sat.J_noRW
                n_rw_acts = len(self.est_sat.rw_actuators)
                n_mtq_acts = len(self.est_sat.mtq_actuators)
                n_u = n_mtq_acts + n_rw_acts
                m_max = self.est_sat.mtq_actuators[0].u_max if n_mtq_acts > 0 else 1.0
                rw_torq_max = self.est_sat.rw_actuators[0].u_max if n_rw_acts > 0 else None
                rw_axes = np.array([rw.axis for rw in self.est_sat.rw_actuators]) if n_rw_acts > 0 else None
                
                # 3. Closed-loop inverse dynamics: solve controls from ACTUAL state
                #    at each step, targeting the NEXT reference state
                Xset_fine = np.zeros((result1.Xset.shape[0], N_fine))
                Uset_fine_ws = np.zeros((n_u, N_fine))
                
                x_sim = result1.Xset[:, 0].copy()
                Xset_fine[:, 0] = x_sim
                
                for k in range(N_fine - 1):
                    w_sim = x_sim[0:3]
                    q_sim = x_sim[3:7]
                    q_sim = q_sim / np.linalg.norm(q_sim)
                    
                    # Target: reach reference ω at next timestep
                    w_target = Xset_ref[0:3, k+1]
                    w_dot_desired = (w_target - w_sim) / dt_fine
                    
                    # Limit angular acceleration to prevent huge torque commands
                    # from large ω errors (e.g., gyroscopic coupling or bad reference)
                    w_dot_mag = np.linalg.norm(w_dot_desired)
                    w_dot_max = 0.1  # rad/s² — generous limit for warm-start
                    if w_dot_mag > w_dot_max:
                        w_dot_desired = w_dot_desired * (w_dot_max / w_dot_mag)
                    
                    # Required torque from Euler equation
                    tau_needed = J @ w_dot_desired + np.cross(w_sim, J @ w_sim)
                    
                    # Split torque between RW and MTQ using inverse dynamics
                    # RW: project needed torque onto RW axes (they can produce torque in any direction along axis)
                    # MTQ: handle the remainder (perpendicular to B-field)
                    tau_rw = np.zeros(3)
                    if n_rw_acts > 0:
                        for i in range(n_rw_acts):
                            ax = rw_axes[i]
                            # Project needed torque onto RW axis
                            rw_torque = np.dot(tau_needed, ax)
                            if rw_torq_max is not None:
                                rw_torque = np.clip(rw_torque, -rw_torq_max, rw_torq_max)
                            Uset_fine_ws[n_mtq_acts + i, k] = rw_torque
                            tau_rw += rw_torque * ax
                    
                    # MTQ: handle whatever the RW can't
                    tau_mtq_needed = tau_needed - tau_rw
                    R = rot_mat(q_sim)
                    k_idx = min(k, B_eci_fine.shape[1] - 1)
                    B_body = R.T @ B_eci_fine[:, k_idx]
                    B_sq = np.dot(B_body, B_body)
                    
                    if B_sq > 1e-20:
                        m = np.cross(B_body, tau_mtq_needed) / B_sq
                        if np.any(np.isnan(m)) or np.any(np.isinf(m)):
                            m = np.zeros(3)
                        m = np.clip(m, -m_max, m_max)
                    else:
                        m = np.zeros(3)
                    Uset_fine_ws[0:n_mtq_acts, k] = m[:n_mtq_acts]
                    
                    # Forward simulate one step using C++ rk4z
                    u_for_cpp = Uset_fine_ws[:, k].copy()
                    NONMTQ_TORQ_SCALE = self.planner.get_nonmtq_torq_scale()
                    if n_rw_acts > 0:
                        u_for_cpp[n_mtq_acts:n_mtq_acts+n_rw_acts] /= NONMTQ_TORQ_SCALE
                    
                    # Build dynamics info tuples for rk4z
                    k_next = min(k + 1, N_fine - 1)
                    # vecs_dt_fine: 0=t, 1=R, 2=V, 3=B, 4=S, 5=satvec, 6=ECIvec, 7=p, 8=rho
                    dyn_k = (vecs_dt_fine[3][:, k], vecs_dt_fine[1][:, k],
                             int(vecs_dt_fine[7][k]),
                             vecs_dt_fine[2][:, k], vecs_dt_fine[4][:, k], 0)
                    dyn_kp1 = (vecs_dt_fine[3][:, k_next], vecs_dt_fine[1][:, k_next],
                               int(vecs_dt_fine[7][k_next]),
                               vecs_dt_fine[2][:, k_next], vecs_dt_fine[4][:, k_next], 0)
                    x_next = np.array(self.planner.rk4z(
                        dt_fine, x_sim, u_for_cpp, dyn_k, dyn_kp1
                    ))
                    x_sim = x_next
                    x_sim[3:7] /= np.linalg.norm(x_sim[3:7])  # normalize quat
                    Xset_fine[:, k+1] = x_sim
                    
                    if np.any(np.isnan(x_sim)):
                        if verbose:
                            print(f"  WARNING: NaN at k={k}, falling back")
                        _ws_failed = True
                        break
                else:
                    _ws_failed = False
                
                if not _ws_failed:  # loop completed without NaN
                    Uset_fine_ws[:, -1] = Uset_fine_ws[:, -2]
                    # Scale RW controls to optimizer units
                    Uset_opt = Uset_fine_ws.copy()
                    if n_rw_acts > 0:
                        Uset_opt[n_mtq_acts:n_mtq_acts+n_rw_acts, :] /= NONMTQ_TORQ_SCALE
                    
                    times_fine = np.linspace(t_start, t_end, N_fine)
                    TQset_fine = np.zeros((3, N_fine))
                    traj_fine = (
                        np.asfortranarray(Xset_fine, dtype=np.float64),
                        np.asfortranarray(Uset_opt, dtype=np.float64),
                        np.asfortranarray(times_fine, dtype=np.float64),
                        np.asfortranarray(TQset_fine, dtype=np.float64)
                    )
                    if verbose:
                        print(f"  Closed-loop warm start complete")
                    
            except Exception as e:
                import traceback
                if verbose:
                    print(f"  ERROR in physics-based warm start: {e}")
                    traceback.print_exc()
                traj_fine = None
        
        # Fallback: ZOH forward simulation
        if traj_fine is None:
            if verbose:
                print(f"  Using ZOH warm start (fallback)")
            try:
                traj_fine = self.planner.generateInitialTrajectory(
                    dt_fine, result1.Xset[:, 0].copy(), Uset_fine, vecs_dt_fine
                )
                Xset_check, _, _, _ = traj_fine
                if np.any(np.isnan(Xset_check)) or np.any(np.isinf(Xset_check)):
                    if verbose:
                        print("  WARNING: ZOH warm start produced NaN, using fresh init")
                    traj_fine, _, _ = initial_result_2
            except Exception as e:
                if verbose:
                    print(f"  WARNING: generateInitialTrajectory failed ({e}), using fresh init")
                traj_fine, _, _ = initial_result_2
        
        # Save warm-start trajectory diagnostic (what Pass 2 receives as initial guess)
        try:
            Xset_ws, Uset_ws, _, _ = traj_fine
            self._save_trajectory_snapshot(
                Xset_ws, Uset_ws, dt_fine, duration,
                "/tmp/planner_pass2_warmstart.png", "Pass 2 Warm-Start"
            )
        except Exception:
            pass
        
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
