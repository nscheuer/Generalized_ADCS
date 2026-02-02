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
        
    def find_u(
        self,
        x_hat: NDArray[np.float64],
        sens: NDArray[np.float64],
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        goal_vector_eci: Optional[NDArray[np.float64]] = None,
        w_ref: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        """Return open-loop control from active trajectory."""
        current_time = os_hat.J2000
        
        if self.active_trajectory is None:
            raise RuntimeError("No active trajectory set")
            
        if not self.active_trajectory.is_valid_time(current_time):
            raise RuntimeError(f"Active trajectory expired at t={current_time}")
            
        return self.active_trajectory.get_control_at(current_time)
    
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
        bdotOn = self.planner_settings.bdot_on
        initial_result = self.planner.prepareForAlilqr(
            vecsPy_coarse, dt_coarse, t_start, t_end, x_0_clean, int(bdotOn)
        )
        initial_traj_1, vecs_dt_coarse, _ = initial_result
        
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
            print(f"\n=== PASS 2: Refinement (dt={dt_fine}s, penalty_init={self.planner_settings.pass2.aug_lag.penalty_init}) ===")
        
        # Propagate environment at fine timestep
        N_fine = int(np.ceil(duration / dt_fine)) + 1
        vecsPy_fine = self._propagate_environment(os_0, t_start, t_end, dt_fine, N_fine, goals)
        
        # Get pass2 settings (higher penalty for strict constraints)
        cost_settings_2 = self.planner_settings.optSecondCostSettings()
        alilqr_settings_2 = self.planner_settings.secondAlilqrSettings()
        
        # Get TVLQR-specific cost settings for gain computation
        # These typically have higher control costs to produce smaller, more practical gains
        tvlqr_cost_settings = self.planner_settings.optTVLQRCostSettings(tracking_LQR_formulation=0)
        
        # Interpolate coarse trajectory to fine timestep (like C++ trajOptAfter)
        interp_ratio = int(dt_coarse / dt_fine)
        if interp_ratio > 1:
            Xset_coarse = result1.Xset
            Uset_coarse = result1.Uset
            
            # Replicate controls (zero-order hold) like C++ repelem
            # UsetLong = join_rows(repelem(Uset.cols(0,Uset.n_cols-3), 1, ratio), ...)
            if Uset_coarse.shape[1] >= 3:
                Uset_main = np.repeat(Uset_coarse[:, :-2], interp_ratio, axis=1)
            else:
                Uset_main = np.repeat(Uset_coarse[:, :-1], interp_ratio, axis=1)
            
            # Handle end padding to reach N_fine
            cols_needed = N_fine - Uset_main.shape[1] - 1
            if cols_needed > 0:
                Uset_pad = np.tile(Uset_coarse[:, -2:-1], (1, cols_needed))
                Uset_fine = np.hstack([Uset_main, Uset_pad, Uset_coarse[:, -1:]])
            else:
                Uset_fine = np.hstack([Uset_main[:, :N_fine-1], Uset_coarse[:, -1:]])
            
            if verbose:
                print(f"  Interpolated controls: {Uset_coarse.shape[1]} -> {Uset_fine.shape[1]}")
        else:
            Uset_fine = result1.Uset
        
        # Prepare vecs at fine timestep
        initial_result_2 = self.planner.prepareForAlilqr(
            vecsPy_fine, dt_fine, t_start, t_end, x_0_clean, 0  # bdot=0
        )
        _, vecs_dt_fine, _ = initial_result_2
        
        # Generate warm-started trajectory from interpolated Pass1 controls
        # This matches C++ trajOptAfter: generateInitialTrajectory(dt_tvlqr, Xset.col(0), UsetLong, vecs_tvlqr)
        # dt_fine=1s is small enough that this should be stable
        #
        # NEW: Option to use SLERP interpolation of states instead of forward simulation.
        # This preserves the Pass1 trajectory shape better, especially for quaternions.
        use_slerp_interpolation = True  # Re-enable with debugging
        
        if use_slerp_interpolation and interp_ratio > 1:
            from ADCS.controller.helpers.mtq_warm_start import (
                interpolate_trajectory_to_finer_grid, 
                solve_controls_from_trajectory
            )
            
            # SLERP interpolate states from Pass1
            Xset_coarse = result1.Xset
            tf = duration
            
            try:
                Xset_fine = interpolate_trajectory_to_finer_grid(
                    Xset_coarse, dt_coarse, dt_fine, tf, use_slerp=True
                )
                
                # Validate interpolated states
                if np.any(np.isnan(Xset_fine)) or np.any(np.isinf(Xset_fine)):
                    print("  WARNING: SLERP produced NaN/Inf, falling back to forward simulation")
                    use_slerp_interpolation = False
                else:
                    # Ensure Xset_fine has correct number of columns
                    if Xset_fine.shape[1] != N_fine:
                        # Pad or trim
                        if Xset_fine.shape[1] < N_fine:
                            pad = np.tile(Xset_fine[:, -1:], (1, N_fine - Xset_fine.shape[1]))
                            Xset_fine = np.hstack([Xset_fine, pad])
                        else:
                            Xset_fine = Xset_fine[:, :N_fine]
                    
                    # Verify quaternion norms
                    q_norms = np.linalg.norm(Xset_fine[3:7, :], axis=0)
                    if np.any(np.abs(q_norms - 1.0) > 0.01):
                        print(f"  WARNING: SLERP quaternions not normalized (range: {q_norms.min():.3f}-{q_norms.max():.3f})")
                    
                    # Recompute controls to match SLERP-interpolated states
                    # Extract B-field from vecs (vecs[4] is B_ECI)
                    B_eci = vecs_dt_fine[4]  # shape (3, N_fine)
                    
                    # Get satellite parameters
                    J = self.est_sat.Jcom
                    
                    # Get RW axes and limits
                    rw_axes = []
                    rw_torq_max = None
                    for act in self.est_sat.actuators:
                        if hasattr(act, 'J_rw'):  # It's a RW
                            rw_axes.append(act.axis)
                            if rw_torq_max is None:
                                rw_torq_max = act.u_max
                    rw_axes = np.array(rw_axes) if rw_axes else None
                    
                    # Get MTQ limit
                    m_max = None
                    for act in self.est_sat.actuators:
                        if hasattr(act, 'm_max'):  # It's an MTQ
                            m_max = act.m_max
                            break
                    
                    # Solve for controls that approximate the SLERP trajectory
                    # Note: MTQ can only produce torque perpendicular to B, so we can't
                    # exactly match arbitrary trajectories. We compute best-effort controls
                    # then forward-simulate to get consistent states.
                    Uset_slerp = solve_controls_from_trajectory(
                        Xset_fine, B_eci, dt_fine, J, rw_axes,
                        m_max=m_max, rw_torq_max=rw_torq_max
                    )
                    
                    print(f"  SLERP: Computed controls from interpolated states")
                    print(f"    MTQ range: [{Uset_slerp[0:3,:].min():.4f}, {Uset_slerp[0:3,:].max():.4f}]")
                    if rw_axes is not None and len(rw_axes) > 0:
                        print(f"    RW range: [{Uset_slerp[3:,:].min():.6f}, {Uset_slerp[3:,:].max():.6f}]")
                    
                    # Forward simulate with these controls to get consistent states
                    # This gives a trajectory that's close to SLERP but dynamically feasible
                    print(f"  SLERP: Forward simulating for consistency...")
                    traj_fwd = self.planner.generateInitialTrajectory(
                        dt_fine, Xset_fine[:, 0].copy(), Uset_slerp, vecs_dt_fine
                    )
                    Xset_fwd, _, times_fwd, TQset_fwd = traj_fwd
                    
                    # Check how close we got to SLERP
                    diff_q = np.abs(Xset_fine[3:7, :] - Xset_fwd[3:7, :])
                    print(f"    Max q diff from SLERP: {diff_q.max():.4f}")
                    
                    # Use the forward-simulated trajectory (consistent dynamics)
                    traj_fine = (Xset_fwd, Uset_slerp, times_fwd, TQset_fwd)
                    
                    if verbose:
                        print(f"  Used SLERP interpolation for states: {Xset_coarse.shape[1]} -> {Xset_fine.shape[1]}")
            except Exception as e:
                import traceback
                print(f"  WARNING: SLERP interpolation failed ({e})")
                traceback.print_exc()
                use_slerp_interpolation = False
        
        if not use_slerp_interpolation or interp_ratio <= 1:
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
