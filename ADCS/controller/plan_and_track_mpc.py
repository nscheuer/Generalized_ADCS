"""
Plan-and-Track controller using MPC-TVLQR hybrid tracking.

This controller uses Model Predictive Control with actual B-field measurements
for MTQ control, combined with TVLQR feedback for reaction wheels. This hybrid
approach fixes the fundamental limitation of pure TVLQR for MTQ-only systems.

Key insight: MTQ torque depends on B-field which depends on attitude.
TVLQR uses planned B-field which diverges from actual when attitude drifts.
MPC uses actual B-field, solving this problem.
"""
from __future__ import annotations

__all__ = ["Plan_and_Track_MPC", "Plan_and_Track_MPC_Python"]

import numpy as np
from typing import Optional
from numpy.typing import NDArray

from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_base import PlanAndTrackBase
from ADCS.controller.helpers import (
    PlannerSettings, Trajectory, reorder_controls_cpp_to_python,
    reorder_gains_cpp_to_python, PythonALILQRv2, OptimizationResult,
    IterationData, LivePlannerViz
)
from ADCS.controller.helpers.mpc_tracker import (
    MPCTracker, MPCParams, mpc_tvlqr_hybrid, quat_error_vec, solve_mtq_for_torque
)
from typing import Callable
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite


class Plan_and_Track_MPC(PlanAndTrackBase):
    r"""
    Plan-and-Track controller using C++ ALTRO planning with MPC-TVLQR hybrid tracking.

    This controller plans trajectories using the C++ ALTRO planner but executes
    them using a hybrid MPC-TVLQR approach:
    
    - **MTQs**: MPC with actual B-field (fixes TVLQR limitation)
    - **RWs**: TVLQR feedback (works because RW torque is attitude-independent)
    
    Why MPC for MTQs?
    -----------------
    MTQ torque is τ = m × B, where B depends on the current attitude. TVLQR
    computes gains assuming the planned B-field, but when attitude diverges
    from the plan, the actual B-field differs, causing TVLQR to command
    torques in the wrong direction.
    
    MPC re-solves for optimal control at each timestep using the actual
    measured B-field, fixing this fundamental problem.
    
    Results
    -------
    With aggressive tuning on a 200s trajectory:
    
    | System     | Open-loop | TVLQR  | MPC-hybrid |
    |------------|-----------|--------|------------|
    | MTQ-only   | 10.9°     | 55.3°  | **9.6°**   |
    | 3MTQ+1RW   | 87.7°     | 28.2°  | **14.0°**  |
    
    Usage
    -----
    >>> controller = Plan_and_Track_MPC(est_sat, planner_settings)
    >>> traj = controller.calculate_trajectory(t_start, duration, x_0, os_0, goals)
    >>> controller.set_active_trajectory(traj)
    >>> 
    >>> # In control loop
    >>> u = controller.find_u(x_hat, sens, est_sat, os_hat, B_body=B_measured)
    
    Parameters
    ----------
    est_sat : EstimatedSatellite
        Estimated satellite model
    planner_settings : PlannerSettings
        ALTRO planner configuration
    mpc_params : MPCParams, optional
        MPC parameters (default: balanced settings)
    """

    def __init__(
        self,
        est_sat: EstimatedSatellite,
        planner_settings: PlannerSettings,
        mpc_params: Optional[MPCParams] = None
    ) -> None:
        # Initialize base planner
        self._init_planner(est_sat, planner_settings, tracking_lqr_formulation=0, quat_to_3vec_mode=2)
        
        # Store satellite info for MPC
        self._J = est_sat.J_noRW
        self._m_max = est_sat.mtq_actuators[0].u_max if est_sat.mtq_actuators else 0.2
        self._has_rw = len(est_sat.rw_actuators) > 0
        self._n_mtq = len(est_sat.mtq_actuators)
        self._n_rw = len(est_sat.rw_actuators)
        
        if self._has_rw:
            self._rw_axis = est_sat.rw_actuators[0].axis
            self._rw_u_max = est_sat.rw_actuators[0].u_max
        
        # MPC parameters
        self.mpc_params = mpc_params if mpc_params is not None else MPCParams.balanced()
        
        # Internal MPC tracker (created when trajectory is set)
        self._mpc_tracker: Optional[MPCTracker] = None
        
        # Store K gains for RW feedback
        self._K_3d: Optional[np.ndarray] = None
        self._t_plan: Optional[np.ndarray] = None

    def find_u(
        self,
        x_hat: NDArray[np.float64],
        sens: NDArray[np.float64],
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        goal_vector_eci: Optional[NDArray[np.float64]] = None,
        w_ref: Optional[NDArray[np.float64]] = None,
        B_body: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        r"""
        Compute MPC-TVLQR hybrid tracking control.

        For MTQs, uses MPC with actual B-field. For RWs, uses TVLQR feedback.

        Parameters
        ----------
        x_hat : ndarray
            Estimated state vector [omega, quaternion, (h_rw)]
        sens : ndarray
            Sensor measurements (not used directly)
        est_sat : EstimatedSatellite
            Estimated satellite model
        os_hat : Orbital_State
            Current orbital state (provides time and B-field if B_body not given)
        goal_vector_eci : ndarray, optional
            Goal vector (not used - taken from trajectory)
        w_ref : ndarray, optional
            Reference angular velocity (not used - taken from trajectory)
        B_body : ndarray, optional
            Magnetic field in body frame [T]. If None, computed from os_hat.

        Returns
        -------
        u : ndarray
            Control vector [MTQ dipoles, (RW torques)]
        """
        current_time = os_hat.J2000

        if self.active_trajectory is None:
            raise RuntimeError(f"Plan_and_Track_MPC: No active trajectory set at t={current_time}")

        if not self.active_trajectory.is_valid_time(current_time):
            raise RuntimeError(
                f"Plan_and_Track_MPC: Active trajectory expired. "
                f"Current: {current_time}, Traj: [{self.active_trajectory.start_time}, {self.active_trajectory.end_time}]"
            )

        # Get B-field in body frame
        if B_body is None:
            # Compute from orbital state and current attitude
            from scipy.spatial.transform import Rotation
            q = x_hat[3:7]
            R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            B_body = R.T @ os_hat.B

        # Get reference from trajectory
        x_ref = self.active_trajectory.get_state_at(current_time)
        u_ref = self.active_trajectory.get_control_at(current_time)
        
        # Time in seconds from trajectory start
        t_sec = (current_time - self.active_trajectory.start_time) / TimeConstants.sec2cent
        dt = self.planner_settings.dt_tvlqr

        # === MTQ control via MPC ===
        # Get next reference for MPC lookahead
        t_next = current_time + dt * TimeConstants.sec2cent
        if self.active_trajectory.is_valid_time(t_next):
            x_ref_next = self.active_trajectory.get_state_at(t_next)
            K_next = self.active_trajectory.get_gain_at(t_next)
        else:
            x_ref_next = x_ref
            K_next = self.active_trajectory.get_gain_at(current_time)

        # Use MPC-TVLQR hybrid for MTQ control
        u_mtq = mpc_tvlqr_hybrid(
            x_hat[:7], x_ref[:7], x_ref_next[:7], 
            K_next[:self._n_mtq, :6],  # MTQ gains on attitude states
            B_body, self._J, self._m_max, dt
        )

        # === RW control via TVLQR (if present) ===
        if self._has_rw:
            # TVLQR works for RW because torque is attitude-independent
            K = self.active_trajectory.get_gain_at(current_time)
            dx = self._state_error(x_hat, x_ref)
            
            # Full TVLQR feedback for RW portion
            u_full_tvlqr = u_ref - K @ dx
            u_rw = u_full_tvlqr[self._n_mtq:self._n_mtq + self._n_rw]
            u_rw = np.clip(u_rw, -self._rw_u_max, self._rw_u_max)
            
            u = np.concatenate([u_mtq, u_rw])
        else:
            u = u_mtq

        return u

    def _state_error(self, x_curr: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        """Compute reduced state error for TVLQR."""
        n_rw = len(x_curr) - 7
        error_dim = 6 + n_rw
        dx = np.zeros(error_dim)
        
        # Angular velocity error
        dx[0:3] = x_curr[0:3] - x_ref[0:3]
        
        # Attitude error (quaternion to 3-vector)
        dx[3:6] = quat_error_vec(x_curr[3:7], x_ref[3:7])
        
        # RW momentum error
        if n_rw > 0:
            dx[6:6+n_rw] = x_curr[7:7+n_rw] - x_ref[7:7+n_rw]
        
        return dx

    def calculate_trajectory(
        self,
        t_start: float,
        duration: float,
        x_0: np.ndarray,
        os_0: Orbital_State,
        goals: GoalList,
        verbose: bool = False
    ) -> Trajectory:
        """
        Plan trajectory using C++ ALTRO and prepare for MPC-TVLQR tracking.

        Parameters
        ----------
        t_start : float
            Start time in J2000 centuries
        duration : float
            Trajectory duration in seconds
        x_0 : ndarray
            Initial state
        os_0 : Orbital_State
            Initial orbital state
        goals : GoalList
            Pointing goals
        verbose : bool
            Enable verbose output

        Returns
        -------
        Trajectory
            Planned trajectory with states, controls, and gains
        """
        lqr_times, Xset, Uset, Kset, Sset = self._calculate_trajectory_common(
            t_start, duration, x_0, os_0, goals, verbose
        )
        
        # Store trajectory info for MPC
        self._t_plan = (lqr_times - lqr_times[0]) / TimeConstants.sec2cent
        
        return Trajectory(lqr_times, Xset, Uset, Kset, Sset)


class Plan_and_Track_MPC_Python(PlanAndTrackBase):
    r"""
    Plan-and-Track controller using Python ALILQR planning with MPC-TVLQR tracking.

    Same as Plan_and_Track_MPC but uses Python ALILQR optimizer for trajectory
    planning, allowing step-by-step inspection of the optimization process.

    This is useful for:
    - Debugging optimization issues
    - Visualizing convergence
    - Research into algorithm modifications

    Parameters
    ----------
    est_sat : EstimatedSatellite
        Estimated satellite model
    planner_settings : PlannerSettings
        ALTRO planner configuration
    mpc_params : MPCParams, optional
        MPC parameters
    verbose : bool
        Enable verbose output from optimizer
    """

    def __init__(
        self,
        est_sat: EstimatedSatellite,
        planner_settings: PlannerSettings,
        mpc_params: Optional[MPCParams] = None,
        verbose: bool = False
    ) -> None:
        # Initialize base planner
        self._init_planner(est_sat, planner_settings, tracking_lqr_formulation=0, quat_to_3vec_mode=2)
        
        # Create Python optimizer
        self.py_alilqr = PythonALILQRv2(self.planner, verbose=verbose)
        
        # Store satellite info
        self._J = est_sat.J_noRW
        self._m_max = est_sat.mtq_actuators[0].u_max if est_sat.mtq_actuators else 0.2
        self._has_rw = len(est_sat.rw_actuators) > 0
        self._n_mtq = len(est_sat.mtq_actuators)
        self._n_rw = len(est_sat.rw_actuators)
        
        if self._has_rw:
            self._rw_axis = est_sat.rw_actuators[0].axis
            self._rw_u_max = est_sat.rw_actuators[0].u_max
        
        # MPC parameters
        self.mpc_params = mpc_params if mpc_params is not None else MPCParams.balanced()
        
        # Storage for optimization results and callbacks
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
        """
        self.iteration_callback = callback
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
        w_ref: Optional[NDArray[np.float64]] = None,
        B_body: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        """Compute MPC-TVLQR hybrid control (same as Plan_and_Track_MPC)."""
        current_time = os_hat.J2000

        if self.active_trajectory is None:
            raise RuntimeError(f"Plan_and_Track_MPC_Python: No active trajectory set")

        if not self.active_trajectory.is_valid_time(current_time):
            raise RuntimeError(f"Plan_and_Track_MPC_Python: Trajectory expired")

        # Get B-field
        if B_body is None:
            from scipy.spatial.transform import Rotation
            q = x_hat[3:7]
            R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            B_body = R.T @ os_hat.B

        # Get references
        x_ref = self.active_trajectory.get_state_at(current_time)
        u_ref = self.active_trajectory.get_control_at(current_time)
        dt = self.planner_settings.dt_tvlqr

        # Next reference for MPC
        t_next = current_time + dt * TimeConstants.sec2cent
        if self.active_trajectory.is_valid_time(t_next):
            x_ref_next = self.active_trajectory.get_state_at(t_next)
            K_next = self.active_trajectory.get_gain_at(t_next)
        else:
            x_ref_next = x_ref
            K_next = self.active_trajectory.get_gain_at(current_time)

        # MPC for MTQ
        u_mtq = mpc_tvlqr_hybrid(
            x_hat[:7], x_ref[:7], x_ref_next[:7],
            K_next[:self._n_mtq, :6],
            B_body, self._J, self._m_max, dt
        )

        # TVLQR for RW
        if self._has_rw:
            K = self.active_trajectory.get_gain_at(current_time)
            dx = self._state_error(x_hat, x_ref)
            u_full = u_ref - K @ dx
            u_rw = np.clip(u_full[self._n_mtq:self._n_mtq + self._n_rw], 
                          -self._rw_u_max, self._rw_u_max)
            u = np.concatenate([u_mtq, u_rw])
        else:
            u = u_mtq

        return u

    def _state_error(self, x_curr: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        """Compute reduced state error."""
        n_rw = len(x_curr) - 7
        dx = np.zeros(6 + n_rw)
        dx[0:3] = x_curr[0:3] - x_ref[0:3]
        dx[3:6] = quat_error_vec(x_curr[3:7], x_ref[3:7])
        if n_rw > 0:
            dx[6:6+n_rw] = x_curr[7:7+n_rw] - x_ref[7:7+n_rw]
        return dx

    def calculate_trajectory(
        self,
        t_start: float,
        duration: float,
        x_0: np.ndarray,
        os_0: Orbital_State,
        goals: GoalList,
        verbose: bool = False,
        collect_all_iterations: bool = False,
        visualize: bool = False,
        viz_save_path: Optional[str] = None
    ) -> Trajectory:
        """
        Plan trajectory using Python ALILQR with two-pass optimization.

        Parameters
        ----------
        t_start : float
            Start time in J2000 centuries
        duration : float
            Duration in seconds
        x_0 : ndarray
            Initial state
        os_0 : Orbital_State
            Initial orbital state
        goals : GoalList
            Pointing goals
        verbose : bool
            Enable verbose output
        collect_all_iterations : bool
            Store all iteration data for analysis
        visualize : bool
            If True, show live visualization of optimization convergence
        viz_save_path : str, optional
            If provided, save final visualization to this path

        Returns
        -------
        Trajectory
            Planned trajectory
        """
        sec2cent = TimeConstants.sec2cent
        t_end = t_start + duration * sec2cent

        # Get timesteps
        dt_coarse = self.planner_settings.dt_tp
        dt_fine = self.planner_settings.dt_tvlqr

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
            goal_vectors = vecsPy_viz[5]  # E vectors shape (3, N)
            
            # Generate actuator names for plot labels
            actuator_names = []
            for act in self.est_sat.mtq_actuators:
                actuator_names.append(f'MTQ_{["x","y","z"][np.argmax(np.abs(act.axis))]}')
            for i, act in enumerate(self.est_sat.rw_actuators):
                actuator_names.append(f'RW{i+1}')
            
            # Create live visualization
            live_viz = LivePlannerViz(
                goal_vector_eci=goal_vectors,
                body_vector=np.array([0, 1, 0]),  # Boresight
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

        # === PASS 1: Coarse exploration ===
        if verbose:
            print(f"=== PASS 1: Exploration (dt={dt_coarse}s) ===")

        N_coarse = int(np.ceil(duration / dt_coarse)) + 1
        vecsPy_coarse = self._propagate_environment(os_0, t_start, t_end, dt_coarse, N_coarse, goals)

        cost_settings_1 = self.planner_settings.optMainCostSettings()
        alilqr_settings_1 = self.planner_settings.mainAlilqrSettings()

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
            print(f"Pass 1: {result1.total_inner_iters} iters, cost={result1.final_cost:.2e}, cmax={result1.final_cmax:.2e}")

        # === PASS 2: Fine refinement ===
        if verbose:
            print(f"=== PASS 2: Refinement (dt={dt_fine}s) ===")

        N_fine = int(np.ceil(duration / dt_fine)) + 1
        vecsPy_fine = self._propagate_environment(os_0, t_start, t_end, dt_fine, N_fine, goals)

        cost_settings_2 = self.planner_settings.optSecondCostSettings()
        alilqr_settings_2 = self.planner_settings.secondAlilqrSettings()
        tvlqr_cost_settings = self.planner_settings.optTVLQRCostSettings(tracking_LQR_formulation=0)

        # Interpolate controls
        interp_ratio = int(dt_coarse / dt_fine)
        if interp_ratio > 1:
            Uset_coarse = result1.Uset
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
        else:
            Uset_fine = result1.Uset

        initial_result_2 = self.planner.prepareForAlilqr(
            vecsPy_fine, dt_fine, t_start, t_end, x_0_clean, 0
        )
        _, vecs_dt_fine, _ = initial_result_2

        try:
            traj_fine = self.planner.generateInitialTrajectory(
                dt_fine, result1.Xset[:, 0].copy(), Uset_fine, vecs_dt_fine
            )
            Xset_check, _, _, _ = traj_fine
            if np.any(np.isnan(Xset_check)) or np.any(np.isinf(Xset_check)):
                traj_fine, _, _ = initial_result_2
        except:
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

        if verbose:
            print(f"Pass 2: {result2.total_inner_iters} iters, cost={result2.final_cost:.2e}, cmax={result2.final_cmax:.2e}")

        # Store result
        self.last_optimization_result = result2

        # Build trajectory
        lqr_times = np.linspace(t_start, t_end, N_fine)
        Xset = result2.Xset
        Uset = result2.Uset
        Kset = result2.Kset

        # Reshape K
        if Kset.ndim == 2:
            ctrl_dim = Uset.shape[0]
            state_dim = Xset.shape[0] - 1
            Kset = Kset.reshape(ctrl_dim, state_dim, -1)

        Sset = np.zeros((Xset.shape[0] - 1, N_fine))

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

        return Trajectory(lqr_times, Xset, Uset, Kset, Sset)


