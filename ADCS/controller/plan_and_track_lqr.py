from __future__ import annotations

__all__ = ["Plan_and_Track_LQR"]

import numpy as np
from typing import Optional
from numpy.typing import NDArray

from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_base import PlanAndTrackBase
from ADCS.controller.helpers import PlannerSettings, Trajectory
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite


class Plan_and_Track_LQR(PlanAndTrackBase):
    r"""
    Plan-and-Track controller using ALTRO planning with TVLQR feedback tracking.

    This controller plans an optimal attitude trajectory using the ALTRO
    (Augmented Lagrangian TRajectory Optimizer) and executes it in closed loop
    using a time-varying linear quadratic regulator (TVLQR). The TVLQR gains are
    computed along the nominal trajectory returned by the planner and are used
    to stabilize the spacecraft about that trajectory.

    Relationship to the Plan-and-Track framework
    ---------------------------------------------
    This class derives from
    :class:`~ADCS.controller.plan_and_track_base.PlanAndTrackBase`, which
    provides:

    - Construction and configuration of the C++ ALTRO planner.
    - Orbit and environment propagation into planner-compatible arrays.
    - A shared trajectory-optimization pipeline.

    Mathematical formulation
    ------------------------
    Let the nonlinear spacecraft attitude dynamics be linearized about the
    nominal planned trajectory, yielding a discrete-time linear time-varying
    system:

    .. math::

       \mathbf{x}_{k+1} = A_k \mathbf{x}_k + B_k \mathbf{u}_k,

    where :math:`\mathbf{x}_k` is the attitude state deviation and
    :math:`\mathbf{u}_k` is the control input deviation at time step
    :math:`k`.

    The planner computes a nominal state-control sequence
    :math:`(\mathbf{x}_k^\ast, \mathbf{u}_k^\ast)` and associated TVLQR gains
    :math:`K_k`. The tracking control law applied by this controller is:

    .. math::

       \mathbf{u}_k =
       \mathbf{u}_k^\ast -
       K_k
       \left(
           \mathbf{x}_k - \mathbf{x}_k^\ast
       \right).

    This feedback stabilizes the system about the planned trajectory while
    compensating for moderate disturbances and modeling errors.

    Intended use
    ------------
    This controller is the standard closed-loop Plan-and-Track variant and is
    appropriate for nominal mission operations where feedback tracking is
    required but explicit disturbance estimation is not needed.

    :param est_sat: Estimated satellite model with actuators and sensors.
    :type est_sat: :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
    :param planner_settings: ALTRO trajectory planner configuration bundle.
    :type planner_settings: :class:`~ADCS.controller.helpers.PlannerSettings`
    :return: None.
    :rtype: None

    """

    def __init__(self, est_sat: EstimatedSatellite, planner_settings: PlannerSettings,
                 settings_factory=None) -> None:
        r"""
        Construct the Plan-and-Track LQR controller.

        This initializes the underlying C++ ALTRO planner using the standard TVLQR
        tracking formulation. The planner is configured through the provided
        :class:`~ADCS.controller.helpers.PlannerSettings`.

        No trajectory is generated during construction. A trajectory must be
        planned using :meth:`~Plan_and_Track_LQR.calculate_trajectory` and installed
        via :meth:`~ADCS.controller.plan_and_track_base.PlanAndTrackBase.set_active_trajectory`
        before control commands can be generated.

        Planner configuration
        ---------------------
        - ``tracking_lqr_formulation = 0`` selects standard TVLQR gains.
        - The quaternion-to-vector mode defaults to the reduced representation
        defined by the base class.

        :param est_sat: Estimated satellite model with actuator and sensor models.
        :type est_sat: :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        :param planner_settings: ALTRO planner configuration settings.
        :type planner_settings: :class:`~ADCS.controller.helpers.PlannerSettings`
        :return: None.
        :rtype: None

        """
        # tracking_lqr_formulation=0 is standard TVLQR
        self._init_planner(est_sat, planner_settings, tracking_lqr_formulation=0)
        
        # MTQ-only gain scale: auto-detect if no reaction wheels
        mtq_gs = getattr(planner_settings, 'mtq_gain_scale', None)
        if mtq_gs is None:
            has_rw = any(hasattr(a, 'J') for a in est_sat.actuators)
            mtq_gs = 1.0 if has_rw else 0.5
        self._gain_scale = mtq_gs
        
        self._max_K_norm = None  # No clamping by default
        self._warmup_time = 0.0  # No warmup by default
        self._trajectory_start_time = None  # Set when trajectory is assigned
        # Optional factory: callable(dt) -> PlannerSettings, for auto_refine_dt
        self._settings_factory = settings_factory
    
    def set_gain_scale(self, scale: float) -> None:
        """
        Set the feedback gain scaling factor.
        
        The TVLQR gains computed by the planner may be too aggressive for
        closed-loop tracking. This scaling factor allows tuning:
        
        - scale=1.0: Full TVLQR gains (default)
        - scale=0.0: Open-loop (just feedforward u_ref)
        - scale=0.01-0.1: Often works better for MTQ-based systems
        
        :param scale: Gain scaling factor (0.0 to 1.0 recommended).
        :type scale: float
        """
        self._gain_scale = scale
    
    def set_gain_clamp(self, max_K_norm: float) -> None:
        """
        Set maximum allowed K gain Frobenius norm.
        
        The TVLQR gains can be very large at the start of the trajectory
        (e.g., |K| > 5000), causing oscillations. Clamping to a reasonable
        value (e.g., 100-500) often improves tracking.
        
        :param max_K_norm: Maximum allowed Frobenius norm of K. Set to None to disable.
        :type max_K_norm: float
        """
        self._max_K_norm = max_K_norm
    
    def set_warmup_time(self, warmup_seconds: float) -> None:
        """
        Set feedback warmup period.
        
        During the warmup period, the gain scale ramps from 0 to 1.
        This prevents aggressive initial feedback that can cause oscillations.
        
        A warmup of 5-10 seconds often works well for MTQ-based systems.
        
        :param warmup_seconds: Duration of warmup period in seconds.
        :type warmup_seconds: float
        """
        self._warmup_time = warmup_seconds

    def find_u(
        self,
        x_hat: NDArray[np.float64],
        sens: NDArray[np.float64],
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        goal_vector_eci: Optional[NDArray[np.float64]] = None,
        w_ref: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        r"""
        Compute the TVLQR tracking control input at the current time.

        This method evaluates the time-varying LQR feedback law associated with the
        active trajectory. The current time is obtained from the orbital state
        estimate and used to interpolate the nominal trajectory and TVLQR gains.

        Control evaluation
        ------------------
        Let :math:`t` be the current time extracted from ``os_hat.J2000``. If the
        active trajectory is valid at :math:`t`, the controller computes:

        .. math::

        \mathbf{u}(t) = \mathbf{u}^\ast(t) - K(t) \left(\mathbf{x}(t) - \mathbf{x}^\ast(t)\right),

        where:

        - :math:`\mathbf{x}(t)` is the estimated state ``x_hat``,
        - :math:`\mathbf{x}^\ast(t)` is the nominal state from the trajectory,
        - :math:`\mathbf{u}^\ast(t)` is the nominal control,
        - :math:`K(t)` is the time-varying LQR gain.

        Validity checks
        ---------------
        A runtime error is raised if:

        - No active trajectory has been set using :meth:`~ADCS.controller.plan_and_track_base.PlanAndTrackBase.set_active_trajectory`.
        - The current time lies outside the valid time interval of the trajectory, as determined by :meth:`~ADCS.controller.helpers.Trajectory.is_valid_time`.

        Parameter usage
        ---------------
        The parameters ``sens``, ``est_sat``, ``goal_vector_eci``, and ``w_ref`` are
        included to satisfy the
        :class:`~ADCS.controller.Controller` interface but are not used directly in
        the control computation, since all references are taken from the active
        trajectory.

        :param x_hat: Estimated state vector.
        :type x_hat: numpy.typing.NDArray[numpy.float64]
        :param sens: Sensor measurement vector. Not directly used.
        :type sens: numpy.typing.NDArray[numpy.float64]
        :param est_sat: Estimated satellite model. Not directly used.
        :type est_sat: :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        :param os_hat: Estimated orbital state providing the current time.
        :type os_hat: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :param goal_vector_eci: Goal vector in ECI frame. Not directly used.
        :type goal_vector_eci: typing.Optional[numpy.typing.NDArray[numpy.float64]]
        :param w_ref: Reference angular velocity. Not directly used.
        :type w_ref: typing.Optional[numpy.typing.NDArray[numpy.float64]]
        :return: Control vector computed by TVLQR tracking.
        :rtype: numpy.typing.NDArray[numpy.float64]

        """

        current_time = os_hat.J2000

        if self.active_trajectory is None:
            raise RuntimeError(f"Plan_and_Track_LQR: No active trajectory set at t={current_time}")

        if not self.active_trajectory.is_valid_time(current_time):
            raise RuntimeError(f"Plan_and_Track_LQR: Active trajectory expired or not started. "
                                f"Current: {current_time}, Traj: [{self.active_trajectory.start_time}, {self.active_trajectory.end_time}]")

        # Get trajectory data
        x_ref = self.active_trajectory.get_state_at(current_time)
        u_ref = self.active_trajectory.get_control_at(current_time)
        K = self.active_trajectory.get_gain_at(current_time)
        
        # Apply gain clamping if configured
        if self._max_K_norm is not None:
            K_norm = np.linalg.norm(K)
            if K_norm > self._max_K_norm:
                K = K * (self._max_K_norm / K_norm)
        
        # Compute effective gain scale (including warmup)
        effective_scale = self._gain_scale
        if self._warmup_time > 0 and self._trajectory_start_time is not None:
            from ADCS.orbits.universal_constants import TimeConstants
            elapsed = (current_time - self._trajectory_start_time) / TimeConstants.sec2cent
            if elapsed < self._warmup_time:
                warmup_scale = elapsed / self._warmup_time
                effective_scale = effective_scale * warmup_scale
        
        # Compute state error
        dx = self.active_trajectory._state_diff(x_hat, x_ref)
        
        # Adaptive gain scaling is available but disabled by default.
        # Enable via controller._adaptive_gain_scaling = True if K-gains are
        # computed at coarse dt and tracking uses finer dt.
        if getattr(self, '_adaptive_gain_scaling', False):
            w_ref_norm = np.linalg.norm(x_ref[0:3])
            w_thresh_low = 0.002; w_thresh_high = 0.015
            if w_ref_norm > w_thresh_low:
                alpha = min(1.0, (w_ref_norm - w_thresh_low) / (w_thresh_high - w_thresh_low))
                effective_scale *= (1.0 - 0.8 * alpha)
        
        # Apply scaled feedback: u = u_ref - scale * K @ dx
        # Standard LQR/TVLQR uses negative feedback for stability
        # 
        # K-gains are in optimizer units. For RW/magic actuators, the optimizer uses
        # scaled controls (u_opt = u_phys / NONMTQ_TORQ_SCALE). So du = K @ dx gives
        # optimizer units, and we need to convert RW portion to physical units.
        du = effective_scale * K @ dx
        
        # TVLQR feedback: u = u_ref - K @ dx
        # 
        # K-gains are computed in optimizer units where RW controls are scaled.
        # u_ref is already converted to physical units by C++.
        # We need to scale du for RW/magic: du_phys = du_opt * NONMTQ_TORQ_SCALE
        NONMTQ_TORQ_SCALE = 1.0  # Must match C++ Satellite.hpp value (was 3e-5)
        n_mtq = len([a for a in est_sat.actuators if hasattr(a, 'axis') and not hasattr(a, 'J')])
        if len(du) > n_mtq:
            du[n_mtq:] *= NONMTQ_TORQ_SCALE
        
        u = u_ref - du
        
        # Saturate control to actuator limits
        u_max = np.array([act.u_max for act in est_sat.actuators])
        u = np.clip(u, -u_max, u_max)
        
        return u

    def set_active_trajectory(self, traj) -> None:
        """
        Set the active trajectory and record its start time for warmup.
        
        Overrides base class to track trajectory start time for warmup period.
        """
        super().set_active_trajectory(traj)
        if traj is not None:
            self._trajectory_start_time = traj.start_time
        else:
            self._trajectory_start_time = None

    @staticmethod
    def _plan_quality_score(Xset: np.ndarray, traj_times: np.ndarray,
                            goals: GoalList, q_goal: np.ndarray = None,
                            orbit=None, t_start: float = None,
                            settle_thresh_deg: float = 5.0,
                            tail_frac: float = 0.5,
                            body_boresight: np.ndarray = None) -> float:
        """Compute trajectory quality score: lower is better.

        Score = settle_frac(settle_thresh) + tail_mean / 180°

        - **settle_frac**: Fraction of trajectory before error permanently drops
          below ``settle_thresh_deg``. 0 = converged instantly, 1 = never settled.
        - **tail_mean / 180°**: Mean error over the last ``tail_frac`` of the
          trajectory, normalized to [0, 1]. 0 = perfect tracking, 1 = worst.

        The score lives in [0, 2]. A ★★★ trajectory scores < 0.3
        (settled by ~20% of the horizon, tail mean < ~5°).

        Works for both quaternion goals (q_goal is not None) and ECI/vector
        goals (q_goal is None, uses boresight error via goals + orbit).

        Parameters
        ----------
        orbit : Orbit, optional
            Required for time-varying ECI goals (Nadir, Sun, etc.) to compute
            the goal direction at each timestep via ``goal.to_ref(os)``.
        t_start : float, optional
            Planning start time in J2000 centuries. Required with orbit.
        """
        from ADCS.helpers.math_helpers import rot_mat
        from ADCS.CONOPS.goals import No_Goal

        if body_boresight is None:
            body_boresight = np.array([0.0, 1.0, 0.0])

        N = Xset.shape[1]
        T = traj_times[-1] - traj_times[0]
        if T <= 0:
            return 2.0  # degenerate

        # --- Compute per-timestep error ---
        errors = np.full(N, 180.0)
        if q_goal is not None:
            # Quaternion goal: geodesic distance
            for k in range(N):
                q = Xset[3:7, k]
                q = q / np.linalg.norm(q)
                cos_half = min(abs(np.dot(q, q_goal)), 1.0)
                errors[k] = np.degrees(2 * np.arccos(cos_half))
        else:
            # ECI/vector goal: boresight angle
            for k in range(N):
                t_k = traj_times[min(k, len(traj_times) - 1)]
                goal = goals.get_active_goal(t_k)
                if goal is None or isinstance(goal, No_Goal):
                    errors[k] = 0.0  # no goal → no error
                    continue
                try:
                    # traj_times are in J2000 centuries — use directly
                    if orbit is not None:
                        os_k = orbit.get_os(t_k)
                        eci_ref, _ = goal.to_ref(os0=os_k)
                    else:
                        eci_ref, _ = goal.to_ref(os0=None)
                except Exception:
                    continue
                nrm = np.linalg.norm(eci_ref)
                if nrm < 1e-10:
                    continue
                eci_ref = eci_ref / nrm
                q = Xset[3:7, k]
                q = q / np.linalg.norm(q)
                bore_eci = rot_mat(q) @ body_boresight
                cos_a = np.clip(np.dot(bore_eci, eci_ref), -1.0, 1.0)
                errors[k] = np.degrees(np.arccos(cos_a))

        # --- settle_frac: first time error < thresh AND stays below ---
        settle_frac = 1.0
        for k in range(N):
            if errors[k] < settle_thresh_deg and np.all(errors[k:] < settle_thresh_deg):
                settle_frac = (traj_times[k] - traj_times[0]) / T
                break

        # --- tail_mean: mean error over last tail_frac of trajectory ---
        start_idx = max(0, int(N * (1.0 - tail_frac)))
        tail_mean = np.mean(errors[start_idx:])

        return settle_frac + tail_mean / 180.0

    @staticmethod
    def _plan_final_angle(Xset: np.ndarray, q_goal: np.ndarray) -> float:
        """Compute final quaternion angle to goal, in degrees.
        
        For MTQ-only fixed-attitude goals, the final error is the right metric:
        feasible trajectories may wobble through large transients but converge
        at the end. These are perfectly trackable (K-gains on feasible trajectory
        are correct), unlike SLERP plans which converge to 0° but can't be tracked.
        """
        q = Xset[3:7, -1]
        q = q / np.linalg.norm(q)
        cos_half = min(abs(np.dot(q, q_goal)), 1.0)
        return np.degrees(2 * np.arccos(cos_half))

    @staticmethod
    def _plan_max_2nd_half_error(Xset: np.ndarray, q_goal: np.ndarray) -> float:
        """Compute max quaternion angle to goal in the 2nd half of the trajectory.
        
        This measures trajectory SHAPE quality: a good trajectory converges
        monotonically, so the max error in the 2nd half should be near 0°.
        A "bounce" trajectory may have 0° final error but >50° max in the
        2nd half, indicating the optimizer found a bad local minimum.
        """
        N = Xset.shape[1]
        start = N // 2
        max_ang = 0.0
        for k in range(start, N):
            q = Xset[3:7, k]
            q = q / np.linalg.norm(q)
            cos_half = min(abs(np.dot(q, q_goal)), 1.0)
            ang = np.degrees(2 * np.arccos(cos_half))
            if ang > max_ang:
                max_ang = ang
        return max_ang

    @staticmethod
    def _plan_max_angle(Xset: np.ndarray, q_goal: np.ndarray, skip_fraction: float = 0.2) -> float:
        """Compute max angle to goal after initial transient, in degrees.
        
        For quaternion goals (4-element q_goal): measures quaternion geodesic distance.
        """
        N = Xset.shape[1]
        skip = max(1, int(N * skip_fraction))
        max_ang = 0.0
        for k in range(skip, N):
            q = Xset[3:7, k]
            cos_half = min(abs(np.dot(q, q_goal)), 1.0)
            ang = np.degrees(2 * np.arccos(cos_half))
            if ang > max_ang:
                max_ang = ang
        return max_ang

    @staticmethod
    def _plan_max_boresight_error(Xset: np.ndarray, goals, traj_times: np.ndarray,
                                   body_boresight: np.ndarray = None,
                                   skip_fraction: float = 0.2) -> float:
        """Compute max boresight error after convergence (winding detection).
        
        Instead of simple max-after-skip, this detects trajectory winding:
        the boresight error is expected to decrease monotonically during a
        maneuver. If it drops below a convergence threshold and then rises
        back above a winding threshold, the plan is wound.
        
        Returns the max error after first convergence. For non-wound plans
        this is the steady-state tracking error (~0°). For wound plans this
        is the peak re-divergence (>>45°).
        
        Works for any goal type (ECI, Nadir, Sun, etc.).
        Only measures during active goal periods (skips No_Goal).
        """
        from ADCS.helpers.math_helpers import rot_mat
        from ADCS.CONOPS.goals import No_Goal
        if body_boresight is None:
            body_boresight = np.array([0.0, 1.0, 0.0])
        
        N = Xset.shape[1]
        
        # Step 1: Compute full boresight error trajectory
        errors = np.full(N, np.nan)
        for k in range(N):
            t_k = traj_times[min(k, len(traj_times) - 1)]
            goal = goals.get_active_goal(t_k)
            if goal is None or isinstance(goal, No_Goal):
                continue
            try:
                eci_ref, _ = goal.to_ref(os0=None)
            except:
                continue
            if np.linalg.norm(eci_ref) < 1e-10:
                continue
            eci_ref = eci_ref / np.linalg.norm(eci_ref)
            q = Xset[3:7, k]
            q = q / np.linalg.norm(q)
            bore_eci = rot_mat(q) @ body_boresight
            cos_a = np.clip(np.dot(bore_eci, eci_ref), -1.0, 1.0)
            errors[k] = np.degrees(np.arccos(cos_a))
        
        # Filter out NaN (No_Goal periods)
        valid = ~np.isnan(errors)
        if not np.any(valid):
            return 0.0
        
        # Step 2: Find convergence point — first time error drops below threshold
        # Use 10° as convergence threshold: "near goal" for boresight tracking.
        # For non-wound plans, error monotonically decreases after this point.
        # For wound plans, error goes back up to >>10° after convergence.
        converge_thresh = 10.0
        converge_idx = None
        for k in range(N):
            if valid[k] and errors[k] < converge_thresh:
                converge_idx = k
                break
        
        if converge_idx is None:
            # Never converged — return max error after skip
            skip = max(1, int(N * skip_fraction))
            valid_after_skip = valid.copy()
            valid_after_skip[:skip] = False
            if np.any(valid_after_skip):
                return np.nanmax(errors[valid_after_skip])
            return 180.0
        
        # Step 3: Return max error after convergence
        # This catches winding (error goes back up) while ignoring
        # the initial maneuver transient
        post_converge = errors[converge_idx:]
        valid_post = ~np.isnan(post_converge)
        if np.any(valid_post):
            return np.nanmax(post_converge[valid_post])
        return 0.0

    def _try_plan_fresh(
        self,
        t_start: float,
        duration: float,
        x_0: np.ndarray,
        os_0: Orbital_State,
        goals: GoalList,
        dt_plan: float,
        bdot_on: int,
        use_slacks: bool,
        tuning: str,
        cross_term_fraction: float,
        verbose: bool = False
    ) -> Optional[Trajectory]:
        """Try planning with completely fresh settings for the given dt.
        
        Creates a new planner with settings auto-scaled for the given dt_plan,
        runs the optimization, and returns the Trajectory. This avoids
        cross-contamination of settings between dt values.
        """
        try:
            from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
            import copy
            
            # Create fresh settings properly scaled for this dt
            settings_factory = getattr(self, '_settings_factory', None)
            if settings_factory is not None:
                new_settings = settings_factory(dt_plan)
            else:
                # Fallback: clone current settings and adjust dt
                new_settings = copy.deepcopy(self.planner_settings)
                new_settings.dt_tp = dt_plan
                new_settings.dt_tvlqr = dt_plan
            
            new_settings.bdot_on = bdot_on
            new_settings.use_infeasible_start = use_slacks
            # Respect the user's skip_pass2 setting; only force-skip when
            # slacks are active (Pass 2 without slacks would wound).
            if use_slacks:
                new_settings.skip_pass2_optimization = True
            # else: keep whatever the settings/factory produced
            if hasattr(self.planner_settings, 'infeasible_ctrl_mode'):
                new_settings.infeasible_ctrl_mode = self.planner_settings.infeasible_ctrl_mode
            
            # Create a temporary controller with the new settings
            temp_controller = Plan_and_Track_LQR(est_sat=self.est_sat, planner_settings=new_settings)
            
            lqr_times, Xset, Uset, Kset, Sset = temp_controller._calculate_trajectory_common(
                t_start, duration, x_0, os_0, goals, verbose
            )
            return Trajectory(lqr_times, Xset, Uset, Kset, Sset)
        except Exception as e:
            if verbose:
                print(f"  Plan attempt (dt={dt_plan}, mode={bdot_on}, slacks={use_slacks}) failed: {e}")
            return None

    def calculate_trajectory(
        self,
        t_start: float,
        duration: float,
        x_0: np.ndarray,
        os_0: Orbital_State,
        goals: GoalList,
        verbose: bool = False
    ) -> Trajectory:
        r"""
        Plan an optimal trajectory using ALTRO and prepare it for TVLQR tracking.

        If ``auto_refine_dt`` is enabled (via ``planner_settings``), this method
        implements a multi-resolution fallback strategy:

        1. Try planning at the configured dt (e.g., dt=10s) — fast (~1s).
        2. If the plan is "wound" (max angle to goal > threshold after transient),
           try alternative warm-start modes at the same dt.
        3. If still wound, fall back to finer dt (dt=1s) — slower (~10s) but
           resolves 180° bifurcation issues.
        4. As a last resort, try dt=1s with SLERP warm-start (slacks).

        The best trajectory (spike-free with lowest max angle) is returned.

        :param t_start: Planning start time in J2000 centuries.
        :type t_start: float
        :param duration: Planning horizon length in seconds.
        :type duration: float
        :param x_0: Initial state vector used to seed the optimizer.
        :type x_0: numpy.ndarray
        :param os_0: Initial orbital state used to seed environment propagation.
        :type os_0: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :param goals: Goal list defining the pointing objectives.
        :type goals: :class:`~ADCS.CONOPS.goallist.GoalList`
        :param verbose: If true, enable planner verbosity and diagnostic output.
        :type verbose: bool
        :return: Planned trajectory with nominal states, controls, and TVLQR gains.
        :rtype: :class:`~ADCS.controller.helpers.Trajectory`

        """
        auto_refine = getattr(self.planner_settings, 'auto_refine_dt', False)
        
        if not auto_refine:
            # Standard single-attempt planning
            lqr_times, Xset, Uset, Kset, Sset = self._calculate_trajectory_common(
                t_start, duration, x_0, os_0, goals, verbose
            )
            return Trajectory(lqr_times, Xset, Uset, Kset, Sset)
        
        # Multi-resolution fallback strategy:
        # 1. Try coarse dt (dt=10) with multiple init modes — fast (~1s each)
        # 2. If all wound, try fine dt (dt=1) — resolves 180° bifurcation (~10s each)
        # 3. Last resort: fine dt with SLERP slacks (~10s)
        #
        # Each attempt uses a completely independent controller to avoid any
        # cross-contamination from C++ global state (Armadillo RNG, etc.)
        dt_coarse = self.planner_settings.dt_tp
        dt_fine = getattr(self.planner_settings, 'auto_refine_dt_fine', 1.0)
        
        # Determine goal type early (needed for metric selection)
        q_goal = self._extract_goal_quaternion(goals, t_start, duration, x_0, os_0)
        is_quat_goal = (q_goal is not None)
        has_rw = any(hasattr(a, 'J') for a in self.est_sat.actuators)
        
        # Adjust gain scale for ECI goals with RW.
        # ECI vector goals only constrain 2 DOFs (boresight pointing).
        # The 3rd DOF (rotation around boresight) is free, but TVLQR
        # K-gains try to correct it anyway, wasting control authority
        # and causing instability. Reducing gain_scale prevents these
        # parasitic corrections from destabilizing the tracker.
        # Empirically validated: 0.3 transforms ★ → ★★★ on mediocre
        # seeds without degrading good seeds (0.142 → 0.145).
        if has_rw and not is_quat_goal:
            self._gain_scale = 0.3
        
        # Create orbit for ECI goal quality evaluation (Nadir, Sun, etc.)
        # Reused across all planning attempts for consistency and speed.
        eval_orbit = None
        if not is_quat_goal:
            from ADCS.orbits.orbit import Orbit
            from ADCS.orbits.universal_constants import TimeConstants
            t_end = t_start + duration * TimeConstants.sec2cent
            eval_orbit = Orbit(os0=os_0, end_time=t_end, dt=1.0, 
                               use_J2=True, fast=True)
        
        # Quality score threshold for cascade decisions.
        # score = settle_frac(5°) + tail_mean/180°  (lower = better, range [0,2])
        # ★★★ < 0.3 (settled by 20%, tail < 5°)
        # "good enough" < 0.5 (no need to cascade to expensive dt=1)
        # "needs refinement" > 0.5 (cascade to finer dt)
        good_thresh = 0.3   # ★★★ — stop immediately
        retry_thresh = 0.5  # Worth cascading below this
        # No special case for has_rw — the quality score handles everything.
        # RW configs can still produce bouncing plans (optimizer local minima
        # from the extra control dimension) that need dt=1 refinement.
        
        settings_factory = getattr(self, '_settings_factory', None)
        if settings_factory is None:
            # No factory — just do standard single-attempt
            lqr_times, Xset, Uset, Kset, Sset = self._calculate_trajectory_common(
                t_start, duration, x_0, os_0, goals, verbose
            )
            return Trajectory(lqr_times, Xset, Uset, Kset, Sset)
        
        best_traj = None
        best_score = 999.0
        
        def evaluate_plan(traj):
            """Evaluate trajectory quality — lower is better.
            
            Uses composite score = settle_frac(5°) + tail_mean/180°.
            Captures both convergence speed and steady-state quality.
            ★★★ trajectories score < 0.3.
            """
            return self._plan_quality_score(
                traj.states, traj.times, goals, q_goal,
                orbit=eval_orbit, t_start=t_start)
        
        def try_config(dt, mode, slacks, label, slerp_rate=None):
            """Create an independent controller and plan with it."""
            nonlocal best_traj, best_score
            try:
                s = settings_factory(dt)
                s.bdot_on = mode
                s.use_infeasible_start = slacks
                # For SLERP (slacks=True), skip Pass 2 — without slacks it wounds.
                # Otherwise, respect the setting from the factory.
                if slacks:
                    s.skip_pass2_optimization = True
                    s.infeasible_ctrl_mode = 1  # SLERP warm-start
                elif hasattr(self.planner_settings, 'infeasible_ctrl_mode'):
                    s.infeasible_ctrl_mode = self.planner_settings.infeasible_ctrl_mode
                if slerp_rate is not None:
                    s.slerp_init_rate = slerp_rate
                # MTQ-only quaternion goals: boost terminal angle cost 100×.
                # fast_slew sets angle_N = angle * 10, too weak for MTQ-only
                # to push through the last 10-20° of B-field-coupled maneuvers.
                # Applied AFTER cross-term so it doesn't inflate the cross-term.
                if is_quat_goal and not has_rw:
                    s.cost_main.angle_N *= 100
                    s.cost_second.angle_N *= 100
                
                ctrl = Plan_and_Track_LQR(est_sat=self.est_sat, planner_settings=s)
                lqr_times, Xset, Uset, Kset, Sset = ctrl._calculate_trajectory_common(
                    t_start, duration, x_0, os_0, goals, verbose
                )
                traj = Trajectory(lqr_times, Xset, Uset, Kset, Sset)
                score = evaluate_plan(traj)
                if verbose:
                    print(f"  {label}: score={score:.3f} (best={best_score:.3f})")
                if score < best_score:
                    best_score = score
                    best_traj = traj
                return score
            except Exception as e:
                if verbose:
                    print(f"  {label}: FAILED ({e})")
                return 999.0
        
        # Phase 1: Coarse dt (fast, ~1-3s each)
        #
        # NOTE: SLERP+slacks produces beautiful PLANNED trajectories but the
        # K-gains are computed along infeasible states/controls, causing
        # catastrophic tracking failure in real simulation (~148° error).
        # Must use no-slacks for physically valid K-gains.
        #
        # For RW configs, try mode 6 (SLERP warm-start) first — it avoids
        # bounce local minima by starting from the geometric SLERP path.
        # The extra control dimension from the RW creates a richer optimization
        # landscape with more (worse) local minima. Mode 6 keeps the optimizer
        # near the smooth-convergence basin. Mode 0 fallback for different basin.
        # Mode 6 (SLERP warm-start) is ideal for quaternion goals with RW:
        # it avoids bounce local minima by starting near the geometric path.
        # For ECI goals, mode 6 plans look "prettier" (plan ★★★) but their
        # K-gains are fragile (sim ✗). Mode 0 (random init) produces plans
        # with slower convergence but much more robust tracking. This is
        # because the SLERP path over-constrains the quaternion trajectory
        # for a reduced-attitude (2-DOF) goal, biasing the optimizer toward
        # a solution that's optimal only for that specific quaternion path.
        phase1_modes = [6, 0] if (has_rw and is_quat_goal) else [0, 4]
        for mode in phase1_modes:
            score = try_config(dt_coarse, mode, False, f"dt={dt_coarse} mode={mode}")
            if score <= good_thresh:
                return best_traj  # ★★★ — no need to try anything else
        
        # Phase 1b: Try complementary mode if first didn't reach good_thresh.
        # Mode 0 and Mode 6 find different local minima — complementary.
        # The quality score already captures bounce/shape issues, so just
        # try the other mode and keep whichever scored better.
        if best_score > good_thresh and len(phase1_modes) >= 1:
            alt_mode = 6 if phase1_modes[0] in [0, 4] else 0
            if alt_mode not in phase1_modes:
                try_config(dt_coarse, alt_mode, False,
                           f"dt={dt_coarse} mode={alt_mode} (complement)")
                if best_score <= good_thresh:
                    return best_traj
        
        # Phase 2: Finer dt attempts for plans that didn't reach retry_thresh.
        if best_score > retry_thresh:
            if has_rw:
                # RW fallback: plan with MTQ-only (0RW) model, then pad to 1RW.
                # The RW adds an extra control dimension that creates worse local
                # minima — the optimizer finds "lazy" trajectories that coast and
                # bounce. The 0RW model avoids this (simpler landscape, ★★★).
                # The padded trajectory has zero RW control/momentum/K-gains,
                # so TVLQR uses MTQs only for tracking (which is sufficient).
                traj_0rw = self._plan_with_mtq_only_model(
                    t_start, duration, x_0, os_0, goals, settings_factory,
                    dt_coarse, verbose)
                if traj_0rw is not None:
                    score_0rw = evaluate_plan(traj_0rw)
                    if verbose:
                        print(f"  0RW fallback: score={score_0rw:.3f} (best={best_score:.3f})")
                    if score_0rw < best_score:
                        best_score = score_0rw
                        best_traj = traj_0rw
                    if best_score <= good_thresh:
                        return best_traj
            else:
                # MTQ-only: try multiple modes at dt=1 (~30-60s each)
                for mode in [0, 4, 6]:
                    score = try_config(dt_fine, mode, False, f"dt={dt_fine} mode={mode}")
                    if score <= good_thresh:
                        return best_traj
        
        # Phase 3: SLERP warm-start WITHOUT slacks (mode 6) at dt=2
        if best_score > retry_thresh and not has_rw:
            dt_slerp = getattr(self.planner_settings, 'auto_refine_dt_slerp', 2.0)
            try_config(dt_slerp, 6, False, f"dt={dt_slerp} mode=6 (SLERP no-slack)")
        
        # Phase 4: SLERP with slacks (absolute last resort, MTQ-only)
        # Produces 0° plan error but K-gains may be poor for tracking.
        if best_score > retry_thresh and not has_rw:
            dt_slerp = getattr(self.planner_settings, 'auto_refine_dt_slerp', 2.0)
            try_config(dt_slerp, 0, True, f"dt={dt_slerp} slacks (last resort)")
        
        if best_traj is None:
            raise RuntimeError("All planning attempts failed")
        
        return best_traj

    def _plan_with_mtq_only_model(self, t_start, duration, x_0, os_0, goals,
                                    settings_factory, dt, verbose=False):
        """Plan with an MTQ-only (0RW) model, then pad to 1RW dimensions.

        When the 1RW optimizer finds bouncing local minima for ECI goals, the
        0RW optimizer consistently converges smoothly (simpler 3-control
        landscape). The resulting trajectory is padded with zero RW
        state/control/K-gains. TVLQR tracking uses MTQs only, which is
        sufficient (proven by 0RW Reduced ★★★ results).

        Returns a Trajectory with 1RW dimensions, or None on failure.
        """
        try:
            from ADCS.satellite_factory.satellites.create_cubesats import (
                create_beavercube1_cubesat)

            # Create 0RW satellite (MTQ-only) with same MTQ config
            sat_0rw = create_beavercube1_cubesat(estimated=False)

            # Strip RW state from x_0: [ω(3), q(4), h_rw(1)] → [ω(3), q(4)]
            x_0_mtq = x_0[:7]

            # Create settings for 0RW planning
            s = settings_factory(dt)
            s.bdot_on = 0
            s.use_infeasible_start = False

            ctrl_0rw = Plan_and_Track_LQR(est_sat=sat_0rw, planner_settings=s)
            lqr_times, Xset, Uset, Kset, Sset = ctrl_0rw._calculate_trajectory_common(
                t_start, duration, x_0_mtq, os_0, goals, verbose
            )

            # Pad to 1RW dimensions:
            # States: (7, N) → (8, N): append h_rw=0 row
            N_states = Xset.shape[1]
            Xset_1rw = np.vstack([Xset, np.zeros((1, N_states))])

            # Controls: (3, N) → (4, N): append τ_rw=0 row
            N_ctrl = Uset.shape[1]
            Uset_1rw = np.vstack([Uset, np.zeros((1, N_ctrl))])

            # K-gains: embed 0RW K(3×6) into 1RW K(4×7) with zeros for RW
            if Kset is not None and Kset.shape[0] > 0:
                N_K = Kset.shape[1]
                Kset_1rw = np.zeros((28, N_K))  # 4 controls × 7 reduced states
                for k in range(N_K):
                    K_0 = Kset[:, k].reshape(3, 6)
                    K_1 = np.zeros((4, 7))
                    K_1[:3, :6] = K_0  # MTQ gains preserved
                    # RW row (index 3) and h_rw column (index 6) stay zero
                    Kset_1rw[:, k] = K_1.ravel()
            else:
                Kset_1rw = Kset

            if verbose:
                print(f"  0RW fallback: planned with 0RW, padded to 1RW dims")

            return Trajectory(lqr_times, Xset_1rw, Uset_1rw, Kset_1rw, Sset)

        except Exception as e:
            if verbose:
                print(f"  0RW fallback: FAILED ({e})")
            return None

    def _extract_goal_quaternion(self, goals, t_start, duration, x_0, os_0):
        """Extract target quaternion from goals for plan quality evaluation.
        
        Returns the goal quaternion for Fixed_Attitude_Goal, or None for
        vector goals (ECI, Nadir, Sun, etc.) which use boresight error instead.
        """
        from ADCS.orbits.universal_constants import TimeConstants
        
        # Check all goals in the list for quaternion goals
        for goal in goals.goals:
            if hasattr(goal, 'q_ref'):
                return goal.q_ref
        
        # No quaternion goal found — will use boresight error metric instead
        return None
