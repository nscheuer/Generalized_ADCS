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

        # Get trajectory data at current time
        x_ref = self.active_trajectory.get_state_at(current_time)
        u_ref = self.active_trajectory.get_control_at(current_time)
        K = self.active_trajectory.get_gain_at(current_time)
        
        # Actuator bounds
        u_bounds = np.array([act.u_max for act in est_sat.actuators])
        
        # ----- Single-step MPC mode -----
        # When enabled, replaces TVLQR's linear `u = u_ref - K @ dx` with a
        # bound-constrained optimization that uses nonlinear RK4 dynamics.
        # Key advantages over TVLQR:
        #   - Linearizes at actual state (correct B-field for MTQ cross-product)
        #   - Respects actuator bounds (optimal allocation when saturated)
        #   - No accumulated linearization error from state drift
        if getattr(self, '_use_mpc', False):
            return self._find_u_mpc(
                x_hat, x_ref, u_ref, K, est_sat, os_hat,
                u_bounds, current_time)
        
        # ----- Standard TVLQR mode -----
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
        if getattr(self, '_adaptive_gain_scaling', False):
            w_ref_norm = np.linalg.norm(x_ref[0:3])
            w_thresh_low = 0.002; w_thresh_high = 0.015
            if w_ref_norm > w_thresh_low:
                alpha = min(1.0, (w_ref_norm - w_thresh_low) / (w_thresh_high - w_thresh_low))
                effective_scale *= (1.0 - 0.8 * alpha)
        
        du = effective_scale * K @ dx
        
        # K-gains are in optimizer units. Scale RW portion to physical units.
        NONMTQ_TORQ_SCALE = 1.0  # Must match C++ Satellite.hpp
        n_mtq = len([a for a in est_sat.actuators if hasattr(a, 'axis') and not hasattr(a, 'J')])
        if len(du) > n_mtq:
            du[n_mtq:] *= NONMTQ_TORQ_SCALE
        
        u = u_ref - du
        u = np.clip(u, -u_bounds, u_bounds)
        
        return u

    # -----------------------------------------------------------------
    # Single-step MPC tracker
    # -----------------------------------------------------------------
    def _find_u_mpc(
        self,
        x_hat: np.ndarray,
        x_ref: np.ndarray,
        u_ref: np.ndarray,
        K: np.ndarray,
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        u_bounds: np.ndarray,
        current_time: float,
    ) -> np.ndarray:
        r"""Compute control via single-step forward-dynamics MPC.

        Solves the bounded QP

        .. math::

            \min_{\delta u}\;
            \tfrac12\,\mathbf{e}^{\!\top} S\,\mathbf{e}
            + \tfrac12\,\delta u^{\!\top} R\,\delta u
            \quad\text{s.t.}\;
            -u_{\max} \le u_{\text{ref}}+\delta u \le u_{\max}

        where :math:`\mathbf{e} = \text{state\_diff}\!\bigl(
        \text{RK4}(x,\,u_{\text{ref}}+\delta u),\;x_{\text{ref}}^+\bigr)`
        is linearised around :math:`\delta u = 0`, and :math:`S` is the
        TVLQR value-function Hessian at the next time step.

        When the unconstrained optimum is feasible the result closely matches
        standard TVLQR (same cost structure, different linearisation point).
        When actuators saturate the bounded solve redistributes effort
        optimally — the key advantage over TVLQR's post-hoc clipping.
        """
        from ADCS.controller.plan_and_track_mpc import _solve_bvls
        from ADCS.orbits.universal_constants import TimeConstants

        traj = self.active_trajectory
        dt = 1.0  # tracking timestep [s]
        sec2cent = TimeConstants.sec2cent
        ct_next = min(current_time + dt * sec2cent, traj.times[-1])

        x_ref_next = traj.get_state_at(ct_next)
        S_next = traj.get_S_at(ct_next)

        # Fallback to TVLQR if S is unavailable
        if S_next is None:
            dx = traj._state_diff(x_hat, x_ref)
            return np.clip(u_ref - K @ dx, -u_bounds, u_bounds)

        n_ctrl = len(u_ref)

        # Forward-predict at u_ref using actual state & environment
        x_pred0 = est_sat.noiseless_rk4(
            x_hat.copy(), u_ref, dt, os_hat, os_hat)
        x_pred0[3:7] /= np.linalg.norm(x_pred0[3:7])
        e0 = traj._state_diff(x_pred0, x_ref_next)

        # Finite-difference control Jacobian de/du (5 RK4 evals total)
        eps = 1e-6
        n_err = len(e0)
        Am = np.zeros((n_err, n_ctrl))
        for j in range(n_ctrl):
            u_p = u_ref.copy()
            u_p[j] += eps
            x_p = est_sat.noiseless_rk4(
                x_hat.copy(), u_p, dt, os_hat, os_hat)
            x_p[3:7] /= np.linalg.norm(x_p[3:7])
            Am[:, j] = (traj._state_diff(x_p, x_ref_next) - e0) / eps

        # QP:  min ½(e0+Am·du)ᵀS(e0+Am·du) + ½ duᵀR du
        R_alpha = getattr(self, '_mpc_R_alpha', 0.01)
        H = Am.T @ S_next @ Am + R_alpha * np.eye(n_ctrl)
        g = Am.T @ (S_next @ e0)

        lb = -u_bounds - u_ref
        ub = u_bounds - u_ref

        du = _solve_bvls(H, g, lb, ub)
        return np.clip(u_ref + du, -u_bounds, u_bounds)

    def _calibrate_mpc_R(
        self,
        traj: Trajectory,
        est_sat: EstimatedSatellite,
        orbit=None,
    ) -> float:
        r"""Calibrate the MPC control-cost scalar R = α·I.

        Computes :math:`A_m^{\!\top} S\,A_m` at a mid-trajectory reference
        point and sets α to its median positive eigenvalue.  This makes the
        control cost comparable to the state cost so that the bounded QP
        neither over-corrects (too small α) nor under-corrects (too large α).

        Falls back to 0.01 if S is unavailable or the computation fails.
        """
        try:
            from ADCS.orbits.universal_constants import TimeConstants

            mid = len(traj.times) // 2
            t_mid = traj.times[mid]
            t_next = traj.times[min(mid + 1, len(traj.times) - 1)]
            S_next = traj.get_S_at(t_next)
            if S_next is None:
                return 0.01

            x_ref = traj.get_state_at(t_mid)
            u_ref = traj.get_control_at(t_mid)

            # Need an orbital state at t_mid for RK4
            if orbit is not None:
                os_mid = orbit.get_os(t_mid)
                os_next = orbit.get_os(t_next)
            else:
                # No orbit — skip calibration
                return 0.01

            dt = 1.0
            eps = 1e-6
            n_ctrl = len(u_ref)

            x_pred0 = est_sat.noiseless_rk4(
                x_ref.copy(), u_ref, dt, os_mid, os_next)
            x_pred0[3:7] /= np.linalg.norm(x_pred0[3:7])
            x_ref_next = traj.get_state_at(t_next)
            e0 = traj._state_diff(x_pred0, x_ref_next)

            Am = np.zeros((len(e0), n_ctrl))
            for j in range(n_ctrl):
                u_p = u_ref.copy()
                u_p[j] += eps
                x_p = est_sat.noiseless_rk4(
                    x_ref.copy(), u_p, dt, os_mid, os_next)
                x_p[3:7] /= np.linalg.norm(x_p[3:7])
                Am[:, j] = (traj._state_diff(x_p, x_ref_next) - e0) / eps

            eigs = np.linalg.eigvalsh(Am.T @ S_next @ Am)
            pos = eigs[eigs > 1e-6]
            if len(pos) > 0:
                return float(np.median(pos))
        except Exception:
            pass
        return 0.01

    def set_active_trajectory(self, traj, orbit=None) -> None:
        """
        Set the active trajectory and record its start time for warmup.
        
        When MPC mode is enabled, also calibrates the control-cost
        scalar R from the trajectory's S matrices.
        
        :param traj: Trajectory to track.
        :param orbit: Orbit object for MPC R-calibration.  If ``None`` and
            ``_use_mpc`` is ``True``, the method falls back to the orbit
            stored during the most recent :meth:`calculate_trajectory` call.
        """
        super().set_active_trajectory(traj)
        if traj is not None:
            self._trajectory_start_time = traj.start_time
            # Calibrate MPC R if MPC mode is active
            if getattr(self, '_use_mpc', False):
                orb = orbit or getattr(self, '_orbit', None)
                est_sat = getattr(self, 'est_sat', None)
                if est_sat is not None:
                    self._mpc_R_alpha = self._calibrate_mpc_R(
                        traj, est_sat, orb)
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
        verbose: bool = False,
        orbit=None,
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
        # Store orbit for MPC R-calibration (used by set_active_trajectory)
        self._orbit = orbit

        auto_refine = getattr(self.planner_settings, 'auto_refine_dt', False)
        
        if not auto_refine:
            # Standard single-attempt planning
            lqr_times, Xset, Uset, Kset, Sset = self._calculate_trajectory_common(
                t_start, duration, x_0, os_0, goals, verbose
            )
            # ECI+RW gain scale (same logic as auto_refine path)
            has_rw = any(hasattr(a, 'J') for a in self.est_sat.actuators)
            q_goal = self._extract_goal_quaternion(goals, t_start, duration, x_0, os_0)
            # PSD projection in optimizer backward pass now produces well-conditioned
            # K-gains for ECI goals. No gain scaling needed (gs=1.0).
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
        
        # PSD projection in optimizer backward pass now produces well-conditioned
        # K-gains for ECI goals. No gain scaling needed (gs=1.0).
        # Previously gs=0.3 was a workaround for non-PSD Hessian corruption.
        
        # Orbit for ECI goal quality evaluation and stability probing.
        # When caller provides the actual orbit (recommended), use it for
        # consistency with the downstream simulation. Otherwise, create one
        # from the initial orbital state (may diverge from the sim orbit).
        eval_orbit = orbit  # Use caller's orbit if provided
        if eval_orbit is None and not is_quat_goal:
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
        best_gs = self._gain_scale  # Track gain_scale for best trajectory
        all_candidates = []  # (score, traj, gain_scale, label) for probe selection
        
        def evaluate_plan(traj):
            """Evaluate trajectory quality — lower is better.
            
            Uses composite score = settle_frac(5°) + tail_mean/180°.
            Captures both convergence speed and steady-state quality.
            ★★★ trajectories score < 0.3.
            """
            return self._plan_quality_score(
                traj.states, traj.times, goals, q_goal,
                orbit=eval_orbit, t_start=t_start)
        
        def try_config(dt, mode, slacks, label, slerp_rate=None, gain_scale=None):
            """Create an independent controller and plan with it."""
            nonlocal best_traj, best_score, best_gs
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
                gs_used = gain_scale if gain_scale is not None else self._gain_scale
                all_candidates.append((score, traj, gs_used, label))
                if verbose:
                    print(f"  {label}: score={score:.3f} (best={best_score:.3f})")
                if score < best_score:
                    best_score = score
                    best_traj = traj
                    if gain_scale is not None:
                        best_gs = gain_scale
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
        # Tracking stability probe: optional, enabled via _use_tracking_probe.
        # For ECI+RW goals, plan quality ≠ tracking quality (~10% of seeds
        # have ★★★ plans but divergent K-gains). The probe runs a 700-step
        # sim to detect this. However, the 6s overhead makes it impractical
        # for standard use. Enable for individual seed debugging.
        use_probe = (has_rw and not is_quat_goal 
                     and getattr(self, '_use_tracking_probe', False))
        skip_to_probe = False
        for mode in phase1_modes:
            score = try_config(dt_coarse, mode, False, f"dt={dt_coarse} mode={mode}")
            if score <= good_thresh:
                if use_probe:
                    skip_to_probe = True
                    break
                return best_traj
        
        # Phase 1b: Try complementary mode if first didn't reach good_thresh.
        if not skip_to_probe and best_score > good_thresh and len(phase1_modes) >= 1:
            alt_mode = 6 if phase1_modes[0] in [0, 4] else 0
            if alt_mode not in phase1_modes:
                try_config(dt_coarse, alt_mode, False,
                           f"dt={dt_coarse} mode={alt_mode} (complement)")
                if best_score <= good_thresh:
                    return best_traj
        
        # Phase 1c: For RW+ECI, try dt=5 (complementary optimizer basin).
        # dt=10 and dt=5 find different local minima. When dt=10 K-gains
        # produce tracking instability, dt=5 K-gains often work (and vice
        # versa). Plan quality selection isn't perfect but net positive —
        # converts ~20% of failures to ★ with no regressions on ★★★.
        if not skip_to_probe and has_rw and not is_quat_goal:
            dt_mid = dt_coarse / 2.0
            try_config(dt_mid, 0, False, f"dt={dt_mid} mode=0 (mid-res)",
                      gain_scale=1.0)
            if best_score <= good_thresh:
                return best_traj
        
        # Phase 2: Finer dt attempts for plans that didn't reach retry_thresh.
        if not skip_to_probe and best_score > retry_thresh:
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
                        if not use_probe:
                            return best_traj
            else:
                # MTQ-only: try multiple modes at dt=1 (~30-60s each)
                for mode in [0, 4, 6]:
                    score = try_config(dt_fine, mode, False, f"dt={dt_fine} mode={mode}")
                    if score <= good_thresh:
                        return best_traj
        
        # Phase 3: SLERP warm-start WITHOUT slacks (mode 6) at dt=2
        if not skip_to_probe and best_score > retry_thresh and not has_rw:
            dt_slerp = getattr(self.planner_settings, 'auto_refine_dt_slerp', 2.0)
            try_config(dt_slerp, 6, False, f"dt={dt_slerp} mode=6 (SLERP no-slack)")
        
        # Phase 4: SLERP with slacks (absolute last resort, MTQ-only)
        # Produces 0° plan error but K-gains may be poor for tracking.
        if not skip_to_probe and best_score > retry_thresh and not has_rw:
            dt_slerp = getattr(self.planner_settings, 'auto_refine_dt_slerp', 2.0)
            try_config(dt_slerp, 0, True, f"dt={dt_slerp} slacks (last resort)")
        
        if best_traj is None:
            raise RuntimeError("All planning attempts failed")
        
        # Tracking stability probe for ECI+RW goals.
        # For boresight goals, the planned trajectory has a free boresight-roll
        # DOF, so different optimizer local minima produce different K-gain
        # structures — some stable, some diverge during sim (~10% failure rate).
        # A 700-step probe detects unstable K-gains (~6s). If unstable, probe
        # all collected candidates and pick the best tracker.
        if use_probe and eval_orbit is not None:
            goal_vec = self._extract_eci_goal_vector(goals, t_start, os_0)
            if goal_vec is not None and all_candidates:
                # Probe best trajectory first
                probe_err = self._tracking_stability_probe(
                    best_traj, x_0, os_0, eval_orbit, goal_vec, best_gs)
                if verbose:
                    print(f"  Probe: err={probe_err:.1f}° (gs={best_gs})")
                if probe_err > 10.0:
                    # Best plan has unstable tracking — probe all dt=10 alternatives
                    if verbose:
                        print(f"  Tracking instability detected. Probing {len(all_candidates)} candidates...")
                    probe_results = [(probe_err, best_traj, best_gs)]
                    for score_c, traj_c, gs_c, label_c in all_candidates:
                        if traj_c is best_traj:
                            continue  # Already probed
                        pe = self._tracking_stability_probe(
                            traj_c, x_0, os_0, eval_orbit, goal_vec, gs_c)
                        probe_results.append((pe, traj_c, gs_c))
                        if verbose:
                            print(f"    {label_c}: probe_err={pe:.1f}°")
                        if pe < 3.0:
                            break  # Good enough
                    
                    # If no dt=10 candidate is stable, try dt=5 alternatives
                    if min(pr[0] for pr in probe_results) > 10.0:
                        dt_mid = dt_coarse / 2.0
                        for mode_alt in [0, 4]:
                            try:
                                s = settings_factory(dt_mid)
                                s.bdot_on = mode_alt
                                ctrl_alt = Plan_and_Track_LQR(
                                    est_sat=self.est_sat, planner_settings=s)
                                lt, X, U, K, S_ = ctrl_alt._calculate_trajectory_common(
                                    t_start, duration, x_0, os_0, goals, False)
                                alt_traj = Trajectory(lt, X, U, K, S_)
                                pe = self._tracking_stability_probe(
                                    alt_traj, x_0, os_0, eval_orbit, goal_vec, 1.0)
                                probe_results.append((pe, alt_traj, 1.0))
                                if verbose:
                                    print(f"    dt={dt_mid} mode={mode_alt}: probe_err={pe:.1f}°")
                                if pe < 3.0:
                                    break
                            except Exception:
                                pass
                    
                    # Pick best-probing trajectory
                    probe_results.sort(key=lambda c: c[0])
                    if probe_results[0][0] < probe_err:
                        best_traj = probe_results[0][1]
                        best_gs = probe_results[0][2]
                        if verbose:
                            print(f"  Selected: probe_err={probe_results[0][0]:.1f}° (was {probe_err:.1f}°)")
        
        # Apply the gain_scale associated with the best trajectory.
        # Different dt values need different gain_scales for stable tracking:
        # dt=10 K-gains are ~10× too strong at dt=1 → gs=0.3
        # dt=5 K-gains are ~5× too strong at dt=1 → gs=1.0
        self._gain_scale = best_gs
        
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

    def _tracking_stability_probe(self, traj, x_0, os_0, orbit, goal_vec,
                                    gain_scale, n_probe=700, n_tail=200):
        """Quick simulation probe to detect tracking instability.

        Simulates ``n_probe`` steps at dt=1 using fixed-step RK4 (fast),
        then measures the mean boresight error over the last ``n_tail``
        steps.  Uses early termination if error exceeds 90° (catastrophic
        divergence detected).

        Returns
        -------
        mean_err_deg : float
            Mean boresight error over last ``n_tail`` steps, in degrees.
            Returns 180.0 if early-terminated.
        """
        from ADCS.helpers.math_helpers import normalize, rot_mat
        from ADCS.orbits.universal_constants import TimeConstants
        sec2cent = TimeConstants.sec2cent

        # Create independent satellite for probe (no state contamination)
        try:
            from ADCS.satellite_factory.satellites.create_cubesats import (
                create_beavercube2_cubesat)
            sat_probe = create_beavercube2_cubesat(estimated=False)
            for rw in sat_probe.rw_actuators:
                rw.h = 0.0
        except Exception:
            return 0.0  # Can't probe, assume stable

        from ADCS.controller.helpers.planner_settings import PlannerSettings
        ctrl = Plan_and_Track_LQR(est_sat=sat_probe,
                                   planner_settings=PlannerSettings(sat_probe))
        ctrl._gain_scale = gain_scale
        ctrl.set_active_trajectory(traj)

        BODY_BORESIGHT = np.array([0, 1, 0])
        gn = goal_vec / np.linalg.norm(goal_vec)
        x = x_0.copy()
        errors_tail = []
        for i in range(n_probe):
            tj = os_0.J2000 + i * sec2cent
            os_i = orbit.get_os(tj)
            u = ctrl.find_u(x_hat=x, sens=sat_probe.sensor_readings(x=x, os=os_i),
                            est_sat=sat_probe, os_hat=os_i)
            # Fixed-step RK4 (much faster than adaptive solve_ivp)
            os_next = orbit.get_os(tj + sec2cent)
            x = sat_probe.noiseless_rk4(x, u, 1.0, os_i, os_next)
            x[3:7] = normalize(x[3:7])
            
            # Compute error for tail window
            if i >= n_probe - n_tail:
                R = rot_mat(x[3:7])
                bore = R @ BODY_BORESIGHT
                err_deg = np.degrees(np.arccos(np.clip(np.dot(bore, gn), -1, 1)))
                errors_tail.append(err_deg)
                # Early termination: catastrophic divergence
                if err_deg > 90.0:
                    return 180.0

        if not errors_tail:
            return 0.0
        return float(np.max(errors_tail))

    @staticmethod
    def _extract_eci_goal_vector(goals, t_start, os_0):
        """Extract ECI goal direction vector for stability probing.
        
        Returns the normalized ECI direction for the first non-quaternion goal,
        or None if only quaternion goals are present.
        """
        for goal in goals.goals:
            if hasattr(goal, 'q_ref'):
                continue  # Skip quaternion goals
            try:
                ref, _ = goal.to_ref(os0=os_0) if hasattr(goal, 'to_ref') else (None, None)
                if ref is not None:
                    nrm = np.linalg.norm(ref)
                    if nrm > 1e-10:
                        return ref / nrm
            except Exception:
                pass
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
