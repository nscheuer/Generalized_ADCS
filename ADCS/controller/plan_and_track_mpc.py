from __future__ import annotations

__all__ = ["Plan_and_Track_SingleStepMPC", "_solve_bvls"]

import numpy as np
from typing import Optional
from numpy.typing import NDArray

from ADCS.CONOPS.goals import Goal
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite


def _solve_bvls(H: np.ndarray, g: np.ndarray,
                lb: np.ndarray, ub: np.ndarray,
                max_iter: int = 10) -> np.ndarray:
    r"""
    Solve a box-constrained QP via active-set iteration.

    .. math::

        \min_x \; \tfrac12 x^\top H x + g^\top x
        \quad\text{s.t.}\quad \text{lb} \le x \le \text{ub}

    For *n* <= 5 this converges in at most *n* iterations (one variable
    fixed to a bound per iteration).

    Parameters
    ----------
    H : (n, n) positive-semi-definite matrix.
    g : (n,)   linear cost vector.
    lb, ub : (n,) element-wise bounds.

    Returns
    -------
    x : (n,) bounded-optimal solution.
    """
    n = len(g)
    try:
        x = np.linalg.solve(H, -g)
    except np.linalg.LinAlgError:
        x = np.zeros(n)

    for _ in range(max_iter):
        at_lo = x < lb
        at_hi = x > ub
        x[at_lo] = lb[at_lo]
        x[at_hi] = ub[at_hi]

        free = ~(at_lo | at_hi)
        if not np.any(free):
            break

        idx = np.where(free)[0]
        g_eff = g.copy()
        fixed = np.where(~free)[0]
        for j in fixed:
            g_eff += H[:, j] * x[j]

        H_f = H[np.ix_(idx, idx)]
        g_f = g_eff[idx]
        try:
            x_f = np.linalg.solve(H_f, -g_f)
        except np.linalg.LinAlgError:
            break

        x_new = x.copy()
        x_new[idx] = x_f
        if np.allclose(x_new, x, atol=1e-12):
            x = x_new
            break
        x = x_new

    return np.clip(x, lb, ub)


# =============================================================================
# Single-Step MPC Tracker (K-Gain Weighted Forward Dynamics)
# =============================================================================

class Plan_and_Track_SingleStepMPC(Plan_and_Track_LQR):
    r"""
    Plan-and-Track with single-step forward-dynamics MPC and K-gain weighting.

    This tracker shares the *identical* ALTRO/SALTRO planning path as
    :class:`Plan_and_Track_LQR` (it subclasses it and reuses
    :meth:`calculate_trajectory` / :meth:`set_active_trajectory`), so a TVLQR
    run and an MPC run plan the **same** nominal trajectory and TVLQR gains.
    Only the per-step tracking law in :meth:`find_u` differs.

    At each control step this tracker:

    1. Forward-predicts the next state using ``noiseless_rk4`` with the
       **actual** orbital environment (actual B-field, not planned).
    2. Computes the predicted reduced-state error at *t + dt*.
    3. Finite-differences the dynamics w.r.t. control at the current state
       (4 extra RK4 evals) -- which uses the **actual** body-frame B-field.
    4. Solves a small **bounded** least-squares problem that minimises the
       K-gain-weighted next-step error subject to actuator saturation limits.

    The K-gain at *t + dt* encodes the trajectory's cost-to-go gradient: it
    tells us which state directions matter most for tracking the rest of the
    planned path.  Weighting the one-step prediction error by *K* couples the
    local correction to the global trajectory structure (the surrogate
    ``S ~ K^T K`` for the value-function Hessian).

    **Why bounded optimisation beats clipping:**

    When actuators saturate (the exact situation during high-error
    transients), clipping the unconstrained optimum preserves its
    *direction* but sacrifices optimality.  The bounded solve redistributes
    effort to unsaturated axes, producing a better achievable torque
    direction.  For MTQ where ``tau = m x B``, the direction of the magnetic
    moment matters as much as its magnitude.

    **Cost function:**

    .. math::

        \min_{\delta m}\;
        \bigl\| K(t{+}dt)\,\bigl[ e + A_m\,\delta m \bigr] \bigr\|^2
        + \lambda\,\|\delta m\|^2
        \quad\text{s.t.}\quad
        -m_{\max} \le u_{\text{ref}} + \delta m \le m_{\max}

    Parameters
    ----------
    est_sat : EstimatedSatellite
        Satellite model (provides ``dynJacCore``, ``noiseless_rk4``).
    planner_settings : PlannerSettings
        Planner configuration (used for trajectory generation).
    control_reg : float
        Regularisation weight lambda on ``||delta m||^2``.
    gain_scale : float, optional
        Scale applied to the K-gain weighting.  Defaults to 1.0 when reaction
        wheels are present, 0.5 for MTQ-only (matches the canonical tuning).
    """

    def __init__(
        self,
        est_sat: EstimatedSatellite,
        planner_settings,
        control_reg: float = 1e-3,
        gain_scale: Optional[float] = None,
        disable_goal_blend: bool = False,
    ) -> None:
        super().__init__(est_sat, planner_settings)

        self.est_sat = est_sat
        self._control_reg = control_reg
        # When True, find_u skips the goal-directed blend above 30 deg and runs
        # pure K-gain-weighted trajectory tracking (P2.4 tuning knob).
        self._disable_goal_blend = disable_goal_blend

        # MTQ-only gain scale: auto-detect if no reaction wheels
        if gain_scale is None:
            has_rw = len(est_sat.rw_actuators) > 0
            gain_scale = 1.0 if has_rw else 0.5
        self._gain_scale = gain_scale

        # Cache actuator info
        self._n_mtq = len(est_sat.mtq_actuators)
        self._m_max = (est_sat.mtq_actuators[0].u_max
                       if est_sat.mtq_actuators else 0.2)
        self._n_rw = len(est_sat.rw_actuators)
        self._has_rw = self._n_rw > 0
        self._n_ctrl = self._n_mtq + self._n_rw

        if self._has_rw:
            self._rw_axes = np.array([rw.axis for rw in est_sat.rw_actuators])
            self._rw_u_max = np.array([rw.u_max for rw in est_sat.rw_actuators])

    # ------------------------------------------------------------------
    def find_u(
        self,
        x_hat: NDArray[np.float64],
        sens: NDArray[np.float64],
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        goal: Optional[Goal] = None,
        **kwargs,
    ) -> NDArray[np.float64]:
        r"""
        Compute control via single-step forward-dynamics MPC.

        Hybrid strategy:

        - **Near the planned trajectory** (tracking error < ``switch_deg``):
          K-gain-weighted tracking that exploits the trajectory structure.
        - **Far from the planned trajectory**: goal-directed PD control with
          bounded MTQ allocation using the actual B-field.  Uses a desired
          torque ``tau = -kp*theta_err - kd*omega`` and solves for the best
          achievable MTQ moment.

        The transition between modes uses a smooth blending weight.
        """
        traj = self.active_trajectory
        if traj is None:
            return np.zeros(self._n_ctrl)

        ct = os_hat.J2000
        if not traj.is_valid_time(ct):
            raise RuntimeError("Plan_and_Track_SingleStepMPC: Trajectory expired")

        # ---- Trajectory look-ups ----------------------------------- #
        u_ref = traj.get_control_at(ct)
        u_nom = u_ref[:self._n_ctrl]

        sec2cent = TimeConstants.sec2cent
        dt = 1.0                                           # tracking dt [s]
        ct_next = min(ct + dt * sec2cent, traj.times[-1])
        x_ref_next = traj.get_state_at(ct_next)
        K_next = traj.get_gain_at(ct_next)                 # (n_ctrl, n_err)

        # ---- Sanitise current state -------------------------------- #
        x_curr = x_hat.copy()
        x_curr[3:7] /= np.linalg.norm(x_curr[3:7])
        os_next = kwargs.get('os_next', os_hat)

        # ---- Measure tracking error magnitude ---------------------- #
        e_now = traj._state_diff(x_curr, traj.get_state_at(ct))
        att_err_deg = np.degrees(np.linalg.norm(e_now[3:6]))  # ~rad for small

        # ---- 1. Forward-predict at u = u_nom ----------------------- #
        x_pred = est_sat.noiseless_rk4(
            x_curr.copy(), u_nom, dt, os_hat, os_next,
        )
        x_pred[3:7] /= np.linalg.norm(x_pred[3:7])

        # ---- 2. Finite-difference control Jacobian (RK4) ----------- #
        n_err = 6 + self._n_rw
        eps = 1e-6

        def _fd_Am(x0, u0, e0, x_ref):
            """Compute Am via finite-diff, error relative to x_ref."""
            Am = np.zeros((n_err, self._n_ctrl))
            for j in range(self._n_ctrl):
                u_p = u0.copy()
                u_p[j] += eps
                x_p = est_sat.noiseless_rk4(x0.copy(), u_p, dt, os_hat, os_next)
                x_p[3:7] /= np.linalg.norm(x_p[3:7])
                e_p = traj._state_diff(x_p, x_ref)
                Am[:, j] = (e_p - e0) / eps
            return Am

        # ----- Trajectory-tracking mode (K-weighted) ---------------- #
        e_traj = traj._state_diff(x_pred, x_ref_next)
        Am_traj = _fd_Am(x_curr, u_nom, e_traj, x_ref_next)

        if K_next is not None:
            K = self._gain_scale * K_next
        else:
            K = np.eye(n_err)
        KAm = K @ Am_traj
        Ke = K @ e_traj
        lam = self._control_reg
        H_traj = KAm.T @ KAm + lam * np.eye(self._n_ctrl)
        g_traj = KAm.T @ Ke

        # ----- Goal-directed mode (PD toward final goal) ------------- #
        # Use trajectory's FINAL state as the goal (best proxy for the
        # actual target without needing explicit goal access).
        x_final = traj.get_state_at(traj.times[-1])
        x_goal = np.zeros_like(x_curr)
        x_goal[3:7] = x_final[3:7]    # final goal quaternion
        x_goal[0:3] = 0.0             # zero angular velocity (rest at goal)
        if self._n_rw > 0:
            x_goal[7:] = x_final[7:]  # final RW momentum

        e_goal = traj._state_diff(x_pred, x_goal)
        # For goal mode, reuse same Am (same Jacobian, different error)
        Am_goal = _fd_Am(x_curr, u_nom, e_goal, x_goal)

        # PD weighting: attitude matters more than rate
        W_pd = np.diag(
            [1.0] * 3 + [10.0] * 3 + [1.0] * self._n_rw
        )
        WAm = W_pd @ Am_goal
        We = W_pd @ e_goal
        H_goal = WAm.T @ WAm + lam * np.eye(self._n_ctrl)
        g_goal = WAm.T @ We

        # ----- Smooth blend based on tracking error ----------------- #
        switch_deg = 30.0    # start blending
        full_goal_deg = 60.0  # fully goal-directed
        alpha = np.clip(
            (att_err_deg - switch_deg) / (full_goal_deg - switch_deg),
            0.0, 1.0
        )
        # Pure trajectory-tracking mode: skip the goal-directed blend entirely.
        if getattr(self, "_disable_goal_blend", False):
            alpha = 0.0
        H_blend = (1.0 - alpha) * H_traj + alpha * H_goal
        g_blend = (1.0 - alpha) * g_traj + alpha * g_goal

        # ---- Per-actuator bounds on dm (delta from u_nom) ---------- #
        lb = np.empty(self._n_ctrl)
        ub = np.empty(self._n_ctrl)
        lb[:self._n_mtq] = -self._m_max - u_nom[:self._n_mtq]
        ub[:self._n_mtq] = self._m_max - u_nom[:self._n_mtq]
        if self._has_rw:
            lb[self._n_mtq:] = -self._rw_u_max - u_nom[self._n_mtq:]
            ub[self._n_mtq:] = self._rw_u_max - u_nom[self._n_mtq:]

        # ---- Bounded active-set solve ------------------------------ #
        dm = _solve_bvls(H_blend, g_blend, lb, ub)

        # ---- Assemble output --------------------------------------- #
        u_out = u_nom + dm
        u_out[:self._n_mtq] = np.clip(
            u_out[:self._n_mtq], -self._m_max, self._m_max)
        if self._has_rw:
            u_out[self._n_mtq:] = np.clip(
                u_out[self._n_mtq:], -self._rw_u_max, self._rw_u_max)
        return u_out
