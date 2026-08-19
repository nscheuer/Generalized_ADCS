"""Disturbance feedforward -- the controller side of Section IV-A's cancellation claim.

**None of the library controllers do any disturbance feedforward.** ``MTQ_w_RW_LP`` and the
base ``Controller`` contain no reference to the estimated disturbance state at all. So the
campaign was carrying a residual dipole in the estimator's augmented state, converging it to
~10% error over an orbit, and then never using it. Section IV-A's cancellation result was
untested by the campaign built to demonstrate it, and the PD steady-state floor
(``tau_dist / kp`` ~ 2.7 deg) was measured against the **full uncancelled** disturbance.

Two feedforward routes, and the distinction matters:

**Dipole feedforward (exact).** A residual dipole produces ``tau = m_res x B``. The
magnetorquers produce ``tau = m_cmd x B`` through the *same* cross product, so commanding
``m_cmd = -m_res_est`` cancels it **identically at every instant and every field
orientation** -- no inversion, no geometry loss, nothing left along B. This is what IV-A
actually claims and it is exact up to the estimate error.

**Torque feedforward (approximate).** Adding ``-tau_dist_est`` to the requested torque before
allocation is the general route for disturbances with no actuator sharing their structure. It
is *not* equivalent for a dipole: the allocator can only deliver the component perpendicular
to **B**, so the along-field part is silently dropped, and a direction-preserving LP will
scale the whole request rather than deliver part of it.

Both are provided. The dipole route is the default because it is the one the paper claims.

The cost is real and should be reported rather than hidden: cancelling permanently reserves
``|m_res| / m_max`` of the magnetorquer command box -- 25% at the reference dipole against
0.2 A m^2, 8% against 0.6 A m^2 -- which is unavailable for control for the whole mission.
"""

from __future__ import annotations

__all__ = ["FeedforwardLP"]

from typing import Any, Optional

import numpy as np

from ADCS.controller import MTQ_w_RW_LP
from ADCS.satellite_hardware.actuators import MTQ
from ADCS.satellite_hardware.disturbances import Dipole_Disturbance


class FeedforwardLP(MTQ_w_RW_LP):
    """LP-allocated PD with estimated-disturbance feedforward.

    :param mode: ``"dipole"`` cancels the estimated residual dipole by commanding
        ``-m_res_est`` on the magnetorquers (exact). ``"torque"`` subtracts the estimated
        disturbance torque from the request before allocation (approximate, and lossy along
        **B**). ``"none"`` reproduces the stock controller exactly.
    :param ff_gain: Fraction of the estimate applied, 0..1. Below 1 it trades cancellation
        for robustness to a bad estimate; the campaign reports the sweep rather than assuming
        1.0 is safe.
    """

    def __init__(self, *args, mode: str = "dipole", ff_gain: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if mode not in ("dipole", "torque", "both", "none"):
            raise ValueError(f"mode must be dipole|torque|both|none, got {mode!r}")
        self.ff_mode = mode
        self.ff_gain = float(ff_gain)
        #: Dipole actually commanded for cancellation last step [A m^2] -- report the duty
        #: it consumes rather than leaving the reservation implicit.
        self.last_ff_dipole = np.zeros(3)

    # -- helpers ------------------------------------------------------------------------
    @staticmethod
    def _estimated_dipole(est_sat) -> Optional[np.ndarray]:
        for d in getattr(est_sat, "disturbances", []):
            if isinstance(d, Dipole_Disturbance) and getattr(d, "estimate_dist", False):
                return np.ravel(np.asarray(d.main_param, float))[:3]
        return None

    @staticmethod
    def _estimated_dist_torque(est_sat, x_hat, os_hat) -> np.ndarray:
        tau = np.zeros(3)
        for d in getattr(est_sat, "disturbances", []):
            if not getattr(d, "estimate_dist", False):
                continue
            try:
                tau = tau + np.ravel(d.torque(sat=est_sat, x=x_hat, os=os_hat))[:3]
            except TypeError:
                tau = tau + np.ravel(d.torque(x=x_hat, os=os_hat))[:3]
        return tau

    # -- control ------------------------------------------------------------------------
    def find_u(self, x_hat, sens, est_sat, os_hat, goal, *args, **kwargs):
        u = np.asarray(super().find_u(x_hat, sens, est_sat, os_hat, goal,
                                      *args, **kwargs), float)
        self.last_ff_dipole = np.zeros(3)
        if self.ff_mode == "none" or self.ff_gain == 0.0:
            return u

        mtq_idx = [i for i, a in enumerate(est_sat.actuators) if isinstance(a, MTQ)]
        if not mtq_idx:
            return u

        if self.ff_mode in ("dipole", "both"):
            m_est = self._estimated_dipole(est_sat)
            if m_est is None:
                m_est = np.zeros(3)
            # Command the negative of the estimated dipole. Exact: the disturbance and the
            # actuator share the cross product, so this cancels at every field orientation
            # rather than at one instant.
            m_ff = -self.ff_gain * m_est
            if self.ff_mode == "both":
                # Lumped remainder (drag + GG + SRP) has no actuator sharing its structure,
                # so it goes through the approximate torque route on top of the exact dipole
                # cancellation. Only its perpendicular-to-B part is deliverable.
                tau_g = np.zeros(3)
                for d in getattr(est_sat, "disturbances", []):
                    if (getattr(d, "estimate_dist", False)
                            and not isinstance(d, Dipole_Disturbance)):
                        tau_g = tau_g + np.ravel(np.asarray(d.main_param, float))[:3]
                B = np.ravel(np.asarray(os_hat.B, float))[:3]
                Bn2 = float(B @ B)
                if Bn2 > 1e-24:
                    m_ff = m_ff - self.ff_gain * np.cross(B, -tau_g) / Bn2
        else:
            tau_est = self._estimated_dist_torque(est_sat, x_hat, os_hat)
            # Minimum-norm dipole giving -tau_est: m = (B x tau) / |B|^2. Only the component
            # perpendicular to B is achievable at all; the rest is unavoidably lost.
            B = np.ravel(np.asarray(os_hat.B, float))[:3]
            Bn2 = float(B @ B)
            if Bn2 < 1e-24:
                return u
            m_ff = -self.ff_gain * np.cross(B, -tau_est) / Bn2

        # Add onto the allocator's command, then re-clip: the feedforward shares the same
        # box as the control effort and must not silently exceed it.
        axes = np.column_stack([np.ravel(est_sat.actuators[i].axis)[:3] for i in mtq_idx])
        cmd = np.linalg.pinv(axes) @ m_ff
        u[mtq_idx] = u[mtq_idx] + cmd
        for j, i in enumerate(mtq_idx):
            lim = float(np.ravel(est_sat.actuators[i].u_max)[0])
            u[mtq_idx[j]] = float(np.clip(u[mtq_idx[j]], -lim, lim))
        self.last_ff_dipole = m_ff
        # Controller-side wheel envelope on the ESTIMATED momentum: do not command torque a
        # saturated wheel cannot produce (mirrors the harness driver; prevents windup).
        from ADCS.satellite_hardware.actuators import RW as _RW
        rw_idx = [i for i, a in enumerate(est_sat.actuators) if isinstance(a, _RW)]
        xh = np.ravel(np.asarray(x_hat, float))
        for j, i in enumerate(rw_idx):
            if xh.size >= 8 + j:
                h_j = float(xh[7 + j])
                hmx = float(np.ravel(est_sat.actuators[i].h_max)[0])
                # hdot = -u (empirical; see enforce_wheel_envelope): at +h_max forbid
                # NEGATIVE u, which is what raises h.
                if (h_j >= hmx * (1 - 1e-12) and u[i] < 0) or \
                   (h_j <= -hmx * (1 - 1e-12) and u[i] > 0):
                    u[i] = 0.0
        return u
