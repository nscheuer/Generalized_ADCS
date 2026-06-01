"""Framework-allocated (MTQ+RW) variants of the Lovera and Wisniewski laws.

The published Lovera and Wisniewski controllers are *magnetic-only*: they
compute a desired body torque, then project it onto the plane perpendicular to
B and command magnetorquers only (the reaction wheel is never actuated). These
subclasses keep the **control law** identical but route the desired torque
through :class:`MTQ_w_RW_LP`'s LP allocation + torque-free desaturation, so the
reaction wheel *is* used and all four Paper-1 laws share one allocation
interface (apples-to-apples on actuation). This departs from the strict
published formulations — use the plain ``MTQ_Lovera`` / ``MTQ_Wisniewski`` for
the magnetic-only versions.
"""

__all__ = ["MTQ_Lovera_LP", "MTQ_Wisniewski_LP"]

import numpy as np

from ADCS.controller.mtq_w_rw_LP import MTQ_w_RW_LP
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.helpers.math_helpers import Wmat


class MTQ_Lovera_LP(MTQ_w_RW_LP):
    r"""Lovera-Astolfi PD law (``-eps^2 k_p q_err - eps k_d w_err``) routed
    through the framework MTQ+RW LP allocation. Gyroscopic compensation is
    added by the base controller."""

    def __init__(self, est_sat: EstimatedSatellite, p_gain: float,
                 d_gain: float, eps: float, c_gain: float = 1e-3,
                 h_target=np.zeros(3)) -> None:
        super().__init__(est_sat, p_gain=p_gain, d_gain=d_gain,
                         c_gain=c_gain, h_target=h_target)
        self.eps = eps

    def _feedback_torque(self, q_err, w_err, est_sat, w=None):
        return -(self.eps ** 2 * self.p_gain * q_err
                 + self.eps * self.d_gain * w_err)


class MTQ_Wisniewski_LP(MTQ_w_RW_LP):
    r"""Wisniewski LTV sliding-mode law routed through the framework MTQ+RW LP
    allocation.

    Feedback torque (gyro added by the base):

    .. math::
        s = J w_{err} + \Lambda_q q_{err}, \quad
        \tau_{fb} = J(\omega\times w_{err}) - \Lambda_q \dot q_{err}
                    - \Lambda_s s
    """

    def __init__(self, est_sat: EstimatedSatellite, lambda_s: np.ndarray,
                 lambda_q: np.ndarray, c_gain: float = 1e-3,
                 h_target=np.zeros(3)) -> None:
        # p_gain/d_gain unused (control law overridden); kept for base init.
        super().__init__(est_sat, p_gain=0.0, d_gain=0.0,
                         c_gain=c_gain, h_target=h_target)
        self.lambda_s = np.asarray(lambda_s, float)
        self.lambda_q = np.asarray(lambda_q, float)

    def _feedback_torque(self, q_err, w_err, est_sat, w=None):
        if w is None:
            w = w_err
        J = np.asarray(est_sat.J_0, float)
        s = J @ w_err + self.lambda_q @ q_err
        q_err_full = np.hstack(
            ([np.sqrt(max(0.0, 1.0 - np.dot(q_err, q_err)))], q_err))
        q_err_dot = 0.5 * w_err @ Wmat(q_err_full).T
        tau_frame = J @ np.cross(w, w_err)
        tau_q_err_dot = self.lambda_q @ q_err_dot[1:4]
        tau_sliding = self.lambda_s @ s
        return tau_frame - tau_q_err_dot - tau_sliding
