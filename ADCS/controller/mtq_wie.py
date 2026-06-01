__all__ = ["MTQ_Wie"]

import numpy as np

from ADCS.controller.mtq_w_rw_LP import MTQ_w_RW_LP
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite


class MTQ_Wie(MTQ_w_RW_LP):
    r"""
    Wie eigenaxis quaternion-feedback regulator, routed through the framework
    LP torque allocation.

    This controller reuses :class:`MTQ_w_RW_LP`'s magnetic/RW LP allocation and
    torque-free desaturation verbatim (so it is the *same* actuator interface
    as the other framework laws) and overrides only the attitude control law in
    :meth:`_feedback_torque`. It implements the **inertia-weighted** quaternion
    feedback of

      B. Wie, *Space Vehicle Dynamics and Control*, 2nd ed., AIAA, 2008
      (quaternion-feedback regulator for eigenaxis rotations; see also
      Wie, Weiss & Arapostathis, "Quaternion Feedback Regulator for Spacecraft
      Eigenaxis Rotations," JGCD 12(3), 1989).

    .. math::

        \boldsymbol{\tau}_{\mathrm{fb}}
        = -J\,(k_p\,\mathbf{q}_{\mathrm{err}})
          - J\,(k_d\,\boldsymbol{\omega}_{\mathrm{err}})

    where :math:`\mathbf{q}_{\mathrm{err}}` is the (shortest-rotation) vector
    part of the attitude-error quaternion and :math:`J` is the spacecraft
    inertia. Pre-multiplying by :math:`J` is what distinguishes Wie's eigenaxis
    regulator from a plain *scalar* PD: on an asymmetric inertia it commands
    different per-axis torques (the low-inertia axis is driven less hard),
    yielding the near-eigenaxis closed-loop response Wie derives. The shared
    gyroscopic-decoupling term :math:`\boldsymbol{\omega}\times(J\boldsymbol{
    \omega}+\mathbf{h}_{rw})` is added by the base controller.

    The constructor signature matches :class:`MTQ_w_RW_LP` so the law is a
    drop-in swap in the framework's control pipeline.
    """

    def _feedback_torque(self, q_err: np.ndarray, w_err: np.ndarray,
                         est_sat: EstimatedSatellite,
                         w: np.ndarray = None) -> np.ndarray:
        J = np.asarray(est_sat.J_0, dtype=float)
        return -(J @ (self.p_gain * q_err)) - (J @ (self.d_gain * w_err))
