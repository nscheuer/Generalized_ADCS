"""
Lovera-Astolfi magnetic attitude control law, as published.

    M. Lovera and A. Astolfi, "Global Magnetic Attitude Control of
    Inertially Pointing Spacecraft," Journal of Guidance, Control and
    Dynamics, Vol. 28, No. 5, 2005, pp. 1065-1072.

    tau = -(eps^2 * kp * q_err + eps * kd * w_err)
          + omega x (J @ omega + h_rw)

Unlike :class:`~ADCS.pipeline.control_law.pd_law.PD_Law`, this law performs
its **own** gyroscopic compensation, exactly as the paper writes it. It
therefore declares ``includes_gyroscopic=True`` so the pipeline's
compensation stage skips that term rather than adding it a second time.

That declaration is the whole point: the published law goes in unmodified,
and the adapter works out which feedforward terms it still owes. Wrapping
this law and asking for gyroscopic compensation anyway produces identical
output to not asking for it -- see
``testing/test_pipeline/test_lovera_law.py``.

The law itself is magnetorquer-agnostic: it emits a body torque. Whether
that torque reaches magnetorquers only (``method='magnetic_cross'``, which
reproduces the published controller exactly) or is spread across
magnetorquers *and* a reaction wheel (``method='lp'``) is Stage 5's
decision, not the law's.
"""

__all__ = ["Lovera_Law"]

import numpy as np
from typing import Optional

from ADCS.pipeline.control_law.law_interface import ControlLaw
from ADCS.pipeline.data import LawInterface


class Lovera_Law(ControlLaw):
    """Lovera-Astolfi magnetic PD law, gyroscopic term included internally.

    Parameters
    ----------
    J : ndarray, shape (3, 3)
        Spacecraft inertia matrix, needed for the law's own gyroscopic term.
    kp : float
        Proportional gain.
    kd : float
        Derivative gain.
    eps : float
        Time-scale separation parameter.
    """

    def __init__(self, J: np.ndarray, kp: float, kd: float,
                 eps: float = 1.0) -> None:
        self.J = np.asarray(J, dtype=float)
        self.kp = kp
        self.kd = kd
        self.eps = eps
        self._interface = LawInterface(
            attitude_type='full',
            omega_type='omega_error',
            output_type='torque',
            includes_gyroscopic=True,      # <- the law does this itself
            includes_frame_rotation=False,
            includes_disturbance_ff=False,
            includes_damping=True,
        )

    @property
    def interface(self) -> LawInterface:
        return self._interface

    def compute(
        self,
        attitude_input: np.ndarray,
        omega_input: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """PD feedback plus the law's own gyroscopic decoupling term.

        Parameters
        ----------
        attitude_input : ndarray, shape (3,)
            Quaternion error vector part.
        omega_input : ndarray, shape (3,)
            Angular velocity error.
        **kwargs
            ``omega_raw`` (body rate) and ``h_rw_body`` (wheel momentum in
            body frame), supplied by the pipeline controller.

        Returns
        -------
        ndarray, shape (3,)
            Desired torque in the body frame.
        """
        q_err = attitude_input
        w_err = omega_input if omega_input is not None else np.zeros(3)
        omega = kwargs.get('omega_raw')
        h_rw_body = kwargs.get('h_rw_body')
        if omega is None:
            omega = np.zeros(3)
        if h_rw_body is None:
            h_rw_body = np.zeros(3)

        tau_pd = -(self.eps ** 2 * self.kp * q_err
                   + self.eps * self.kd * w_err)
        tau_gyro = np.cross(omega, self.J @ omega + h_rw_body)
        return tau_pd + tau_gyro
