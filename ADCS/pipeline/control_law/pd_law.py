"""
Proportional-Derivative (PD) attitude control law.

Implements the Lovera/Wie-style PD law:

    tau = -(eps^2 * kp * q_err + eps * kd * omega_err)

This law expects full attitude error (quaternion vector part) and
angular velocity error. It does NOT include gyroscopic compensation
internally -- that is handled by the compensation block.
"""

__all__ = ["PD_Law"]

import numpy as np
from typing import Optional

from ADCS.pipeline.control_law.law_interface import ControlLaw
from ADCS.pipeline.data import LawInterface


class PD_Law(ControlLaw):
    """PD attitude control law with time-scale separation parameter.

    Parameters
    ----------
    kp : float
        Proportional gain.
    kd : float
        Derivative gain.
    eps : float
        Time-scale separation parameter (typically small, e.g. 0.1-1.0).
    """

    def __init__(self, kp: float, kd: float, eps: float) -> None:
        self.kp = kp
        self.kd = kd
        self.eps = eps
        self._interface = LawInterface(
            attitude_type='full',
            omega_type='omega_error',
            output_type='torque',
            includes_gyroscopic=False,
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
    ) -> np.ndarray:
        """Compute PD control torque.

        tau = -(eps^2 * kp * q_err + eps * kd * omega_err)

        Parameters
        ----------
        attitude_input : ndarray, shape (3,)
            Quaternion error vector part.
        omega_input : ndarray, shape (3,)
            Angular velocity error.

        Returns
        -------
        ndarray, shape (3,)
            Desired control torque in body frame.
        """
        q_err = attitude_input
        w_err = omega_input if omega_input is not None else np.zeros(3)
        return -(self.eps**2 * self.kp * q_err + self.eps * self.kd * w_err)
