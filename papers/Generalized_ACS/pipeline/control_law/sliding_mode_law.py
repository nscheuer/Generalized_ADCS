"""
Wisniewski-style sliding mode attitude control law.

Implements the control law from Wisniewski (2004)::

    s = J @ w_err + Lambda_q @ q_err
    tau = cross(w, J @ w + h_rw)
        + J @ cross(w, w_err)
        - Lambda_q @ q_err_dot[1:4]
        - Lambda_s @ s

This law includes gyroscopic compensation and its own
frame-rotation-like term internally, so the pipeline compensation
block should skip those terms.

The law requires extra state beyond the standard error signals:
``omega_raw`` and ``h_rw_body`` are passed via keyword arguments to
:meth:`compute`.
"""

__all__ = ["SlidingMode_Law"]

import numpy as np
from typing import Optional

from ADCS.pipeline.control_law.law_interface import ControlLaw
from ADCS.pipeline.data import LawInterface
from ADCS.helpers.math_helpers import Wmat


class SlidingMode_Law(ControlLaw):
    """Wisniewski sliding mode attitude control law.

    Parameters
    ----------
    J : ndarray, shape (3, 3)
        Spacecraft inertia matrix.
    lambda_q : ndarray, shape (3, 3)
        Positive definite attitude error gain matrix.
    lambda_s : ndarray, shape (3, 3)
        Positive definite sliding surface gain matrix.
    """

    def __init__(
        self,
        J: np.ndarray,
        lambda_q: np.ndarray,
        lambda_s: np.ndarray,
    ) -> None:
        self.J = np.asarray(J, dtype=float)
        self.lambda_q = np.asarray(lambda_q, dtype=float)
        self.lambda_s = np.asarray(lambda_s, dtype=float)
        self._interface = LawInterface(
            attitude_type='full',
            omega_type='omega_error',
            output_type='torque',
            includes_gyroscopic=True,
            includes_frame_rotation=True,
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
        """Compute sliding mode control torque.

        Parameters
        ----------
        attitude_input : ndarray, shape (3,)
            Quaternion error vector part q_err[1:4].
        omega_input : ndarray, shape (3,)
            Angular velocity error (omega - omega_ref_body).
        omega_raw : ndarray, shape (3,)
            Raw body angular velocity (passed via kwargs).
        h_rw_body : ndarray, shape (3,)
            Total RW momentum in body frame (passed via kwargs).

        Returns
        -------
        ndarray, shape (3,)
            Desired control torque in body frame.
        """
        q_err = attitude_input
        w_err = omega_input if omega_input is not None else np.zeros(3)
        w = kwargs.get('omega_raw', w_err)
        h_rw_body = kwargs.get('h_rw_body', np.zeros(3))

        J = self.J

        # Reconstruct full error quaternion from vector part
        q_err_sq = np.dot(q_err, q_err)
        q0 = np.sqrt(max(0.0, 1.0 - q_err_sq))
        q_err_full = np.array([q0, q_err[0], q_err[1], q_err[2]])

        # Quaternion error derivative: q_err_dot = 0.5 * w_err @ W(q_err)^T
        q_err_dot = 0.5 * w_err @ Wmat(q_err_full).T

        # Sliding surface
        s = J @ w_err + self.lambda_q @ q_err

        # Torque terms
        tau_gyro = np.cross(w, J @ w + h_rw_body)
        tau_frame = J @ np.cross(w, w_err)
        tau_q_err_dot = self.lambda_q @ q_err_dot[1:4]
        tau_sliding = self.lambda_s @ s

        return tau_gyro + tau_frame - tau_q_err_dot - tau_sliding
