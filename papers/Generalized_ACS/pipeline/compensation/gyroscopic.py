"""
Gyroscopic compensation torque.

Computes the gyroscopic coupling term that appears in the Euler
equation and must be compensated to decouple the rotational axes:

    tau_gyro = omega x (J @ omega + h_rw_body)
"""

__all__ = ["compute_gyroscopic_torque"]

import numpy as np


def compute_gyroscopic_torque(
    omega: np.ndarray,
    J: np.ndarray,
    h_rw_body: np.ndarray,
) -> np.ndarray:
    """Compute the gyroscopic compensation torque.

    Parameters
    ----------
    omega : ndarray, shape (3,)
        Body angular velocity.
    J : ndarray, shape (3, 3)
        Spacecraft inertia matrix.
    h_rw_body : ndarray, shape (3,)
        Total reaction wheel angular momentum in body frame.

    Returns
    -------
    ndarray, shape (3,)
        Gyroscopic torque: omega x (J @ omega + h_rw_body).
    """
    return np.cross(omega, J @ omega + h_rw_body)
