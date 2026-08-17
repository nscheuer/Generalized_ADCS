"""
Frame rotation feedforward torque.

Compensates for the reference frame's rotational dynamics so the
control law can treat tracking as regulation.

    tau_frame = J @ omega_ref_dot - cross(omega_ref, J @ omega_ref)

The reference angular-velocity derivative is computed via finite
differencing or (when available) analytically.
"""

__all__ = ["compute_frame_rotation_torque"]

import numpy as np


def compute_frame_rotation_torque(
    omega_ref_body: np.ndarray,
    omega_ref_body_prev: np.ndarray,
    J: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Compute the frame rotation feedforward torque.

    Parameters
    ----------
    omega_ref_body : ndarray, shape (3,)
        Current reference angular velocity in body frame.
    omega_ref_body_prev : ndarray, shape (3,)
        Previous-step reference angular velocity in body frame.
    J : ndarray, shape (3, 3)
        Spacecraft inertia matrix.
    dt : float
        Timestep for finite differencing (s).

    Returns
    -------
    ndarray, shape (3,)
        Frame rotation feedforward torque.
    """
    omega_ref_dot = (omega_ref_body - omega_ref_body_prev) / dt
    return J @ omega_ref_dot - np.cross(omega_ref_body, J @ omega_ref_body)
