"""
Disturbance feedforward compensation.

Cancels known environmental disturbance torques using model-based
estimates.  The total feedforward is the negation of all enabled
disturbance models:

    tau_ff = -(tau_gg + tau_mag + tau_aero + tau_srp)

Each term is individually toggleable.
"""

__all__ = ["compute_disturbance_feedforward"]

import numpy as np

from ADCS.helpers.math_helpers import rot_mat, norm


# Earth gravitational parameter [km^3/s^2]
_MU = 398600.4418


def compute_disturbance_feedforward(
    q: np.ndarray,
    r_eci: np.ndarray,
    J: np.ndarray,
    B_body: np.ndarray = None,
    m_residual: np.ndarray = None,
    enable_gravity_gradient: bool = True,
    enable_magnetic: bool = False,
) -> np.ndarray:
    """Compute the disturbance feedforward torque.

    Parameters
    ----------
    q : ndarray, shape (4,)
        Attitude quaternion (Hamilton, scalar-first, body-to-ECI).
    r_eci : ndarray, shape (3,)
        Spacecraft position in ECI frame (km).
    J : ndarray, shape (3, 3)
        Spacecraft inertia matrix (kg*m^2).
    B_body : ndarray, shape (3,) or None
        Magnetic field vector in body frame (T).
    m_residual : ndarray, shape (3,) or None
        Residual magnetic dipole moment in body frame (A*m^2).
    enable_gravity_gradient : bool
        Include gravity-gradient torque.
    enable_magnetic : bool
        Include residual-dipole torque.

    Returns
    -------
    ndarray, shape (3,)
        Feedforward torque (negative of estimated disturbances).
    """
    tau_dist = np.zeros(3)

    if enable_gravity_gradient:
        r_mag = norm(r_eci)
        if r_mag > 1e-6:
            R_b2i = rot_mat(q)
            r_hat_body = R_b2i.T @ (np.asarray(r_eci).flatten() / r_mag)
            # tau_gg = (3 mu / r^3) * r_hat_b x (J @ r_hat_b)
            # r is in km, but J is kg*m^2 — keep units consistent:
            # mu/r^3 has units 1/s^2 when r in km and mu in km^3/s^2
            tau_gg = (3.0 * _MU / r_mag**3) * np.cross(r_hat_body, J @ r_hat_body)
            tau_dist += tau_gg

    if enable_magnetic and B_body is not None and m_residual is not None:
        tau_mag = np.cross(m_residual, B_body)
        tau_dist += tau_mag

    return -tau_dist
