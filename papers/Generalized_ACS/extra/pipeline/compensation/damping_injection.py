"""
Damping injection for control laws that do not use angular velocity.

When a control law declares ``omega_type='no_omega'``, the goal
formulation stage provides no angular velocity error to the law.
This compensator injects damping externally:

    tau_damp = -k_d * P @ (omega - omega_ref_body)

The projection matrix P ensures that damping is only applied in
the constrained directions:

    full attitude:      P = I        (3-axis damping)
    reduced attitude:   P = I - b b^T  (2-axis, boresight axis free)
    no goal:            P = 0        (no damping)
"""

__all__ = ["compute_damping_injection"]

import numpy as np


def compute_damping_injection(
    omega: np.ndarray,
    omega_ref_body: np.ndarray,
    P: np.ndarray,
    k_d: float,
) -> np.ndarray:
    """Compute the damping injection torque.

    Parameters
    ----------
    omega : ndarray, shape (3,)
        Current body angular velocity (rad/s).
    omega_ref_body : ndarray, shape (3,)
        Reference angular velocity in body frame (rad/s).
    P : ndarray, shape (3, 3)
        Projection matrix from goal formulation.
    k_d : float
        Damping gain (scalar applied to all projected axes).

    Returns
    -------
    ndarray, shape (3,)
        Damping torque: -k_d * P @ (omega - omega_ref_body).
    """
    omega_err = omega - omega_ref_body
    return -k_d * (P @ omega_err)
