"""Feedback-linearized MRP attitude tracking controller."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from numba import njit
from numpy.typing import ArrayLike

from GenADCS.backends.python.helpers.math.rotations.mrp.kernels import (
    quaternion_to_mrp,
)
from GenADCS.backends.python.helpers.math.vector import skew

from .feedback import FB_MRP_FEEDBACK_LINEARIZED, Feedback, to_gain_matrix


@njit(cache=True, inline="always")
def _mrp_b_matrix(sigma: np.ndarray) -> np.ndarray:
    """MRP kinematic matrix B(sigma) = (1 - sigma^T sigma) I + 2 [sigma]_x + 2 sigma sigma^T."""
    s2 = np.dot(sigma, sigma)
    return (
        (1.0 - s2) * np.eye(3)
        + 2.0 * skew(sigma)
        + 2.0 * np.outer(sigma, sigma)
    )


@njit(cache=True, inline="always")
def _mrp_b_dot_matrix(sigma: np.ndarray, sigma_dot: np.ndarray) -> np.ndarray:
    """Time derivative B_dot(sigma, sigma_dot)."""
    s_dot_s = np.dot(sigma, sigma_dot)
    return (
        -2.0 * s_dot_s * np.eye(3)
        + 2.0 * skew(sigma_dot)
        + 2.0 * (np.outer(sigma_dot, sigma) + np.outer(sigma, sigma_dot))
    )


@njit(cache=True)
def mrp_feedback_linearized_step(
    q: np.ndarray,
    omega: np.ndarray,
    q_d: np.ndarray,
    omega_d: np.ndarray,
    omega_d_dot: np.ndarray,
    ff_torque: np.ndarray,
    wheel_momenta: np.ndarray,
    total_inertia: np.ndarray,
    bus_inertia: np.ndarray,
    wheel_axes: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    """Compute feedback-linearized reference torque tau_c = omega x H + J_noRW * nu."""
    # GenADCS quaternions are body-from-inertial (q_b_from_i), so the MRP whose
    # kinematics satisfy sigma_dot = +1/4 B(sigma) omega is -quaternion_to_mrp(q).
    sigma = -quaternion_to_mrp(q)
    sigma_d = -quaternion_to_mrp(q_d)

    # Keep desired MRP on the nearest branch to avoid shadow-set discontinuities.
    sd2 = np.dot(sigma_d, sigma_d)
    if sd2 > 1e-12:
        sigma_d_shadow = -sigma_d / sd2
        diff_direct = sigma - sigma_d
        diff_shadow = sigma - sigma_d_shadow
        if np.dot(diff_shadow, diff_shadow) < np.dot(diff_direct, diff_direct):
            sigma_d = sigma_d_shadow

    b = _mrp_b_matrix(sigma)
    sigma_dot = 0.25 * (b @ omega)
    b_dot = _mrp_b_dot_matrix(sigma, sigma_dot)

    b_d = _mrp_b_matrix(sigma_d)
    sigma_d_dot = 0.25 * (b_d @ omega_d)
    b_d_dot = _mrp_b_dot_matrix(sigma_d, sigma_d_dot)
    sigma_d_ddot = 0.25 * (b_d_dot @ omega_d) + 0.25 * (b_d @ omega_d_dot)

    e = sigma - sigma_d
    e_dot = sigma_dot - sigma_d_dot

    s2 = np.dot(sigma, sigma)
    b_inv = b.T / ((1.0 + s2) ** 2)
    nu = 4.0 * (b_inv @ (-0.25 * (b_dot @ omega) + sigma_d_ddot - kd @ e_dot - kp @ e))

    angular_momentum = total_inertia @ omega
    for w in range(wheel_axes.shape[0]):
        angular_momentum += wheel_axes[w] * wheel_momenta[w]

    return np.cross(omega, angular_momentum) + bus_inertia @ nu + ff_torque


class MRPFeedbackLinearized(Feedback):
    """Feedback-linearized MRP tracking controller: ``e_ddot + K_D e_dot + K_P e = 0``."""

    kind: ClassVar[int] = FB_MRP_FEEDBACK_LINEARIZED

    def __init__(self, kp: float | ArrayLike, kd: float | ArrayLike) -> None:
        self.kp = to_gain_matrix(kp, "kp", require_spd=True)
        self.kd = to_gain_matrix(kd, "kd", require_spd=False)

    def to_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        return self.kp, self.kd
