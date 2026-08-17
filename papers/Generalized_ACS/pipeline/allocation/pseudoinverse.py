"""
Pseudoinverse allocation: Moore-Penrose pinv + clip.

Computes u = pinv(B_tau) @ tau_desired and clips to bounds.
Simple baseline for fully actuated systems where bounds are rarely hit.
"""

__all__ = ["allocate_pseudoinverse"]

import numpy as np

from ADCS.pipeline.data import AllocationResult


def allocate_pseudoinverse(
    tau_desired: np.ndarray,
    B_tau: np.ndarray,
    u_min: np.ndarray,
    u_max: np.ndarray,
    n_actuators: int,
    group_indices: np.ndarray,
) -> AllocationResult:
    """Allocate torque via pseudoinverse with clipping.

    Parameters
    ----------
    tau_desired : ndarray, shape (3,)
        Desired torque in body frame.
    B_tau : ndarray, shape (3, n)
        Torque effectiveness matrix.
    u_min, u_max : ndarray, shape (n,)
        Actuator command bounds.
    n_actuators : int
        Total number of actuators in the spacecraft.
    group_indices : ndarray, shape (n,)
        Mapping from B_tau columns to actuator command vector indices.

    Returns
    -------
    AllocationResult
    """
    n = B_tau.shape[1]
    tau_norm = np.linalg.norm(tau_desired)

    if tau_norm < 1e-12 or n == 0:
        return AllocationResult(
            u=np.zeros(n_actuators),
            tau_achieved=np.zeros(3),
            alpha=1.0 if tau_norm < 1e-12 else 0.0,
            direction_error=0.0,
            feasible=True,
        )

    # Pseudoinverse solution
    u_opt = np.linalg.pinv(B_tau) @ tau_desired

    # Clip to bounds
    u_clipped = np.clip(u_opt, u_min, u_max)
    clipped = not np.allclose(u_opt, u_clipped, atol=1e-12)

    # Achieved torque
    tau_achieved = B_tau @ u_clipped
    tau_ach_norm = np.linalg.norm(tau_achieved)

    # Direction error
    if tau_ach_norm > 1e-12:
        cos_angle = np.clip(
            np.dot(tau_achieved, tau_desired) / (tau_ach_norm * tau_norm),
            -1.0, 1.0,
        )
        direction_error = float(np.arccos(cos_angle))
    else:
        direction_error = 0.0

    # Alpha: projection onto desired direction
    tau_hat = tau_desired / tau_norm
    alpha = float(np.dot(tau_achieved, tau_hat)) / tau_norm

    # Pack into full command vector
    u_out = np.zeros(n_actuators)
    u_out[group_indices] = u_clipped

    return AllocationResult(
        u=u_out,
        tau_achieved=tau_achieved,
        alpha=max(0.0, alpha),
        direction_error=direction_error,
        feasible=not clipped,
    )
