"""
QP allocation: bounded least-squares minimization of torque error.

    min  ||B_tau @ u - tau_desired||^2_W + lambda_reg * ||u||^2
    s.t. u_min <= u <= u_max

Uses scipy.optimize.lsq_linear (TRF method), which handles
rank-deficient B_tau robustly.
"""

__all__ = ["allocate_qp"]

import numpy as np
from scipy.optimize import lsq_linear

from ADCS.pipeline.data import AllocationConfig, AllocationResult


def allocate_qp(
    tau_desired: np.ndarray,
    B_tau: np.ndarray,
    u_min: np.ndarray,
    u_max: np.ndarray,
    n_actuators: int,
    group_indices: np.ndarray,
    config: AllocationConfig,
) -> AllocationResult:
    """Bounded least-squares QP allocation.

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
    config : AllocationConfig
        QP parameters (W, lambda_reg).

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

    # Build weighted system
    W = config.W
    lambda_reg = config.lambda_reg

    if W is not None:
        # W is [3x3] weighting — compute sqrt via eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(W)
        sqrt_W = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0.0))) @ eigvecs.T
        A_sys = sqrt_W @ B_tau
        b_sys = sqrt_W @ tau_desired
    else:
        A_sys = B_tau
        b_sys = tau_desired

    if lambda_reg > 0:
        # Augmented system for regularization
        A_aug = np.vstack([A_sys, np.sqrt(lambda_reg) * np.eye(n)])
        b_aug = np.concatenate([b_sys, np.zeros(n)])
    else:
        A_aug = A_sys
        b_aug = b_sys

    # Solve bounded least squares
    res = lsq_linear(A_aug, b_aug, bounds=(u_min, u_max), method='trf')

    if not res.success:
        return AllocationResult(
            u=np.zeros(n_actuators),
            tau_achieved=np.zeros(3),
            alpha=0.0,
            direction_error=0.0,
            feasible=False,
        )

    u_sol = res.x

    # Compute achieved torque and metrics
    tau_achieved = B_tau @ u_sol
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

    # Alpha
    tau_hat = tau_desired / tau_norm
    alpha = float(np.dot(tau_achieved, tau_hat)) / tau_norm

    # Feasibility
    feasible = np.linalg.norm(tau_achieved - tau_desired) < 1e-6

    u_out = np.zeros(n_actuators)
    u_out[group_indices] = u_sol

    return AllocationResult(
        u=u_out,
        tau_achieved=tau_achieved,
        alpha=max(0.0, alpha),
        direction_error=direction_error,
        feasible=feasible,
    )
