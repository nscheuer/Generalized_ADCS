"""
QPW allocation: direction-weighted bounded least-squares.

Decomposes the torque error into parallel and perpendicular components
relative to the desired direction and penalizes them differently:

    min  w_par * ||e_par||^2 + w_perp * ||e_perp||^2
    s.t. u_min <= u <= u_max

where e = B_tau @ u - tau_desired.

With w_perp >> w_par, this biases the solution toward preserving the
torque direction even when the exact magnitude is unachievable.

Implements the same formulation as MTQ_w_RW_QPW in the existing codebase.
"""

__all__ = ["allocate_qpw"]

import numpy as np
from scipy.optimize import lsq_linear

from ADCS.pipeline.data import AllocationConfig, AllocationResult


def allocate_qpw(
    tau_desired: np.ndarray,
    B_tau: np.ndarray,
    u_min: np.ndarray,
    u_max: np.ndarray,
    n_actuators: int,
    group_indices: np.ndarray,
    config: AllocationConfig,
) -> AllocationResult:
    """Direction-weighted QP allocation.

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
        QPW parameters (w_parallel, w_perpendicular).

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

    tau_hat = tau_desired / tau_norm

    # Projection operators
    P_par = np.outer(tau_hat, tau_hat)
    P_perp = np.eye(3) - P_par

    # Square-root weighting matrix
    w_par = config.w_parallel
    w_perp = config.w_perpendicular
    W = np.sqrt(w_par) * P_par + np.sqrt(w_perp) * P_perp

    # Weighted system: min ||W @ (B_tau @ u - tau_desired)||^2
    A_w = W @ B_tau
    b_w = W @ tau_desired

    # Solve bounded least squares
    res = lsq_linear(A_w, b_w, bounds=(u_min, u_max), method='trf')

    if not res.success:
        return AllocationResult(
            u=np.zeros(n_actuators),
            tau_achieved=np.zeros(3),
            alpha=0.0,
            direction_error=0.0,
            feasible=False,
        )

    u_sol = res.x

    # Achieved torque (use unweighted B_tau)
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
