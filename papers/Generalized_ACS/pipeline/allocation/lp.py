"""
LP allocation: direction-preserving linear program.

Maximizes torque magnitude along the desired direction::

    max   T_avail
    s.t.  B_tau @ u = T_avail * tau_hat
          u_min <= u <= u_max
          T_avail >= 0

where ``tau_hat = tau_desired / ||tau_desired||``.

Direction error is zero by construction. If the desired direction
is unachievable (e.g., along B for MTQ-only), optionally projects
onto the achievable subspace.
"""

__all__ = ["allocate_lp"]

import numpy as np
from scipy.optimize import linprog

from ADCS.pipeline.data import AllocationConfig, AllocationResult


def allocate_lp(
    tau_desired: np.ndarray,
    B_tau: np.ndarray,
    u_min: np.ndarray,
    u_max: np.ndarray,
    n_actuators: int,
    group_indices: np.ndarray,
    config: AllocationConfig,
) -> AllocationResult:
    """Direction-preserving LP allocation.

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
        LP configuration (project_when_infeasible, etc.).

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

    # Decision variables: x = [u_1, ..., u_n, T_avail]
    # Maximize T_avail => minimize -T_avail
    c = np.zeros(n + 1)
    c[-1] = -1.0

    # Equality constraint: B_tau @ u - T_avail * tau_hat = 0
    A_eq = np.hstack([B_tau, -tau_hat.reshape(3, 1)])
    b_eq = np.zeros(3)

    # Bounds: u_min <= u <= u_max, T_avail >= 0
    bounds = list(zip(u_min, u_max)) + [(0, None)]

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if res.success and res.x[-1] > 1e-12:
        u_sol = res.x[:n]
        T_max = res.x[-1]

        # Scale down if more capacity than requested
        if T_max >= tau_norm:
            scale = tau_norm / T_max
            u_sol = u_sol * scale
            alpha = 1.0
            feasible = True
        else:
            alpha = T_max / tau_norm
            feasible = False

        tau_achieved = B_tau @ u_sol

        u_out = np.zeros(n_actuators)
        u_out[group_indices] = u_sol

        return AllocationResult(
            u=u_out,
            tau_achieved=tau_achieved,
            alpha=alpha,
            direction_error=0.0,  # LP preserves direction by construction
            feasible=feasible,
        )

    # LP returned zero or failed — try projection if configured
    if config.lp_project_when_infeasible:
        return _allocate_lp_projected(
            tau_desired, B_tau, u_min, u_max,
            n_actuators, group_indices,
        )

    return AllocationResult(
        u=np.zeros(n_actuators),
        tau_achieved=np.zeros(3),
        alpha=0.0,
        direction_error=0.0,
        feasible=False,
    )


def _allocate_lp_projected(
    tau_desired: np.ndarray,
    B_tau: np.ndarray,
    u_min: np.ndarray,
    u_max: np.ndarray,
    n_actuators: int,
    group_indices: np.ndarray,
) -> AllocationResult:
    """Project tau_desired onto achievable subspace and re-solve LP."""
    n = B_tau.shape[1]
    tau_norm = np.linalg.norm(tau_desired)

    # Column space projection via SVD
    U, S, Vt = np.linalg.svd(B_tau, full_matrices=False)
    rank = np.sum(S > 1e-10)
    U_range = U[:, :rank]

    tau_projected = U_range @ (U_range.T @ tau_desired)
    proj_norm = np.linalg.norm(tau_projected)

    if proj_norm < 1e-12:
        return AllocationResult(
            u=np.zeros(n_actuators),
            tau_achieved=np.zeros(3),
            alpha=0.0,
            direction_error=0.0,
            feasible=False,
        )

    tau_hat_proj = tau_projected / proj_norm

    # Re-run LP with projected direction
    c = np.zeros(n + 1)
    c[-1] = -1.0

    A_eq = np.hstack([B_tau, -tau_hat_proj.reshape(3, 1)])
    b_eq = np.zeros(3)
    bounds = list(zip(u_min, u_max)) + [(0, None)]

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if res.success and res.x[-1] > 1e-12:
        u_sol = res.x[:n]
        T_max = res.x[-1]

        # Scale down if more capacity than projected magnitude
        if T_max >= proj_norm:
            scale = proj_norm / T_max
            u_sol = u_sol * scale

        tau_achieved = B_tau @ u_sol
        tau_ach_norm = np.linalg.norm(tau_achieved)

        # Direction error from original desired
        if tau_ach_norm > 1e-12:
            cos_angle = np.clip(
                np.dot(tau_achieved, tau_desired) / (tau_ach_norm * tau_norm),
                -1.0, 1.0,
            )
            direction_error = float(np.arccos(cos_angle))
        else:
            direction_error = 0.0

        tau_hat_des = tau_desired / tau_norm
        alpha = float(np.dot(tau_achieved, tau_hat_des)) / tau_norm

        u_out = np.zeros(n_actuators)
        u_out[group_indices] = u_sol

        return AllocationResult(
            u=u_out,
            tau_achieved=tau_achieved,
            alpha=max(0.0, alpha),
            direction_error=direction_error,
            feasible=False,  # had to project
        )

    return AllocationResult(
        u=np.zeros(n_actuators),
        tau_achieved=np.zeros(3),
        alpha=0.0,
        direction_error=0.0,
        feasible=False,
    )
