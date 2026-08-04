"""
QPC allocation: energy-constrained quadratic programming.

Bounded least-squares with a Lyapunov-style power gate constraint::

    min  ||B_tau @ u - tau_desired||^2
    s.t. u_min <= u <= u_max
         omega^T @ B_tau @ u <= max(0, omega^T @ tau_desired)

The power gate prevents the allocator from producing torque that injects
rotational energy when the control law intends to dissipate it.

Falls back to LP if the constrained optimizer fails.

Implements the same formulation as MTQ_w_RW_QPC in the existing codebase.
"""

__all__ = ["allocate_qpc"]

import numpy as np
from scipy.optimize import minimize, Bounds, lsq_linear

from ADCS.pipeline.data import AllocationConfig, AllocationResult


def allocate_qpc(
    tau_desired: np.ndarray,
    B_tau: np.ndarray,
    u_min: np.ndarray,
    u_max: np.ndarray,
    n_actuators: int,
    group_indices: np.ndarray,
    config: AllocationConfig,
    omega: np.ndarray = None,
) -> AllocationResult:
    """Energy-constrained QP allocation.

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
        QPC configuration parameters.
    omega : ndarray, shape (3,) or None
        Body angular velocity for the power gate. If None, falls back
        to unconstrained QP.

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

    # If no omega provided, fall back to unconstrained QP
    if omega is None or np.linalg.norm(omega) < 1e-12:
        return _solve_unconstrained(
            tau_desired, B_tau, u_min, u_max,
            n_actuators, group_indices, tau_norm,
        )

    omega = np.asarray(omega, float).reshape(3,)

    # Energy gate: omega^T @ tau_achieved <= max(0, omega^T @ tau_desired)
    power_desired = float(np.dot(omega, tau_desired))
    ub_constraint = max(0.0, power_desired)

    # Constraint vector: C @ u <= ub_constraint
    # where C = omega^T @ B_tau
    C = omega @ B_tau  # (n,)

    # Objective: min ||B_tau @ u - tau_desired||^2
    def fun(u):
        r = B_tau @ u - tau_desired
        return 0.5 * np.dot(r, r)

    def jac(u):
        r = B_tau @ u - tau_desired
        return B_tau.T @ r

    # Starting point: center of bounds
    u0 = 0.5 * (u_min + u_max)

    bounds = Bounds(u_min, u_max)
    constraints = [{
        'type': 'ineq',
        'fun': lambda u: ub_constraint - float(C @ u),
        'jac': lambda u: -C,
    }]

    res = minimize(
        fun, u0, jac=jac,
        method='SLSQP',
        constraints=constraints,
        bounds=bounds,
    )

    if res.success:
        u_sol = res.x
    else:
        # Fall back to unconstrained QP
        return _solve_unconstrained(
            tau_desired, B_tau, u_min, u_max,
            n_actuators, group_indices, tau_norm,
        )

    # Compute metrics
    tau_achieved = B_tau @ u_sol
    tau_ach_norm = np.linalg.norm(tau_achieved)

    if tau_ach_norm > 1e-12:
        cos_angle = np.clip(
            np.dot(tau_achieved, tau_desired) / (tau_ach_norm * tau_norm),
            -1.0, 1.0,
        )
        direction_error = float(np.arccos(cos_angle))
    else:
        direction_error = 0.0

    tau_hat = tau_desired / tau_norm
    alpha = float(np.dot(tau_achieved, tau_hat)) / tau_norm
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


def _solve_unconstrained(
    tau_desired, B_tau, u_min, u_max,
    n_actuators, group_indices, tau_norm,
):
    """Fallback: unconstrained bounded least-squares."""
    res = lsq_linear(B_tau, tau_desired, bounds=(u_min, u_max), method='trf')

    if not res.success:
        return AllocationResult(
            u=np.zeros(n_actuators),
            tau_achieved=np.zeros(3),
            alpha=0.0,
            direction_error=0.0,
            feasible=False,
        )

    u_sol = res.x
    tau_achieved = B_tau @ u_sol
    tau_ach_norm = np.linalg.norm(tau_achieved)

    if tau_ach_norm > 1e-12:
        cos_angle = np.clip(
            np.dot(tau_achieved, tau_desired) / (tau_ach_norm * tau_norm),
            -1.0, 1.0,
        )
        direction_error = float(np.arccos(cos_angle))
    else:
        direction_error = 0.0

    tau_hat = tau_desired / tau_norm
    alpha = float(np.dot(tau_achieved, tau_hat)) / tau_norm
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
