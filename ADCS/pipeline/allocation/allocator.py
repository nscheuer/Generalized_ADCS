"""
Allocation stage: routes the desired torque to the appropriate
allocation method based on configuration.

Supported methods:
    - magnetic_cross : Cross-product inversion for MTQ-only
    - lp             : Direction-preserving LP
    - qp             : Bounded least-squares QP
    - qpw            : Direction-weighted QP
    - qpc            : Energy-constrained QP
    - pseudoinverse  : Moore-Penrose pinv + clip

Momentum management (desaturation):
    - nullspace  : Post-allocation nullspace projection (zero torque impact)
    - weighted   : Augmented QP cost (trades pointing for desat)
    - scheduled  : Pre-allocation tau_desired modification (orbit-gated)

Actuator failure:
    - Zero out failed columns in B_tau and set bounds to zero.
"""

__all__ = ["allocation_step"]

import numpy as np
from typing import List, Optional

from ADCS.pipeline.data import (
    AllocationConfig, AllocationResult, ActuatorGroup, DesaturationConfig,
)
from ADCS.pipeline.allocation.magnetic_cross import allocate_magnetic_cross
from ADCS.pipeline.allocation.actuator_set import assemble_B_tau, mask_failed_actuators
from ADCS.pipeline.allocation.lp import allocate_lp
from ADCS.pipeline.allocation.qp import allocate_qp
from ADCS.pipeline.allocation.qpw import allocate_qpw
from ADCS.pipeline.allocation.qpc import allocate_qpc
from ADCS.pipeline.allocation.pseudoinverse import allocate_pseudoinverse
from ADCS.pipeline.allocation.momentum import (
    apply_nullspace_desaturation,
    build_weighted_desat_system,
    apply_scheduled_desaturation,
)


def allocation_step(
    tau_desired: np.ndarray,
    actuator_groups: List[ActuatorGroup],
    alloc_config: AllocationConfig,
    B_body: np.ndarray,
    n_actuators: int,
    omega: Optional[np.ndarray] = None,
    h_rw_body: Optional[np.ndarray] = None,
    failed_actuators: Optional[np.ndarray] = None,
) -> AllocationResult:
    """Route desired torque to the configured allocation method.

    Parameters
    ----------
    tau_desired : ndarray, shape (3,)
        Desired torque in body frame.
    actuator_groups : list of ActuatorGroup
        Actuator groups (RW, MTQ, etc.).
    alloc_config : AllocationConfig
        Allocation method and configuration.
    B_body : ndarray, shape (3,)
        Magnetic field vector in body frame.
    n_actuators : int
        Total number of actuators.
    omega : ndarray, shape (3,) or None
        Body angular velocity (needed for QPC energy gate).
    h_rw_body : ndarray, shape (3,) or None
        Total RW angular momentum in body frame (needed for desaturation).
    failed_actuators : ndarray of int or None
        Indices of failed actuators in the full command vector.

    Returns
    -------
    AllocationResult
        Actuator commands and metadata.
    """
    method = alloc_config.method

    # ---- magnetic_cross: special-case MTQ-only allocator ----
    if method == 'magnetic_cross':
        mtq_group = None
        for group in actuator_groups:
            if group.group_type == 'mtq':
                mtq_group = group
                break
        if mtq_group is None:
            return AllocationResult(
                u=np.zeros(n_actuators),
                tau_achieved=np.zeros(3),
                alpha=0.0,
            )
        return allocate_magnetic_cross(
            tau_desired=tau_desired,
            B_body=B_body,
            mtq_group=mtq_group,
            n_actuators=n_actuators,
        )

    # ---- General allocation methods via B_tau ----
    # Step 1: Assemble torque effectiveness matrix
    B_tau, u_min, u_max = assemble_B_tau(actuator_groups, B_body)

    n = B_tau.shape[1]
    if n == 0:
        return AllocationResult(
            u=np.zeros(n_actuators),
            tau_achieved=np.zeros(3),
            alpha=0.0,
            feasible=False,
        )

    # Build combined group_indices mapping
    group_indices = _build_group_indices(actuator_groups)

    # Step 1b: Remove failed actuators from the problem
    if failed_actuators is not None and len(failed_actuators) > 0:
        B_tau, u_min, u_max, group_indices = mask_failed_actuators(
            B_tau, u_min, u_max, failed_actuators, group_indices,
        )
        n = B_tau.shape[1]
        if n == 0:
            return AllocationResult(
                u=np.zeros(n_actuators),
                tau_achieved=np.zeros(3),
                alpha=0.0,
                feasible=False,
            )

    # Step 1c: Desaturation pre-processing
    desat_config = alloc_config.desat_config
    do_desat = (
        alloc_config.enable_desaturation
        and desat_config is not None
        and h_rw_body is not None
    )

    if do_desat and desat_config.strategy == 'scheduled':
        # Scheduled: modify tau_desired before allocation
        tau_desired = apply_scheduled_desaturation(
            tau_desired, h_rw_body, B_body, actuator_groups, desat_config,
        )

    if do_desat and desat_config.strategy == 'weighted':
        # Weighted: replace B_tau/tau_desired with augmented system,
        # then solve via bounded least-squares (QP path)
        A_aug, b_aug = build_weighted_desat_system(
            B_tau, tau_desired, u_min, u_max,
            h_rw_body, actuator_groups, desat_config,
        )
        return _solve_weighted_desat(
            A_aug, b_aug, B_tau, tau_desired,
            u_min, u_max, n_actuators, group_indices,
        )

    # Step 2: Route to solver
    result = _route_to_solver(
        method, tau_desired, B_tau, u_min, u_max,
        n_actuators, group_indices, alloc_config, omega,
    )

    # Step 3: Nullspace desaturation post-processing
    if do_desat and desat_config.strategy == 'nullspace':
        # Identify RW columns in the B_tau column ordering
        rw_columns = _find_rw_columns(actuator_groups)

        # Extract the B_tau-indexed solution from the full command vector
        u_btau = result.u[group_indices]

        u_desat = apply_nullspace_desaturation(
            u_btau, B_tau, u_min, u_max,
            h_rw_body, rw_columns, desat_config,
        )

        # Recompute metrics with desaturated commands
        tau_achieved = B_tau @ u_desat
        tau_norm = np.linalg.norm(tau_desired)

        u_out = np.zeros(n_actuators)
        u_out[group_indices] = u_desat

        if tau_norm > 1e-12:
            tau_ach_norm = np.linalg.norm(tau_achieved)
            tau_hat = tau_desired / tau_norm
            alpha = float(np.dot(tau_achieved, tau_hat)) / tau_norm
            if tau_ach_norm > 1e-12:
                cos_a = np.clip(
                    np.dot(tau_achieved, tau_desired) / (tau_ach_norm * tau_norm),
                    -1.0, 1.0,
                )
                direction_error = float(np.arccos(cos_a))
            else:
                direction_error = 0.0
        else:
            alpha = 1.0
            direction_error = 0.0

        return AllocationResult(
            u=u_out,
            tau_achieved=tau_achieved,
            alpha=max(0.0, alpha),
            direction_error=direction_error,
            feasible=result.feasible,
        )

    return result


def _route_to_solver(
    method, tau_desired, B_tau, u_min, u_max,
    n_actuators, group_indices, alloc_config, omega,
) -> AllocationResult:
    """Route to the appropriate solver based on method string."""
    if method == 'lp':
        return allocate_lp(
            tau_desired, B_tau, u_min, u_max,
            n_actuators, group_indices, alloc_config,
        )
    elif method == 'qp':
        return allocate_qp(
            tau_desired, B_tau, u_min, u_max,
            n_actuators, group_indices, alloc_config,
        )
    elif method == 'qpw':
        return allocate_qpw(
            tau_desired, B_tau, u_min, u_max,
            n_actuators, group_indices, alloc_config,
        )
    elif method == 'qpc':
        return allocate_qpc(
            tau_desired, B_tau, u_min, u_max,
            n_actuators, group_indices, alloc_config,
            omega=omega,
        )
    elif method == 'pseudoinverse':
        return allocate_pseudoinverse(
            tau_desired, B_tau, u_min, u_max,
            n_actuators, group_indices,
        )
    else:
        raise ValueError(f"Unknown allocation method: {method}")


def _solve_weighted_desat(
    A_aug, b_aug, B_tau, tau_desired,
    u_min, u_max, n_actuators, group_indices,
) -> AllocationResult:
    """Solve weighted desaturation via augmented bounded least-squares."""
    from scipy.optimize import lsq_linear

    res = lsq_linear(A_aug, b_aug, bounds=(u_min, u_max), method='trf')

    if not res.success:
        return AllocationResult(
            u=np.zeros(n_actuators),
            tau_achieved=np.zeros(3),
            alpha=0.0,
            feasible=False,
        )

    u_sol = res.x
    tau_achieved = B_tau @ u_sol
    tau_norm = np.linalg.norm(tau_desired)

    if tau_norm > 1e-12:
        tau_ach_norm = np.linalg.norm(tau_achieved)
        tau_hat = tau_desired / tau_norm
        alpha = float(np.dot(tau_achieved, tau_hat)) / tau_norm
        if tau_ach_norm > 1e-12:
            cos_a = np.clip(
                np.dot(tau_achieved, tau_desired) / (tau_ach_norm * tau_norm),
                -1.0, 1.0,
            )
            direction_error = float(np.arccos(cos_a))
        else:
            direction_error = 0.0
    else:
        alpha = 1.0
        direction_error = 0.0

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


def _find_rw_columns(actuator_groups: List[ActuatorGroup]) -> np.ndarray:
    """Find column indices in B_tau that correspond to RW actuators."""
    rw_cols = []
    col_offset = 0
    for group in actuator_groups:
        n_g = group.axes.shape[1]
        if group.group_type == 'rw':
            rw_cols.extend(range(col_offset, col_offset + n_g))
        col_offset += n_g
    return np.array(rw_cols, dtype=int)


def _build_group_indices(actuator_groups: List[ActuatorGroup]) -> np.ndarray:
    """Concatenate group indices into a single mapping array.

    Returns an array where entry i gives the index into the full
    actuator command vector for B_tau column i.
    """
    all_indices = []
    for group in actuator_groups:
        if group.indices is not None:
            all_indices.append(group.indices)
        else:
            # If no indices set, assume sequential from 0
            n_g = group.axes.shape[1]
            offset = sum(len(idx) for idx in all_indices)
            all_indices.append(np.arange(offset, offset + n_g))
    if not all_indices:
        return np.array([], dtype=int)
    return np.concatenate(all_indices)
