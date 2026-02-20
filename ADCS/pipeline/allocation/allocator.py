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
"""

__all__ = ["allocation_step"]

import numpy as np
from typing import List, Optional

from ADCS.pipeline.data import AllocationConfig, AllocationResult, ActuatorGroup
from ADCS.pipeline.allocation.magnetic_cross import allocate_magnetic_cross
from ADCS.pipeline.allocation.actuator_set import assemble_B_tau
from ADCS.pipeline.allocation.lp import allocate_lp
from ADCS.pipeline.allocation.qp import allocate_qp
from ADCS.pipeline.allocation.qpw import allocate_qpw
from ADCS.pipeline.allocation.qpc import allocate_qpc
from ADCS.pipeline.allocation.pseudoinverse import allocate_pseudoinverse


def allocation_step(
    tau_desired: np.ndarray,
    actuator_groups: List[ActuatorGroup],
    alloc_config: AllocationConfig,
    B_body: np.ndarray,
    n_actuators: int,
    omega: Optional[np.ndarray] = None,
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

    # Step 2: Route to solver
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
