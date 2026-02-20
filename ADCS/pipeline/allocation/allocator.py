"""
Allocation stage: routes the desired torque to the appropriate
allocation method based on configuration.

Phase 1: only magnetic_cross is implemented.
Future phases add LP, QP, pseudoinverse, and momentum management.
"""

__all__ = ["allocation_step"]

import numpy as np

from ADCS.pipeline.data import AllocationConfig, AllocationResult, ActuatorGroup
from ADCS.pipeline.allocation.magnetic_cross import allocate_magnetic_cross
from typing import List


def allocation_step(
    tau_desired: np.ndarray,
    actuator_groups: List[ActuatorGroup],
    alloc_config: AllocationConfig,
    B_body: np.ndarray,
    n_actuators: int,
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

    Returns
    -------
    AllocationResult
        Actuator commands and metadata.
    """
    if alloc_config.method == 'magnetic_cross':
        # Find the MTQ group
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

    # Phase 4+: LP, QP, pseudoinverse
    raise ValueError(f"Unknown allocation method: {alloc_config.method}")
