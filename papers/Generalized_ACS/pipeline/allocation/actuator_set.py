"""
Actuator set assembly: builds the time-varying torque effectiveness
matrix B_tau from actuator groups.

For reaction wheels:
    tau = A_rw @ u_rw    (static, body-fixed axes)

For magnetorquers:
    tau = -skew(B_body) @ A_mtq @ u_mtq    (time-varying via B field)

The combined matrix is:
    B_tau = [A_rw | -skew(B_body) @ A_mtq]
"""

__all__ = ["assemble_B_tau", "mask_failed_actuators"]

import numpy as np
from typing import List, Tuple

from ADCS.pipeline.data import ActuatorGroup
from ADCS.helpers.math_helpers import skewsym


def assemble_B_tau(
    actuator_groups: List[ActuatorGroup],
    B_body: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble the full torque effectiveness matrix from actuator groups.

    Parameters
    ----------
    actuator_groups : list of ActuatorGroup
        Actuator groups (RW, MTQ, etc.).
    B_body : ndarray, shape (3,)
        Magnetic field vector in body frame (Tesla).

    Returns
    -------
    B_tau : ndarray, shape (3, n_total)
        Combined torque effectiveness matrix.
    u_min : ndarray, shape (n_total,)
        Lower bounds on actuator commands.
    u_max : ndarray, shape (n_total,)
        Upper bounds on actuator commands.
    """
    B_columns = []
    u_min_parts = []
    u_max_parts = []

    for group in actuator_groups:
        if group.group_type == 'rw':
            # Static mapping: tau = A_rw @ u_rw
            B_group = group.axes  # [3 x n_rw]

        elif group.group_type == 'mtq':
            # Time-varying: tau = -skew(B_body) @ A_mtq @ u_mtq
            B_skew = skewsym(B_body)
            B_group = -B_skew @ group.axes  # [3 x n_mtq]

        elif group.group_type == 'thruster':
            # Static for body-fixed thrusters
            B_group = group.axes  # [3 x n_thr]

        else:
            # Custom: use provided effectiveness function or axes
            if hasattr(group, 'custom_effectiveness') and group.custom_effectiveness is not None:
                B_group = group.custom_effectiveness
            else:
                B_group = group.axes

        B_columns.append(B_group)
        u_min_parts.append(group.u_min)
        u_max_parts.append(group.u_max)

    if not B_columns:
        return np.zeros((3, 0)), np.zeros(0), np.zeros(0)

    B_tau = np.hstack(B_columns)
    u_min = np.concatenate(u_min_parts)
    u_max = np.concatenate(u_max_parts)

    return B_tau, u_min, u_max


def mask_failed_actuators(
    B_tau: np.ndarray,
    u_min: np.ndarray,
    u_max: np.ndarray,
    failed_indices: np.ndarray,
    group_indices: np.ndarray,
) -> tuple:
    """Remove failed actuators from B_tau, bounds, and group_indices.

    Drops the columns corresponding to failed actuators entirely
    (rather than zeroing, which would break solvers that require
    strict u_min < u_max).

    Parameters
    ----------
    B_tau : ndarray, shape (3, n)
        Torque effectiveness matrix.
    u_min, u_max : ndarray, shape (n,)
        Actuator command bounds.
    failed_indices : ndarray of int
        Indices in the *full actuator command vector* that have failed.
    group_indices : ndarray of int
        Mapping from B_tau column index to full command vector index.

    Returns
    -------
    B_tau, u_min, u_max, group_indices :
        Copies with failed columns removed.
    """
    failed_set = set(int(i) for i in failed_indices)

    # Find which B_tau columns to keep
    keep_mask = np.array([
        int(full_idx) not in failed_set
        for full_idx in group_indices
    ], dtype=bool)

    return (
        B_tau[:, keep_mask],
        u_min[keep_mask],
        u_max[keep_mask],
        group_indices[keep_mask],
    )
