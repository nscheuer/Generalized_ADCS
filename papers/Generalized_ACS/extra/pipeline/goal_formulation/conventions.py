"""
Quaternion convention conversion layer.

Internal convention: Hamilton, scalar-first, error = q_g * q^{-1}.
This module converts to/from the law's preferred convention.
"""

__all__ = ["convert_quat_convention", "quat_conjugate"]

import numpy as np

from ADCS.pipeline.data import LawInterface


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate: [q0, -q1, -q2, -q3]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def convert_quat_convention(
    q_e: np.ndarray,
    from_convention: str,
    law_flags: LawInterface,
) -> np.ndarray:
    """Convert a quaternion error from internal convention to the law's convention.

    Internal convention:
        Hamilton, scalar-first, q_e = q_g * q^{-1}

    Parameters
    ----------
    q_e : ndarray, shape (4,)
        Error quaternion in internal convention.
    from_convention : str
        Convention of q_e (always 'hamilton_scalar_first' internally).
    law_flags : LawInterface
        Target conventions declared by the control law.

    Returns
    -------
    ndarray, shape (4,)
        Error quaternion in the law's convention.
    """
    q_out = q_e.copy()

    # Handle error direction convention
    # Internal: goal_times_current_inv (q_g * q^{-1})
    # Alt: current_inv_times_goal (q^{-1} * q_g) = conjugate of (q_g * q^{-1})
    if law_flags.error_convention != 'goal_times_current_inv':
        q_out = quat_conjugate(q_out)

    # Handle quaternion storage convention
    if law_flags.quat_convention == 'hamilton_scalar_first':
        pass  # already in this format
    elif law_flags.quat_convention == 'hamilton_scalar_last':
        q_out = np.array([q_out[1], q_out[2], q_out[3], q_out[0]])
    elif law_flags.quat_convention == 'jpl':
        q_out = quat_conjugate(q_out)

    return q_out
