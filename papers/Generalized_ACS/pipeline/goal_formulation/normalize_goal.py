"""
Goal normalization: convert any goal format to the internal canonical form.

Full goals -> canonical quaternion (Hamilton, scalar-first, scalar-positive).
Reduced goals -> canonical (b_hat, u_hat) pair.
"""

__all__ = ["normalize_full_goal", "normalize_reduced_goal"]

import numpy as np
import warnings

from ADCS.pipeline.data import GoalSpec
from ADCS.helpers.math_helpers import normalize, dcm_to_quat, mrp_to_quat


def normalize_full_goal(goal_spec: GoalSpec) -> np.ndarray:
    """Convert any full-attitude goal format to a canonical quaternion.

    Accepts quaternion, DCM, or MRP representations. Returns a
    normalized, scalar-positive Hamilton quaternion.

    Parameters
    ----------
    goal_spec : GoalSpec
        Goal with goal_type == 'full' and one attitude field populated.

    Returns
    -------
    ndarray, shape (4,)
        Canonical goal quaternion (Hamilton, scalar-first, scalar-positive).
    """
    if goal_spec.q_goal is not None:
        q_g = goal_spec.q_goal.copy()
    elif goal_spec.dcm_goal is not None:
        q_g = dcm_to_quat(goal_spec.dcm_goal)
    else:
        raise ValueError(
            "Full-attitude goal must have q_goal or dcm_goal populated."
        )

    q_g = normalize(q_g)

    # Enforce scalar-positive convention
    if q_g[0] < 0:
        q_g = -q_g

    return q_g


def normalize_reduced_goal(
    goal_spec: GoalSpec,
    u_hat_resolved: np.ndarray,
) -> tuple:
    """Normalize a reduced-attitude goal to a canonical (b_hat, u_hat) pair.

    Parameters
    ----------
    goal_spec : GoalSpec
        Goal with goal_type == 'reduced' and b_hat populated.
    u_hat_resolved : ndarray, shape (3,)
        Already-resolved world-frame target direction (from world_vectors).

    Returns
    -------
    b_hat : ndarray, shape (3,)
        Normalized body-frame direction.
    u_hat : ndarray, shape (3,)
        Normalized world-frame target direction.
    """
    b_raw = goal_spec.b_hat
    if b_raw is None:
        raise ValueError("Reduced goal must have b_hat populated.")

    b_norm = np.linalg.norm(b_raw)
    if abs(b_norm - 1.0) > 0.01:
        warnings.warn(
            f"b_hat norm is {b_norm:.4f}, far from 1.0. Normalizing.",
            stacklevel=2,
        )
    b_hat = normalize(b_raw)

    u_norm = np.linalg.norm(u_hat_resolved)
    if abs(u_norm - 1.0) > 0.01:
        warnings.warn(
            f"u_hat norm is {u_norm:.4f}, far from 1.0. Normalizing.",
            stacklevel=2,
        )
    u_hat = normalize(u_hat_resolved)

    return b_hat, u_hat
