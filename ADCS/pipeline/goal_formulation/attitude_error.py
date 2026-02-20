"""
Attitude error computation: the 2x2 conversion table.

Handles all combinations of (goal_type x law_attitude_type):
  - full goal    x full law    -> direct quaternion error
  - reduced goal x full law    -> quaternion set selection
  - reduced goal x reduced law -> pass-through with frame transform
  - full goal    x reduced law -> alternating sub-goal decomposition
  - none goal    x any law     -> identity/zero error
"""

__all__ = [
    "attitude_full_to_full",
    "attitude_reduced_to_full",
    "attitude_reduced_to_reduced",
    "attitude_full_to_reduced",
    "attitude_none",
    "convert_error_representation",
    "zero_attitude",
    "AlternatingState",
]

from dataclasses import dataclass, field
import numpy as np
import warnings

from ADCS.helpers.math_helpers import (
    quat_mult,
    quat_inv,
    normalize,
    rot_mat,
    quat_to_mrp,
    quat_to_cayley,
    quat_to_euler,
)
from ADCS.pipeline.data import LawInterface
from ADCS.pipeline.goal_formulation.quat_set import (
    compute_set_basis,
    select_nearest_quaternion,
)


_VALID_REPRESENTATIONS = {
    'quaternion_vector', 'quaternion_full', 'mrp', 'cayley',
    'dcm', 'euler_321', '2mrp',
}


def convert_error_representation(
    q_e: np.ndarray,
    representation: str,
) -> np.ndarray:
    """Convert a shortest-path error quaternion to the requested representation.

    The input must be a Hamilton scalar-first error quaternion with
    positive scalar part (shortest-path enforced).

    Parameters
    ----------
    q_e : ndarray, shape (4,)
        Error quaternion (Hamilton, scalar-first, q_e[0] >= 0).
    representation : str
        Target representation (see LawInterface.attitude_representation).

    Returns
    -------
    ndarray
        Attitude error in requested form:
        - 'quaternion_vector' → shape (3,)
        - 'quaternion_full'   → shape (4,)
        - 'mrp'               → shape (3,)
        - 'cayley'            → shape (3,)
        - 'dcm'               → shape (3, 3)
        - 'euler_321'         → shape (3,)  [degrees]
        - '2mrp'              → shape (3,)
    """
    if representation == 'quaternion_vector':
        return q_e[1:4]
    elif representation == 'quaternion_full':
        return q_e
    elif representation == 'mrp':
        return quat_to_mrp(q_e)
    elif representation == '2mrp':
        return 2.0 * quat_to_mrp(q_e)
    elif representation == 'cayley':
        return quat_to_cayley(q_e)
    elif representation == 'dcm':
        return rot_mat(q_e)
    elif representation == 'euler_321':
        return quat_to_euler(q_e)
    else:
        raise ValueError(
            f"Unknown attitude_representation: {representation!r}. "
            f"Valid options: {sorted(_VALID_REPRESENTATIONS)}"
        )


def zero_attitude(representation: str) -> np.ndarray:
    """Return the identity / zero-error value for a given representation."""
    if representation == 'quaternion_vector':
        return np.zeros(3)
    elif representation == 'quaternion_full':
        return np.array([1.0, 0.0, 0.0, 0.0])
    elif representation in ('mrp', 'cayley', '2mrp'):
        return np.zeros(3)
    elif representation == 'dcm':
        return np.eye(3)
    elif representation == 'euler_321':
        return np.zeros(3)
    else:
        return np.zeros(3)


@dataclass
class AlternatingState:
    """Persistent state for full->reduced alternating decomposition."""
    active_index: int = 0
    step_counter: int = 0


def attitude_full_to_full(
    q_g: np.ndarray,
    q: np.ndarray,
    law_flags: LawInterface,
) -> np.ndarray:
    """Full goal x full law: direct quaternion error.

    Computes q_e = q_g^{-1} * q (body-frame error), then converts
    to the representation requested by law_flags.attitude_representation.

    Parameters
    ----------
    q_g : ndarray, shape (4,)
        Goal quaternion (Hamilton, scalar-first).
    q : ndarray, shape (4,)
        Current attitude quaternion.
    law_flags : LawInterface
        Declares attitude_representation (default 'quaternion_vector').

    Returns
    -------
    ndarray
        Attitude error in the requested representation.
    """
    q_e = quat_mult(quat_inv(q_g), q)

    # Enforce short rotation path
    if q_e[0] < 0:
        q_e = -q_e

    return convert_error_representation(q_e, law_flags.attitude_representation)


def attitude_reduced_to_full(
    b_hat: np.ndarray,
    u_hat: np.ndarray,
    q: np.ndarray,
    law_flags: LawInterface,
    epsilon_reg: float = 1e-6,
) -> np.ndarray:
    """Reduced goal x full law: quaternion set selection.

    Selects the nearest quaternion from the goal set that aligns
    b_hat with u_hat, then computes the error as q_g^{-1} * q,
    converted to the representation requested by
    law_flags.attitude_representation.

    Parameters
    ----------
    b_hat : ndarray, shape (3,)
        Body-frame direction to align.
    u_hat : ndarray, shape (3,)
        World-frame target direction.
    q : ndarray, shape (4,)
        Current attitude quaternion.
    law_flags : LawInterface
        Declares attitude_representation.
    epsilon_reg : float
        Anti-parallel regularization strength.

    Returns
    -------
    ndarray
        Attitude error in the requested representation.
    """
    # Compute quaternion set basis
    x_bar, y_bar = compute_set_basis(b_hat, u_hat, epsilon_reg)

    # Select nearest quaternion
    q_g = select_nearest_quaternion(x_bar, y_bar, q)

    # Compute error as q_g^{-1} * q (body-frame error)
    q_e = quat_mult(quat_inv(q_g), q)
    if q_e[0] < 0:
        q_e = -q_e

    return convert_error_representation(q_e, law_flags.attitude_representation)


def attitude_reduced_to_reduced(
    b_hat: np.ndarray,
    u_hat: np.ndarray,
    q: np.ndarray,
    law_flags: LawInterface,
) -> tuple:
    """Reduced goal x reduced law: pass-through with optional frame transform.

    Parameters
    ----------
    b_hat : ndarray, shape (3,)
        Body-frame direction.
    u_hat : ndarray, shape (3,)
        World-frame target (ECI).
    q : ndarray, shape (4,)
        Current attitude quaternion.
    law_flags : LawInterface
        Declares world_vector_frame preference.

    Returns
    -------
    tuple of (ndarray, ndarray)
        (b_hat, r_target) where r_target is in body or world frame.
    """
    if law_flags.world_vector_frame == 'body':
        R_b2i = rot_mat(q)
        r_body = R_b2i.T @ u_hat
        return (b_hat, r_body)
    else:  # 'world'
        return (b_hat, u_hat)


def attitude_full_to_reduced(
    q_g: np.ndarray,
    q: np.ndarray,
    law_flags: LawInterface,
    alternating_state: AlternatingState,
    alternating_body_vectors: tuple = None,
    alternating_switch: str = 'every_step',
    alternating_threshold: float = 0.01,
    alternating_period: int = 10,
) -> tuple:
    """Full goal x reduced law: alternating sub-goal decomposition.

    Decomposes the full attitude goal into two reduced sub-goals
    and alternates between them.

    Parameters
    ----------
    q_g : ndarray, shape (4,)
        Goal quaternion.
    q : ndarray, shape (4,)
        Current attitude quaternion.
    law_flags : LawInterface
        Declares world_vector_frame preference.
    alternating_state : AlternatingState
        Persistent state (active_index, step_counter).
    alternating_body_vectors : tuple or None
        (b1, b2) custom body vectors. Default: body X, Y.
    alternating_switch : str
        Switching strategy: 'every_step', 'threshold', 'time_based'.
    alternating_threshold : float
        Error threshold for threshold-based switching (rad).
    alternating_period : int
        Period for time-based switching (timesteps).

    Returns
    -------
    attitude_output : tuple of (ndarray, ndarray)
        (b_active, r_target) for the active sub-goal.
    P_updated : ndarray, shape (3, 3)
        Projection matrix for the active sub-goal's body vector.
    alternating_state : AlternatingState
        Updated persistent state.
    """
    # Step 1: Decompose into two reduced sub-goals
    if alternating_body_vectors is not None:
        b1, b2 = alternating_body_vectors
    else:
        b1 = np.array([1.0, 0.0, 0.0])
        b2 = np.array([0.0, 1.0, 0.0])

    b1 = normalize(b1)
    b2 = normalize(b2)

    if abs(np.dot(b1, b2)) > 0.9:
        warnings.warn(
            "Alternating body vectors are nearly parallel.",
            stacklevel=2,
        )

    # Compute world targets from goal orientation
    R_g = rot_mat(q_g)  # body-to-inertial at goal
    u1 = R_g @ b1       # where b1 should point
    u2 = R_g @ b2       # where b2 should point

    # Step 2: Determine active sub-goal
    active_idx = _determine_active_subgoal(
        alternating_state, q, b1, u1, b2, u2,
        alternating_switch, alternating_threshold, alternating_period,
    )
    alternating_state.active_index = active_idx
    alternating_state.step_counter += 1

    if active_idx == 0:
        b_active, u_active = b1, u1
    else:
        b_active, u_active = b2, u2

    # Step 3: Update projection matrix
    P_updated = np.eye(3) - np.outer(b_active, b_active)

    # Step 4: Compute output
    if law_flags.world_vector_frame == 'body':
        R_b2i = rot_mat(q)
        r_body = R_b2i.T @ u_active
        attitude_output = (b_active, r_body)
    else:
        attitude_output = (b_active, u_active)

    return attitude_output, P_updated, alternating_state


def attitude_none(law_flags: LawInterface):
    """No goal: return zero/identity error.

    Parameters
    ----------
    law_flags : LawInterface
        Law's attitude type and representation.

    Returns
    -------
    attitude_output
        Zero error in requested representation (full) or
        aligned vectors (reduced).
    """
    if law_flags.attitude_type == 'full':
        return zero_attitude(law_flags.attitude_representation)
    else:  # reduced
        b_default = np.array([0.0, 0.0, 1.0])
        return (b_default, b_default)


def _determine_active_subgoal(
    state: AlternatingState,
    q: np.ndarray,
    b1: np.ndarray, u1: np.ndarray,
    b2: np.ndarray, u2: np.ndarray,
    switch_mode: str,
    threshold: float,
    period: int,
) -> int:
    """Determine which sub-goal should be active."""
    if switch_mode == 'every_step':
        return (state.active_index + 1) % 2

    elif switch_mode == 'threshold':
        R_b2i = rot_mat(q)
        if state.active_index == 0:
            r_body = R_b2i.T @ u1
            error = np.arccos(np.clip(np.dot(b1, r_body), -1.0, 1.0))
        else:
            r_body = R_b2i.T @ u2
            error = np.arccos(np.clip(np.dot(b2, r_body), -1.0, 1.0))

        if error < threshold:
            return (state.active_index + 1) % 2
        return state.active_index

    elif switch_mode == 'time_based':
        if state.step_counter % period == 0:
            return (state.active_index + 1) % 2
        return state.active_index

    return state.active_index
