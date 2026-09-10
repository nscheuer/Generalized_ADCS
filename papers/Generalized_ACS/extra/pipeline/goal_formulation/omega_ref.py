"""
Reference angular velocity computation.

Computes omega_ref in the world (ECI) frame using:
  1. User-explicit override (highest priority)
  2. Analytical formulas for named goals (nadir, zenith, etc.)
  3. Finite differencing as a fallback

The result is transformed to body frame by the caller.
"""

__all__ = ["compute_omega_ref_world"]

import numpy as np
import warnings

from ADCS.pipeline.data import GoalSpec, WorldVectorSpec
from ADCS.helpers.math_helpers import normalize, quat_mult, quat_inv
from ADCS.orbits.orbital_state import Orbital_State


def compute_omega_ref_world(
    goal_spec: GoalSpec,
    goal_spec_next: GoalSpec | None,
    goal_type: str,
    q: np.ndarray,
    os: Orbital_State,
    dt: float = 1.0,
) -> np.ndarray:
    """Compute the reference angular velocity in the ECI frame.

    Priority:
      1. User-explicit omega_ref
      2. Analytical formula for named goals
      3. Finite difference (requires goal_spec_next)
      4. Zero (with warning)

    Parameters
    ----------
    goal_spec : GoalSpec
        Current goal specification.
    goal_spec_next : GoalSpec or None
        Next-step goal for finite differencing.
    goal_type : str
        'full', 'reduced', or 'none'.
    q : ndarray, shape (4,)
        Current attitude quaternion.
    os : Orbital_State
        Current orbital state.
    dt : float
        Control timestep for finite differencing.

    Returns
    -------
    ndarray, shape (3,)
        Reference angular velocity in ECI frame.
    """
    # Priority 1: User explicit
    if goal_spec.omega_ref_eci is not None:
        return goal_spec.omega_ref_eci.copy()

    # Priority 2: No goal
    if goal_type == 'none':
        return np.zeros(3)

    # Priority 3: Analytical for named reduced goals
    if goal_type == 'reduced' and goal_spec.u_spec is not None:
        if goal_spec.u_spec.type == 'named':
            omega_analytical = _compute_analytical(goal_spec.u_spec.name, os)
            if omega_analytical is not None:
                return omega_analytical

    # Priority 4: Finite difference
    if goal_spec_next is None:
        # No next-step goal -> zero omega_ref
        return np.zeros(3)

    if goal_type == 'full':
        return _finite_diff_full(goal_spec, goal_spec_next, dt)
    elif goal_type == 'reduced':
        return _finite_diff_reduced(goal_spec, goal_spec_next, q, os, dt)

    return np.zeros(3)


def _compute_analytical(name: str, os: Orbital_State) -> np.ndarray | None:
    """Compute analytical omega_ref for named goals.

    Returns None if no analytical formula is available (caller falls
    through to finite differencing).
    """
    r = np.asarray(os.R).flatten()
    v = np.asarray(os.V).flatten()

    if name in ('nadir', 'zenith'):
        # Orbital angular velocity: omega = (r x v) / |r|^2
        return np.cross(r, v) / np.dot(r, r)

    elif name in ('normal', 'anti_normal'):
        # Orbit normal is nearly constant (exactly constant for Keplerian)
        return np.zeros(3)

    elif name in ('ram', 'anti_ram'):
        # Velocity direction rate depends on acceleration; fall through
        return None

    elif name in ('sun', 'anti_sun'):
        # Sun direction changes ~1 deg/day; near-zero for control
        return None

    elif name in ('bfield', 'anti_bfield', 'perp_bfield'):
        # Magnetic field changes too fast for a simple analytical formula
        return None

    return None


def _finite_diff_full(
    goal_spec: GoalSpec,
    goal_spec_next: GoalSpec,
    dt: float,
) -> np.ndarray:
    """Finite-difference omega_ref for full-attitude goals.

    Uses the incremental rotation between consecutive goal quaternions.
    """
    from ADCS.pipeline.goal_formulation.normalize_goal import normalize_full_goal

    q_g = normalize_full_goal(goal_spec)
    q_g_next = normalize_full_goal(goal_spec_next)

    # Incremental rotation: dq = q_g_next * q_g^{-1}
    dq = quat_mult(q_g_next, quat_inv(q_g))

    # Enforce short path
    if dq[0] < 0:
        dq = -dq

    # Extract axis-angle
    sin_half = np.linalg.norm(dq[1:4])
    if sin_half < 1e-10:
        return np.zeros(3)

    angle = 2.0 * np.arctan2(sin_half, dq[0])
    axis = dq[1:4] / sin_half

    return axis * angle / dt


def _finite_diff_reduced(
    goal_spec: GoalSpec,
    goal_spec_next: GoalSpec,
    q: np.ndarray,
    os: Orbital_State,
    dt: float,
) -> np.ndarray:
    """Finite-difference omega_ref for reduced-attitude goals.

    Handles both time-varying u_hat and time-varying b_hat by computing
    the rotation from where b_hat_next currently points (in world frame)
    to where it needs to point (u_hat_next).
    """
    from ADCS.helpers.math_helpers import rot_mat
    from ADCS.pipeline.goal_formulation.world_vectors import resolve_world_vector

    # Resolve next-step target
    if goal_spec_next.u_hat_eci is not None:
        u_hat_next = normalize(goal_spec_next.u_hat_eci)
    elif goal_spec_next.u_spec is not None:
        u_hat_next = resolve_world_vector(goal_spec_next.u_spec, os)
    else:
        return np.zeros(3)

    # Determine b_hat for next step
    b_hat_next = goal_spec.b_hat_next if goal_spec.b_hat_next is not None else goal_spec.b_hat
    if b_hat_next is None:
        return np.zeros(3)

    # Where b_hat_next currently points in world frame
    R_b2i = rot_mat(q)
    c_hat = normalize(R_b2i @ b_hat_next)

    # Rotation from c_hat to u_hat_next
    cos_delta = np.clip(np.dot(c_hat, u_hat_next), -1.0, 1.0)
    cross_vec = np.cross(c_hat, u_hat_next)
    sin_delta = np.linalg.norm(cross_vec)

    if sin_delta < 1e-10:
        if cos_delta > 0:
            return np.zeros(3)  # already aligned
        else:
            # Anti-parallel: 180° rotation about any perp axis
            from ADCS.pipeline.goal_formulation.quat_set import find_perpendicular
            e_perp = find_perpendicular(c_hat)
            return e_perp * np.pi / dt

    n_hat = cross_vec / sin_delta
    delta_theta = np.arccos(cos_delta)
    return n_hat * delta_theta / dt
