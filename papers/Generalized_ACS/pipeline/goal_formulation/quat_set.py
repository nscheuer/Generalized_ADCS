"""
Quaternion set parameterization for reduced-attitude goals.

Given a body vector b_hat and a world target u_hat, the set of all
quaternions that align b_hat with u_hat forms a one-parameter family:

    f(beta) = x_bar * cos(beta) + y_bar * sin(beta)

where x_bar and y_bar are basis quaternions computed from the
(b_hat, u_hat) pair.

The nearest-quaternion selection finds the beta that minimizes
geodesic distance from the current attitude q to the goal set,
yielding a unique goal quaternion for full-attitude control laws.

Reference: McKeen thesis, quaternion set parameterization.
"""

__all__ = ["compute_set_basis", "select_nearest_quaternion", "find_perpendicular"]

import numpy as np

from ADCS.helpers.math_helpers import normalize


def find_perpendicular(v: np.ndarray) -> np.ndarray:
    """Find an arbitrary unit vector perpendicular to v.

    Uses the axis least aligned with v for numerical stability.

    Parameters
    ----------
    v : ndarray, shape (3,)
        Input vector (need not be unit).

    Returns
    -------
    ndarray, shape (3,)
        Unit vector perpendicular to v.
    """
    abs_v = np.abs(v)
    # Choose the canonical axis least aligned with v
    if abs_v[0] <= abs_v[1] and abs_v[0] <= abs_v[2]:
        candidate = np.array([1.0, 0.0, 0.0])
    elif abs_v[1] <= abs_v[2]:
        candidate = np.array([0.0, 1.0, 0.0])
    else:
        candidate = np.array([0.0, 0.0, 1.0])
    return normalize(np.cross(v, candidate))


def compute_set_basis(
    b_hat: np.ndarray,
    u_hat: np.ndarray,
    epsilon_reg: float = 1e-6,
) -> tuple:
    """Compute the quaternion set basis vectors (x_bar, y_bar).

    The goal set is parameterized as:
        f(beta) = x_bar * cos(beta) + y_bar * sin(beta)

    This function includes anti-parallel regularization: when
    b_hat and u_hat are nearly anti-parallel, a small perturbation
    ensures well-defined basis vectors.

    Parameters
    ----------
    b_hat : ndarray, shape (3,)
        Body-frame direction to align (unit vector).
    u_hat : ndarray, shape (3,)
        World-frame target direction (unit vector).
    epsilon_reg : float
        Anti-parallel regularization strength.

    Returns
    -------
    x_bar : ndarray, shape (4,)
        First basis quaternion.
    y_bar : ndarray, shape (4,)
        Second basis quaternion (pure quaternion, scalar = 0).
    """
    cos_theta = np.clip(np.dot(b_hat, u_hat), -1.0, 1.0)
    cross_vu = np.cross(b_hat, u_hat)
    sin_theta = np.linalg.norm(cross_vu)

    # Regularized rotation axis: perpendicular to both b and u
    # With epsilon perturbation for anti-parallel case
    e_perp = find_perpendicular(b_hat)
    x_hat = normalize(cross_vu + epsilon_reg * e_perp)

    # y_hat: perpendicular to b_hat and x_hat
    # Since b_hat and x_hat are orthogonal unit vectors, y_hat is unit
    y_hat = np.cross(b_hat, x_hat)

    theta = np.arccos(cos_theta)
    half_theta = theta / 2.0

    # Basis quaternions (from set parameterization)
    x_bar = np.array([
        np.cos(half_theta),
        x_hat[0] * np.sin(half_theta),
        x_hat[1] * np.sin(half_theta),
        x_hat[2] * np.sin(half_theta),
    ])

    y_bar = np.array([
        0.0,
        y_hat[0],
        y_hat[1],
        y_hat[2],
    ])

    return x_bar, y_bar


def select_nearest_quaternion(
    x_bar: np.ndarray,
    y_bar: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    """Select the quaternion from the goal set nearest to q.

    Finds beta that maximizes ``|q . f(beta)|``, where::

        f(beta) = x_bar * cos(beta) + y_bar * sin(beta)

    The inner product ``q . f(beta) = (q.x_bar) cos(beta) + (q.y_bar)
    sin(beta)`` is maximized when ``beta = atan2(q.y_bar, q.x_bar)``.

    Parameters
    ----------
    x_bar : ndarray, shape (4,)
        First basis quaternion.
    y_bar : ndarray, shape (4,)
        Second basis quaternion.
    q : ndarray, shape (4,)
        Current attitude quaternion.

    Returns
    -------
    ndarray, shape (4,)
        Goal quaternion nearest to q in the geodesic sense.
    """
    qx = np.dot(q, x_bar)
    qy = np.dot(q, y_bar)
    beta_opt = np.arctan2(qy, qx)

    q_g = x_bar * np.cos(beta_opt) + y_bar * np.sin(beta_opt)
    return normalize(q_g)
