__all__ = ["quat_continuity", "interp_quat", "hermite_position"]

import numpy as np

from ADCS.helpers.math_helpers import interpolate_quat


def quat_continuity(q: np.ndarray) -> np.ndarray:
    r"""
    Flip quaternion signs so consecutive rows lie in the same hemisphere.

    ``q`` and ``-q`` are the same rotation, but component-wise interpolation or
    plotting across a sign flip produces a spurious tumble. Each row is
    negated when its dot product with the (already corrected) previous row is
    negative.

    :param q: Quaternions, shape ``(N, 4)``.
    :return: Sign-continuous copy, shape ``(N, 4)``.
    """
    q = np.array(q, dtype=float, copy=True).reshape(-1, 4)
    if len(q) < 2:
        return q
    # Row k flips relative to its *original* predecessor; the cumulative parity
    # of those flips gives the sign relative to row 0.
    flips = np.einsum("ij,ij->i", q[1:], q[:-1]) < 0.0
    sign = np.concatenate([[1.0], np.where(np.cumsum(flips) % 2 == 1, -1.0, 1.0)])
    return q * sign[:, None]


def interp_quat(t: np.ndarray, q: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    r"""
    Slerp a quaternion time series onto new times (shortest arc per interval).

    :param t: Sample times, strictly increasing, shape ``(N,)``.
    :param q: Unit quaternions at ``t``, shape ``(N, 4)``.
    :param t_new: Query times inside ``[t[0], t[-1]]``, shape ``(M,)``.
    :return: Interpolated unit quaternions, shape ``(M, 4)``.
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    t_new = np.atleast_1d(np.asarray(t_new, dtype=float))
    if t_new.min() < t[0] or t_new.max() > t[-1]:
        raise ValueError("t_new must lie within the sampled interval")
    idx = np.clip(np.searchsorted(t, t_new, side="right") - 1, 0, len(t) - 2)
    alpha = (t_new - t[idx]) / (t[idx + 1] - t[idx])
    return np.array([
        interpolate_quat(q[i], q[i + 1], float(a)) for i, a in zip(idx, alpha)
    ])


def hermite_position(t: np.ndarray, r: np.ndarray, v: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    r"""
    Cubic Hermite interpolation of position using the sampled velocity.

    Linear interpolation of position cuts the chord of the orbit, an error of
    about ``a dθ²/8``: ~1 m at 1 s sampling and ~1 km at 30 s in LEO, which is
    visible next to a metre-sized spacecraft. The Hermite error scales as
    ``a dθ⁴/384``: ~1e-10 m at 1 s and ~2 cm at 30 s.

    :param t: Sample times [s], strictly increasing, shape ``(N,)``.
    :param r: Positions, shape ``(N, 3)``.
    :param v: Velocities in position units per second, shape ``(N, 3)``.
    :param t_new: Query times inside ``[t[0], t[-1]]``, shape ``(M,)``.
    :return: Interpolated positions, shape ``(M, 3)``.
    """
    t = np.asarray(t, dtype=float)
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    t_new = np.atleast_1d(np.asarray(t_new, dtype=float))
    if t_new.min() < t[0] or t_new.max() > t[-1]:
        raise ValueError("t_new must lie within the sampled interval")
    i = np.clip(np.searchsorted(t, t_new, side="right") - 1, 0, len(t) - 2)
    h = (t[i + 1] - t[i])[:, None]
    s = ((t_new - t[i])[:, None]) / h
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    return h00 * r[i] + h10 * h * v[i] + h01 * r[i + 1] + h11 * h * v[i + 1]
