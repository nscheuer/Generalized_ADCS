import numpy as np
from numba import njit


@njit(cache=True)
def _normalize_quaternion(q):
    magnitude = np.sqrt(np.dot(q, q))
    if magnitude == 0.0:
        raise ValueError("quaternion must have nonzero norm")
    return q / magnitude


@njit(cache=True)
def _quaternion_norm(q):
    return np.sqrt(np.dot(q, q))


@njit(cache=True)
def _jpl_to_hamilton(q_jpl):
    return np.array((q_jpl[3], -q_jpl[0], -q_jpl[1], -q_jpl[2]))


@njit(cache=True)
def _hamilton_to_jpl(q_hamilton):
    return np.array((-q_hamilton[1], -q_hamilton[2], -q_hamilton[3], q_hamilton[0]))


@njit(cache=True)
def _conjugate(q):
    return np.array((q[0], -q[1], -q[2], -q[3]))


@njit(cache=True)
def _inverse(q):
    squared_norm = np.dot(q, q)
    if squared_norm == 0.0:
        raise ValueError("zero quaternion has no inverse")
    return _conjugate(q) / squared_norm


@njit(cache=True)
def _multiply(p, q):
    return np.array(
        (
            p[0] * q[0] - np.dot(p[1:4], q[1:4]),
            p[0] * q[1] + q[0] * p[1] + p[2] * q[3] - p[3] * q[2],
            p[0] * q[2] + q[0] * p[2] + p[3] * q[1] - p[1] * q[3],
            p[0] * q[3] + q[0] * p[3] + p[1] * q[2] - p[2] * q[1],
        )
    )


@njit(cache=True)
def _relative_quaternion(reference, target):
    relative = _multiply(target, _inverse(reference))
    relative = _normalize_quaternion(relative)
    if relative[0] < 0.0:
        relative = -relative
    return relative
