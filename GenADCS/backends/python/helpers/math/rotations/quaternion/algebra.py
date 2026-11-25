import numpy as np
from numba import njit


@njit(cache=True)
def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    magnitude = np.sqrt(np.dot(q, q))
    if magnitude == 0.0:
        raise ValueError("quaternion must have nonzero norm")
    return q / magnitude


@njit(cache=True)
def quaternion_norm(q: np.ndarray) -> float:
    return np.sqrt(np.dot(q, q))


@njit(cache=True)
def jpl_to_hamilton(q_jpl: np.ndarray) -> np.ndarray:
    return np.array((q_jpl[3], -q_jpl[0], -q_jpl[1], -q_jpl[2]))


@njit(cache=True)
def hamilton_to_jpl(q_hamilton: np.ndarray) -> np.ndarray:
    return np.array((-q_hamilton[1], -q_hamilton[2], -q_hamilton[3], q_hamilton[0]))


@njit(cache=True)
def conjugate(q: np.ndarray) -> np.ndarray:
    return np.array((q[0], -q[1], -q[2], -q[3]))


@njit(cache=True)
def inverse(q: np.ndarray) -> np.ndarray:
    squared_norm = np.dot(q, q)
    if squared_norm == 0.0:
        raise ValueError("zero quaternion has no inverse")
    return conjugate(q) / squared_norm


@njit(cache=True)
def multiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.array(
        (
            p[0] * q[0] - np.dot(p[1:4], q[1:4]),
            p[0] * q[1] + q[0] * p[1] + p[2] * q[3] - p[3] * q[2],
            p[0] * q[2] + q[0] * p[2] + p[3] * q[1] - p[1] * q[3],
            p[0] * q[3] + q[0] * p[3] + p[1] * q[2] - p[2] * q[1],
        )
    )


@njit(cache=True)
def relative_quaternion(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = multiply(target, inverse(reference))
    relative = normalize_quaternion(relative)
    if relative[0] < 0.0:
        relative = -relative
    return relative
