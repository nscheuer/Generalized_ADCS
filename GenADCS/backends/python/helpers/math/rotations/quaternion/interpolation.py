import numpy as np
from numba import njit

from .algebra import normalize_quaternion


@njit(cache=True)
def slerp(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    a = normalize_quaternion(q0)
    b = normalize_quaternion(q1)
    cosine = np.dot(a, b)
    if cosine < 0.0:
        b = -b
        cosine = -cosine
    cosine = min(cosine, 1.0)
    if cosine > 0.9995:
        return normalize_quaternion((1.0 - fraction) * a + fraction * b)
    angle = np.arccos(cosine)
    denominator = np.sin(angle)
    return normalize_quaternion(
        (np.sin((1.0 - fraction) * angle) * a + np.sin(fraction * angle) * b)
        / denominator
    )


@njit(cache=True)
def rotation_vector_to_quaternion(rotation_vector: np.ndarray) -> np.ndarray:
    angle = np.sqrt(np.dot(rotation_vector, rotation_vector))
    if angle == 0.0:
        return np.array((1.0, 0.0, 0.0, 0.0))
    half = 0.5 * angle
    return np.concatenate(
        (np.array((np.cos(half),)), rotation_vector * (np.sin(half) / angle))
    )


@njit(cache=True)
def quaternion_to_rotation_vector(q: np.ndarray) -> np.ndarray:
    unit_q = normalize_quaternion(q)
    if unit_q[0] < 0.0:
        unit_q = -unit_q
    vector_norm = np.sqrt(np.dot(unit_q[1:4], unit_q[1:4]))
    if vector_norm == 0.0:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(vector_norm, unit_q[0])
    return unit_q[1:4] * (angle / vector_norm)
