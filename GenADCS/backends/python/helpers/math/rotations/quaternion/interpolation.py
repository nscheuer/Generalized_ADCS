import numpy as np
from numba import njit

from .algebra import _normalize_quaternion


@njit(cache=True)
def slerp(q0, q1, fraction):
    a = _normalize_quaternion(q0)
    b = _normalize_quaternion(q1)
    cosine = np.dot(a, b)
    if cosine < 0.0:
        b = -b
        cosine = -cosine
    cosine = min(cosine, 1.0)
    if cosine > 0.9995:
        return _normalize_quaternion((1.0 - fraction) * a + fraction * b)
    angle = np.arccos(cosine)
    denominator = np.sin(angle)
    return _normalize_quaternion(
        (np.sin((1.0 - fraction) * angle) * a + np.sin(fraction * angle) * b)
        / denominator
    )


@njit(cache=True)
def rotation_vector_to_quaternion(rotation_vector):
    angle = np.sqrt(np.dot(rotation_vector, rotation_vector))
    if angle == 0.0:
        return np.array((1.0, 0.0, 0.0, 0.0))
    half = 0.5 * angle
    return np.concatenate(
        (np.array((np.cos(half),)), rotation_vector * (np.sin(half) / angle))
    )


@njit(cache=True)
def quaternion_to_rotation_vector(q):
    unit_q = _normalize_quaternion(q)
    if unit_q[0] < 0.0:
        unit_q = -unit_q
    vector_norm = np.sqrt(np.dot(unit_q[1:4], unit_q[1:4]))
    if vector_norm == 0.0:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(vector_norm, unit_q[0])
    return unit_q[1:4] * (angle / vector_norm)
