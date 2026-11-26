import numpy as np
from numba import njit

from .algebra import conjugate, multiply, normalize_quaternion


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


@njit(cache=True)
def quaternion_mean(quaternions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted manifold mean solving sum(w_i * log(q_i * mean^-1)) = 0."""
    count = quaternions.shape[0]
    normalized = np.empty((count, 4))
    for i in range(count):
        normalized[i] = normalize_quaternion(quaternions[i])
    mean = normalized[0].copy()
    deviations = np.empty((count, 3))
    for _ in range(32):
        mean_conj = conjugate(mean)
        scale = 0.0
        for i in range(count):
            deviations[i] = quaternion_to_rotation_vector(
                multiply(normalized[i], mean_conj)
            )
            scale += abs(weights[i]) * np.sqrt(np.dot(deviations[i], deviations[i]))
        residual = weights @ deviations
        residual_norm = np.sqrt(np.dot(residual, residual))
        tolerance = 1.0e-12 + 64.0 * 2.220446049250313e-16 * scale
        if residual_norm <= tolerance:
            return mean
        step = residual * min(1.0, (0.5 * np.pi) / max(residual_norm, 1.0e-15))
        improved = False
        alpha = 1.0
        for _ in range(16):
            candidate = normalize_quaternion(
                multiply(rotation_vector_to_quaternion(alpha * step), mean)
            )
            cand_conj = conjugate(candidate)
            cand_residual = np.zeros(3)
            for i in range(count):
                cand_residual += weights[i] * quaternion_to_rotation_vector(
                    multiply(normalized[i], cand_conj)
                )
            if np.sqrt(np.dot(cand_residual, cand_residual)) < residual_norm:
                mean = candidate
                improved = True
                break
            alpha *= 0.5
        if not improved:
            break
    return mean
