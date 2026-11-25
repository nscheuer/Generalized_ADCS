import numpy as np
from numba import njit


@njit(cache=True)
def rotate_vector(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    qv = q[1:4]
    return vector + 2.0 * (
        q[0] * np.cross(qv, vector) + np.cross(qv, np.cross(qv, vector))
    )


@njit(cache=True)
def quaternion_to_dcm(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        (
            (
                w * w + x * x - y * y - z * z,
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ),
            (
                2.0 * (x * y + w * z),
                w * w - x * x + y * y - z * z,
                2.0 * (y * z - w * x),
            ),
            (
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                w * w - x * x - y * y + z * z,
            ),
        )
    )


@njit(cache=True)
def dcm_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    q = np.empty(4, dtype=np.float64)
    trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        q[0] = 0.25 * scale
        q[1] = (matrix[2, 1] - matrix[1, 2]) / scale
        q[2] = (matrix[0, 2] - matrix[2, 0]) / scale
        q[3] = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        q[0] = (matrix[2, 1] - matrix[1, 2]) / scale
        q[1] = 0.25 * scale
        q[2] = (matrix[0, 1] + matrix[1, 0]) / scale
        q[3] = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        q[0] = (matrix[0, 2] - matrix[2, 0]) / scale
        q[1] = (matrix[0, 1] + matrix[1, 0]) / scale
        q[2] = 0.25 * scale
        q[3] = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        q[0] = (matrix[1, 0] - matrix[0, 1]) / scale
        q[1] = (matrix[0, 2] + matrix[2, 0]) / scale
        q[2] = (matrix[1, 2] + matrix[2, 1]) / scale
        q[3] = 0.25 * scale
    return q / np.sqrt(np.dot(q, q))


@njit(cache=True)
def axis_angle_to_quaternion(axis: np.ndarray, angle: float) -> np.ndarray:
    axis_norm = np.sqrt(np.dot(axis, axis))
    if axis_norm == 0.0:
        raise ValueError("axis must be nonzero")
    half_angle = 0.5 * angle
    scale = np.sin(half_angle) / axis_norm
    return np.array(
        (np.cos(half_angle), axis[0] * scale, axis[1] * scale, axis[2] * scale)
    )
