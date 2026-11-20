import numpy as np
from numba import njit


@njit(cache=True)
def quaternion_derivative(q_b_from_i, angular_velocity_b):
    q = q_b_from_i
    w = angular_velocity_b
    return 0.5 * np.array((
        np.dot(q[1:4], w),
        -(q[0] * w[0] + w[1] * q[3] - w[2] * q[2]),
        -(q[0] * w[1] + w[2] * q[1] - w[0] * q[3]),
        -(q[0] * w[2] + w[0] * q[2] - w[1] * q[1]),
    ))


@njit(cache=True)
def quaternion_rate_matrix(q_b_from_i):
    w, x, y, z = q_b_from_i
    return 0.5 * np.array(((x, y, z), (-w, -z, y), (z, -w, -x), (-y, x, -w)))


@njit(cache=True)
def angular_velocity_from_quaternion_derivative(q_b_from_i, qdot_b_from_i):
    q = q_b_from_i
    qdot = qdot_b_from_i
    return 2.0 * np.array((
        q[1] * qdot[0] - q[0] * qdot[1] + q[3] * qdot[2] - q[2] * qdot[3],
        q[2] * qdot[0] - q[3] * qdot[1] - q[0] * qdot[2] + q[1] * qdot[3],
        q[3] * qdot[0] + q[2] * qdot[1] - q[1] * qdot[2] - q[0] * qdot[3],
    ))


@njit(cache=True)
def quaternion_normalization_jacobian(quaternion):
    magnitude = np.sqrt(np.dot(quaternion, quaternion))
    if magnitude == 0.0:
        raise ValueError("quaternion must have nonzero norm")
    return np.eye(4) / magnitude - np.outer(quaternion, quaternion) / magnitude**3


@njit(cache=True)
def rotated_vector_quaternion_jacobian(q_b_from_i, vector_i):
    qv = q_b_from_i[1:4]
    jacobian = np.empty((3, 4))
    jacobian[:, 0] = 2.0 * np.cross(qv, vector_i)
    jacobian[:, 1:4] = 2.0 * (
        (np.dot(qv, vector_i) * np.eye(3))
        + np.outer(qv, vector_i)
        - 2.0 * np.outer(vector_i, qv)
        - q_b_from_i[0] * np.array(((0.0, -vector_i[2], vector_i[1]), (vector_i[2], 0.0, -vector_i[0]), (-vector_i[1], vector_i[0], 0.0)))
    )
    return jacobian


@njit(cache=True)
def quaternion_to_euler321(q_b_from_i):
    w, x, y, z = q_b_from_i
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = np.arcsin(min(max(sin_pitch, -1.0), 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array((roll, pitch, yaw))
