import numpy as np
from numba import njit


@njit(cache=True)
def rodrigues_to_quaternion(parameters):
    denominator = np.sqrt(1.0 + np.dot(parameters, parameters))
    return np.concatenate((np.array((1.0 / denominator,)), parameters / denominator))


@njit(cache=True)
def quaternion_to_rodrigues(quaternion):
    q = quaternion / np.sqrt(np.dot(quaternion, quaternion))
    if q[0] < 0.0:
        q = -q
    if abs(q[0]) <= 1e-12:
        raise ValueError("Rodrigues parameters are singular at a 180-degree rotation")
    return q[1:4] / q[0]
