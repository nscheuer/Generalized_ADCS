import numpy as np
from numba import njit


@njit(cache=True)
def mrp_to_quaternion(parameters: np.ndarray) -> np.ndarray:
    squared_norm = np.dot(parameters, parameters)
    denominator = 1.0 + squared_norm
    return np.concatenate(
        (
            np.array(((1.0 - squared_norm) / denominator,)),
            2.0 * parameters / denominator,
        )
    )


@njit(cache=True)
def quaternion_to_mrp(quaternion: np.ndarray) -> np.ndarray:
    q = quaternion / np.sqrt(np.dot(quaternion, quaternion))
    if q[0] < 0.0:
        q = -q
    parameters = q[1:4] / (1.0 + q[0])
    squared_norm = np.dot(parameters, parameters)
    if squared_norm > 1.0:
        parameters = -parameters / squared_norm
    return parameters


@njit(cache=True)
def mrp_shadow_set(parameters: np.ndarray) -> np.ndarray:
    squared_norm = np.dot(parameters, parameters)
    if squared_norm == 0.0:
        raise ValueError("the identity MRP has no finite shadow set")
    return -parameters / squared_norm
