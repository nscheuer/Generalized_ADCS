import numpy as np
from numba import njit


@njit(cache=True)
def cross(a, b):
    return np.array(
        (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )
    )


@njit(cache=True)
def norm(vector):
    # A scalar loop handles strided views without Numba's np.dot contiguity
    # performance warning or an implicit copy.
    squared = 0.0
    for index in range(vector.size):
        squared += vector[index] * vector[index]
    return np.sqrt(squared)


@njit(cache=True)
def normalize(vector):
    magnitude = norm(vector)
    if magnitude == 0.0:
        return np.zeros_like(vector)
    return vector / magnitude


@njit(cache=True)
def unit(vector):
    magnitude = norm(vector)
    if magnitude == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return vector / magnitude


@njit(cache=True)
def skew(vector):
    return np.array(
        (
            (0.0, -vector[2], vector[1]),
            (vector[2], 0.0, -vector[0]),
            (-vector[1], vector[0], 0.0),
        )
    )


@njit(cache=True)
def saturate_vector(vector, limit):
    """Clamp each component of a vector to symmetric limits."""
    return np.minimum(np.maximum(vector, -limit), limit)
