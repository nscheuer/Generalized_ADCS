"""Attitude reference from two vector correspondences."""

import numpy as np
from numba import njit

from GenADCS.backends.python.helpers.math.rotations.quaternion.conversions import dcm_to_quaternion
from .goal import Goal, validate_reference_state, validate_vector


@njit(cache=True)
def _two_vector_reference(state, epoch_jd, runtime, inertial_first, inertial_second,
                          body_first, body_second):
    i0 = inertial_first
    i1 = inertial_second - np.dot(inertial_second, i0) * i0
    i1 /= np.linalg.norm(i1)
    i2 = np.cross(i0, i1)
    b0 = body_first
    b1 = body_second - np.dot(body_second, b0) * b0
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(b0, b1)
    q = dcm_to_quaternion(np.column_stack((b0, b1, b2)) @ np.column_stack((i0, i1, i2)).T)
    return q, np.zeros(3)


class TwoVectorGoal(Goal):
    """Track two nonparallel inertial-to-body vector correspondences."""

    kernel = staticmethod(_two_vector_reference)

    def __init__(self, inertial_vectors, body_vectors):
        inertial = np.asarray(inertial_vectors, dtype=np.float64)
        body = np.asarray(body_vectors, dtype=np.float64)
        if inertial.shape != (2, 3) or body.shape != (2, 3):
            raise ValueError("inertial_vectors and body_vectors must each have shape (2, 3)")
        self.inertial_first = validate_vector(inertial[0], "inertial_vectors[0]")
        self.inertial_second = validate_vector(inertial[1], "inertial_vectors[1]")
        self.body_first = validate_vector(body[0], "body_vectors[0]")
        self.body_second = validate_vector(body[1], "body_vectors[1]")
        if abs(np.dot(self.inertial_first, self.inertial_second)) > 1.0 - 1e-10:
            raise ValueError("inertial vectors must not be parallel")
        if abs(np.dot(self.body_first, self.body_second)) > 1.0 - 1e-10:
            raise ValueError("body vectors must not be parallel")

    def reference(self, state, epoch_jd, runtime):
        value = validate_reference_state(state)
        return self.kernel(value, float(epoch_jd), runtime, self.inertial_first,
                           self.inertial_second, self.body_first, self.body_second)


twovectorgoal = TwoVectorGoal
