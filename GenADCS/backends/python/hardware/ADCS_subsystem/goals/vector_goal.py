"""Attitude reference aligning one body axis with an inertial direction."""

import numpy as np
from numba import njit

from GenADCS.backends.python.helpers.math.rotations.quaternion.conversions import dcm_to_quaternion
from .goal import Goal, validate_reference_state, validate_vector


@njit(cache=True)
def _vector_reference(state, epoch_jd, runtime, inertial_axis, body_axis, inertial_roll, body_roll):
    # Build corresponding right-handed triads, resolving the otherwise-free roll.
    i0 = inertial_axis
    i1 = inertial_roll - np.dot(inertial_roll, i0) * i0
    i1 /= np.linalg.norm(i1)
    i2 = np.cross(i0, i1)
    b0 = body_axis
    b1 = body_roll - np.dot(body_roll, b0) * b0
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(b0, b1)
    inertial_basis = np.column_stack((i0, i1, i2))
    body_basis = np.column_stack((b0, b1, b2))
    return dcm_to_quaternion(body_basis @ inertial_basis.T), np.zeros(3)


class VectorGoal(Goal):
    """Align ``body_axis`` to ``inertial_vector`` and resolve roll with references.

    The default body axis is +Z and the default roll references are +X in their
    respective frames. Choose nonparallel roll reference vectors.
    """

    kernel = staticmethod(_vector_reference)

    def __init__(self, inertial_vector, body_axis=(0.0, 0.0, 1.0), *,
                 inertial_roll=(1.0, 0.0, 0.0), body_roll=(1.0, 0.0, 0.0)):
        self.inertial_vector = validate_vector(inertial_vector, "inertial_vector")
        self.body_axis = validate_vector(body_axis, "body_axis")
        self.inertial_roll = validate_vector(inertial_roll, "inertial_roll")
        self.body_roll = validate_vector(body_roll, "body_roll")
        if abs(np.dot(self.inertial_vector, self.inertial_roll)) > 1.0 - 1e-10:
            raise ValueError("inertial_roll must not be parallel to inertial_vector")
        if abs(np.dot(self.body_axis, self.body_roll)) > 1.0 - 1e-10:
            raise ValueError("body_roll must not be parallel to body_axis")

    def reference(self, state, epoch_jd, runtime):
        value = validate_reference_state(state)
        q, _ = self.kernel(value, float(epoch_jd), runtime, self.inertial_vector,
                           self.body_axis, self.inertial_roll, self.body_roll)
        # A fixed inertial direction defines a fixed attitude reference here.
        return q, np.zeros(3)


vectorgoal = VectorGoal
