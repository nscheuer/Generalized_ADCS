"""Fixed quaternion and angular-rate reference."""

import numpy as np
from numba import njit

from .goal import Goal, validate_quaternion, validate_reference_state


@njit(cache=True)
def _quaternion_reference(state, epoch_jd, runtime, quaternion, rate):
    return quaternion, rate


class QuaternionGoal(Goal):
    """Track a fixed inertial-to-body quaternion and optional body rate."""

    kernel = staticmethod(_quaternion_reference)

    def __init__(self, quaternion, rate=None):
        self.quaternion = validate_quaternion(quaternion)
        self.rate = np.zeros(3) if rate is None else np.asarray(rate, dtype=np.float64)
        if self.rate.shape != (3,) or not np.all(np.isfinite(self.rate)):
            raise ValueError("rate must be a finite 3-element vector")
        self.rate = np.ascontiguousarray(self.rate)

    def reference(self, state, epoch_jd, runtime):
        value = validate_reference_state(state)
        return self.kernel(value, float(epoch_jd), runtime, self.quaternion, self.rate)


quaterniongoal = QuaternionGoal
