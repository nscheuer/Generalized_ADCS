"""Body angular-rate reference without an attitude target."""

import numpy as np
from numba import njit

from .goal import Goal, validate_reference_state


@njit(cache=True)
def _angular_rate_reference(state, epoch_jd, runtime, rate):
    return None, rate


class AngularRateGoal(Goal):
    """Track a constant body-frame rate without specifying an attitude target."""

    kernel = staticmethod(_angular_rate_reference)

    def __init__(self, rate):
        self.rate = np.asarray(rate, dtype=np.float64)
        if self.rate.shape != (3,) or not np.all(np.isfinite(self.rate)):
            raise ValueError("rate must be a finite 3-element vector")
        self.rate = np.ascontiguousarray(self.rate)

    def reference(self, state, epoch_jd, runtime):
        value = validate_reference_state(state)
        return self.kernel(value, float(epoch_jd), runtime, self.rate)


angularrategoal = AngularRateGoal
