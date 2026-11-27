"""Shared validation and dispatch for orbit-relative goal classes."""

import math
from typing import Any

import numpy as np

from .goal import Goal, validate_reference_state


class OrbitGoal(Goal):
    """Base implementation for goals backed by orbit-geometry kernels."""

    def reference(
        self, state: np.ndarray, epoch_jd: float, runtime: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        epoch = float(epoch_jd)
        if not math.isfinite(epoch):
            raise ValueError("epoch_jd must be finite")
        return self.kernel(validate_reference_state(state), epoch, runtime)
