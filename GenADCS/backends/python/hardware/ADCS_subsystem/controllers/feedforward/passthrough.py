"""Passthrough feedforward component."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numba import njit

from .feedforward import FF_PASSTHROUGH, Feedforward


@njit(cache=True, inline="always")
def passthrough_step(
    q_goal: np.ndarray,
    w_goal: np.ndarray,
    q_current: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pass goal attitude/rate through as desired trajectory with zero feedforward torque."""
    q_d = q_current.copy() if np.isnan(q_goal[0]) else q_goal.copy()
    w_d = np.zeros(3, dtype=np.float64) if np.isnan(w_goal[0]) else w_goal.copy()
    w_d_dot = np.zeros(3, dtype=np.float64)
    ff_torque = np.zeros(3, dtype=np.float64)
    return q_d, w_d, w_d_dot, ff_torque


@dataclass(slots=True)
class Passthrough(Feedforward):
    """Pass the goal attitude and rate directly without slew shaping."""

    kind: ClassVar[int] = FF_PASSTHROUGH

    def to_params(self) -> np.ndarray:
        return np.zeros(1, dtype=np.float64)
