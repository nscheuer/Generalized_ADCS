"""Base class and shared saturation helper for reaction-wheel torque allocators."""

from __future__ import annotations

from abc import ABC
from typing import ClassVar

import numpy as np
from numba import njit

ALLOC_L2 = 0
ALLOC_LINFINITY = 1


@njit(cache=True, inline="always")
def saturate_wheels(u_rw: np.ndarray, max_torque: np.ndarray) -> np.ndarray:
    """Clamp wheel torque commands to per-wheel limits."""
    for i in range(u_rw.size):
        lim = max_torque[i]
        if u_rw[i] > lim:
            u_rw[i] = lim
        elif u_rw[i] < -lim:
            u_rw[i] = -lim
    return u_rw


class Allocator(ABC):
    """Base class for reaction-wheel torque allocators."""

    kind: ClassVar[int]
