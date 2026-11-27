"""L2 minimum-norm reaction-wheel torque allocator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numba import njit

from .allocator import ALLOC_L2, Allocator, saturate_wheels


@njit(cache=True)
def allocate_l2(
    tau_c: np.ndarray,
    a_pinv: np.ndarray,
    null_proj: np.ndarray,
    u_0: np.ndarray,
    max_torque: np.ndarray,
) -> np.ndarray:
    """L2 minimum-norm torque allocation with nullspace desaturation."""
    u_rw = a_pinv @ tau_c + null_proj @ u_0
    return saturate_wheels(u_rw, max_torque)


@dataclass(slots=True)
class L2Allocator(Allocator):
    """Minimum-L2-norm Moore-Penrose pseudoinverse allocator."""

    kind: ClassVar[int] = ALLOC_L2
