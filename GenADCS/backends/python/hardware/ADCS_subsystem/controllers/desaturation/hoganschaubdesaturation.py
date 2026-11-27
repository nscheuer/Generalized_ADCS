"""Hogan & Schaub continuous nullspace reaction-wheel momentum management."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar

import numpy as np
from numba import njit

from .desaturation import DESAT_HOGAN_SCHAUB, Desaturation


@njit(cache=True, inline="always")
def hogan_schaub_u0(
    wheel_momenta: np.ndarray,
    gain: float,
    target_momentum: float,
) -> np.ndarray:
    """Continuous nullspace momentum management: ``u_0 = +k * (h - h*)``."""
    return gain * (wheel_momenta - target_momentum)


@dataclass(slots=True)
class HoganSchaubManagement(Desaturation):
    """Continuous nullspace reaction-wheel momentum management (Hogan & Schaub)."""

    gain: float
    target_momentum: float = 0.0
    kind: ClassVar[int] = DESAT_HOGAN_SCHAUB

    def __post_init__(self) -> None:
        self.gain = float(self.gain)
        self.target_momentum = float(self.target_momentum)
        if not math.isfinite(self.gain) or self.gain <= 0.0:
            raise ValueError("HoganSchaubManagement gain must be finite and positive")
        if not math.isfinite(self.target_momentum):
            raise ValueError("target_momentum must be finite")

    def to_params(self) -> np.ndarray:
        return np.array([self.gain, self.target_momentum], dtype=np.float64)
