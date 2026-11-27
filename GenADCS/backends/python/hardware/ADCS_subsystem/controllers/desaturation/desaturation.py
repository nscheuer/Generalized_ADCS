"""Desaturation base class and NoDesaturation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

DESAT_NONE = 0
DESAT_HOGAN_SCHAUB = 1


class Desaturation(ABC):
    """Base class for nullspace reaction-wheel momentum management."""

    kind: ClassVar[int]

    @abstractmethod
    def to_params(self) -> np.ndarray:
        """Return ``[gain, target_momentum]`` parameters."""


@dataclass(slots=True)
class NoDesaturation(Desaturation):
    """Disable nullspace wheel momentum management (``u_0 = 0``)."""

    kind: ClassVar[int] = DESAT_NONE

    def to_params(self) -> np.ndarray:
        return np.zeros(3, dtype=np.float64)
