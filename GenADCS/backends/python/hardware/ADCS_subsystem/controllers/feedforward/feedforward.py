"""Feedforward base class for AttitudeController."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

FF_PASSTHROUGH = 0


class Feedforward(ABC):
    """Base class for AttitudeController feedforward trajectory generators."""

    kind: ClassVar[int]

    @abstractmethod
    def to_params(self) -> np.ndarray:
        """Return fixed-width ``float64`` parameter vector."""
