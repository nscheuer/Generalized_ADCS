"""Base contract for compiled attitude-reference goals.

Goal kernels consume the coupled-dynamics state layout
``[q_body_from_inertial (4), omega_body (3), r_inertial (3), v_inertial (3)]``
and the compiled runtime used by the propagators. They return a scalar-first
Hamilton quaternion from inertial to body and the corresponding body-frame
angular rate, both as float64 arrays, or ``None`` for either component when
that reference is unspecified.

Python goal objects hold validated configuration. Their ``kernel`` is a Numba
dispatcher so the numerical reference calculation can run inside compiled
simulation loops without passing Python objects into nopython code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
from numba.core.registry import CPUDispatcher


class Goal(ABC):
    """Configured attitude goal with a compiled reference-generation kernel."""

    kernel: ClassVar[CPUDispatcher]

    @abstractmethod
    def reference(
        self, state: np.ndarray, epoch_jd: float, runtime: Any
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return desired attitude/rate; ``None`` means that component is unset."""

    def __call__(
        self, state: np.ndarray, epoch_jd: float, runtime: Any
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self.reference(state, epoch_jd, runtime)


def validate_reference_state(state: np.ndarray) -> np.ndarray:
    """Validate and return a spacecraft state with the canonical prefix."""
    value = np.asarray(state, dtype=np.float64)
    if value.ndim != 1 or value.size < 13 or not np.all(np.isfinite(value)):
        raise ValueError("goal state must have a finite 13-element spacecraft-state prefix")
    if np.linalg.norm(value[:4]) == 0.0:
        raise ValueError("goal state quaternion must be nonzero")
    return np.ascontiguousarray(value)


def validate_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """Validate and normalize a scalar-first Hamilton quaternion."""
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError("quaternion must be a finite 4-element vector")
    norm = np.linalg.norm(value)
    if norm == 0.0:
        raise ValueError("quaternion must be nonzero")
    return np.ascontiguousarray(value / norm)


def validate_vector(vector: np.ndarray, name: str = "vector") -> np.ndarray:
    """Validate and normalize a finite nonzero three-vector."""
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite 3-element vector")
    norm = np.linalg.norm(value)
    if norm == 0.0:
        raise ValueError(f"{name} must be nonzero")
    return np.ascontiguousarray(value / norm)

