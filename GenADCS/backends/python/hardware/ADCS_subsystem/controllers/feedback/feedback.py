"""Feedback base class and shared gain matrix validation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike

FB_MRP_FEEDBACK_LINEARIZED = 1


def to_gain_matrix(value: float | ArrayLike, name: str, *, require_spd: bool) -> np.ndarray:
    """Validate and convert a scalar, 3-vector, or 3x3 matrix into a symmetric gain matrix."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        mat = float(arr) * np.eye(3, dtype=np.float64)
    elif arr.shape == (3,):
        mat = np.diag(arr)
    elif arr.shape == (3, 3):
        mat = np.array(arr, dtype=np.float64, copy=True)
    else:
        raise ValueError(f"{name} must be a scalar, 3-vector, or 3x3 matrix")
    if not np.all(np.isfinite(mat)):
        raise ValueError(f"{name} must be finite")
    if not np.allclose(mat, mat.T):
        raise ValueError(f"{name} must be symmetric")
    min_eig = float(np.linalg.eigvalsh(mat).min())
    if require_spd and min_eig <= 0.0:
        raise ValueError(f"{name} must be positive definite")
    if not require_spd and min_eig < 0.0:
        raise ValueError(f"{name} must be positive semidefinite")
    return np.ascontiguousarray(mat)


class Feedback(ABC):
    """Base class for AttitudeController feedback laws."""

    kind: ClassVar[int]

    @abstractmethod
    def to_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(kp, kd)`` 3x3 gain matrices."""
