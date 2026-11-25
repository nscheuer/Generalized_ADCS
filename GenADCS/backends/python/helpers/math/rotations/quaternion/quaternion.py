from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .algebra import (
    _conjugate,
    _hamilton_to_jpl,
    _inverse,
    _jpl_to_hamilton,
    _multiply,
    _normalize_quaternion,
    _quaternion_norm,
    _relative_quaternion,
)
from .conversions import (
    _axis_angle_to_quaternion,
    _dcm_to_quaternion,
    _quaternion_to_dcm,
    _rotate_vector,
)
from .interpolation import rotation_vector_to_quaternion


class Quaternion:
    """Convention-aware public quaternion with Hamilton q_B<-I storage.

    Hamilton inputs use scalar-first ordering and active rotation. JPL inputs
    use scalar-last ordering and the passive JPL convention; they are converted
    at construction with q_H = [w_JPL, -v_JPL]. Stored values and numerical
    kernels use normalized scalar-first Hamilton quaternions.
    """

    __slots__ = ("_convention", "_hamilton")

    def __init__(
        self,
        values: ArrayLike = (1.0, 0.0, 0.0, 0.0),
        convention: str = "Hamilton",
    ) -> None:
        raw = np.asarray(values, dtype=np.float64)
        if raw.shape != (4,) or not np.all(np.isfinite(raw)):
            raise ValueError("quaternion values must be a finite 4-vector")
        convention_key = str(convention).strip().lower()
        if convention_key in ("hamilton", "h"):
            hamilton = raw.copy()
            self._convention = "Hamilton"
        elif convention_key in ("jpl", "j"):
            hamilton = _jpl_to_hamilton(raw)
            self._convention = "JPL"
        else:
            raise ValueError("convention must be 'Hamilton' or 'JPL'")
        self._hamilton = _normalize_quaternion(hamilton)

    @classmethod
    def identity(cls) -> "Quaternion":
        return cls()

    @classmethod
    def from_axis_angle(cls, axis: ArrayLike, angle: float) -> "Quaternion":
        axis = np.asarray(axis, dtype=np.float64)
        if (
            axis.shape != (3,)
            or not np.all(np.isfinite(axis))
            or not np.isfinite(angle)
        ):
            raise ValueError("axis must be a finite 3-vector and angle finite")
        return cls(_axis_angle_to_quaternion(axis, float(angle)))

    @classmethod
    def from_rotation_vector(cls, rotation_vector: ArrayLike) -> "Quaternion":
        vector = np.asarray(rotation_vector, dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("rotation_vector must be a finite 3-vector")
        return cls(rotation_vector_to_quaternion(vector))

    @classmethod
    def from_dcm(cls, matrix: ArrayLike) -> "Quaternion":
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("matrix must be a finite 3x3 DCM")
        if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-8) or not np.isclose(
            np.linalg.det(matrix), 1.0, atol=1e-8
        ):
            raise ValueError("matrix must be a proper rotation matrix")
        return cls(_dcm_to_quaternion(matrix))

    @property
    def components(self) -> np.ndarray:
        return self.as_convention(self._convention)

    @property
    def hamilton_components(self) -> np.ndarray:
        return self._hamilton.copy()

    @property
    def convention(self) -> str:
        return self._convention

    @property
    def norm(self) -> float:
        return float(_quaternion_norm(self._hamilton))

    def as_convention(self, convention: str = "Hamilton") -> np.ndarray:
        convention_key = str(convention).strip().lower()
        if convention_key in ("hamilton", "h"):
            return self._hamilton.copy()
        if convention_key in ("jpl", "j"):
            return _hamilton_to_jpl(self._hamilton)
        raise ValueError("convention must be 'Hamilton' or 'JPL'")

    def to_convention(self, convention: str = "Hamilton") -> "Quaternion":
        return Quaternion(self.as_convention(convention), convention=convention)

    def normalized(self) -> "Quaternion":
        return Quaternion(_normalize_quaternion(self._hamilton))

    def conjugate(self) -> "Quaternion":
        return Quaternion(_conjugate(self._hamilton))

    def inverse(self) -> "Quaternion":
        return Quaternion(_inverse(self._hamilton))

    def multiply(self, other: ArrayLike | "Quaternion") -> "Quaternion":
        if not isinstance(other, Quaternion):
            other = Quaternion(other)
        return Quaternion(_multiply(self._hamilton, other._hamilton))

    def rotate(self, vector_i: ArrayLike) -> np.ndarray:
        vector_i = np.asarray(vector_i, dtype=np.float64)
        if vector_i.shape != (3,) or not np.all(np.isfinite(vector_i)):
            raise ValueError("vector must be a finite 3-vector")
        return _rotate_vector(self._hamilton, vector_i)

    def inertial_to_body(self, vector_i: ArrayLike) -> np.ndarray:
        return self.rotate(vector_i)

    def body_to_inertial(self, vector_b: ArrayLike) -> np.ndarray:
        vector_b = np.asarray(vector_b, dtype=np.float64)
        if vector_b.shape != (3,) or not np.all(np.isfinite(vector_b)):
            raise ValueError("vector must be a finite 3-vector")
        return _rotate_vector(_conjugate(self._hamilton), vector_b)

    def to_dcm(self) -> np.ndarray:
        return _quaternion_to_dcm(self._hamilton)

    def relative_to(self, reference: ArrayLike | "Quaternion") -> "Quaternion":
        if not isinstance(reference, Quaternion):
            reference = Quaternion(reference)
        return Quaternion(_relative_quaternion(reference._hamilton, self._hamilton))

    def __mul__(self, other: ArrayLike | "Quaternion") -> "Quaternion":
        return self.multiply(other)

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        return np.asarray(self._hamilton, dtype=dtype).copy()

    def __repr__(self) -> str:
        return f"Quaternion({self._hamilton.tolist()}, convention='Hamilton', frame='q_B<-I')"
