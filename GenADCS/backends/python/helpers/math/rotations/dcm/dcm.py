import numpy as np

from ..quaternion.conversions import _dcm_to_quaternion, _quaternion_to_dcm
from ..quaternion.interpolation import rotation_vector_to_quaternion
from .kernels import (
    axis_angle_to_dcm,
    compose_dcm,
    dcm_orthogonality_error,
    inverse_dcm,
    normalize_dcm,
    relative_dcm,
    rotate_vector,
)


class DCM:
    """Direction cosine matrix for an active inertial-to-body q_B<-I rotation."""

    __slots__ = ("_matrix",)

    def __init__(self, matrix=None):
        value = (
            np.eye(3, dtype=np.float64)
            if matrix is None
            else np.asarray(matrix, dtype=np.float64)
        )
        if value.shape != (3, 3) or not np.all(np.isfinite(value)):
            raise ValueError("DCM must be a finite 3x3 matrix")
        if not np.allclose(value.T @ value, np.eye(3), atol=1e-8) or not np.isclose(
            np.linalg.det(value), 1.0, atol=1e-8
        ):
            raise ValueError("DCM must be a proper rotation matrix")
        self._matrix = value.copy()

    @classmethod
    def identity(cls):
        return cls()

    @classmethod
    def from_axis_angle(cls, axis, angle):
        axis = np.asarray(axis, dtype=np.float64)
        if (
            axis.shape != (3,)
            or not np.all(np.isfinite(axis))
            or not np.isfinite(angle)
        ):
            raise ValueError("axis must be a finite 3-vector and angle finite")
        return cls(axis_angle_to_dcm(axis, float(angle)))

    @classmethod
    def from_rotation_vector(cls, rotation_vector):
        vector = np.asarray(rotation_vector, dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("rotation_vector must be a finite 3-vector")
        return cls(_quaternion_to_dcm(rotation_vector_to_quaternion(vector)))

    @classmethod
    def from_quaternion(cls, quaternion):
        from ..quaternion.quaternion import Quaternion

        if not isinstance(quaternion, Quaternion):
            quaternion = Quaternion(quaternion)
        return cls(_quaternion_to_dcm(quaternion.hamilton_components))

    @classmethod
    def from_dcm(cls, matrix):
        return cls(matrix)

    @property
    def components(self):
        return self._matrix.copy()

    @property
    def hamilton_components(self):
        return _dcm_to_quaternion(self._matrix)

    @property
    def norm(self):
        return float(dcm_orthogonality_error(self._matrix))

    @property
    def orthogonality_error(self):
        return self.norm

    def as_matrix(self):
        return self._matrix.copy()

    def to_dcm(self):
        return self._matrix.copy()

    def to_quaternion(self):
        from ..quaternion.quaternion import Quaternion

        return Quaternion(_dcm_to_quaternion(self._matrix))

    def normalized(self):
        return DCM(normalize_dcm(self._matrix))

    def inverse(self):
        return DCM(inverse_dcm(self._matrix))

    def conjugate(self):
        return self.inverse()

    def multiply(self, other):
        if not isinstance(other, DCM):
            other = DCM(other)
        return DCM(compose_dcm(self._matrix, other._matrix))

    def rotate(self, vector_i):
        vector_i = np.asarray(vector_i, dtype=np.float64)
        if vector_i.shape != (3,) or not np.all(np.isfinite(vector_i)):
            raise ValueError("vector must be a finite 3-vector")
        return rotate_vector(self._matrix, vector_i)

    def inertial_to_body(self, vector_i):
        return self.rotate(vector_i)

    def body_to_inertial(self, vector_b):
        vector_b = np.asarray(vector_b, dtype=np.float64)
        if vector_b.shape != (3,) or not np.all(np.isfinite(vector_b)):
            raise ValueError("vector must be a finite 3-vector")
        return rotate_vector(inverse_dcm(self._matrix), vector_b)

    def relative_to(self, reference):
        if not isinstance(reference, DCM):
            reference = DCM(reference)
        return DCM(relative_dcm(self._matrix, reference._matrix))

    def __mul__(self, other):
        return self.multiply(other)

    def __array__(self, dtype=None):
        return np.asarray(self._matrix, dtype=dtype).copy()

    def __repr__(self):
        return f"DCM({self._matrix.tolist()}, frame='q_B<-I')"
