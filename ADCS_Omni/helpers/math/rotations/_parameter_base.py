import numpy as np

from .quaternion import Quaternion
from ..vector import norm as _vector_norm


class _RotationParameter:
    __slots__ = ("_parameters",)

    def __init__(self, values):
        parameters = np.asarray(values, dtype=np.float64)
        if parameters.shape != (3,) or not np.all(np.isfinite(parameters)):
            raise ValueError("rotation parameters must be a finite 3-vector")
        self._parameters = self._canonicalize(parameters)

    @classmethod
    def from_quaternion(cls, quaternion):
        if not isinstance(quaternion, Quaternion):
            quaternion = Quaternion(quaternion)
        return cls(cls._from_hamilton(quaternion.components))

    @classmethod
    def from_axis_angle(cls, axis, angle):
        return cls.from_quaternion(Quaternion.from_axis_angle(axis, angle))

    @classmethod
    def from_rotation_vector(cls, rotation_vector):
        return cls.from_quaternion(Quaternion.from_rotation_vector(rotation_vector))

    @classmethod
    def from_dcm(cls, matrix):
        return cls.from_quaternion(Quaternion.from_dcm(matrix))

    @classmethod
    def identity(cls):
        return cls(np.zeros(3))

    @property
    def components(self):
        return self._parameters.copy()

    @property
    def norm(self):
        return float(_vector_norm(self._parameters))

    def as_vector(self):
        return self._parameters.copy()

    def to_quaternion(self):
        return Quaternion(self._to_hamilton(self._parameters))

    def to_dcm(self):
        return self.to_quaternion().to_dcm()

    def rotate(self, vector_i):
        return self.to_quaternion().rotate(vector_i)

    def inertial_to_body(self, vector_i):
        return self.to_quaternion().inertial_to_body(vector_i)

    def body_to_inertial(self, vector_b):
        return self.to_quaternion().body_to_inertial(vector_b)

    def normalized(self):
        return type(self)(self._parameters)

    def conjugate(self):
        return type(self).from_quaternion(self.to_quaternion().conjugate())

    def inverse(self):
        return self.conjugate()

    def multiply(self, other):
        if not isinstance(other, _RotationParameter):
            raise TypeError("other must be a rotation-parameter object")
        return type(self).from_quaternion(self.to_quaternion() * other.to_quaternion())

    def relative_to(self, reference):
        if not isinstance(reference, _RotationParameter):
            raise TypeError("reference must be a rotation-parameter object")
        return type(self).from_quaternion(self.to_quaternion().relative_to(reference.to_quaternion()))

    def __mul__(self, other):
        return self.multiply(other)

    def __array__(self, dtype=None):
        return np.asarray(self._parameters, dtype=dtype).copy()

    @classmethod
    def _canonicalize(cls, parameters):
        raise NotImplementedError

    @staticmethod
    def _from_hamilton(quaternion):
        raise NotImplementedError

    @staticmethod
    def _to_hamilton(parameters):
        raise NotImplementedError
