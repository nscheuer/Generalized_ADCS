import numpy as np

from .kernels import mrp_shadow_set, mrp_to_quaternion, quaternion_to_mrp
from .._parameter_base import _RotationParameter


class Modified_Rodriguez_Parameters(_RotationParameter):
    """MRP input/output representation with automatic primary-set switching."""

    @classmethod
    def _canonicalize(cls, parameters):
        squared_norm = float(parameters @ parameters)
        if squared_norm > 1.0:
            return mrp_shadow_set(parameters)
        return parameters.copy()

    @staticmethod
    def _from_hamilton(quaternion):
        return quaternion_to_mrp(quaternion)

    @staticmethod
    def _to_hamilton(parameters):
        return mrp_to_quaternion(parameters)

    def switch_shadow_set(self):
        switched = object.__new__(type(self))
        switched._parameters = mrp_shadow_set(self._parameters)
        return switched

    def __repr__(self):
        return f"Modified_Rodriguez_Parameters({self._parameters.tolist()}, frame='q_B<-I')"


Modified_Rodrigues_Parameters = Modified_Rodriguez_Parameters
