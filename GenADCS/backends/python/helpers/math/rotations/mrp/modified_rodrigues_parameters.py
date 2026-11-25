import numpy as np

from .._parameter_base import _RotationParameter
from .kernels import mrp_shadow_set, mrp_to_quaternion, quaternion_to_mrp


class Modified_Rodriguez_Parameters(_RotationParameter):
    """MRP input/output representation with automatic primary-set switching."""

    @classmethod
    def _canonicalize(cls, parameters: np.ndarray) -> np.ndarray:
        squared_norm = float(parameters @ parameters)
        if squared_norm > 1.0:
            return mrp_shadow_set(parameters)
        return parameters.copy()

    @staticmethod
    def _from_hamilton(quaternion: np.ndarray) -> np.ndarray:
        return quaternion_to_mrp(quaternion)

    @staticmethod
    def _to_hamilton(parameters: np.ndarray) -> np.ndarray:
        return mrp_to_quaternion(parameters)

    def switch_shadow_set(self) -> "Modified_Rodriguez_Parameters":
        switched = object.__new__(type(self))
        switched._parameters = mrp_shadow_set(self._parameters)
        return switched

    def __repr__(self) -> str:
        return f"Modified_Rodriguez_Parameters({self._parameters.tolist()}, frame='q_B<-I')"


Modified_Rodrigues_Parameters = Modified_Rodriguez_Parameters
