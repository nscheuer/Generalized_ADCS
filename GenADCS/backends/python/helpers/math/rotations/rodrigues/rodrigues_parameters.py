import numpy as np

from .._parameter_base import RotationParameter
from .kernels import quaternion_to_rodrigues, rodrigues_to_quaternion


class Rodriguez_Parameters(RotationParameter):
    """Rodrigues input/output representation of the active q_B<-I rotation."""

    @classmethod
    def _canonicalize(cls, parameters: np.ndarray) -> np.ndarray:
        return parameters.copy()

    @staticmethod
    def _from_hamilton(quaternion: np.ndarray) -> np.ndarray:
        return quaternion_to_rodrigues(quaternion)

    @staticmethod
    def _to_hamilton(parameters: np.ndarray) -> np.ndarray:
        return rodrigues_to_quaternion(parameters)

    def __repr__(self) -> str:
        return f"Rodriguez_Parameters({self._parameters.tolist()}, frame='q_B<-I')"


Rodrigues_Parameters = Rodriguez_Parameters
