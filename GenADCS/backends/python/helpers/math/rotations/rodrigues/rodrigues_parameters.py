from .._parameter_base import _RotationParameter
from .kernels import quaternion_to_rodrigues, rodrigues_to_quaternion


class Rodriguez_Parameters(_RotationParameter):
    """Rodrigues input/output representation of the active q_B<-I rotation."""

    @classmethod
    def _canonicalize(cls, parameters):
        return parameters.copy()

    @staticmethod
    def _from_hamilton(quaternion):
        return quaternion_to_rodrigues(quaternion)

    @staticmethod
    def _to_hamilton(parameters):
        return rodrigues_to_quaternion(parameters)

    def __repr__(self):
        return f"Rodriguez_Parameters({self._parameters.tolist()}, frame='q_B<-I')"


Rodrigues_Parameters = Rodriguez_Parameters
