from .kernels import (
    quaternion_to_rodrigues as quaternion_to_rodrigues,
)
from .kernels import (
    rodrigues_to_quaternion as rodrigues_to_quaternion,
)
from .rodrigues_parameters import Rodrigues_Parameters, Rodriguez_Parameters

RodriguesParameters = Rodriguez_Parameters

__all__ = ["RodriguesParameters", "Rodrigues_Parameters", "Rodriguez_Parameters"]
