from .kernels import mrp_shadow_set, mrp_to_quaternion, quaternion_to_mrp
from .modified_rodrigues_parameters import (
    Modified_Rodrigues_Parameters,
    Modified_Rodriguez_Parameters,
)

ModifiedRodriguesParameters = Modified_Rodriguez_Parameters

__all__ = [
    "Modified_Rodriguez_Parameters",
    "Modified_Rodrigues_Parameters",
    "ModifiedRodriguesParameters",
]
