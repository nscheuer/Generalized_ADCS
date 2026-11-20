"""Python names backed by native geomagnetic kernels."""
from .._environment import (
    MagneticFieldModel, magnetic_field_inertial_tesla, read_shc,
)

__all__ = ["MagneticFieldModel", "magnetic_field_inertial_tesla", "read_shc"]
