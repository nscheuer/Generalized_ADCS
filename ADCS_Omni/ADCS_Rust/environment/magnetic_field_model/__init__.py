"""Rust-backed IGRF-14 and CHAOS-8.7 magnetic field models."""

from .magnetic_field_model import MagneticFieldModel, magnetic_field_inertial_tesla, read_shc
from .igrf14 import IGRF14
from .chaos8 import CHAOS8
from .wmm import WMM

__all__ = ["CHAOS8", "IGRF14", "MagneticFieldModel", "WMM", "magnetic_field_inertial_tesla", "read_shc"]
