"""IGRF-14 geomagnetic model and coefficient kernel."""
from .igrf14 import IGRF14, igrf14_field_inertial_tesla

__all__ = ["IGRF14", "igrf14_field_inertial_tesla"]
