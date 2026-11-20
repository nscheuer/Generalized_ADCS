"""Spherical harmonic primitives supplied by the native extension."""
from .._environment import _schmidt_table

RE = 6371.2

__all__ = ["RE", "_schmidt_table"]
