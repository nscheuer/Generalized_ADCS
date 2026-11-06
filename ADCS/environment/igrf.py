"""Compatibility imports for the relocated IGRF-14 kernel."""

from .magnetic_field.igrf14_numba import NMAX, RE, coefficients_at, igrf_gc

__all__ = ["NMAX", "RE", "coefficients_at", "igrf_gc"]
