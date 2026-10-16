"""IGRF-14 magnetic field model backed by the existing selectable kernels."""

from . import DEFAULT_MAGNETIC_MODEL, _normalize_magnetic_model, field_gc
from .magnetic_field import MagneticField


class IGRF14(MagneticField):
    """IGRF-14 with either the Numba or ppigrf backend.

    ``field(position_ecef_km, date)`` accepts one ECEF position or an ``(N, 3)``
    array and returns ECEF field vectors in tesla. Dates may be a single
    ``datetime`` or one date per position.
    """

    def __init__(self, implementation: str = DEFAULT_MAGNETIC_MODEL, *, name: str | None = None) -> None:
        if name is not None:
            implementation = name
        self.name = _normalize_magnetic_model(implementation)

    def field_gc(self, r, theta, phi, date):
        return field_gc(self.name, r, theta, phi, date)
