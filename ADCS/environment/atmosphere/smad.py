"""SMAD tabulated atmospheric density model."""

import numpy as np

from ADCS.orbits.universal_constants import EarthConstants

from .atmosphere import Atmosphere
from .density_model import DensityModel


class SMADAtmosphere(Atmosphere):
    """Use the existing SMAD density table and interpolation implementation.

    ``density(position_eci_km)`` accepts a single ECI position or an ``(N, 3)``
    array and returns density in kg/m³. The model depends only on distance
    from Earth's center, so the same value is obtained for an ECEF position.
    """

    def __init__(self, altitude_range=None, rho_range=None) -> None:
        kwargs = {}
        if altitude_range is not None:
            kwargs["altitude_range"] = altitude_range
        if rho_range is not None:
            kwargs["rho_range"] = rho_range
        self._model = DensityModel(**kwargs)

    @property
    def altitude_range(self):
        return self._model.altitude_range

    @property
    def rho_range(self):
        return self._model.rho_range

    def interpolate(self, altitude_km):
        return self._model.interpolate(altitude_km)

    def density(self, position_eci_km, datetime_utc=None, geodetic_lla=None):
        del datetime_utc, geodetic_lla
        position = np.asarray(position_eci_km, dtype=float)
        if position.ndim not in (1, 2) or position.shape[-1] != 3:
            raise ValueError("position_eci_km must have shape (3,) or (N, 3)")
        altitude = np.linalg.norm(position, axis=-1) - EarthConstants.R_e
        if np.ndim(altitude) == 0:
            return self.interpolate(float(altitude))
        return self._model.interpolate_many(altitude)

    def __repr__(self) -> str:
        return f"SMADAtmosphere(n={len(self._model.altitude_range)})"
