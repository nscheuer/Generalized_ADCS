"""SMAD tabulated atmosphere with log-linear density interpolation.

Reference data are from *Space Mission Analysis and Design* (Wertz and Larson,
eds.). See the project citation guide for bibliographic details.
"""

import numpy as np

from ADCS.orbits.universal_constants import EarthConstants
from .atmosphere import Atmosphere


SMAD_ALTITUDE_KM = np.array([
    0, 100, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 450,
    500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000,
], dtype=float)
SMAD_DENSITY_KG_M3 = np.array([
    1.2, 5.69e-7, 2.02e-9, 7.66e-10, 2.90e-10, 1.46e-10, 7.30e-11,
    4.10e-11, 2.30e-11, 1.38e-11, 8.33e-12, 5.24e-12, 3.29e-12,
    1.39e-12, 6.15e-13, 2.84e-13, 1.37e-13, 6.87e-14, 3.63e-14,
    2.02e-14, 1.21e-14, 7.69e-15, 5.24e-15, 3.78e-15, 2.86e-15,
], dtype=float)


class SMADAtmosphere(Atmosphere):
    """Tabulated SMAD density profile, evaluated from spacecraft position.

    The tabulated reference profile follows Wertz and Larson (eds.),
    *Space Mission Analysis and Design*; see the project citation guide for
    bibliographic details. Intermediate values are log-linearly interpolated.
    """

    def __init__(self, altitude_range=None, rho_range=None):
        self.altitude_range = np.array(
            SMAD_ALTITUDE_KM if altitude_range is None else altitude_range, dtype=float
        )
        self.rho_range = np.array(
            SMAD_DENSITY_KG_M3 if rho_range is None else rho_range, dtype=float
        )
        if (self.altitude_range.ndim != 1 or self.rho_range.shape != self.altitude_range.shape
                or self.altitude_range.size < 2 or np.any(np.diff(self.altitude_range) <= 0)
                or np.any(self.altitude_range < 0) or np.any(self.rho_range <= 0)
                or not np.all(np.isfinite(self.altitude_range)) or not np.all(np.isfinite(self.rho_range))):
            raise ValueError("altitude and density samples must be finite, positive, and strictly ordered")
        self._log_rho = np.log(self.rho_range)

    def interpolate(self, altitude_km):
        """Interpolate density at one or more altitudes [km]."""
        altitude = np.asarray(altitude_km, dtype=float)
        density = np.exp(np.interp(altitude, self.altitude_range, self._log_rho))
        above = altitude >= self.altitude_range[-1]
        if np.any(above):
            scale_height = (self.altitude_range[-1] - self.altitude_range[-2]) / (
                self._log_rho[-2] - self._log_rho[-1]
            )
            density = np.where(
                above,
                self.rho_range[-1] * np.exp(-(altitude - self.altitude_range[-1]) / scale_height),
                density,
            )
        return float(density) if density.ndim == 0 else density

    def density(self, position_eci_km, datetime_utc=None, geodetic_lla=None):
        """Evaluate density at ECI positions; date and geodetic data are unused."""
        del datetime_utc, geodetic_lla
        position = np.asarray(position_eci_km, dtype=float)
        if position.ndim not in (1, 2) or position.shape[-1] != 3:
            raise ValueError("position_eci_km must have shape (3,) or (N, 3)")
        return self.interpolate(np.linalg.norm(position, axis=-1) - EarthConstants.R_e)

    def __repr__(self):
        return f"SMADAtmosphere(n={self.altitude_range.size})"
