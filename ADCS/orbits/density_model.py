"""Deprecated compatibility path for the default atmospheric density model."""

import warnings

warnings.warn(
    "ADCS.orbits.density_model is deprecated; import SMADAtmosphere from ADCS.environment.atmosphere instead.",
    DeprecationWarning,
    stacklevel=2,
)

from ADCS.environment.atmosphere import SMADAtmosphere as DensityModel

__all__ = ["DensityModel"]
