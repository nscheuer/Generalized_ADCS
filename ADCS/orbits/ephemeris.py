"""Deprecated compatibility path for the default ephemeris."""

import warnings

warnings.warn(
    "ADCS.orbits.ephemeris is deprecated; import DE421 from ADCS.environment.ephemeris instead "
    "(the Ephemeris name there is the abstract base class).",
    DeprecationWarning,
    stacklevel=2,
)

from ADCS.environment.ephemeris import DE421 as Ephemeris

__all__ = ["Ephemeris"]
