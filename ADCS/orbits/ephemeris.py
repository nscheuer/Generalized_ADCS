"""Deprecated compatibility path for the default ephemeris."""

import warnings

warnings.warn(
    "ADCS.orbits.ephemeris is deprecated; import Ephemeris from ADCS.environment.ephemeris instead.",
    DeprecationWarning,
    stacklevel=2,
)

from ADCS.environment.ephemeris import DE421 as Ephemeris

__all__ = ["Ephemeris"]
