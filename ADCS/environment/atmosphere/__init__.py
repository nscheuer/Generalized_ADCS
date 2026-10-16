"""Atmospheric model classes."""

from .atmosphere import Atmosphere
from .smad import SMADAtmosphere
from .nrlmsise import NRLMSISE00, NRLMSISE

AtmosphericModel = SMADAtmosphere
Atmosphere.SMADAtmosphere = SMADAtmosphere
Atmosphere.NRLMSISE00 = NRLMSISE00
Atmosphere.NRLMSISE = NRLMSISE

__all__ = ["Atmosphere", "SMADAtmosphere", "NRLMSISE00", "NRLMSISE", "AtmosphericModel"]
