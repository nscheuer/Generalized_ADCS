"""Atmospheric model classes."""

from .atmosphere import Atmosphere
from .smad import SMADAtmosphere

AtmosphericModel = SMADAtmosphere
Atmosphere.SMADAtmosphere = SMADAtmosphere

__all__ = ["Atmosphere", "SMADAtmosphere", "AtmosphericModel"]
