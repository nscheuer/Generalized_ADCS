"""Ephemeris model classes."""

from .ephemeris import Ephemeris
from .de421 import DE421
from .de442 import DE442
from .de442s import DE442s, DE442S
from .de441 import DE441
EphemerisModel = DE421
Ephemeris.DE421 = DE421
Ephemeris.DE442 = DE442
Ephemeris.DE442s = DE442s
Ephemeris.DE442S = DE442S
Ephemeris.DE441 = DE441
__all__ = ["Ephemeris", "DE421", "DE442", "DE442s", "DE442S", "DE441", "EphemerisModel"]
