"""Parent interface for atmospheric models."""

from abc import ABC, abstractmethod


class Atmosphere(ABC):
    """Provide atmospheric density at a spacecraft position in km."""

    @abstractmethod
    def density(self, position_eci_km, datetime_utc=None, geodetic_lla=None):
        """Return density in kg/m³ at ECI positions and optional UTC/LLA data."""
        raise NotImplementedError

    @abstractmethod
    def interpolate(self, altitude_km):
        """Return atmospheric density at ``altitude_km``."""
        raise NotImplementedError
