"""Parent interface for ephemeris models."""

from abc import ABC, abstractmethod


class Ephemeris(ABC):
    """Base type for ephemeris models consumed by orbital states and sensors."""

    @property
    @abstractmethod
    def available_bodies(self) -> tuple[str, ...]:
        """Names and aliases for bodies available in the loaded kernel."""
        raise NotImplementedError

    @abstractmethod
    def position_eci(self, target, j2000, *, observer="earth", apparent=True):
        """Return target position relative to observer [km]."""
        raise NotImplementedError

    @abstractmethod
    def sun_position_eci(self, j2000):
        """Return Earth-to-Sun position in ECI coordinates [km]."""
        raise NotImplementedError

