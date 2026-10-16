"""Parent interface for magnetic field models."""

from abc import ABC, abstractmethod


class MagneticField(ABC):
    """Evaluate Earth's magnetic field from an ECEF position and date."""

    @abstractmethod
    def field(self, position_ecef_km, date):
        """Return an ECEF magnetic field vector in tesla."""
        raise NotImplementedError

    @abstractmethod
    def field_gc(self, r, theta, phi, date):
        """Return ``(Br, Btheta, Bphi)`` in nT for radius km and angles deg."""
        raise NotImplementedError
