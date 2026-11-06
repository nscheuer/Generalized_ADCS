"""Parent interface for ephemeris models."""

from abc import ABC, abstractmethod

import numpy as np


def sunlit_from_vectors(position_eci_km, sun_eci_km):
    """Test whether any part of the solar disk clears Earth's limb."""
    from ADCS.orbits.universal_constants import EarthConstants

    position, sun = np.broadcast_arrays(
        np.asarray(position_eci_km, dtype=float), np.asarray(sun_eci_km, dtype=float)
    )
    if position.shape[-1] != 3:
        raise ValueError("position and Sun vectors must end in three coordinates")
    to_sun = sun - position
    r = np.linalg.norm(position, axis=-1)
    d = np.linalg.norm(to_sun, axis=-1)
    cos_separation = -np.sum(position * to_sun, axis=-1) / (r * d)
    separation = np.arccos(np.clip(cos_separation, -1.0, 1.0))
    earth_radius = np.arcsin(np.clip(EarthConstants.R_e / r, 0.0, 1.0))
    sun_radius = np.arcsin(np.clip(695700.0 / d, 0.0, 1.0))
    return separation > earth_radius - sun_radius


class Ephemeris(ABC):
    """Base type for ephemeris models consumed by orbital states and sensors."""

    @property
    @abstractmethod
    def available_bodies(self) -> tuple[str, ...]:
        r"""List bodies exposed by the selected ephemeris.

        :return: Body names and supported aliases.
        :rtype: tuple[str, ...]
        """
        raise NotImplementedError

    @abstractmethod
    def position_eci(self, target, j2000, *, observer="earth", apparent=True):
        r"""Find the target relative to an observer in the inertial frame.

        :param target: Body name or identifier to locate.
        :type target: str or int
        :param j2000: Epoch or epochs in Julian centuries since J2000.
        :type j2000: float or numpy.ndarray
        :param observer: Body name or identifier used as the origin.
        :type observer: str or int
        :param apparent: Apply light time and aberration when supported.
        :type apparent: bool
        :return: Relative ECI position vector or vectors [km].
        :rtype: numpy.ndarray
        """
        raise NotImplementedError

    @abstractmethod
    def sun_position_eci(self, j2000):
        r"""Find the Earth to Sun vector in the inertial frame.

        :param j2000: Epoch or epochs in Julian centuries since J2000.
        :type j2000: float or numpy.ndarray
        :return: Earth to Sun ECI position vector or vectors [km].
        :rtype: numpy.ndarray
        """
        raise NotImplementedError

