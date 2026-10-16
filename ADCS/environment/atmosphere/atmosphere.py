"""Parent interface for atmospheric models."""

from abc import ABC, abstractmethod


class Atmosphere(ABC):
    """Provide atmospheric density at a spacecraft position in km."""

    @abstractmethod
    def density(self, position_eci_km, datetime_utc=None, geodetic_lla=None):
        r"""Evaluate atmospheric mass density at ECI positions.

        :param position_eci_km: One ECI position or ``(N, 3)`` positions [km].
        :type position_eci_km: numpy.ndarray
        :param datetime_utc: One UTC date or one date per position.
        :type datetime_utc: datetime.datetime or sequence or None
        :param geodetic_lla: Optional latitude, longitude [deg] and altitude [km].
        :type geodetic_lla: numpy.ndarray or None
        :return: Atmospheric mass density [kg/m³].
        :rtype: float or numpy.ndarray
        """
        raise NotImplementedError

    @abstractmethod
    def interpolate(self, altitude_km):
        r"""Evaluate a one-dimensional altitude density profile.

        :param altitude_km: Altitude above Earth's mean surface [km].
        :type altitude_km: float or numpy.ndarray
        :return: Atmospheric mass density [kg/m³].
        :rtype: float or numpy.ndarray
        """
        raise NotImplementedError
