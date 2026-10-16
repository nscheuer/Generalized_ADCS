"""Parent interface for ephemeris models."""

from abc import ABC, abstractmethod


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

