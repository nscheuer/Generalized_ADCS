"""SMAD tabulated atmospheric density model."""

import numpy as np

from ADCS.orbits.universal_constants import EarthConstants

from .atmosphere import Atmosphere
from .density_model import DensityModel


class SMADAtmosphere(Atmosphere):
    """Use the existing SMAD density table and interpolation implementation.

    ``density(position_eci_km)`` accepts a single ECI position or an ``(N, 3)``
    array and returns density in kg/m³. The model depends only on distance
    from Earth's center, so the same value is obtained for an ECEF position.
    """

    def __init__(self, altitude_range=None, rho_range=None) -> None:
        r"""Set the tabulated altitude and density samples.

        :param altitude_range: Optional altitude samples [km].
        :type altitude_range: numpy.ndarray or None
        :param rho_range: Optional density samples [kg/m³].
        :type rho_range: numpy.ndarray or None
        :return: None
        :rtype: None
        """
        kwargs = {}
        if altitude_range is not None:
            kwargs["altitude_range"] = altitude_range
        if rho_range is not None:
            kwargs["rho_range"] = rho_range
        self._model = DensityModel(**kwargs)

    @property
    def altitude_range(self):
        r"""Expose the model's tabulated altitudes.

        :return: Altitude samples [km].
        :rtype: numpy.ndarray
        """
        return self._model.altitude_range

    @property
    def rho_range(self):
        r"""Expose the model's tabulated mass densities.

        :return: Density samples [kg/m³].
        :rtype: numpy.ndarray
        """
        return self._model.rho_range

    def interpolate(self, altitude_km):
        r"""Interpolate the SMAD density profile by altitude.

        :param altitude_km: Altitude above Earth's mean radius [km].
        :type altitude_km: float
        :return: Atmospheric mass density [kg/m³].
        :rtype: float
        """
        return self._model.interpolate(altitude_km)

    def density(self, position_eci_km, datetime_utc=None, geodetic_lla=None):
        r"""Evaluate SMAD density from the radius of ECI positions.

        Date and geodetic inputs are accepted for interface compatibility; this
        altitude-only model does not use them.

        :param position_eci_km: One ECI position or ``(N, 3)`` positions [km].
        :type position_eci_km: numpy.ndarray
        :param datetime_utc: Unused UTC date input.
        :type datetime_utc: object or None
        :param geodetic_lla: Unused geodetic coordinate input.
        :type geodetic_lla: numpy.ndarray or None
        :return: Atmospheric mass density [kg/m³].
        :rtype: float or numpy.ndarray
        :raises ValueError: If position shape is invalid.
        """
        del datetime_utc, geodetic_lla
        position = np.asarray(position_eci_km, dtype=float)
        if position.ndim not in (1, 2) or position.shape[-1] != 3:
            raise ValueError("position_eci_km must have shape (3,) or (N, 3)")
        altitude = np.linalg.norm(position, axis=-1) - EarthConstants.R_e
        if np.ndim(altitude) == 0:
            return self.interpolate(float(altitude))
        return self._model.interpolate_many(altitude)

    def __repr__(self) -> str:
        r"""Summarize the number of tabulated altitude samples.

        :return: Short model representation.
        :rtype: str
        """
        return f"SMADAtmosphere(n={len(self._model.altitude_range)})"
