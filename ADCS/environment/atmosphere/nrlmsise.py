"""NRLMSISE-00 neutral-atmosphere density model via the optional PyMSIS package."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from .atmosphere import Atmosphere


class NRLMSISE00(Atmosphere):
    """Evaluate NRLMSISE-00 mass density at ECI positions.

    Install the optional backend with ``pip install Generalized_ADCS[nrlmsise]``.
    UTC dates and geodetic latitude, longitude, and altitude are normally
    supplied by ``Orbital_State`` and ``Orbit``. If solar indices are omitted,
    PyMSIS retrieves and caches historical F10.7 and Ap data when needed.

    ``interpolate`` is provided for altitude-profile plotting and evaluates at
    the configured reference date and location (defaults: J2000-01-01, 0°, 0°).
    """

    def __init__(
        self,
        *,
        version: float = 0,
        f107: float | None = None,
        f107a: float | None = None,
        ap: float | None = None,
        reference_datetime: datetime = datetime(2000, 1, 1, 12),
        reference_latitude_deg: float = 0.0,
        reference_longitude_deg: float = 0.0,
    ) -> None:
        r"""Configure NRLMSISE version, solar inputs, and profile location.

        :param version: MSIS version (0 for NRLMSISE-00, 2.0, or 2.1).
        :type version: float
        :param f107: Optional daily 10.7 cm solar flux.
        :type f107: float or None
        :param f107a: Optional 81-day mean solar flux.
        :type f107a: float or None
        :param ap: Optional geomagnetic activity index.
        :type ap: float or None
        :param reference_datetime: UTC date for altitude profile queries.
        :type reference_datetime: datetime.datetime
        :param reference_latitude_deg: Profile geodetic latitude [deg].
        :type reference_latitude_deg: float
        :param reference_longitude_deg: Profile east longitude [deg].
        :type reference_longitude_deg: float
        :return: None
        :rtype: None
        :raises ValueError: If the selected MSIS version is unsupported.
        """
        if version not in (0, 2.0, 2.1):
            raise ValueError("NRLMSISE version must be 0 (NRLMSISE-00), 2.0, or 2.1")
        self.version = version
        self.f107 = f107
        self.f107a = f107a
        self.ap = ap
        self.reference_datetime = reference_datetime
        self.reference_latitude_deg = float(reference_latitude_deg)
        self.reference_longitude_deg = float(reference_longitude_deg)

    @staticmethod
    def _backend():
        r"""Import the optional PyMSIS backend when needed.

        :return: Imported ``pymsis`` module.
        :rtype: module
        :raises ImportError: If PyMSIS is not installed.
        """
        try:
            import pymsis
        except ImportError as exc:
            raise ImportError(
                "NRLMSISE00 requires the optional PyMSIS backend; install it with "
                "`pip install Generalized_ADCS[nrlmsise]`."
            ) from exc
        return pymsis

    @staticmethod
    def _date_array(datetime_utc, count: int) -> np.ndarray:
        r"""Align UTC dates with a batch of position samples.

        :param datetime_utc: One UTC date or one date per position.
        :type datetime_utc: datetime.datetime or sequence
        :param count: Number of positions.
        :type count: int
        :return: UTC ``datetime64`` array with ``count`` entries.
        :rtype: numpy.ndarray
        :raises ValueError: If dates are missing or counts do not match.
        """
        if datetime_utc is None:
            raise ValueError("datetime_utc is required for NRLMSISE density evaluation")
        dates = [datetime_utc] if isinstance(datetime_utc, datetime) else list(datetime_utc)
        if len(dates) == 1 and count > 1:
            dates *= count
        if len(dates) != count:
            raise ValueError(f"datetime_utc must contain one date or {count} dates")
        normalized = []
        for date in dates:
            if not isinstance(date, datetime):
                date = np.datetime64(date, "us").astype(datetime)
            if date.tzinfo is not None:
                date = date.astimezone(timezone.utc).replace(tzinfo=None)
            normalized.append(np.datetime64(date, "us"))
        return np.asarray(normalized)

    @staticmethod
    def _to_geodetic(position, dates) -> np.ndarray:
        r"""Convert aligned ECI positions to WGS84 geodetic coordinates.

        :param position: ECI Cartesian positions ``(N, 3)`` [km].
        :type position: numpy.ndarray
        :param dates: Matching UTC ``datetime64`` entries.
        :type dates: numpy.ndarray
        :return: Latitude, east longitude [deg], and altitude [km].
        :rtype: numpy.ndarray
        """
        from skyfield.api import load, wgs84
        from skyfield.positionlib import ICRF
        from skyfield.units import Distance

        ts = load.timescale()
        datetimes = [
            date.astype("datetime64[us]").astype(datetime).replace(tzinfo=timezone.utc)
            for date in dates
        ]
        times = ts.from_datetimes(datetimes)
        distance = Distance(km=position.T)
        inertial = ICRF(distance.au, t=times, center=399, target=0)
        geo = wgs84.geographic_position_of(inertial)
        return np.column_stack((geo.latitude.degrees, geo.longitude.degrees, geo.elevation.km))

    def _calculate(self, lla: np.ndarray, dates: np.ndarray) -> np.ndarray:
        r"""Evaluate neutral mass density through PyMSIS.

        :param lla: Geodetic latitude, longitude [deg], and altitude [km].
        :type lla: numpy.ndarray
        :param dates: Matching UTC dates.
        :type dates: numpy.ndarray
        :return: Atmospheric mass density per sample [kg/m³].
        :rtype: numpy.ndarray
        """
        pymsis = self._backend()
        kwargs = {"version": self.version}
        if self.f107 is not None:
            kwargs["f107s"] = self.f107
        if self.f107a is not None:
            kwargs["f107as"] = self.f107a
        if self.ap is not None:
            kwargs["aps"] = self.ap
        result = pymsis.calculate(
            dates,
            lla[:, 1],
            lla[:, 0],
            lla[:, 2],
            **kwargs,
        )
        return np.asarray(result, dtype=float).reshape(-1, 11)[:, 0]

    def density(self, position_eci_km, datetime_utc=None, geodetic_lla=None):
        r"""Evaluate NRLMSISE mass density at one or many ECI positions.

        :param position_eci_km: ECI position ``(3,)`` or positions ``(N, 3)`` [km].
        :type position_eci_km: numpy.ndarray
        :param datetime_utc: One UTC date or one per position.
        :type datetime_utc: datetime.datetime or sequence
        :param geodetic_lla: Optional latitude, longitude [deg], altitude [km].
        :type geodetic_lla: numpy.ndarray or None
        :return: Atmospheric mass density [kg/m³].
        :rtype: float or numpy.ndarray
        :raises ValueError: If positions, dates, or geodetic data do not align.
        """
        position = np.asarray(position_eci_km, dtype=float)
        scalar = position.ndim == 1
        if scalar:
            position = position.reshape(1, 3)
        if position.ndim != 2 or position.shape[1] != 3:
            raise ValueError("position_eci_km must have shape (3,) or (N, 3)")
        dates = self._date_array(datetime_utc, len(position))
        if geodetic_lla is None:
            lla = self._to_geodetic(position, dates)
        else:
            lla = np.asarray(geodetic_lla, dtype=float)
            if lla.ndim == 1:
                lla = lla.reshape(1, 3)
            if lla.shape != (len(position), 3):
                raise ValueError("geodetic_lla must have shape (3,) or (N, 3), in degrees, degrees, km")
        density = self._calculate(lla, dates)
        return float(density[0]) if scalar else density

    def interpolate(self, altitude_km):
        r"""Evaluate the reference-date density profile by altitude.

        :param altitude_km: One altitude or an array of altitudes [km].
        :type altitude_km: float or numpy.ndarray
        :return: Atmospheric mass density [kg/m³].
        :rtype: float or numpy.ndarray
        """
        altitude = np.asarray(altitude_km, dtype=float)
        scalar = altitude.ndim == 0
        lla = np.column_stack((
            np.full(altitude.size or 1, self.reference_latitude_deg),
            np.full(altitude.size or 1, self.reference_longitude_deg),
            altitude.reshape(-1) if not scalar else [float(altitude)],
        ))
        dates = self._date_array(self.reference_datetime, len(lla))
        density = self._calculate(lla, dates)
        return float(density[0]) if scalar else density.reshape(altitude.shape)

    def __repr__(self) -> str:
        r"""Identify the selected PyMSIS model version.

        :return: Short model representation.
        :rtype: str
        """
        return f"NRLMSISE00(version={self.version})"


NRLMSISE = NRLMSISE00

