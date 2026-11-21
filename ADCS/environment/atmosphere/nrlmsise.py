"""NRLMSISE-00 neutral-atmosphere density model via the optional PyMSIS package."""

from __future__ import annotations

import re
import warnings
from datetime import datetime, timezone

import numpy as np

from .atmosphere import Atmosphere

# PyMSIS reports the span of its space-weather record only in the error it
# raises for a date outside it ("Dates should be between <start> and <end>.").
_RECORD_SPAN_RE = re.compile(r"between (\S+) and (\S+?)\.?\s*$")
# (first date, first date past the end) of that record, once a request has
# fallen outside it. Later requests then never hand PyMSIS an out-of-range
# date, which would make it re-download the record each time.
_record_span: tuple[np.datetime64, np.datetime64] | None = None


class NRLMSISE00(Atmosphere):
    """Evaluate NRLMSISE-00 mass density at ECI positions.

    Install the optional backend with ``pip install Generalized_ADCS[nrlmsise]``.
    UTC dates and geodetic latitude, longitude, and altitude are normally
    supplied by ``Orbital_State`` and ``Orbit``. If solar indices are omitted,
    PyMSIS retrieves and caches historical F10.7 and Ap data when needed.
    That record (CelesTrak ``SW-All.csv``) starts in October 1957 and ends
    with daily predictions a few weeks ahead; dates outside it use
    :attr:`FALLBACK_F107`, :attr:`FALLBACK_F107A`, and :attr:`FALLBACK_AP`
    with a warning. Pin ``f107``, ``f107a``, and ``ap`` for future epochs.

    ``interpolate`` is provided for altitude-profile plotting and evaluates at
    the configured reference date and location (defaults: J2000-01-01, 0°, 0°).

    Source: U.S. Naval Research Laboratory NRLMSISE-00 model; Picone et al.
    (2002), doi:10.1029/2002JA009430. See the project citation guide.
    """

    #: F10.7 [sfu] used for dates outside PyMSIS's space-weather record when
    #: ``f107`` is not pinned. With :attr:`FALLBACK_F107A` and
    #: :attr:`FALLBACK_AP` this is the NRLMSISE-00 reference test case:
    #: moderate solar and quiet geomagnetic activity.
    FALLBACK_F107: float = 150.0
    #: 81-day mean F10.7 [sfu] used outside the record when ``f107a`` is not pinned.
    FALLBACK_F107A: float = 150.0
    #: Daily Ap used outside the record when ``ap`` is not pinned.
    FALLBACK_AP: float = 4.0

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
        f107s, f107as, aps = self._indices(pymsis, dates)
        result = pymsis.calculate(
            dates,
            lla[:, 1],
            lla[:, 0],
            lla[:, 2],
            f107s=f107s,
            f107as=f107as,
            aps=aps,
            version=self.version,
        )
        return np.asarray(result, dtype=float).reshape(-1, 11)[:, 0]

    def _indices(self, pymsis, dates: np.ndarray):
        r"""Solar and geomagnetic indices for each date.

        Pinned values are used as given. The rest come from PyMSIS's
        space-weather record, or from the ``FALLBACK_*`` values (with a
        warning) for dates the record does not cover.

        :param pymsis: Imported PyMSIS backend.
        :type pymsis: module
        :param dates: UTC ``datetime64`` dates.
        :type dates: numpy.ndarray
        :return: F10.7 ``(n,)``, 81-day F10.7 ``(n,)``, and ap ``(n, 7)``:
            daily Ap, then the 3-hour history that only
            ``geomagnetic_activity=-1`` reads.
        :rtype: tuple
        """
        n = len(dates)
        f107 = None if self.f107 is None else np.full(n, float(self.f107))
        f107a = None if self.f107a is None else np.full(n, float(self.f107a))
        ap = None if self.ap is None else np.full((n, 7), float(self.ap))
        if f107 is None or f107a is None or ap is None:
            recorded = self._recorded_indices(pymsis, dates)
            f107 = recorded[0] if f107 is None else f107
            f107a = recorded[1] if f107a is None else f107a
            ap = recorded[2] if ap is None else ap
        return f107, f107a, ap

    def _recorded_indices(self, pymsis, dates: np.ndarray):
        r"""Look up indices in PyMSIS's record, falling back outside it.

        :param pymsis: Imported PyMSIS backend.
        :type pymsis: module
        :param dates: UTC ``datetime64`` dates.
        :type dates: numpy.ndarray
        :return: F10.7 ``(n,)``, 81-day F10.7 ``(n,)``, and ap ``(n, 7)``.
        :rtype: tuple
        :raises ValueError: If PyMSIS rejects the dates for a reason other
            than the record's span.
        """
        global _record_span
        get_f107_ap = pymsis.utils.get_f107_ap
        if _record_span is None:
            try:
                found = get_f107_ap(dates)
            except ValueError as exc:
                match = _RECORD_SPAN_RE.search(str(exc))
                if match is None:
                    raise
                # Each record entry covers the three hours from its timestamp.
                _record_span = (
                    np.datetime64(match.group(1)),
                    np.datetime64(match.group(2)) + np.timedelta64(3, "h"),
                )
            else:
                return (
                    np.asarray(found[0], dtype=float).reshape(-1),
                    np.asarray(found[1], dtype=float).reshape(-1),
                    np.asarray(found[2], dtype=float).reshape(-1, 7),
                )

        start, end = _record_span
        inside = (dates >= start) & (dates < end)
        n = len(dates)
        f107 = np.full(n, float(self.FALLBACK_F107))
        f107a = np.full(n, float(self.FALLBACK_F107A))
        ap = np.full((n, 7), float(self.FALLBACK_AP))
        if np.any(inside):
            found = get_f107_ap(dates[inside])
            f107[inside] = np.asarray(found[0], dtype=float).reshape(-1)
            f107a[inside] = np.asarray(found[1], dtype=float).reshape(-1)
            ap[inside] = np.asarray(found[2], dtype=float).reshape(-1, 7)
        if not np.all(inside):
            warnings.warn(
                f"NRLMSISE00: PyMSIS's space-weather record covers {start} to {end}; "
                f"dates outside it use F10.7 = {self.FALLBACK_F107:g}, "
                f"F10.7a = {self.FALLBACK_F107A:g} sfu and Ap = {self.FALLBACK_AP:g}. "
                "Pin f107, f107a and ap to choose these values.",
                stacklevel=2,
            )
        return f107, f107a, ap

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

