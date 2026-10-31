"""NOAA World Magnetic Model 2025 (WMM2025)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from ADCS.helpers.math.spherical_harmonics import _kernel_varying, _schmidt_table

from ADCS.environment.ephemeris._download import kernel_path as default_kernel_path

from ._coefficients import coordinate_dates, ensure_kernel, evaluate_static, shaped_field
from .magnetic_field import MagneticField


WMM2025_URL = "https://www.ngdc.noaa.gov/geomag/calculators/mobileStatic/WMM.COF"


@lru_cache(maxsize=4)
def _load_cof(path: str):
    r"""Parse WMM2025 Gauss coefficients and annual secular variation.

    Schmidt factors are folded into coefficients once, before Numba evaluation.

    :param path: Local path to the official ``WMM.COF`` file.
    :type path: str
    :return: Epoch and normalized main and variation coefficient arrays.
    :rtype: tuple
    :raises ValueError: If the file is incomplete or not WMM2025.
    """
    with open(path, encoding="ascii") as stream:
        header = stream.readline().split()
        if len(header) < 2 or header[1] != "WMM-2025":
            raise ValueError(f"{path} is not a WMM-2025 coefficient file")
        epoch = float(header[0])
        rows = []
        for line in stream:
            parts = line.split()
            if not parts or parts[0].startswith("999"):
                break
            if len(parts) < 6:
                raise ValueError(f"Malformed WMM coefficient row in {path}: {line!r}")
            n, m = int(parts[0]), int(parts[1])
            rows.append((n, m, *(float(v) for v in parts[2:6])))
    if epoch != 2025.0 or len(rows) != 90 or max(n for n, *_ in rows) != 12:
        raise ValueError(f"Incomplete WMM-2025 coefficient file: {path}")
    g0 = np.zeros((13, 13)); h0 = np.zeros_like(g0)
    gd = np.zeros_like(g0); hd = np.zeros_like(g0)
    seen = set()
    for n, m, gv, hv, dgv, dhv in rows:
        if not 1 <= n <= 12 or not 0 <= m <= n or (n, m) in seen:
            raise ValueError(f"Invalid WMM degree/order in {path}: {(n, m)}")
        seen.add((n, m))
        g0[n, m], h0[n, m] = gv, hv
        gd[n, m], hd[n, m] = dgv, dhv
    s = _schmidt_table(12)
    return epoch, g0 * s, h0 * s, gd * s, hd * s


def _decimal_years(dates):
    r"""Convert UTC dates to calendar year fractions for WMM variation.

    :param dates: UTC dates as a NumPy datetime array.
    :type dates: numpy.ndarray
    :return: Decimal calendar years.
    :rtype: numpy.ndarray
    """
    years = dates.astype("datetime64[Y]")
    next_years = years + np.timedelta64(1, "Y")
    return years.astype(int) + 1970 + (
        (dates - years).astype("timedelta64[ns]").astype(np.int64)
        / (next_years - years).astype("timedelta64[ns]").astype(np.int64)
    )


class WMM2025(MagneticField):
    """WMM2025 internal field, valid from 2025-01-01 through 2029-12-31.

    ``field_gc`` returns geocentric spherical components in nT; ``field``
    returns ECEF vectors in tesla. The official WMM COF is downloaded on first
    evaluation if missing from ``environment/kernels``.
    """

    name = "wmm2025"

    def __init__(self, kernel_path: str | Path | None = None):
        r"""Choose the local WMM2025 coefficient file.

        :param kernel_path: Optional local ``.COF`` path; missing files download.
        :type kernel_path: str or pathlib.Path or None
        :return: None
        :rtype: None
        """
        self.kernel_path = Path(kernel_path) if kernel_path is not None else default_kernel_path("WMM2025.COF")

    def field_gc(self, r, theta, phi, date):
        r"""Evaluate WMM2025 in geocentric spherical coordinates.

        :param r: Geocentric radius [km].
        :type r: float or numpy.ndarray
        :param theta: Geocentric colatitude [deg].
        :type theta: float or numpy.ndarray
        :param phi: East longitude [deg].
        :type phi: float or numpy.ndarray
        :param date: One UTC date or one date per position.
        :type date: datetime.datetime or sequence
        :return: Radial, southward, and eastward field components [nT].
        :rtype: tuple
        :raises ValueError: If a date is outside 2025 through 2029.
        """
        shape, r, theta, phi, dates = coordinate_dates(r, theta, phi, date)
        years = _decimal_years(dates)
        if np.any((years < 2025.0) | (years >= 2030.0)):
            raise ValueError("WMM2025 is valid from 2025-01-01 through 2029-12-31")
        path = ensure_kernel(self.kernel_path, WMM2025_URL)
        epoch, g0, h0, gd, hd = _load_cof(str(path))
        if dates.size == 1 or np.all(years == years[0]):
            delta = float(years[0] - epoch)
            return shaped_field(shape, evaluate_static(r, theta, phi, g0 + delta * gd, h0 + delta * hd))

        g = np.stack((g0, g0 + gd))
        h = np.stack((h0, h0 + hd))
        br, bt, bp = (np.empty(r.size) for _ in range(3))
        _kernel_varying(
            r, theta, phi, g, h, np.zeros(r.size, dtype=np.int64),
            np.ascontiguousarray(years - epoch), 12, br, bt, bp, 512,
        )
        return shaped_field(shape, (br, bt, bp))


WMM = WMM2025
