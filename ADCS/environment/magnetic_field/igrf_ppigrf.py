r"""Optional ppigrf backend for IGRF-14.

The adapter pairs each date with its coordinate rather than exposing the
outer-product shape returned by ppigrf itself.
"""

from __future__ import annotations

from datetime import datetime
import numpy as np

from .igrf14 import IGRF14

#: Samples per ppigrf call when dates vary per sample. ppigrf forms the outer
#: product of dates and coordinates, so a chunk of m samples costs m^2 field
#: evaluations; this bounds that cost without changing any value.
_PPIGRF_CHUNK = 256


def _import_ppigrf():
    r"""Import the optional ppigrf reference backend on demand.

    :return: Imported ``ppigrf`` module.
    :rtype: module
    :raises ImportError: If the optional package is unavailable.
    """
    try:
        import ppigrf
    except ImportError as exc:
        raise ImportError(
            "magnetic_model='igrf_ppigrf' needs the optional ppigrf package: "
            "pip install \"Generalized_ADCS[ppigrf]\""
        ) from exc
    return ppigrf


def _ppigrf_gc(r, theta, phi, date):
    r"""
    ``ppigrf.igrf_gc`` with dates paired elementwise with the coordinates.

    Same contract as :func:`ADCS.environment.magnetic_field.igrf14_numba.igrf_gc`. ppigrf returns an
    ``(n_dates, N)`` grid; for one date that is a single row, and for one date
    per sample only the diagonal is wanted, which is taken chunk by chunk so
    the discarded off-diagonal work stays bounded.

    :param r: Geocentric radius [km].
    :type r: float or numpy.ndarray
    :param theta: Geocentric colatitude [deg].
    :type theta: float or numpy.ndarray
    :param phi: East longitude [deg].
    :type phi: float or numpy.ndarray
    :param date: One UTC date or one date per position.
    :type date: datetime.datetime or sequence
    :return: Radial, southward, and eastward components [nT].
    :rtype: tuple
    :raises ValueError: If date count differs from position count.
    """
    ppigrf = _import_ppigrf()

    scalar = all(np.ndim(x) == 0 for x in (r, theta, phi))
    r_a, th_a, ph_a = np.broadcast_arrays(
        np.asarray(r, dtype=float), np.asarray(theta, dtype=float), np.asarray(phi, dtype=float)
    )
    shape = r_a.shape
    r_f, th_f, ph_f = (a.reshape(-1) for a in (r_a, th_a, ph_a))
    N = r_f.size

    single_date = isinstance(date, datetime) or np.ndim(date) == 0
    dates = None if single_date else list(date)
    if dates is not None and len(dates) == 1:
        single_date, date = True, dates[0]
    if dates is not None and len(dates) != N:
        raise ValueError(
            f"date has {len(dates)} entries; expected 1 or {N} to match the coordinates"
        )

    if single_date:
        out = ppigrf.igrf_gc(r_f, th_f, ph_f, date)
        comps = [np.asarray(c, dtype=float).reshape(N) for c in out]
    else:
        comps = [np.empty(N) for _ in range(3)]
        for i0 in range(0, N, _PPIGRF_CHUNK):
            i1 = min(i0 + _PPIGRF_CHUNK, N)
            out = ppigrf.igrf_gc(r_f[i0:i1], th_f[i0:i1], ph_f[i0:i1], dates[i0:i1])
            for dst, c in zip(comps, out):
                dst[i0:i1] = np.diagonal(np.asarray(c, dtype=float).reshape(i1 - i0, i1 - i0))

    if scalar:
        return tuple(float(c[0]) for c in comps)
    return tuple(c.reshape(shape) for c in comps)



class IGRF14Ppigrf(IGRF14):
    r"""IGRF-14 evaluated by the optional ppigrf compatibility backend."""

    def __init__(self) -> None:
        r"""Select the ppigrf IGRF-14 implementation.

        :return: None
        :rtype: None
        """
        super().__init__("igrf_ppigrf")
