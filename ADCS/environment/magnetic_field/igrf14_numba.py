r"""
IGRF geomagnetic field model.

This is the default ``"igrf_numba"`` model of
:mod:`ADCS.environment.magnetic_field`; ``ppigrf`` remains selectable as
``"igrf_ppigrf"``. It evaluates the same model from the same IAGA coefficients
and agrees with ``ppigrf`` to ~1e-12 relative, but it is roughly 50-70x faster
in batch and ~3000x faster per call (``debug/debug_orbit/benchmark_igrf.py``),
which matters because :meth:`ADCS.Orbital_State.get_b_eci` is on the
simulation's inner loop.

Why it was worth writing rather than tuning ``ppigrf``
------------------------------------------------------

``ppigrf`` is correct but its cost structure is wrong for this use:

* ``igrf_gc`` calls ``read_shc`` on *every* invocation -- reopening the ``.shc``
  file and rebuilding two pandas DataFrames. There is no caching anywhere in
  the library. Evaluating one point therefore cost ~31 ms, essentially all of
  it re-reading a file that never changes.
* Dates are deliberately kept out of broadcasting, so N coordinates and N dates
  produce an N x N result. Passing a trajectory produced N^2 field evaluations
  of which only the diagonal was kept.
* The Legendre routine allocates (nmax+1)^2 arrays through a dict, then
  ``hstack``es 104 columns, and the field sum is built from several
  (N, 208) temporaries including a ``**`` over the whole array.

Here the coefficient file is parsed once and cached, the Legendre recursion
keeps only the two previous degrees (so the per-sample scratch is 6 rows of 14
doubles rather than a 14x14 table), and the whole evaluation is a single numba
serial kernel. numba is already a hard dependency of this
package, so this adds no new install burden -- unlike ESA's eoxmagmod, which is
not on PyPI and needs the NASA CDF library plus a Fortran build.

Coefficients
------------

``IGRF14.shc`` is the 14th-generation IGRF from IAGA Working Group V-MOD
(https://doi.org/10.5281/zenodo.14012302), in the ``.shc`` layout used by
``ppigrf`` (MIT, Copyright (c) 2021 Karl M. Laundal). Valid 1900.0-2030.0.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
from ADCS.helpers.math.spherical_harmonics import (
    RE, _schmidt_table, _kernel_static, _kernel_varying, _eval_scalar
)

__all__ = ["IGRF14Numba", "igrf_gc", "coefficients_at", "NMAX", "RE"]

#: Maximum spherical-harmonic degree of IGRF-14.
NMAX = 13

_SHC_FILE = Path(__file__).resolve().parents[1] / "kernels" / "IGRF14.shc"

#: Cached as a string: ``Path.__str__`` is not free, and the scalar path calls
#: this often enough for it to show up.
_SHC_PATH = str(_SHC_FILE)


# ---------------------------------------------------------------------------
# Coefficients
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def _read_shc(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    r"""
    Parse a ``.shc`` spherical-harmonic coefficient file.

    Cached, because this is exactly the work ``ppigrf`` repeated on every call.

    :param path:
        Path to the ``.shc`` file.
    :type path: str
    :return:
        ``(epochs, g, h, nmax)`` where ``epochs`` are the model epochs as
        POSIX seconds with shape ``(n_epochs,)``, and ``g``/``h`` have shape
        ``(n_epochs, nmax + 1, nmax + 1)`` with Schmidt normalisation already
        folded in.
    :rtype: tuple
    """
    header = 2
    times: np.ndarray | None = None
    nmax = 0
    raw: dict[tuple[int, int], np.ndarray] = {}

    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            if header == 2:  # N_MIN N_MAX NTIMES SP_ORDER N_STEPS
                _, nmax, _, _, _ = (int(v) for v in line.split()[:5])
                header -= 1
                continue
            if header == 1:  # the model epochs, as decimal years
                times = np.array([float(v) for v in line.split()])
                header -= 1
                continue
            parts = line.split()
            raw[(int(parts[0]), int(parts[1]))] = np.array(
                [float(v) for v in parts[2:]]
            )

    if times is None:
        raise ValueError(f"{path}: no epoch row found")

    n_ep = times.size
    g = np.zeros((n_ep, nmax + 1, nmax + 1))
    h = np.zeros((n_ep, nmax + 1, nmax + 1))
    for (n, m), vals in raw.items():
        if m >= 0:
            g[:, n, m] = vals
        else:
            h[:, n, -m] = vals

    # Fold Schmidt normalisation in once, here, so the kernel never sees it.
    S = _schmidt_table(nmax)
    g *= S[None, :, :]
    h *= S[None, :, :]

    # IGRF epochs are whole years, so ppigrf's yearfrac_to_datetime puts them at
    # 1 January. Interpolating on POSIX seconds reproduces its pandas
    # ``interpolate(method="time")`` exactly.
    # Built through datetime64 rather than datetime.timestamp(), which would
    # read these naive datetimes as *local* time and shift the epoch axis by
    # the machine's UTC offset.
    epochs = np.array(
        [
            np.datetime64(f"{int(t):04d}-01-01", "ns").astype(np.int64) / 1e9
            for t in times
        ],
        dtype=float,
    )
    return epochs, g, h, nmax


def _as_posix(date) -> np.ndarray:
    r"""
    Normalise one date or a sequence of dates to POSIX seconds.

    Naive datetimes are treated as UTC, matching how the rest of the package
    stores epochs.

    :param date:
        ``datetime``, ``numpy.datetime64``, or a sequence of either.
    :type date: datetime.datetime or numpy.datetime64 or sequence
    :return:
        Array of POSIX seconds, shape ``(n_dates,)``.
    :rtype: numpy.ndarray
    """
    arr = np.atleast_1d(np.asarray(date, dtype="datetime64[ns]"))
    return arr.astype("datetime64[ns]").astype(np.int64) / 1e9


def coefficients_at(date) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Gauss coefficients interpolated to a single date.

    :param date:
        Date at which to evaluate the model.
    :type date: datetime.datetime
    :return:
        ``(g, h)``, each of shape ``(NMAX + 1, NMAX + 1)``, Schmidt normalised.
    :rtype: tuple

    :example:

    >>> from datetime import datetime
    >>> g, h = coefficients_at(datetime(2024, 3, 1))
    >>> g.shape
    (14, 14)
    """
    epochs, g, h, _ = _read_shc(_SHC_PATH)
    k, w = _bracket(_as_posix(date), epochs)
    k0 = int(k[0])
    wt = float(w[0])
    return (
        g[k0] + wt * (g[k0 + 1] - g[k0]),
        h[k0] + wt * (h[k0 + 1] - h[k0]),
    )


def _year_of(posix_seconds: float) -> int:
    r"""Get the calendar year of a POSIX timestamp for warnings.

    :param posix_seconds: Seconds since the POSIX epoch [s].
    :type posix_seconds: float
    :return: UTC calendar year.
    :rtype: int
    """
    return int(
        np.datetime64(int(posix_seconds), "s").astype("datetime64[Y]").astype(int)
        + 1970
    )


def _bracket(t: np.ndarray, epochs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Locate each time between model epochs.

    :param t:
        Times as POSIX seconds.
    :type t: numpy.ndarray
    :param epochs:
        Model epochs as POSIX seconds.
    :type epochs: numpy.ndarray
    :return:
        ``(k, w)`` such that the coefficients are
        ``g[k] + w * (g[k + 1] - g[k])``.
    :rtype: tuple
    """
    if np.any(t < epochs[0]) or np.any(t > epochs[-1]):
        warnings.warn(
            "IGRF evaluated outside the coefficient file's range "
            f"({_year_of(epochs[0])}-{_year_of(epochs[-1])}); "
            "clamping to the nearest epoch.",
            RuntimeWarning,
            stacklevel=3,
        )
    k = np.clip(np.searchsorted(epochs, t, side="right") - 1, 0, epochs.size - 2)
    w = (t - epochs[k]) / (epochs[k + 1] - epochs[k])
    return k, np.clip(w, 0.0, 1.0)


def _posix_of(date: datetime) -> float:
    r"""
    POSIX seconds for one ``datetime``, treating naive input as UTC.

    Avoids the ``datetime64`` round-trip in :func:`_as_posix`, which dominates
    the scalar call.

    :param date:
        The date to convert.
    :type date: datetime.datetime
    :return:
        Seconds since the POSIX epoch.
    :rtype: float
    """
    if date.tzinfo is None:
        return date.replace(tzinfo=timezone.utc).timestamp()
    return date.timestamp()


def _bracket_scalar(t: float, epochs: np.ndarray) -> tuple[int, float]:
    r"""
    Scalar counterpart of :func:`_bracket`, without the numpy round-trips.

    :param t:
        Time as POSIX seconds.
    :type t: float
    :param epochs:
        Model epochs as POSIX seconds.
    :type epochs: numpy.ndarray
    :return:
        ``(k, w)`` such that the coefficients are ``g[k] + w * (g[k+1] - g[k])``.
    :rtype: tuple
    """
    if t < epochs[0] or t > epochs[-1]:
        warnings.warn(
            "IGRF evaluated outside the coefficient file's range "
            f"({_year_of(epochs[0])}-{_year_of(epochs[-1])}); "
            "clamping to the nearest epoch.",
            RuntimeWarning,
            stacklevel=3,
        )
    n = epochs.size
    k = int(np.searchsorted(epochs, t, side="right")) - 1
    if k < 0:
        k = 0
    elif k > n - 2:
        k = n - 2
    w = (t - epochs[k]) / (epochs[k + 1] - epochs[k])
    return k, min(max(w, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def igrf_gc(r, theta, phi, date, chunk: int = 512):
    r"""
    Geomagnetic field in geocentric spherical components.

    Unlike ``ppigrf.igrf_gc``, dates are broadcast *elementwise* against the
    coordinates rather than forming an outer product. Passing N positions and N
    dates returns N field vectors, not an N x N grid -- which is what every
    caller in this package actually wanted, and what the ``np.diagonal``
    workarounds in :mod:`ADCS.orbits.orbit` used to recover by hand.

    :param r:
        Geocentric radius [km].
    :type r: float or numpy.ndarray
    :param theta:
        Geocentric colatitude [deg].
    :type theta: float or numpy.ndarray
    :param phi:
        Geocentric longitude [deg], positive east.
    :type phi: float or numpy.ndarray
    :param date:
        One date, or one date per sample.
    :type date: datetime.datetime or sequence
    :param chunk:
        Samples per block of Legendre scratch reuse. Affects speed only,
        never the result.
    :type chunk: int
    :return:
        ``(Br, Btheta, Bphi)`` in nT, each with the broadcast shape of the
        coordinates. ``Btheta`` points south, ``Bphi`` east.
    :rtype: tuple

    :example:

    >>> from datetime import datetime
    >>> import numpy as np
    >>> br, bt, bp = igrf_gc(6771.2, 45.0, 10.0, datetime(2024, 3, 1))
    >>> bool(np.hypot(np.hypot(br, bt), bp) > 1e4)   # order 10^4 nT in LEO
    True
    """
    # Single point, single date: skip every array allocation below.
    # ``np.float64`` subclasses ``float``, and ``pandas.Timestamp`` subclasses
    # ``datetime``, so isinstance covers the shapes callers actually pass.
    if (
        np.ndim(r) == 0 and np.ndim(theta) == 0
        and np.ndim(phi) == 0 and isinstance(date, datetime)
    ):
        epochs, g, h, nmax = _read_shc(_SHC_PATH)
        k, w = _bracket_scalar(_posix_of(date), epochs)
        return _eval_scalar(float(r), float(theta), float(phi), g, h, k, w, nmax)

    r_a = np.asarray(r, dtype=float)
    th_a = np.asarray(theta, dtype=float)
    ph_a = np.asarray(phi, dtype=float)
    if not (r_a.shape == th_a.shape == ph_a.shape):
        r_a, th_a, ph_a = np.broadcast_arrays(r_a, th_a, ph_a)
    shape = r_a.shape
    r_f = np.ascontiguousarray(r_a.reshape(-1))
    th_f = np.ascontiguousarray(th_a.reshape(-1))
    ph_f = np.ascontiguousarray(ph_a.reshape(-1))
    N = r_f.size

    epochs, g, h, nmax = _read_shc(_SHC_PATH)
    t = _as_posix(date)
    if t.size not in (1, N):
        raise ValueError(
            f"date has {t.size} entries; expected 1 or {N} to match the coordinates"
        )
    k, w = _bracket(t, epochs)

    Br = np.empty(N)
    Bt = np.empty(N)
    Bp = np.empty(N)

    # A single date -- the common case, including every per-step call from the
    # simulation -- collapses to one set of coefficients, which lets the whole
    # secular-variation blend leave the inner loop.
    if t.size == 1 or (np.all(k == k[0]) and np.all(w == w[0])):
        k0 = int(k[0])
        w0 = float(w[0])
        gb = g[k0] + w0 * (g[k0 + 1] - g[k0])
        hb = h[k0] + w0 * (h[k0 + 1] - h[k0])
        _kernel_static(r_f, th_f, ph_f, gb, hb, nmax, Br, Bt, Bp, int(chunk))
    else:
        _kernel_varying(
            r_f, th_f, ph_f, g, h,
            np.ascontiguousarray(k, dtype=np.int64),
            np.ascontiguousarray(w, dtype=float),
            nmax, Br, Bt, Bp, int(chunk),
        )

    if shape == ():
        return Br[0], Bt[0], Bp[0]
    return Br.reshape(shape), Bt.reshape(shape), Bp.reshape(shape)


from .igrf14 import IGRF14


class IGRF14Numba(IGRF14):
    r"""IGRF-14 evaluated by the in-tree Numba spherical harmonic kernels."""

    def __init__(self) -> None:
        r"""Select the Numba IGRF-14 implementation.

        :return: None
        :rtype: None
        """
        super().__init__("igrf_numba")
