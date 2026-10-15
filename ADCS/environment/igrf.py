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
from numba import njit

__all__ = ["igrf_gc", "coefficients_at", "NMAX", "RE"]

#: Reference radius of the IGRF expansion [km]. This is the IGRF's own
#: constant, not an Earth radius from :mod:`ADCS.helpers` -- changing it would
#: silently rescale the field.
RE = 6371.2

#: Maximum spherical-harmonic degree of IGRF-14.
NMAX = 13

_SHC_FILE = Path(__file__).with_name("IGRF14.shc")

#: Cached as a string: ``Path.__str__`` is not free, and the scalar path calls
#: this often enough for it to show up.
_SHC_PATH = str(_SHC_FILE)


# ---------------------------------------------------------------------------
# Coefficients
# ---------------------------------------------------------------------------


def _schmidt_table(nmax: int = NMAX) -> np.ndarray:
    r"""
    Schmidt semi-normalisation factors :math:`S_{n,m}`.

    Folded into the Gauss coefficients rather than applied to the Legendre
    functions, which is algebraically identical and keeps it out of the kernel.

    :param nmax:
        Maximum spherical-harmonic degree.
    :type nmax: int
    :return:
        Array of shape ``(nmax + 1, nmax + 1)``.
    :rtype: numpy.ndarray
    """
    S = np.zeros((nmax + 1, nmax + 1))
    S[0, 0] = 1.0
    for n in range(1, nmax + 1):
        S[n, 0] = S[n - 1, 0] * (2.0 * n - 1.0) / n
        for m in range(1, n + 1):
            S[n, m] = S[n, m - 1] * np.sqrt(
                (n - m + 1.0) * ((2.0 if m == 1 else 1.0)) / (n + m)
            )
    return S


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
    print("✅ Loaded Magnetic Field Model: IGRF14.shc")
    return epochs, g, h, nmax


def _as_posix(date) -> np.ndarray:
    r"""
    Normalise one date or a sequence of dates to POSIX seconds.

    Naive datetimes are treated as UTC, matching how the rest of the package
    stores epochs.

    :param date:
        ``datetime``, ``numpy.datetime64``, or a sequence of either.
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
    r"""Calendar year of a POSIX timestamp, for warning messages."""
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


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True, nogil=True, inline="always")
def _eval_one(ri, th_deg, ph_deg, g, h, nmax, Pc, P1, P2, Dc, D1, D2, cosmp, sinmp):
    r"""
    Evaluate one sample against a single set of Gauss coefficients.

    Shared by both kernels so the recursion exists in exactly one place. The
    scratch arrays are passed in rather than allocated: allocating them per
    sample was most of the cost of the first version of this code.

    :return:
        ``(Br, Btheta, Bphi)`` in nT for this sample.
    """
    if ri <= 0.0:
        # r=0 is non-physical (the expansion is singular at the origin), but
        # some callers construct a throwaway state at the origin purely for
        # its frame-conversion helpers and never read .B (e.g. the GPS
        # estimator's dummy temp_os). ppigrf's numpy division degrades this
        # to nan with a warning rather than raising; match that instead of
        # a hard ZeroDivisionError from RE / ri below, since Python/numba
        # float division (unlike numpy's) raises on an exact-zero divisor.
        nan = np.nan
        return nan, nan, nan

    # Bphi is expressed as a quotient by sin(theta). At the exact poles both
    # its numerator and denominator vanish, although the physical limiting
    # value is finite. Evaluate the pole limit at a tiny angular offset to
    # avoid 0/0 while retaining the radial and southward components to
    # machine precision.
    if th_deg == 0.0:
        th_deg = 1e-7
    elif th_deg == 180.0:
        th_deg = 180.0 - 1e-7
    th = th_deg * (np.pi / 180.0)
    ph = ph_deg * (np.pi / 180.0)
    sinth = np.sin(th)
    costh = np.cos(th)

    # cos(m*phi), sin(m*phi) by angle addition -- 2 trig calls, not 2*nmax.
    cp = np.cos(ph)
    sp = np.sin(ph)
    cosmp[0] = 1.0
    sinmp[0] = 0.0
    for m in range(1, nmax + 1):
        cosmp[m] = cosmp[m - 1] * cp - sinmp[m - 1] * sp
        sinmp[m] = sinmp[m - 1] * cp + cosmp[m - 1] * sp

    for q in range(nmax + 2):
        P1[q] = 0.0
        D1[q] = 0.0
        P2[q] = 0.0
        D2[q] = 0.0
    P1[0] = 1.0  # P[0,0]; acts as degree n-1 on the first pass

    a_r = RE / ri
    arn = a_r * a_r  # (RE/r)^(n+2) at n = 0

    sr = 0.0
    st = 0.0
    sq = 0.0
    for n in range(1, nmax + 1):
        arn *= a_r
        for m in range(0, n + 1):
            if n == m:
                Pc[n] = sinth * P1[m - 1]
                Dc[n] = sinth * D1[m - 1] + costh * P1[n - 1]
            elif n == 1:
                Pc[m] = costh * P1[m]
                Dc[m] = costh * D1[m] - sinth * P1[m]
            else:
                K = ((n - 1.0) * (n - 1.0) - m * m) / (
                    (2.0 * n - 1.0) * (2.0 * n - 3.0)
                )
                Pc[m] = costh * P1[m] - K * P2[m]
                Dc[m] = costh * D1[m] - sinth * P1[m] - K * D2[m]
        # One slot past the diagonal is held at zero so that reading degree
        # n-2 at order m = n-1 stays in range.
        Pc[n + 1] = 0.0
        Dc[n + 1] = 0.0

        for m in range(0, n + 1):
            gnm = g[n, m]
            hnm = h[n, m]
            cc = gnm * cosmp[m] + hnm * sinmp[m]
            sr += arn * (n + 1.0) * cc * Pc[m]
            st -= arn * cc * Dc[m]
            sq -= arn * m * (hnm * cosmp[m] - gnm * sinmp[m]) * Pc[m]

        P2, P1, Pc = P1, Pc, P2
        D2, D1, Dc = D1, Dc, D2

    return sr, st, sq / sinth


@njit(cache=True, fastmath=True, nogil=True)
def _kernel_static(r, theta_deg, phi_deg, g, h, nmax, Br, Bt, Bp, chunk):
    r"""
    All samples share one epoch, so the coefficients are blended once outside.

    Serial on purpose: with ``parallel=True`` numba starts a GNU OpenMP
    thread pool, which is not fork-safe, and the Monte Carlo runner forks its
    workers after the parent has built an Orbit through here -- every worker
    then died. Parallelism belongs to the Monte Carlo layer, one run per
    process. The Legendre scratch is allocated once per chunk.
    """
    N = r.shape[0]
    n_chunks = (N + chunk - 1) // chunk
    for c in range(n_chunks):
        i0 = c * chunk
        i1 = min(i0 + chunk, N)
        Pc = np.zeros(nmax + 2); P1 = np.zeros(nmax + 2); P2 = np.zeros(nmax + 2)
        Dc = np.zeros(nmax + 2); D1 = np.zeros(nmax + 2); D2 = np.zeros(nmax + 2)
        cosmp = np.zeros(nmax + 1); sinmp = np.zeros(nmax + 1)
        for i in range(i0, i1):
            br, bt, bp = _eval_one(
                r[i], theta_deg[i], phi_deg[i], g, h, nmax,
                Pc, P1, P2, Dc, D1, D2, cosmp, sinmp,
            )
            Br[i] = br; Bt[i] = bt; Bp[i] = bp


@njit(cache=True, fastmath=True, nogil=True)
def _kernel_varying(r, theta_deg, phi_deg, g, h, kidx, w, nmax, Br, Bt, Bp, chunk):
    r"""
    Per-sample dates: blend the bracketing epochs into chunk-local scratch.

    IGRF coefficients are piecewise linear in time between 5-year epochs, so
    this reproduces ``ppigrf``'s per-date interpolation exactly. It costs about
    60% more than :func:`_kernel_static`, which is why the single-date case
    does not come through here.
    """
    N = r.shape[0]
    n_chunks = (N + chunk - 1) // chunk
    for c in range(n_chunks):
        i0 = c * chunk
        i1 = min(i0 + chunk, N)
        Pc = np.zeros(nmax + 2); P1 = np.zeros(nmax + 2); P2 = np.zeros(nmax + 2)
        Dc = np.zeros(nmax + 2); D1 = np.zeros(nmax + 2); D2 = np.zeros(nmax + 2)
        cosmp = np.zeros(nmax + 1); sinmp = np.zeros(nmax + 1)
        gb = np.zeros((nmax + 1, nmax + 1)); hb = np.zeros((nmax + 1, nmax + 1))
        for i in range(i0, i1):
            ki = kidx[i]
            wi = w[i]
            for n in range(nmax + 1):
                for m in range(n + 1):
                    gb[n, m] = g[ki, n, m] + wi * (g[ki + 1, n, m] - g[ki, n, m])
                    hb[n, m] = h[ki, n, m] + wi * (h[ki + 1, n, m] - h[ki, n, m])
            br, bt, bp = _eval_one(
                r[i], theta_deg[i], phi_deg[i], gb, hb, nmax,
                Pc, P1, P2, Dc, D1, D2, cosmp, sinmp,
            )
            Br[i] = br; Bt[i] = bt; Bp[i] = bp


@njit(cache=True, fastmath=True, nogil=True)
def _eval_scalar(ri, th_deg, ph_deg, g, h, k, w, nmax):
    r"""
    One sample, one date, with the scratch allocated inside the kernel.

    The array plumbing in :func:`igrf_gc` costs more than the arithmetic when
    N is 1, and N is 1 on every per-step call from the simulation, so that
    path comes straight here instead.
    """
    gb = np.zeros((nmax + 1, nmax + 1))
    hb = np.zeros((nmax + 1, nmax + 1))
    for n in range(nmax + 1):
        for m in range(n + 1):
            gb[n, m] = g[k, n, m] + w * (g[k + 1, n, m] - g[k, n, m])
            hb[n, m] = h[k, n, m] + w * (h[k + 1, n, m] - h[k, n, m])
    Pc = np.zeros(nmax + 2); P1 = np.zeros(nmax + 2); P2 = np.zeros(nmax + 2)
    Dc = np.zeros(nmax + 2); D1 = np.zeros(nmax + 2); D2 = np.zeros(nmax + 2)
    cosmp = np.zeros(nmax + 1); sinmp = np.zeros(nmax + 1)
    return _eval_one(ri, th_deg, ph_deg, gb, hb, nmax,
                     Pc, P1, P2, Dc, D1, D2, cosmp, sinmp)


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
        isinstance(r, float) and isinstance(theta, float)
        and isinstance(phi, float) and isinstance(date, datetime)
    ):
        epochs, g, h, nmax = _read_shc(_SHC_PATH)
        k, w = _bracket_scalar(_posix_of(date), epochs)
        return _eval_scalar(r, theta, phi, g, h, k, w, nmax)

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
