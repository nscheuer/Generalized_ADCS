r"""Numba spherical harmonic helpers shared by geomagnetic field models.

The coefficients passed to these kernels are already Schmidt semi-normalized.
All field components are geocentric spherical and measured in nanotesla.
"""

from __future__ import annotations

import numpy as np
from numba import njit

RE = 6371.2  # Magnetic harmonic reference radius [km].


@njit(cache=True, nogil=True)
def schmidt_table(nmax: int) -> np.ndarray:
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
                (n - m + 1.0) * (2.0 if m == 1 else 1.0) / (n + m)
            )
    return S


@njit(cache=True, fastmath=True, nogil=True, inline="always")
def eval_one(
    ri: float,
    th_deg: float,
    ph_deg: float,
    g: np.ndarray,
    h: np.ndarray,
    nmax: int,
    pc: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    dc: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    cosmp: np.ndarray,
    sinmp: np.ndarray,
) -> tuple[float, float, float]:
    r"""
    Evaluate one sample against a single set of Gauss coefficients.

    Shared by both kernels so the recursion exists in exactly one place. The
    scratch arrays are passed in rather than allocated: allocating them per
    sample was most of the cost of the first version of this code.

    :param ri: Geocentric radius [km].
    :type ri: float
    :param th_deg: Geocentric colatitude [deg].
    :type th_deg: float
    :param ph_deg: East longitude [deg].
    :type ph_deg: float
    :param g: Schmidt normalized cosine Gauss coefficients [nT].
    :type g: numpy.ndarray
    :param h: Schmidt normalized sine Gauss coefficients [nT].
    :type h: numpy.ndarray
    :param nmax: Maximum harmonic degree.
    :type nmax: int
    :param pc: Scratch for current Legendre degree.
    :type pc: numpy.ndarray
    :param p1: Scratch for previous Legendre degree.
    :type p1: numpy.ndarray
    :param p2: Scratch for second previous Legendre degree.
    :type p2: numpy.ndarray
    :param dc: Scratch for current colatitude derivative.
    :type dc: numpy.ndarray
    :param d1: Scratch for previous derivative.
    :type d1: numpy.ndarray
    :param d2: Scratch for second previous derivative.
    :type d2: numpy.ndarray
    :param cosmp: Scratch for longitude cosine multiples.
    :type cosmp: numpy.ndarray
    :param sinmp: Scratch for longitude sine multiples.
    :type sinmp: numpy.ndarray
    :return: Radial, southward, and eastward components [nT].
    :rtype: tuple[float, float, float]
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
        p1[q] = 0.0
        d1[q] = 0.0
        p2[q] = 0.0
        d2[q] = 0.0
    p1[0] = 1.0  # P[0,0]; acts as degree n-1 on the first pass

    a_r = RE / ri
    arn = a_r * a_r  # (RE/r)^(n+2) at n = 0

    sr = 0.0
    st = 0.0
    sq = 0.0
    for n in range(1, nmax + 1):
        arn *= a_r
        for m in range(n + 1):
            if n == m:
                pc[n] = sinth * p1[m - 1]
                dc[n] = sinth * d1[m - 1] + costh * p1[n - 1]
            elif n == 1:
                pc[m] = costh * p1[m]
                dc[m] = costh * d1[m] - sinth * p1[m]
            else:
                K = ((n - 1.0) * (n - 1.0) - m * m) / (
                    (2.0 * n - 1.0) * (2.0 * n - 3.0)
                )
                pc[m] = costh * p1[m] - K * p2[m]
                dc[m] = costh * d1[m] - sinth * p1[m] - K * d2[m]
        # One slot past the diagonal is held at zero so that reading degree
        # n-2 at order m = n-1 stays in range.
        pc[n + 1] = 0.0
        dc[n + 1] = 0.0

        for m in range(n + 1):
            gnm = g[n, m]
            hnm = h[n, m]
            cc = gnm * cosmp[m] + hnm * sinmp[m]
            sr += arn * (n + 1.0) * cc * pc[m]
            st -= arn * cc * dc[m]
            sq -= arn * m * (hnm * cosmp[m] - gnm * sinmp[m]) * pc[m]

        p2, p1, pc = p1, pc, p2
        d2, d1, dc = d1, dc, d2

    return sr, st, sq / sinth
