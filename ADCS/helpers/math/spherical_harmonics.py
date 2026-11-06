r"""Numba spherical harmonic helpers shared by geomagnetic field models.

The coefficients passed to these kernels are already Schmidt semi-normalized.
All field components are geocentric spherical and measured in nanotesla.
"""

from __future__ import annotations

import numpy as np
from numba import njit

RE = 6371.2  # Magnetic harmonic reference radius [km].

@njit(cache=True, nogil=True)
def _schmidt_table(nmax: int) -> np.ndarray:
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

@njit(cache=True, fastmath=True, nogil=True, inline="always")
def _eval_one(ri, th_deg, ph_deg, g, h, nmax, Pc, P1, P2, Dc, D1, D2, cosmp, sinmp):
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
    :param Pc: Scratch for current Legendre degree.
    :type Pc: numpy.ndarray
    :param P1: Scratch for previous Legendre degree.
    :type P1: numpy.ndarray
    :param P2: Scratch for second previous Legendre degree.
    :type P2: numpy.ndarray
    :param Dc: Scratch for current colatitude derivative.
    :type Dc: numpy.ndarray
    :param D1: Scratch for previous derivative.
    :type D1: numpy.ndarray
    :param D2: Scratch for second previous derivative.
    :type D2: numpy.ndarray
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
    Evaluate positions sharing one set of Gauss coefficients.

    Serial on purpose: with ``parallel=True`` numba starts a GNU OpenMP
    thread pool, which is not fork-safe, and the Monte Carlo runner forks its
    workers after the parent has built an Orbit through here. Parallelism
    belongs to the Monte Carlo layer, one run per process. Legendre scratch
    arrays are allocated once per chunk.

    :param r: Geocentric radii [km].
    :type r: numpy.ndarray
    :param theta_deg: Geocentric colatitudes [deg].
    :type theta_deg: numpy.ndarray
    :param phi_deg: East longitudes [deg].
    :type phi_deg: numpy.ndarray
    :param g: Schmidt normalized cosine Gauss coefficients [nT].
    :type g: numpy.ndarray
    :param h: Schmidt normalized sine Gauss coefficients [nT].
    :type h: numpy.ndarray
    :param nmax: Maximum harmonic degree.
    :type nmax: int
    :param Br: Output radial components [nT].
    :type Br: numpy.ndarray
    :param Bt: Output southward components [nT].
    :type Bt: numpy.ndarray
    :param Bp: Output eastward components [nT].
    :type Bp: numpy.ndarray
    :param chunk: Samples per scratch-array reuse block.
    :type chunk: int
    :return: None; field component arrays are filled in place.
    :rtype: None
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
    Blend adjacent coefficient sets and evaluate positions with paired dates.

    The same kernel supports IGRF epoch interpolation and WMM annual variation.

    :param r: Geocentric radii [km].
    :type r: numpy.ndarray
    :param theta_deg: Geocentric colatitudes [deg].
    :type theta_deg: numpy.ndarray
    :param phi_deg: East longitudes [deg].
    :type phi_deg: numpy.ndarray
    :param g: Consecutive Schmidt normalized cosine coefficient sets [nT].
    :type g: numpy.ndarray
    :param h: Consecutive Schmidt normalized sine coefficient sets [nT].
    :type h: numpy.ndarray
    :param kidx: First coefficient index per sample.
    :type kidx: numpy.ndarray
    :param w: Linear blend weight per sample.
    :type w: numpy.ndarray
    :param nmax: Maximum harmonic degree.
    :type nmax: int
    :param Br: Output radial components [nT].
    :type Br: numpy.ndarray
    :param Bt: Output southward components [nT].
    :type Bt: numpy.ndarray
    :param Bp: Output eastward components [nT].
    :type Bp: numpy.ndarray
    :param chunk: Samples per scratch-array reuse block.
    :type chunk: int
    :return: None; field component arrays are filled in place.
    :rtype: None
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

    The array plumbing around these kernels costs more than the arithmetic when
    N is 1, and N is 1 on every per-step call from the simulation, so that
    path comes straight here instead.

    :param ri: Geocentric radius [km].
    :type ri: float
    :param th_deg: Geocentric colatitude [deg].
    :type th_deg: float
    :param ph_deg: East longitude [deg].
    :type ph_deg: float
    :param g: Consecutive Schmidt normalized cosine coefficient sets [nT].
    :type g: numpy.ndarray
    :param h: Consecutive Schmidt normalized sine coefficient sets [nT].
    :type h: numpy.ndarray
    :param k: First coefficient index.
    :type k: int
    :param w: Linear blend weight.
    :type w: float
    :param nmax: Maximum harmonic degree.
    :type nmax: int
    :return: Radial, southward, and eastward components [nT].
    :rtype: tuple[float, float, float]
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

