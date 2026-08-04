from __future__ import annotations
__all__ = [
    "finite_s_coefficients",
    "panel_aero_force_body_finite_s",
    "speed_ratio",
    "wall_speed",
]

from math import erf as _erf

import numpy as np

_K_BOLTZ = 1.380649e-23
_AMU = 1.66054e-27
_SQRT_PI = np.sqrt(np.pi)
_erf_v = np.vectorize(_erf, otypes=[float])


def wall_speed(T_wall: float = 300.0, m_amu: float = 16.0) -> float:
    r"""
    Mean normal re-emission speed :math:`V_w=\sqrt{\pi k T_w/(2m)}`
    (Storch 2002, Eq. 2.2). Atomic oxygen (16 amu) dominates at LEO
    datacenter altitudes.
    """
    return float(np.sqrt(np.pi * _K_BOLTZ * T_wall / (2.0 * m_amu * _AMU)))


def speed_ratio(V: float, T_inf: float = 900.0, m_amu: float = 16.0) -> float:
    r"""
    Molecular speed ratio :math:`S=V/\sqrt{2kT_\infty/m}` — the parameter
    that separates hyperthermal (:math:`S\to\infty`) from finite-temperature
    free-molecular flow. Over a solar cycle (:math:`T_\infty` 600–1500 K,
    :math:`V\approx7.6` km/s) :math:`S` runs roughly 9.6 down to 6.1, so the
    *shape* of the force law changes with no density change at all.
    """
    return float(V / np.sqrt(2.0 * _K_BOLTZ * T_inf / (m_amu * _AMU)))


def finite_s_coefficients(s, S: float, sigma_n: float = 0.90,
                          sigma_t: float = 0.80,
                          Vw_over_V: float = 494.85 / 7602.3):
    r"""
    Storch (2002) finite speed-ratio flat-plate coefficients
    :math:`(A_w, A_n)` resolved on the wind/normal basis, for incidence
    sine :math:`s=|\hat n\cdot\hat v|` (Eqs. 3.18/3.19):

    .. math::

        f(x) = e^{-x^2}/\sqrt\pi + x\,\mathrm{erf}\,x,\qquad x = S s,

        A_w = (C_t/S)\, f, \qquad
        A_n = K s + \frac{C_n}{2S^2} f' + \frac{C_n-C_t}{S}\, s f,

    with :math:`C_n=2(2-\sigma_n)`, :math:`C_t=2\sigma_t`,
    :math:`K=2\sigma_n V_w/V`, :math:`f'=\mathrm{erf}`. The
    :math:`S\to\infty` limit is the classical hyperthermal plate
    (:func:`~ADCS.satellite_hardware.aero.aero_force.panel_aero_force_body`
    with the same :math:`C_n,C_t`). Unlike that limit, the finite-``S``
    form has a **thermal floor**: a feathered panel (:math:`s=0`) still
    feels shear :math:`A_w=(C_t/S)/\sqrt\pi > 0` — the effect that
    dominates near-feather differential-drag authority and is entirely
    absent from the hyperthermal model.

    Force per panel: :math:`\mathbf F/(qA) = A_w\,\hat v - A_n\,\hat n_w`
    with :math:`q=\tfrac12\rho V^2` and :math:`\hat n_w` the upwind-pointing
    normal.
    """
    s = np.asarray(s, dtype=float)
    Cn = 2.0 * (2.0 - sigma_n)
    Ct = 2.0 * sigma_t
    K = 2.0 * sigma_n * Vw_over_V
    x = S * s
    F = np.exp(-x * x) / _SQRT_PI + x * _erf_v(x)
    R = _erf_v(x)
    A_w = (Ct / S) * F
    A_n = s * K + (Cn / (2.0 * S * S)) * R + ((Cn - Ct) / S) * s * F
    return A_w, A_n


def panel_aero_force_body_finite_s(V_b, rho, plate_normals, plate_areas,
                                   sigma_n: float = 0.90,
                                   sigma_t: float = 0.80,
                                   T_inf: float = 900.0,
                                   T_wall: float = 300.0,
                                   m_amu: float = 16.0) -> np.ndarray:
    r"""
    Finite speed-ratio free-molecular force in the body frame, for a
    satellite built from thin **plates** (solar arrays, sails, wings —
    the formation-flying use case).

    **Plate semantics (different from** :func:`panel_aero_force_body`
    **):** each physical plate appears ONCE, with either choice of
    normal; the Storch coefficients describe the whole two-sided plate,
    including the thermal-floor shear a feathered plate still feels.
    Do not list front/back face pairs.

    **Coefficient convention:** this function is parameterized by
    accommodation coefficients and uses the Storch normalization
    :math:`q=\tfrac12\rho V^2`, :math:`C_n=2(2-\sigma_n)`,
    :math:`C_t=2\sigma_t`. The hyperthermal
    :func:`panel_aero_force_body` uses the SAME standard
    :math:`\tfrac12\rho V^2` dynamic pressure (normalized in the
    accompanying fix commit; it briefly carried a :math:`\rho V^2`
    prefactor on this branch), so the coefficient mapping is the
    IDENTITY: the Suncatcher committed pair :math:`(C_n, C_t) =
    (2.2, 1.6)` is passed to either function unchanged, and the
    default ``Cn = 2.0`` corresponds to a fully diffuse plate
    (:math:`\sigma_n = 1`). The :math:`S\to\infty`,
    :math:`T_{\rm wall}\to0` limit of this function equals
    :func:`panel_aero_force_body` with :math:`C_n=2(2-\sigma_n)`,
    :math:`C_t=2\sigma_t` directly (regression-tested).

    :param V_b: Body-frame satellite velocity relative to the
        atmosphere [m/s] (ram direction), shape ``(3,)``.
    :param rho: Atmospheric density [kg/m^3].
    :param plate_normals: One unit normal per physical plate,
        shape ``(M, 3)``.
    :param plate_areas: One-sided plate areas [m^2], shape ``(M,)``.
    :param sigma_n: Normal momentum accommodation (0.90 = Suncatcher
        committed value).
    :param sigma_t: Tangential momentum accommodation (0.80 = WP4
        literature-central; Storch :math:`C_t=1.6`).
    :param T_inf: Exospheric temperature [K] — sets :math:`S`; expose
        it in solar-cycle studies (600–1500 K swings the feather floor
        ~58% with no density change).
    :param T_wall: Wall temperature [K] for the re-emission term.
    :param m_amu: Mean molecular mass [amu].
    :return: Net aerodynamic force in the body frame [N], shape ``(3,)``.
    """
    V_b = np.asarray(V_b, dtype=float).reshape(3)
    V2 = float(V_b @ V_b)
    if rho <= 0.0 or V2 <= 0.0:
        return np.zeros(3)
    V = np.sqrt(V2)
    vhat = V_b / V
    S = speed_ratio(V, T_inf=T_inf, m_amu=m_amu)
    Vw = wall_speed(T_wall=T_wall, m_amu=m_amu)
    plate_normals = np.asarray(plate_normals, dtype=float)
    si = plate_normals @ vhat                 # signed incidence sine
    s = np.abs(si)
    A_w, A_n = finite_s_coefficients(s, S, sigma_n=sigma_n,
                                     sigma_t=sigma_t,
                                     Vw_over_V=Vw / V)
    q = 0.5 * rho * V2 * np.asarray(plate_areas, dtype=float)
    # drag along -vhat; normal pressure pushes each plate toward its
    # leeward side (-sign(si) n); Storch (w, n_w) with w = -vhat.
    F = (-(q * A_w)[:, None] * vhat[None, :]
         - (q * A_n * np.sign(si))[:, None] * plate_normals)
    return F.sum(axis=0)
