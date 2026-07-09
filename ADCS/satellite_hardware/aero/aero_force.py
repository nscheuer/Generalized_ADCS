from __future__ import annotations
__all__ = ["AeroModel", "panel_aero_force_body"]

import numpy as np

from ADCS.helpers.math_helpers import normalize
from ADCS.satellite_hardware.disturbances.helpers.geometry_config import GeometryConfig


def panel_aero_force_body(V_b, rho, normals, areas, Cn, Ct):
    r"""
    Free-molecular panel aerodynamic force in the body frame (drag + lift).

    For each surface panel with outward unit normal :math:`\hat{n}` and area
    :math:`A`, exposed to a body-frame relative wind velocity
    :math:`\mathbf{V}_b` (magnitude :math:`V`, direction :math:`\hat{v}`), define
    the (clipped) incidence cosine :math:`c = \max(0,\hat{n}\cdot\hat{v})`. The
    panel force combines a **normal pressure** term (momentum delivered along the
    surface normal) and a **tangential shear** term (momentum delivered along the
    in-plane flow direction):

    .. math::

        \mathbf{F} = -\tfrac{1}{2}\,\rho V^2 A\, c\,
            \Big[\, C_n\, c\, \hat{n} \;+\; C_t\,(\hat{v} - c\,\hat{n}) \,\Big].

    The **standard** dynamic pressure :math:`\tfrac{1}{2}\rho V^2` is used, so at
    normal incidence (:math:`c=1`, :math:`\hat n=\hat v`) the force reduces to the
    textbook drag :math:`\mathbf{F} = -\tfrac{1}{2} C_n\,\rho V^2 A\,\hat v` — i.e.
    :math:`C_n` is the normal-incidence drag coefficient (:math:`C_D`), about 2.0
    for a diffuse free-molecular flat plate. At oblique incidence the normal-
    pressure term contributes a component perpendicular to :math:`\hat v` — the
    **lift** — whose magnitude scales with :math:`(C_n-C_t)`; setting
    :math:`C_n=C_t` recovers a lift-free (drag-only) panel. Faces in the wake
    (:math:`c\le 0`) contribute nothing.

    :param V_b: Body-frame relative wind velocity [m/s], shape ``(3,)``.
    :param rho: Atmospheric density [kg/m^3].
    :param normals: Unit outward normals, shape ``(M, 3)``.
    :param areas: Panel areas [m^2], shape ``(M,)``.
    :param Cn: Normal (pressure) coefficient.
    :param Ct: Tangential (shear) coefficient.
    :return: Net aerodynamic force in the body frame [N], shape ``(3,)``.
    """
    V_b = np.asarray(V_b, dtype=float).reshape(3)
    V2 = float(V_b @ V_b)
    if rho <= 0.0 or V2 <= 0.0:
        return np.zeros(3)

    V = np.sqrt(V2)
    vhat = V_b / V
    c = normals @ vhat                      # (M,) incidence cosine per face
    c = np.where(c > 0.0, c, 0.0)           # only windward faces contribute

    scale = 0.5 * rho * V2 * np.asarray(areas, dtype=float) * c   # (M,) standard 1/2 rho V^2
    F_normal = -(Cn * scale * c)[:, None] * normals
    F_shear = -(Ct * scale)[:, None] * (vhat[None, :] - c[:, None] * normals)
    return (F_normal + F_shear).sum(axis=0)


# =========================================================================== #
# Sentman / DRIA facet model (WP2 Task 4; NOT the default -- see AeroModel.mode)
# =========================================================================== #
K_BOLTZ = 1.380649e-23          # J/K
AMU = 1.66053907e-27            # kg

# 3-point exospheric-temperature table tied to solar_level (WP2 spec defaults;
# overridable by callers): T_inf = 700 / 900 / 1100 K at level 0 / 0.5 / 1.
EXO_TEMP_LEVELS = np.array([0.0, 0.5, 1.0])
EXO_TEMP_K = np.array([700.0, 900.0, 1100.0])


def exospheric_temperature(solar_level: float, levels=EXO_TEMP_LEVELS, temps=EXO_TEMP_K) -> float:
    r"""Exospheric temperature T_inf [K] vs solar activity level (linear interp)."""
    return float(np.interp(float(solar_level), levels, temps))


def sentman_facet_coeffs(gamma, s: float, vr_ratio):
    r"""
    Sentman (1961) free-molecular facet coefficients with diffuse re-emission,
    per unit area, normalized by the standard dynamic pressure 1/2 rho V^2.

    Derived from the drifting-Maxwellian flux integrals of Bird (1994) Eqs. (4.22)
    (number flux) and (4.27) (energy flux), as quoted in Fuchs & Fasoulas,
    arXiv:2411.11597, Eqs. (10)-(11); the incident normal/tangential momentum
    fluxes follow the same integrals. Diffuse re-emission carries momentum along
    the surface normal only (Sentman's model; Doornbos 2012 Sec. 3.5 concept).
    Validated against the WP2 spec gates: broadside C_D in [2.0, 2.5], grazing
    dC_D/dphi = 2 erf(s sin phi) cos phi (alpha=1), hyperthermal recovery at s=50.

    :param gamma: SIGNED incidence cosine n_hat . v_hat per facet (windward > 0;
        leeward facets receive the thermal tail -- do NOT clip).
    :param s: molecular speed ratio |V| / sqrt(2 k T_inf / m).
    :param vr_ratio: re-emission speed ratio c_m,r / V = sqrt(2 k T_r / m) / V,
        with T_r from the DRIA temperature ratio (see :func:`dria_reemission_temp`).
    :return: (Cp, Ctau_over_l) -- normal-pressure coefficient and tangential
        coefficient divided by the tangential direction cosine l = sqrt(1-gamma^2).
    """
    gamma = np.asarray(gamma, dtype=float)
    P = np.exp(-gamma**2 * s**2) / s
    Z = 1.0 + erf_np(gamma * s)
    G = 1.0 / (2.0 * s**2)
    flux = P / np.sqrt(np.pi) + gamma * Z          # ~ 2 m nu_i / (rho V), Bird Eq. 4.22
    Cp_inc = gamma * P / np.sqrt(np.pi) + (gamma**2 + G) * Z
    Cp_re = 0.5 * np.asarray(vr_ratio) * (np.sqrt(np.pi) * gamma * Z + P)
    Ctau_over_l = flux                              # incident tangential momentum only
    return Cp_inc + Cp_re, Ctau_over_l


def erf_np(x):
    r"""Vectorized error function (numpy has no erf; avoid a scipy dependency)."""
    from math import erf
    return np.vectorize(erf)(np.asarray(x, dtype=float))


def erfc_np(x):
    r"""Vectorized COMPLEMENTARY error function via math.erfc. Numerically stable
    for large x -- computing 1 - erf(x) instead cancels catastrophically for
    x >~ 5 (deep-leeward facets), which flips the sign of the DRIA temperature-
    ratio denominator and produces negative/runaway T_r."""
    from math import erfc
    return np.vectorize(erfc)(np.asarray(x, dtype=float))


def dria_reemission_temp(gamma, s: float, T_inf: float, T_wall: float, alpha_accom: float):
    r"""
    Re-emitted-particle temperature T_r per facet from the generalized DRIA
    temperature ratio of Fuchs & Fasoulas, arXiv:2411.11597, Eq. (19) (valid for
    any speed ratio and facet orientation; erfc form for numerical robustness):

        T_r/T_i = alpha * T_w/T_i
                  + (1-alpha) * (1 + s^2/2 + (1/4) * s*g*erfc(-s*g) /
                                 [ (1/sqrt(pi)) exp(-s^2 g^2) + s*g*erfc(-s*g) ])

    with T_i = m V^2 / (2 k s^2) = T_inf (their Eq. 7). alpha=1 -> full
    accommodation (T_r = T_wall); alpha=0 -> energy-conserving re-emission.
    """
    gamma = np.asarray(gamma, dtype=float)
    sg = s * gamma
    erfc_term = erfc_np(-sg)                                   # = 1 + erf(sg), stable at |sg| >> 1
    denom = np.exp(-sg**2) / np.sqrt(np.pi) + sg * erfc_term   # analytically > 0 for all sg
    denom = np.where(denom < 1e-300, 1e-300, denom)
    Tr_over_Ti = (alpha_accom * (T_wall / T_inf)
                  + (1.0 - alpha_accom) * (1.0 + s**2 / 2.0 + 0.25 * sg * erfc_term / denom))
    return Tr_over_Ti * T_inf


def panel_aero_force_body_sentman(V_b, rho, normals, areas, alpha_accom: float = 1.0,
                                  T_wall: float = 300.0, T_inf: float = 900.0,
                                  m_mean_amu: float = 16.0) -> np.ndarray:
    r"""
    Net body-frame aerodynamic force [N] from the Sentman/DRIA facet model (see
    :func:`sentman_facet_coeffs`). Unlike the hyperthermal model, leeward facets
    receive the thermal tail of the Maxwellian (the ~1/s "thermal floor"), so
    gamma is NOT clipped. Speed ratio s = |V_b| / sqrt(2 k T_inf / m_mean).
    """
    V_b = np.asarray(V_b, dtype=float).reshape(3)
    V2 = float(V_b @ V_b)
    if rho <= 0.0 or V2 <= 0.0:
        return np.zeros(3)
    V = np.sqrt(V2)
    vhat = V_b / V
    m_kg = m_mean_amu * AMU
    c_m = np.sqrt(2.0 * K_BOLTZ * T_inf / m_kg)
    s = V / c_m
    gamma = np.asarray(normals, dtype=float) @ vhat            # SIGNED
    T_r = dria_reemission_temp(gamma, s, T_inf, T_wall, alpha_accom)
    vr_ratio = np.sqrt(2.0 * K_BOLTZ * T_r / m_kg) / V
    Cp, Ctau_over_l = sentman_facet_coeffs(gamma, s, vr_ratio)
    q = 0.5 * rho * V2
    A = np.asarray(areas, dtype=float)
    n_arr = np.asarray(normals, dtype=float)
    t_vec = vhat[None, :] - gamma[:, None] * n_arr             # l * t_hat (tangential flow dir)
    F = -(q * A * Cp)[:, None] * n_arr - (q * A * Ctau_over_l)[:, None] * t_vec
    return F.sum(axis=0)


# =========================================================================== #
# Storch (2002) finite-speed-ratio facet model (WP4/5 Phase A1; ADA410696)
# =========================================================================== #

def storch_gamma1(x):
    r"""Storch (2002) universal function Gamma_1, eq. (3.6):

        Gamma_1(x) = (1/(2 sqrt(pi))) [ exp(-x^2) + sqrt(pi) x (1 + erf x) ]

    evaluated with erfc(-x) = 1 + erf(x) for numerical stability at deep-leeward
    arguments (x << 0 -> Gamma_1 -> 0 without cancellation)."""
    x = np.asarray(x, dtype=float)
    return (np.exp(-x**2) + np.sqrt(np.pi) * x * erfc_np(-x)) / (2.0 * np.sqrt(np.pi))


def storch_gamma2(x):
    r"""Storch (2002) universal function Gamma_2, eq. (3.8), via the exact
    identity below eq. (3.14): Gamma_2(x) = x Gamma_1(x) + (1/4)(1 + erf x)."""
    x = np.asarray(x, dtype=float)
    return x * storch_gamma1(x) + 0.25 * erfc_np(-x)


def storch_facet_coeffs(gamma, s: float, sigma_n: float, sigma_t: float,
                        vw_over_va: float):
    r"""
    Per-facet force coefficients of the Storch (2002) eq. (3.9) single kernel
    (free-molecular flow at FINITE molecular speed ratio, partial normal and
    tangential momentum accommodation, diffuse re-emission at wall temperature),
    normalized by the standard dynamic pressure q = 1/2 rho V^2:

        F / (q A) = -Cn_coeff * n_hat  -  Cv_coeff * v_hat

    with n_hat the OUTWARD facet normal and v_hat the ram direction (satellite
    velocity relative to the atmosphere). With x = s * gamma (gamma = n_hat .
    v_hat, SIGNED -- leeward facets receive the thermal tail; do not clip):

        Cn_coeff = (2/s^2) [ (2 - sigma_n) Gamma_2(x) - sigma_t x Gamma_1(x)
                             + sigma_n (V_w/V_a) Gamma_1(x) ]
        Cv_coeff = (2/s)   sigma_t Gamma_1(x)

    Hyperthermal limit (s -> inf, V_w/V_a fixed): Cn_coeff -> 2(2-sigma_n-sigma_t)
    gamma^2 + 2 sigma_n (V_w/V) gamma and Cv_coeff -> 2 sigma_t gamma (windward),
    i.e. exactly the legacy two-term model with Cn = 2(2-sigma_n), Ct = 2 sigma_t
    plus the re-emission term -- the (0.90, 0.15) row reproduces the legacy
    (2.2, 0.3) coefficients.

    :param gamma: SIGNED incidence cosine n_hat . v_hat per facet.
    :param s: molecular speed ratio |V| / sqrt(2 k T_inf / m)  (Storch's S).
    :param sigma_n: normal momentum accommodation coefficient, eq. (2.1).
    :param sigma_t: tangential momentum accommodation coefficient, eq. (2.1).
    :param vw_over_va: V_w / V_a with V_w = sqrt(pi k T_wall / (2 m)) the mean
        normal re-emission speed (eq. 2.2) and V_a = sqrt(2 k T_inf / m) the
        atmospheric most-probable speed. Note V_w/V = vw_over_va / s.
    """
    gamma = np.asarray(gamma, dtype=float)
    x = s * gamma
    g1 = storch_gamma1(x)
    g2 = storch_gamma2(x)
    Cn_coeff = (2.0 / s**2) * ((2.0 - sigma_n) * g2 - sigma_t * x * g1
                               + sigma_n * vw_over_va * g1)
    Cv_coeff = (2.0 / s) * sigma_t * g1
    return Cn_coeff, Cv_coeff


def panel_aero_force_body_storch(V_b, rho, normals, areas, sigma_n: float = 0.9,
                                 sigma_t: float = 0.7, T_wall: float = 300.0,
                                 T_inf: float = 900.0,
                                 m_mean_amu: float = 16.0) -> np.ndarray:
    r"""
    Net body-frame aerodynamic force [N] from the Storch eq. (3.9) facet kernel
    (see :func:`storch_facet_coeffs`). Like Sentman, leeward facets receive the
    Maxwellian thermal tail, so the incidence cosine is NOT clipped.
    """
    V_b = np.asarray(V_b, dtype=float).reshape(3)
    V2 = float(V_b @ V_b)
    if rho <= 0.0 or V2 <= 0.0:
        return np.zeros(3)
    V = np.sqrt(V2)
    vhat = V_b / V
    m_kg = m_mean_amu * AMU
    v_a = np.sqrt(2.0 * K_BOLTZ * T_inf / m_kg)
    v_w = np.sqrt(np.pi * K_BOLTZ * T_wall / (2.0 * m_kg))
    s = V / v_a
    n_arr = np.asarray(normals, dtype=float)
    gamma = n_arr @ vhat                                       # SIGNED
    Cn_coeff, Cv_coeff = storch_facet_coeffs(gamma, s, sigma_n, sigma_t, v_w / v_a)
    q = 0.5 * rho * V2
    A = np.asarray(areas, dtype=float)
    F = -(q * A * Cn_coeff)[:, None] * n_arr - (q * A * Cv_coeff)[:, None] * vhat[None, :]
    return F.sum(axis=0)


def facet_aero_forces(V_b, rho, normals, areas, mode: str = "hyperthermal_faceted",
                      Cn: float = 2.2, Ct: float = 0.3, sigma_n: float = 0.9,
                      sigma_t: float = 0.7, T_wall: float = 300.0,
                      T_inf: float = 900.0, m_mean_amu: float = 16.0) -> np.ndarray:
    r"""
    PER-FACET aerodynamic force vectors [N], shape ``(M, 3)`` -- the moment-path
    twin of the net-force routines (which are kept untouched for bit-for-bit
    legacy reproducibility). Facets here are ONE-SIDED: in hyperthermal mode a
    leeward facet (incidence cosine <= 0) contributes zero; in storch mode it
    receives the Maxwellian thermal tail.
    """
    V_b = np.asarray(V_b, dtype=float).reshape(3)
    V2 = float(V_b @ V_b)
    n_arr = np.asarray(normals, dtype=float)
    A = np.asarray(areas, dtype=float)
    if rho <= 0.0 or V2 <= 0.0:
        return np.zeros((n_arr.shape[0], 3))
    V = np.sqrt(V2)
    vhat = V_b / V
    q = 0.5 * rho * V2
    gamma = n_arr @ vhat
    if mode == "storch":
        m_kg = m_mean_amu * AMU
        v_a = np.sqrt(2.0 * K_BOLTZ * T_inf / m_kg)
        v_w = np.sqrt(np.pi * K_BOLTZ * T_wall / (2.0 * m_kg))
        Cn_c, Cv_c = storch_facet_coeffs(gamma, V / v_a, sigma_n, sigma_t, v_w / v_a)
        return (-(q * A * Cn_c)[:, None] * n_arr
                - (q * A * Cv_c)[:, None] * vhat[None, :])
    if mode != "hyperthermal_faceted":
        raise ValueError(f"unknown mode {mode!r} for facet_aero_forces")
    c = np.where(gamma > 0.0, gamma, 0.0)
    scale = q * A * c
    return (-(Cn * scale * c)[:, None] * n_arr
            - (Ct * scale)[:, None] * (vhat[None, :] - c[:, None] * n_arr))


def facet_aero_force_moment(V_b, rho, normals, areas, centroids, about=None,
                            **kwargs):
    r"""
    Net aerodynamic force [N] and moment [N m] about ``about`` (default: the
    origin of the centroid frame) for one-sided facets with centroids:
    ``M = sum_i (r_i - about) x F_i``. Kernel selected via ``mode`` (see
    :func:`facet_aero_forces`).
    """
    F_per = facet_aero_forces(V_b, rho, normals, areas, **kwargs)
    r = np.asarray(centroids, dtype=float)
    if about is not None:
        r = r - np.asarray(about, dtype=float)[None, :]
    return F_per.sum(axis=0), np.cross(r, F_per).sum(axis=0)


class AeroModel:
    r"""
    Attitude-dependent orbital aerodynamic force (drag + lift) for a satellite.

    Wraps a free-molecular panel model (see :func:`panel_aero_force_body`) built
    from the satellite surface geometry. Because the force depends on the
    satellite's attitude (how each panel is oriented into the flow), it couples
    the orbit to attitude and must be evaluated in the simulation loop — see
    :class:`~ADCS.formation.constellation.Constellation` (operator-split
    co-integration). This model produces the orbital **force** only; attitude
    aerodynamic torque remains the responsibility of
    :class:`~ADCS.satellite_hardware.disturbances.drag_disturbance.Drag_Disturbance`.

    :param normals: Unit outward panel normals, shape ``(M, 3)``.
    :param areas: Panel areas [m^2], shape ``(M,)``.
    :param Cn: Normal (pressure) coefficient (default 2.0).
    :param Ct: Tangential (shear) coefficient (default 0.0). ``Cn == Ct`` gives a
        lift-free (drag-only) model.
    :param mode: ``"hyperthermal_faceted"`` (DEFAULT -- the legacy Cn/Ct two-term
        panel model, preserved bit-for-bit), ``"sentman"`` (Sentman/DRIA facet
        model with thermal-tail leeward flux; see
        :func:`panel_aero_force_body_sentman`) or ``"storch"`` (Storch 2002
        eq. 3.9 finite-speed-ratio kernel with (sigma_n, sigma_t) momentum
        accommodation; see :func:`storch_facet_coeffs`). Sentman ignores Cn/Ct
        and uses ``alpha_accom``/``T_wall``; storch ignores Cn/Ct and uses
        ``sigma_n``/``sigma_t``/``T_wall``; both take per-call
        ``T_inf``/``m_mean_amu``.
    :param alpha_accom: DRIA energy-accommodation coefficient (sentman mode).
    :param T_wall: Surface temperature [K] (sentman/storch modes; default 300).
    :param sigma_n: normal momentum accommodation (storch mode; baseline 0.90).
    :param sigma_t: tangential momentum accommodation (storch mode; baseline
        0.70 == Ct 1.4; the specular-edge row 0.15 reproduces the legacy 0.3).
    """

    def __init__(self, normals, areas, Cn: float = 2.0, Ct: float = 0.0,
                 mode: str = "hyperthermal_faceted", alpha_accom: float = 1.0,
                 T_wall: float = 300.0, sigma_n: float = 0.9,
                 sigma_t: float = 0.7) -> None:
        self.normals = np.vstack([normalize(np.asarray(n, dtype=float)) for n in normals])
        self.areas = np.asarray(areas, dtype=float).reshape(-1)
        self.Cn = float(Cn)
        self.Ct = float(Ct)
        if mode not in ("hyperthermal_faceted", "sentman", "storch"):
            raise ValueError(f"unknown aero mode {mode!r}")
        self.mode = mode
        self.alpha_accom = float(alpha_accom)
        self.T_wall = float(T_wall)
        self.sigma_n = float(sigma_n)
        self.sigma_t = float(sigma_t)

    @classmethod
    def from_geometry(cls, config: GeometryConfig, Cn: float = 2.0, Ct: float = 0.0) -> "AeroModel":
        r"""
        Build an :class:`AeroModel` from a satellite :class:`GeometryConfig`
        (the same per-face geometry used by the drag-torque disturbance).
        """
        params = config.params
        normals = [p["normal"] for p in params]
        areas = [p["area"] for p in params]
        return cls(normals=normals, areas=areas, Cn=Cn, Ct=Ct)

    def force_body(self, V_b, rho, T_inf: float = 900.0, m_mean_amu: float = 16.0) -> np.ndarray:
        r"""
        Net body-frame aerodynamic force [N].

        :param V_b: Body-frame relative wind velocity [m/s], shape ``(3,)``.
        :param rho: Atmospheric density [kg/m^3].
        :param T_inf: Exospheric temperature [K] (sentman mode only; see
            :func:`exospheric_temperature` for the solar-level table).
        :param m_mean_amu: Mean molecular mass [amu] (sentman mode only).
        """
        if self.mode == "sentman":
            return panel_aero_force_body_sentman(V_b, rho, self.normals, self.areas,
                                                 self.alpha_accom, self.T_wall,
                                                 T_inf, m_mean_amu)
        if self.mode == "storch":
            return panel_aero_force_body_storch(V_b, rho, self.normals, self.areas,
                                                self.sigma_n, self.sigma_t,
                                                self.T_wall, T_inf, m_mean_amu)
        return panel_aero_force_body(V_b, rho, self.normals, self.areas, self.Cn, self.Ct)
