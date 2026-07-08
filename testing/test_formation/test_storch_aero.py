r"""
Closed-form gates for the Storch (2002, ADA410696) eq. (3.9) finite-speed-ratio
facet kernel (WP4/5 Phase A1). Every gate is against a closed form printed in
the report:

1. Flat plate at angle of attack: two-facet kernel sum == eqs. (3.18)/(3.19)
   (C_L and C_D, both faces, all three physical terms exercised).
2. Sphere: Gauss-Legendre quadrature of the kernel == eq. (3.17); hyperthermal
   sphere eq. (2.13) recovered as S -> inf.
3. Cylinder (lateral surface): theta-quadrature of the kernel == eq. (3.10)
   assembled from the Bessel-function closed forms (3.20)-(3.23) -- an
   independent erf/exp-vs-Bessel check on a curved, signed-incidence surface.
4. Hyperthermal/legacy limit: at S -> inf with V_w = 0 the kernel reproduces
   the legacy two-term panel model with Cn = 2(2-sigma_n), Ct = 2 sigma_t;
   the specular-edge row (0.90, 0.15) == the legacy (2.2, 0.3) coefficients.
5. Edge-on plate: nonzero finite-S drag 2 sigma_t / (sqrt(pi) S) (the thermal
   floor the hyperthermal model misses entirely).
"""
import numpy as np
import pytest
from scipy.special import erf, i0, i1

from ADCS.satellite_hardware.aero.aero_force import (
    AeroModel, K_BOLTZ, AMU, panel_aero_force_body, panel_aero_force_body_storch,
    storch_facet_coeffs, storch_gamma1, storch_gamma2)

# baseline / specular-edge / full-accommodation coefficient rows (WP4/5 A1)
ROWS = [(0.90, 0.70), (0.90, 0.15), (1.00, 1.00)]

# canonical thermosphere numbers: T_inf = 900 K, T_wall = 300 K, m = 16 amu
M_KG = 16.0 * AMU
V_A = np.sqrt(2.0 * K_BOLTZ * 900.0 / M_KG)
V_W = np.sqrt(np.pi * K_BOLTZ * 300.0 / (2.0 * M_KG))


def test_gamma_functions_and_identity():
    # compare against the report's literal (1 + erf x) forms only where THEY are
    # accurate -- for x <~ -4 the naive form cancels catastrophically (drops the
    # erf tail entirely by x = -8) while the erfc implementation stays exact
    x = np.linspace(-4.0, 8.0, 121)
    g1 = (np.exp(-x**2) + np.sqrt(np.pi) * x * (1.0 + erf(x))) / (2.0 * np.sqrt(np.pi))
    assert np.allclose(storch_gamma1(x), g1, rtol=2e-6, atol=1e-300)
    # identity below eq. (3.14): Gamma_2 = x Gamma_1 + (1/4)(1 + erf x)
    g2_direct = (x * np.exp(-x**2) + (np.sqrt(np.pi) / 2.0) * (1.0 + 2.0 * x**2)
                 * (1.0 + erf(x))) / (2.0 * np.sqrt(np.pi))
    assert np.allclose(storch_gamma2(x), g2_direct, rtol=2e-6, atol=1e-300)
    # deep leeward: the erfc form recovers the true asymptote
    # Gamma_1(x) ~ e^{-x^2}/(2 sqrt(pi)) * (1/(2x^2) - 3/(4x^4) + ...) instead of
    # the naive form's spurious e^{-x^2}/(2 sqrt(pi))
    for xx in (-6.0, -8.0, -10.0):
        asym = np.exp(-xx**2) / (2.0 * np.sqrt(np.pi)) * (0.5 / xx**2 - 0.75 / xx**4)
        assert np.isclose(float(storch_gamma1(xx)), asym, rtol=5e-2)
    assert storch_gamma1(-40.0) == 0.0 and storch_gamma2(-40.0) == 0.0
    # Gamma_1(x) - Gamma_1(-x) = x (relation on p.24)
    xs = np.array([0.3, 1.7, 4.0])
    assert np.allclose(storch_gamma1(xs) - storch_gamma1(-xs), xs, rtol=1e-12)


def _plate_closed_form(beta, S, sigma_n, sigma_t, vw_over_V):
    r"""Eqs. (3.18)-(3.19): C_L, C_D for a flat plate at angle of attack beta."""
    sb, cb = np.sin(beta), np.cos(beta)
    mu = S * sb
    CL = (((2.0 - sigma_n - sigma_t) / (np.sqrt(np.pi) * S) * np.exp(-mu**2)
           + sigma_n * vw_over_V) * np.sin(2.0 * beta)
          + ((2.0 - sigma_n) / S**2 + 2.0 * (2.0 - sigma_n - sigma_t) * sb**2)
          * cb * erf(mu))
    CD = (2.0 * sigma_n * vw_over_V * sb**2
          + 2.0 / (np.sqrt(np.pi) * S) * ((2.0 - sigma_n) * sb**2
                                          + sigma_t * cb**2) * np.exp(-mu**2)
          + 2.0 * ((2.0 - sigma_n) * (sb**2 + 1.0 / (2.0 * S**2))
                   + sigma_t * cb**2) * sb * erf(mu))
    return CL, CD


@pytest.mark.parametrize("sigma_n,sigma_t", ROWS)
@pytest.mark.parametrize("S", [2.0, 5.0, 7.79])
@pytest.mark.parametrize("beta_deg", [2.0, 10.0, 35.0, 60.0, 90.0])
def test_flat_plate_matches_eqs_318_319(sigma_n, sigma_t, S, beta_deg):
    beta = np.radians(beta_deg)
    V = S * V_A
    vhat = np.array([1.0, 0.0, 0.0])            # ram direction (our convention)
    n = np.array([np.sin(beta), np.cos(beta), 0.0])
    rho, A = 1e-12, 3.7
    F = panel_aero_force_body_storch(V * vhat, rho, [n, -n], [A, A],
                                     sigma_n, sigma_t, 300.0, 900.0, 16.0)
    q = 0.5 * rho * V**2
    CD_num = float(F @ (-vhat)) / (q * A)
    CL_num = float(np.linalg.norm(F - (F @ vhat) * vhat)) / (q * A)
    CL_ref, CD_ref = _plate_closed_form(beta, S, sigma_n, sigma_t, (V_W / V_A) / S)
    assert np.isclose(CD_num, CD_ref, rtol=1e-10), (CD_num, CD_ref)
    assert np.isclose(CL_num, abs(CL_ref), rtol=1e-10, atol=1e-14), (CL_num, CL_ref)


@pytest.mark.parametrize("sigma_n,sigma_t", ROWS)
@pytest.mark.parametrize("S", [2.0, 5.0, 7.79])
def test_sphere_matches_eq_317(sigma_n, sigma_t, S):
    # C_D = (2-sigma_n+sigma_t)/(2 S^3) [ (4S^4+4S^2-1)/(2S) erf S
    #        + (2S^2+1)/sqrt(pi) e^{-S^2} ] + (4/3) sigma_n V_w/V
    vw_over_V = (V_W / V_A) / S
    CD_ref = ((2.0 - sigma_n + sigma_t) / (2.0 * S**3)
              * ((4.0 * S**4 + 4.0 * S**2 - 1.0) / (2.0 * S) * erf(S)
                 + (2.0 * S**2 + 1.0) / np.sqrt(np.pi) * np.exp(-S**2))
              + 4.0 / 3.0 * sigma_n * vw_over_V)
    # quadrature of the facet kernel over the sphere: gamma = u = cos(theta)
    u, w = np.polynomial.legendre.leggauss(240)
    Cn_c, Cv_c = storch_facet_coeffs(u, S, sigma_n, sigma_t, V_W / V_A)
    # F_z / (q pi a^2) = 2 * integral of (-Cn_coeff u - Cv_coeff) du
    CD_num = -2.0 * float(np.sum(w * (-Cn_c * u - Cv_c)))
    assert np.isclose(CD_num, CD_ref, rtol=1e-9), (CD_num, CD_ref)


def test_sphere_hyperthermal_limit_eq_213():
    # D = 1/2 pi a^2 rho V^2 (2 + sigma_t - sigma_n + (4/3) sigma_n Vw/V)
    sigma_n, sigma_t, S = 0.9, 0.7, 1e5
    vw_over_V = (V_W / V_A) / S
    u, w = np.polynomial.legendre.leggauss(400)
    Cn_c, Cv_c = storch_facet_coeffs(u, S, sigma_n, sigma_t, V_W / V_A)
    CD_num = -2.0 * float(np.sum(w * (-Cn_c * u - Cv_c)))
    CD_ref = 2.0 + sigma_t - sigma_n + 4.0 / 3.0 * sigma_n * vw_over_V
    assert np.isclose(CD_num, CD_ref, rtol=5e-4), (CD_num, CD_ref)


@pytest.mark.parametrize("sigma_n,sigma_t", ROWS)
@pytest.mark.parametrize("S", [2.0, 7.79])
@pytest.mark.parametrize("beta_deg", [15.0, 45.0, 80.0])
def test_cylinder_matches_bessel_closed_forms(sigma_n, sigma_t, S, beta_deg):
    r"""Lateral cylinder surface: kernel theta-quadrature == eq. (3.10) with the
    surface integrals replaced by the closed forms (3.20)-(3.23). Storch frame:
    z along the cylinder axis, incident flow vhat_storch = sin(beta) i +
    cos(beta) k, inner normal n_in = -cos(t) i - sin(t) j, cos(alpha) =
    -sin(beta) cos(t). a, l are radius and length."""
    beta = np.radians(beta_deg)
    a, ell = 0.8, 2.5
    mu = S * np.sin(beta)
    lam = mu**2 / 2.0
    F1 = np.sqrt(np.pi) * a * ell * np.exp(-lam) * ((1.0 + mu**2) * i0(lam)
                                                    + mu**2 * i1(lam))
    F2 = (np.sqrt(np.pi) / 3.0) * a * ell * mu * np.exp(-lam) * (
        (2.0 * mu**2 + 3.0) * i0(lam) + (2.0 * mu**2 + 1.0) * i1(lam))
    F3 = (np.sqrt(np.pi) / 6.0) * a * ell * (mu / S) * np.exp(-lam) * (
        (3.0 + 4.0 * mu**2) * i0(lam) + (4.0 * mu**2 - 1.0) * i1(lam))
    F4 = (np.pi / 2.0) * a * ell * mu
    vhat_storch = np.array([np.sin(beta), 0.0, np.cos(beta)])
    xhat = np.array([1.0, 0.0, 0.0])
    vw_over_V = (V_W / V_A) / S
    # eq. (3.10); F2, F3, F4 are along +i by the report's symmetry reduction
    f_ref = ((sigma_t / S) * F1 * vhat_storch
             + ((2.0 - sigma_n) / S**2 * F2 - sigma_t / S * F3
                + sigma_n / S * vw_over_V * F4) * xhat)          # units of rho V^2

    # kernel quadrature: our convention vhat_ram = -vhat_storch, n_out = -n_in
    th = np.linspace(0.0, 2.0 * np.pi, 4001)[:-1]
    dth = 2.0 * np.pi / 4000.0
    n_out = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], axis=1)
    gamma = n_out @ (-vhat_storch)
    Cn_c, Cv_c = storch_facet_coeffs(gamma, S, sigma_n, sigma_t, V_W / V_A)
    dA = a * ell * dth
    F_num = (-(Cn_c * dA)[:, None] * n_out
             - (Cv_c * dA)[:, None] * (-vhat_storch)[None, :]).sum(axis=0)
    # F_num is per q = 1/2 rho V^2; f_ref is per rho V^2
    assert np.allclose(F_num / 2.0, f_ref, rtol=5e-7, atol=1e-12 * a * ell), \
        (F_num / 2.0, f_ref)


def test_hyperthermal_limit_reproduces_legacy_two_term_model():
    r"""S -> inf, V_w = 0: the (0.90, 0.15) specular-edge row == the legacy
    panel model with (Cn, Ct) = (2.2, 0.3) -- the legacy flag equivalence."""
    rng = np.random.default_rng(7)
    normals = rng.normal(size=(6, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    areas = rng.uniform(0.5, 40.0, size=6)
    V_b = 7530.0 * np.array([0.9, -0.32, 0.28]) / np.linalg.norm([0.9, -0.32, 0.28])
    rho = 3e-13
    F_legacy = panel_aero_force_body(V_b, rho, normals, areas, Cn=2.2, Ct=0.3)
    # S -> inf via tiny T_inf; V_w = 0 via T_wall = 0
    F_storch = panel_aero_force_body_storch(V_b, rho, normals, areas,
                                            sigma_n=0.9, sigma_t=0.15,
                                            T_wall=0.0, T_inf=1e-4, m_mean_amu=16.0)
    assert np.allclose(F_storch, F_legacy, rtol=1e-7, atol=1e-16)


def test_edge_on_plate_thermal_floor():
    # beta = 0: C_D = 2 sigma_t / (sqrt(pi) S) (finite-S; hyperthermal gives 0)
    for sigma_n, sigma_t in ROWS:
        for S in (2.0, 7.79):
            V = S * V_A
            vhat = np.array([1.0, 0.0, 0.0])
            n = np.array([0.0, 1.0, 0.0])
            rho, A = 1e-12, 5.0
            F = panel_aero_force_body_storch(V * vhat, rho, [n, -n], [A, A],
                                             sigma_n, sigma_t, 300.0, 900.0, 16.0)
            CD = float(F @ (-vhat)) / (0.5 * rho * V**2 * A)
            assert np.isclose(CD, 2.0 * sigma_t / (np.sqrt(np.pi) * S), rtol=1e-10)
            # normal components cancel exactly by the +/- pair symmetry
            assert abs(F[1]) < 1e-25 and abs(F[2]) < 1e-25


def test_aeromodel_storch_mode_dispatch():
    m = AeroModel(normals=[[0, 0, 1], [0, 0, -1]], areas=[10.0, 10.0],
                  mode="storch", sigma_n=0.9, sigma_t=0.7)
    V_b = np.array([0.0, 0.0, 7530.0])
    F = m.force_body(V_b, 1e-12, T_inf=900.0, m_mean_amu=16.0)
    F_ref = panel_aero_force_body_storch(V_b, 1e-12, m.normals, m.areas,
                                         0.9, 0.7, 300.0, 900.0, 16.0)
    assert np.allclose(F, F_ref, rtol=0, atol=0)
