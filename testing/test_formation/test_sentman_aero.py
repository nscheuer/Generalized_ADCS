r"""
Sentman/DRIA facet model validation gates (WP2 Task 4.1) -- standalone single
plate, run BEFORE any Phase-2 test: (a) broadside C_D in [2.0, 2.5] at alpha=1;
(b) grazing gain dC_D/dphi = 2 erf(s sin phi) cos phi within 5% for phi <= 5 deg;
(c) hyperthermal recovery at s = 50 (approaches the faceted/geometric limit).
"""
import numpy as np
import pytest

from ADCS.satellite_hardware.aero.aero_force import (
    panel_aero_force_body_sentman, panel_aero_force_body, sentman_facet_coeffs,
    dria_reemission_temp, exospheric_temperature, K_BOLTZ, AMU,
)
from math import erf

T_INF, T_WALL, M_AMU = 900.0, 300.0, 16.0
V = 7500.0
RHO = 1e-12
S = V / np.sqrt(2.0 * K_BOLTZ * T_INF / (M_AMU * AMU))     # ~7.5 at 900 K


def _plate(phi, alpha=1.0, s_speed=None, area=1.0):
    r"""Two-sided unit plate at incidence phi to the wind; returns C_D per plate
    area (force along -wind / (q*A))."""
    v = V if s_speed is None else s_speed * np.sqrt(2.0 * K_BOLTZ * T_INF / (M_AMU * AMU))
    vb = v * np.array([np.sin(phi), 0.0, np.cos(phi)])      # wind at angle phi to the +x normal
    normals = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
    F = panel_aero_force_body_sentman(vb, RHO, normals, [area, area],
                                      alpha_accom=alpha, T_wall=T_WALL, T_inf=T_INF,
                                      m_mean_amu=M_AMU)
    q = 0.5 * RHO * v**2
    return float(F @ (-vb / v)) / (q * area)


def test_gate_a_broadside_cd():
    cd = _plate(np.pi / 2, alpha=1.0)                        # wind along the normal
    assert 2.0 <= cd <= 2.5, cd


def test_gate_b_grazing_gain_matches_erf():
    # dC_D/dphi vs 2 erf(s sin phi) cos phi, within 5% for phi <= 5 deg
    for phi_deg in (1.0, 2.0, 3.0, 5.0):
        phi = np.radians(phi_deg)
        eps = 1e-5
        dcd = (_plate(phi + eps) - _plate(phi - eps)) / (2 * eps)
        gate = 2.0 * erf(S * np.sin(phi)) * np.cos(phi)
        assert abs(dcd / gate - 1.0) < 0.05, (phi_deg, dcd, gate)


def test_gate_c_hyperthermal_recovery_at_s50():
    # s = 50: thermal terms die; C_D approaches the geometric limit 2*sin(phi)
    # (per plate area, alpha=1, modulo the small re-emission term ~ vr ~ 1/s0).
    for phi_deg in (10.0, 30.0, 90.0):
        phi = np.radians(phi_deg)
        cd = _plate(phi, s_speed=50.0)
        geo = 2.0 * np.sin(phi)                              # full-accommodation geometric limit
        assert abs(cd / geo - 1.0) < 0.10, (phi_deg, cd, geo)


def test_leeward_thermal_floor_and_edge_on():
    # edge-on plate (gamma = 0 both faces): normal forces cancel; the tangential
    # thermal floor is 2*P/sqrt(pi) = 2/(s sqrt(pi)) per plate area (both faces).
    cd0 = _plate(0.0)
    assert np.isclose(cd0, 2.0 / (S * np.sqrt(np.pi)), rtol=0.15)
    assert cd0 > 0.1                                         # ~0.15: a REAL floor (vs ~0 hyperthermal)


def test_dria_alpha_limits():
    # alpha=1 -> T_r = T_wall; alpha=0, hyperthermal flow-facing -> T_r -> T_i*s^2/2 (energy conserving)
    Tr1 = dria_reemission_temp(1.0, S, T_INF, T_WALL, 1.0)
    assert np.isclose(Tr1, T_WALL, rtol=1e-9)
    Tr0 = dria_reemission_temp(1.0, S, T_INF, T_WALL, 0.0)
    assert np.isclose(Tr0, T_INF * (1 + S**2 / 2 + 0.25), rtol=0.01)   # Eq. 19 limit at large s
    # lower alpha -> hotter re-emission -> more re-emission pressure -> higher broadside C_D
    assert _plate(np.pi / 2, alpha=0.7) > _plate(np.pi / 2, alpha=1.0)


def test_dria_temperature_stable_for_all_incidences():
    # regression: erfc built as 1-erf cancels catastrophically for |s*gamma| >~ 5
    # (deep leeward), flipping the denominator sign -> negative/runaway T_r.
    gammas = np.linspace(-1.0, 1.0, 401)
    for alpha in (0.0, 0.5, 0.7, 1.0):
        Tr = dria_reemission_temp(gammas, S, T_INF, T_WALL, alpha)
        assert np.all(np.isfinite(Tr)) and np.all(Tr > 0.0)
        assert np.all(Tr < T_INF * (1 + S**2 / 2 + 1.0) + 1.0)   # bounded by the energy limit


def test_exospheric_temperature_table():
    assert exospheric_temperature(0.0) == 700.0
    assert exospheric_temperature(0.5) == 900.0
    assert exospheric_temperature(1.0) == 1100.0
    assert exospheric_temperature(0.25) == 800.0             # linear interp


def test_a3_s_inf_limit_matches_three_term_hyperthermal():
    r"""WP4/5 A3: the s -> inf limit of the Sentman module vs the THREE-TERM
    hyperthermal form (Storch eq. 2.9 with sigma_n = sigma_t = 1: normal
    pressure + shear + diffuse re-emission at the DRIA T_r), per plate area:

        C_D = 2 sin(psi) [ 1 + (V_w/V) sin(psi) ],   V_w = sqrt(pi k T_r / 2 m)

    Gate 2% at psi in {2, 5, 10} deg. MEASURED closure (for report_wp45): the
    old gate-c 10% slack vs the bare geometric 2 sin(psi) was conservative --
    actual bare-geometric residuals are 0.24/0.11/0.20% at s = 50; the
    three-term form + s = 500 closes to < 1e-4 (re-emission term removes a
    5-10x bias at psi >= 5 deg, erf saturation does the rest)."""
    s_lim = 500.0
    v = s_lim * np.sqrt(2.0 * K_BOLTZ * T_INF / (M_AMU * AMU))
    for psi_deg in (2.0, 5.0, 10.0):
        psi = np.radians(psi_deg)
        cd = _plate(psi, alpha=1.0, s_speed=s_lim)
        # alpha = 1 -> T_r = T_wall; mean normal re-emission speed (Storch 2.2)
        v_w = np.sqrt(np.pi * K_BOLTZ * T_WALL / (2.0 * M_AMU * AMU))
        cd_ref = 2.0 * np.sin(psi) * (1.0 + (v_w / v) * np.sin(psi))
        assert abs(cd / cd_ref - 1.0) < 0.02, (psi_deg, cd, cd_ref)
        # in fact the closure is ~1e-5; keep the spec's 2% as the formal gate
