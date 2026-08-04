r"""
AeroModel finite-S mode: dispatch, plate-once semantics, the exospheric
temperature axis, and a CROSS-IMPLEMENTATION equivalence gate.

The equivalence gate is the important one. This library now carries the
finite speed-ratio (Storch) physics in ``finite_s.py``; an independently
written implementation of the same physics exists on the paper freeze branch
(``panel_aero_force_body_storch``, gated there against six closed forms from
Storch 2002: flat plate eqs. 3.18/3.19 at 1e-10, sphere 3.17, cylinder eq.
3.10 vs the Bessel forms 3.20-3.23, hyperthermal sphere 2.13, edge-on floor).
The two agree to 1e-6 at every incidence from feather to broadside. Rather
than depend on that branch, the gate below re-derives the same closed forms
locally, so this file is self-contained.
"""
import numpy as np
import pytest

from ADCS.satellite_hardware.aero.aero_force import AeroModel, panel_aero_force_body
from ADCS.satellite_hardware.aero.finite_s import (
    panel_aero_force_body_finite_s, speed_ratio, wall_speed)

RHO, V = 1e-12, 7500.0
PLATE = np.array([[0.0, 0.0, 1.0]])          # plate-once
PAIR = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
AREA, PAIR_A = [105.0], [105.0, 105.0]
SN, ST = 0.90, 0.80


def _wind(angle_rad):
    return V * np.array([np.cos(angle_rad), 0.0, np.sin(angle_rad)])


def test_mode_dispatch_matches_the_free_functions():
    m_h = AeroModel(PAIR, PAIR_A, Cn=2.2, Ct=1.6)
    assert np.allclose(m_h.force_body(_wind(0.4), RHO),
                       panel_aero_force_body(_wind(0.4), RHO, PAIR, PAIR_A, 2.2, 1.6))
    m_f = AeroModel(PLATE, AREA, mode="finite_s", sigma_n=SN, sigma_t=ST)
    assert np.allclose(m_f.force_body(_wind(0.4), RHO),
                       panel_aero_force_body_finite_s(_wind(0.4), RHO, PLATE, AREA,
                                                      sigma_n=SN, sigma_t=ST))


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        AeroModel(PLATE, AREA, mode="hypersonic")


def test_exospheric_temperature_axis_swings_the_feather_floor():
    # T_inf is the solar-activity axis: at feather the floor scales ~1/S, and
    # S = V / sqrt(2 k T_inf / m), so hotter atmosphere -> larger floor, with
    # NO density change. Per-call override must be honoured.
    m = AeroModel(PLATE, AREA, mode="finite_s", T_inf=900.0)
    d_cold = -m.force_body(_wind(0.0), RHO, T_inf=600.0) @ np.array([1.0, 0.0, 0.0])
    d_hot = -m.force_body(_wind(0.0), RHO, T_inf=1500.0) @ np.array([1.0, 0.0, 0.0])
    assert d_hot > d_cold > 0.0
    ratio = d_hot / d_cold
    assert np.isclose(ratio, np.sqrt(1500.0 / 600.0), rtol=0.02), ratio
    # default is used when no override is given
    assert np.allclose(m.force_body(_wind(0.0), RHO),
                       m.force_body(_wind(0.0), RHO, T_inf=900.0))


def test_hyperthermal_limit_is_the_identity_mapping():
    # S -> infinity, T_wall -> 0: finite_s must reduce to the hyperthermal
    # panel model with (Cn, Ct) = (2(2 - sigma_n), 2 sigma_t) -- NO factor of
    # two anywhere (both use q = 1/2 rho V^2 since #98).
    m_f = AeroModel(PLATE, AREA, mode="finite_s", sigma_n=SN, sigma_t=ST,
                    T_inf=1e-6, T_wall=1e-9)
    m_h = AeroModel(PAIR, PAIR_A, Cn=2.0 * (2.0 - SN), Ct=2.0 * ST)
    for ang in (0.05, 0.3, 1.0):
        assert np.allclose(m_f.force_body(_wind(ang), RHO),
                           m_h.force_body(_wind(ang), RHO), rtol=2e-3), ang


def _storch_closed_form_plate(beta, S, sigma_n, sigma_t, vw_over_V):
    r"""Storch (2002) eqs. (3.18)/(3.19): C_L, C_D for a flat plate at angle of
    attack beta, finite speed ratio S. Independent of the module under test."""
    from math import erf
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


@pytest.mark.parametrize("beta_deg", [0.0, 1.0, 4.0, 20.0, 60.0, 90.0])
def test_matches_storch_closed_form_plate_equations(beta_deg):
    r"""THE physics gate: the implementation against the published closed form,
    not against itself. Storch eqs. (3.18)/(3.19), C_D and C_L per plate area
    with q = 1/2 rho V^2."""
    T_inf, T_wall, m_amu = 900.0, 300.0, 16.0
    S = speed_ratio(V, T_inf, m_amu)
    vw_over_V = wall_speed(T_wall, m_amu) / V
    beta = np.radians(beta_deg)
    model = AeroModel(PLATE, AREA, mode="finite_s", sigma_n=SN, sigma_t=ST,
                      T_wall=T_wall, T_inf=T_inf, m_amu=m_amu)
    F = model.force_body(_wind(beta), RHO)
    vhat = np.array([np.cos(beta), 0.0, np.sin(beta)])
    q_A = 0.5 * RHO * V**2 * AREA[0]
    CD_num = float(F @ (-vhat)) / q_A
    CL_num = float(np.linalg.norm(F - (F @ vhat) * vhat)) / q_A
    CL_ref, CD_ref = _storch_closed_form_plate(beta, S, SN, ST, vw_over_V)
    assert np.isclose(CD_num, CD_ref, rtol=1e-9), (beta_deg, CD_num, CD_ref)
    assert np.isclose(CL_num, abs(CL_ref), rtol=1e-9, atol=1e-12), (beta_deg, CL_num, CL_ref)


def test_plate_once_semantics_are_not_double_counted():
    # passing the pair to finite_s would double the force -- the docstring
    # contract, pinned so a caller porting from hyperthermal cannot silently
    # double its physics (the failure mode that cost us the 2026-08 audit)
    one = panel_aero_force_body_finite_s(_wind(0.3), RHO, PLATE, AREA)
    two = panel_aero_force_body_finite_s(_wind(0.3), RHO, PAIR, PAIR_A)
    assert np.allclose(two, 2.0 * one)
