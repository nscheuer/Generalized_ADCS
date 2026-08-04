import numpy as np
import pytest

from ADCS.satellite_hardware.aero.aero_force import panel_aero_force_body
from ADCS.satellite_hardware.aero.finite_s import (
    finite_s_coefficients,
    panel_aero_force_body_finite_s,
    speed_ratio,
)

PLATE = np.array([[0.0, 0.0, 1.0]])       # one entry per plate
AREA = np.array([105.0])
PAIR = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])   # hyperthermal
PAIR_A = np.array([105.0, 105.0])          # model wants face pairs
RHO = 2e-14
V = 7602.3


def _wind(angle):
    return V * np.array([np.cos(angle), 0.0, np.sin(angle)])


def test_hyperthermal_limit_matches_panel_model():
    # S -> infinity (T_inf -> 0) must recover the classical plate with
    # Cn = 2(2 - sigma_n), Ct = 2 sigma_t and no re-emission (T_wall -> 0).
    # Both functions use the standard q = 1/2 rho V^2 (identity mapping;
    # the engine was normalized in the accompanying fix commit).
    sn, st = 0.90, 0.80
    Cn_full, Ct_full = 2.0 * (2.0 - sn), 2.0 * st
    for ang in (0.05, 0.3, 1.0):
        F_fs = panel_aero_force_body_finite_s(
            _wind(ang), RHO, PLATE, AREA, sigma_n=sn, sigma_t=st,
            T_inf=1e-6, T_wall=1e-9)
        F_h = panel_aero_force_body(_wind(ang), RHO, PAIR, PAIR_A,
                                    Cn_full, Ct_full)
        assert np.allclose(F_fs, F_h, rtol=2e-3), (ang, F_fs, F_h)


def test_thermal_floor_at_feather():
    # a feathered plate (wind in-plane) still feels along-wind shear;
    # the hyperthermal model predicts exactly zero.
    F_fs = panel_aero_force_body_finite_s(_wind(0.0), RHO, PLATE, AREA)
    F_h = panel_aero_force_body(_wind(0.0), RHO, PAIR, PAIR_A, 2.2, 1.6)
    assert np.linalg.norm(F_h) == 0.0
    drag = -F_fs @ np.array([1.0, 0.0, 0.0])
    assert drag > 0.0
    S = speed_ratio(V)
    A_w, _ = finite_s_coefficients(0.0, S)
    assert np.isclose(drag, 0.5 * RHO * V**2 * AREA[0] * A_w, rtol=1e-9)


def test_coefficient_symmetry():
    # A_w even in incidence, A_n odd through the n_w flip: force from a
    # +ang wind mirrors a -ang wind.
    Fp = panel_aero_force_body_finite_s(_wind(0.4), RHO, PLATE, AREA)
    Fm = panel_aero_force_body_finite_s(_wind(-0.4), RHO, PLATE, AREA)
    assert np.allclose(Fp[0], Fm[0], rtol=1e-12)
    assert np.allclose(Fp[2], -Fm[2], rtol=1e-12)


def test_solar_cycle_moves_the_floor():
    cold = panel_aero_force_body_finite_s(_wind(0.0), RHO, PLATE, AREA,
                                          T_inf=600.0)
    hot = panel_aero_force_body_finite_s(_wind(0.0), RHO, PLATE, AREA,
                                         T_inf=1500.0)
    assert -hot[0] > -cold[0] * 1.4   # ~58% swing across the cycle
