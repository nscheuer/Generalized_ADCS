from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.legendre import legval

from ADCS.orbits.orbital_state import (
    Orbital_State,
    _legendre_p_and_dp,
    _normalize_zonal_J,
    _zonal_perturbation_accel,
    _zonal_perturbation_accel_jac,
)
from ADCS.orbits.universal_constants import EarthConstants


MU = EarthConstants.mu_e
RE = EarthConstants.R_e
JC = EarthConstants.Jcoeffs
POSITIONS = [np.array([7000.0, 1200.0, -800.0]), np.array([7300.0, -1500.0, 2100.0]), np.array([6800.0, 400.0, 300.0]), np.array([7900.0, -2300.0, -1700.0])]


@pytest.mark.parametrize("value, expected", [(0, 0), (2, 2), (3.0, 3), (5, 5), (6, 6)])
def test_normalize_zonal_degree_accepts_supported_values(value, expected):
    assert _normalize_zonal_J(value) == expected


@pytest.mark.parametrize("value", [-1, 1, 7, 99, "bad"])
def test_normalize_zonal_degree_rejects_unsupported_values(value):
    with pytest.raises((ValueError, TypeError)):
        _normalize_zonal_J(value)


@pytest.mark.parametrize("degree", range(7))
@pytest.mark.parametrize("s", [-0.75, 0.0, 0.4])
def test_legendre_recurrence_matches_numpy_and_its_derivative(degree, s):
    p, dp = _legendre_p_and_dp(s, degree)
    coeff = np.zeros(degree + 1)
    coeff[-1] = 1.0
    assert p[degree] == pytest.approx(legval(s, coeff))
    h = 1.0e-6
    assert dp[degree] == pytest.approx((legval(s + h, coeff) - legval(s - h, coeff)) / (2.0 * h), rel=2e-6, abs=2e-6)


@pytest.mark.parametrize("position", POSITIONS)
def test_two_body_dynamics_has_expected_closed_form(position):
    velocity = np.array([1.0, 7.1, -0.3])
    r_dot, v_dot = Orbital_State._orbit_dynamics_raw(position, velocity, MU, RE, JC[0], zonal_J=0)
    assert np.array_equal(r_dot, velocity)
    assert np.allclose(v_dot, -MU * position / np.linalg.norm(position) ** 3)


@pytest.mark.parametrize("position", POSITIONS)
def test_j2_changes_acceleration_but_higher_zonals_remain_small(position):
    velocity = np.zeros(3)
    two_body = Orbital_State._orbit_dynamics_raw(position, velocity, MU, RE, JC[0], zonal_J=0)[1]
    j2 = Orbital_State._orbit_dynamics_raw(position, velocity, MU, RE, JC[0], zonal_J=2)[1]
    j6 = Orbital_State._orbit_dynamics_raw(position, velocity, MU, RE, JC[0], zonal_J=6)[1]
    assert not np.allclose(j2, two_body, rtol=0.0, atol=1e-12)
    assert 0.0 < np.linalg.norm(j6 - j2) < 0.01 * np.linalg.norm(j2 - two_body)


@pytest.mark.parametrize("degree", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("position", POSITIONS)
def test_each_zonal_acceleration_matches_a_finite_difference_potential_gradient(degree, position):
    coefficient = np.array([JC[degree - 2]])
    accel = _zonal_perturbation_accel(position, MU, RE, coefficient, degree)

    def potential(r):
        radius = np.linalg.norm(r)
        s = r[2] / radius
        coeff = np.zeros(degree + 1)
        coeff[-1] = 1.0
        return -(MU / radius) * coefficient[0] * (RE / radius) ** degree * legval(s, coeff)

    h = 0.1
    reference = np.array([(potential(position + np.eye(3)[i] * h) - potential(position - np.eye(3)[i] * h)) / (2.0 * h) for i in range(3)])
    assert np.allclose(accel, reference, rtol=2e-6, atol=1e-16)


@pytest.mark.parametrize("position", POSITIONS)
def test_higher_zonal_jacobian_matches_central_difference(position):
    coeffs = JC[1:]
    analytic = _zonal_perturbation_accel_jac(position, MU, RE, coeffs, 3)
    h = 1.0e-3
    numerical = np.column_stack([
        (_zonal_perturbation_accel(position + np.eye(3)[i] * h, MU, RE, coeffs, 3) - _zonal_perturbation_accel(position - np.eye(3)[i] * h, MU, RE, coeffs, 3)) / (2.0 * h)
        for i in range(3)
    ])
    assert np.allclose(analytic, numerical, rtol=1e-6, atol=1e-14)


def test_empty_zonal_coefficient_arrays_are_zero():
    assert np.array_equal(_zonal_perturbation_accel(POSITIONS[0], MU, RE, [], 3), np.zeros(3))
    assert np.array_equal(_zonal_perturbation_accel_jac(POSITIONS[0], MU, RE, [], 3), np.zeros((3, 3)))
