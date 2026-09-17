from __future__ import annotations

import numpy as np
import pytest

from ADCS.orbits.orbital_state import Orbital_State


VECTORS = [
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, -2.0, 3.0]),
    np.array([-4.0, 5.0, 6.0]),
    np.array([1.0e-9, -1.0e-9, 2.0e-9]),
    np.array([1.0e3, -2.0e3, 3.0e3]),
    np.array([-7.0, 0.5, 11.0]),
    np.array([2.5, 3.5, -4.5]),
    np.array([9.0, -8.0, 7.0]),
]


@pytest.mark.parametrize("vector", VECTORS)
def test_eci_and_ecef_transforms_are_inverse_pairs(state, vector):
    assert np.allclose(state.ecef_to_eci(state.eci_to_ecef(vector)), vector, atol=1e-12)


@pytest.mark.parametrize("vector", VECTORS)
def test_geocentric_and_ecef_transforms_are_inverse_pairs(state, vector):
    assert np.allclose(state.ecef_to_geocentric(state.geocentric_to_ecef(vector)), vector, atol=1e-12)


@pytest.mark.parametrize("vector", VECTORS)
def test_enu_and_eci_transforms_are_inverse_pairs(state, vector):
    assert np.allclose(state.enu_to_eci(state.eci_to_enu(vector)), vector, atol=1e-12)


def test_frame_matrices_are_orthonormal_and_right_handed(state):
    for matrix in (state._R_eci2ecef, state._ecef_to_geo, state.ECI2ENUmat):
        assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-10)
        assert np.isclose(abs(np.linalg.det(matrix)), 1.0, atol=1e-10)


@pytest.mark.parametrize("j2000", [0.0, 0.01, 0.1, 0.22, 0.5, -0.05, -0.1, 0.9])
def test_state_construction_computes_consistent_geographic_fields(ephem, j2000):
    state = Orbital_State(ephem, j2000, [7000.0, 0.0, 1000.0], [0.0, 7.3, 0.2], S=[1e8, 0, 0], B=[1e-5, 0, 0], rho=1e-12)
    assert np.all(np.isfinite(state.ECEF))
    assert np.all(np.isfinite(state.LLA))
    assert state.TAI == pytest.approx(2451545.0 + 36525.0 * j2000)
