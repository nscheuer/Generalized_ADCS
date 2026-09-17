from __future__ import annotations

import numpy as np
import pytest

from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import EarthConstants, TimeConstants
from ADCS.state import State


@pytest.mark.parametrize("zonal_j", [0, 2, 6])
@pytest.mark.parametrize("dt", [0.0, 0.1, 10.0])
def test_propagators_advance_time_and_return_finite_states(state, zonal_j, dt):
    for propagated in (state.propagate_orbit(dt, zonal_J=zonal_j), state.propagate_orbit_rk4(dt, zonal_J=zonal_j)):
        assert propagated.J2000 == pytest.approx(state.J2000 + dt * TimeConstants.sec2cent)
        assert np.all(np.isfinite(propagated.R))
        assert np.all(np.isfinite(propagated.V))


@pytest.mark.parametrize("dt", [1.0, 10.0, 60.0])
def test_rk4_has_smaller_one_step_error_than_euler(state, dt):
    reference = state.propagate_orbit_rk4(dt / 20.0, zonal_J=0)
    for _ in range(19):
        reference = reference.propagate_orbit_rk4(dt / 20.0, zonal_J=0)
    euler = state.propagate_orbit(dt, zonal_J=0)
    rk4 = state.propagate_orbit_rk4(dt, zonal_J=0)
    assert np.linalg.norm(rk4.R - reference.R) < np.linalg.norm(euler.R - reference.R)


@pytest.mark.parametrize("zonal_j", [0, 2, 6])
def test_dynamics_jacobians_match_finite_difference(state, zonal_j):
    _, _, analytic, _ = state.orbit_dynamics_jacobians(zonal_J=zonal_j)
    h = 1.0e-3
    numerical = np.column_stack([
        (Orbital_State._orbit_dynamics_raw(state.R + np.eye(3)[i] * h, state.V, state.mu_e, state.R_e, state.J2coeff, zonal_j, state.Jcoeffs)[1]
         - Orbital_State._orbit_dynamics_raw(state.R - np.eye(3)[i] * h, state.V, state.mu_e, state.R_e, state.J2coeff, zonal_j, state.Jcoeffs)[1]) / (2.0 * h)
        for i in range(3)
    ])
    assert np.allclose(analytic, numerical, rtol=3e-5, atol=1e-12)


@pytest.mark.parametrize("method", ["propagate_jacobians", "propagate_jacobians_rk4"])
@pytest.mark.parametrize("zonal_j", [0, 2, 6])
def test_propagation_jacobians_have_correct_shapes_and_identity_limit(state, method, zonal_j):
    blocks = getattr(state, method)(0.0, zonal_J=zonal_j)
    assert all(block.shape == (3, 3) for block in blocks)
    assert np.allclose(blocks[0], np.eye(3))
    assert np.allclose(blocks[1], np.zeros((3, 3)))
    assert np.allclose(blocks[2], np.zeros((3, 3)))
    assert np.allclose(blocks[3], np.eye(3))


def test_copy_is_independent_and_roundtrip_serialization_preserves_core_data(state):
    copied = state.copy()
    copied.R[0] += 1.0
    copied.V[1] += 1.0
    assert not np.array_equal(copied.R, state.R)
    assert not np.array_equal(copied.V, state.V)
    restored = Orbital_State.from_dict(state.to_dict(), state.ephem, state.density_model)
    for field in ("R", "V", "S", "B"):
        assert np.array_equal(getattr(restored, field), getattr(state, field))
    assert restored.rho == state.rho


@pytest.mark.parametrize("ratio", [0.0, 0.25, 0.5, 1.0])
def test_average_linearly_blends_state_and_environment_fields(state, ratio):
    other = state.propagate_orbit_rk4(5.0, zonal_J=2)
    averaged = state.average(other, ratio=ratio)
    for field in ("R", "V", "S", "B"):
        assert np.allclose(getattr(averaged, field), (1.0 - ratio) * getattr(state, field) + ratio * getattr(other, field))
    assert averaged.rho == pytest.approx((1.0 - ratio) * state.rho + ratio * other.rho)


def test_body_vector_cache_updates_only_for_a_new_attitude(state):
    attitude = State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0])
    vectors = state.get_state_vector(attitude)
    assert np.array_equal(vectors["r"], state.R)
    assert state.get_state_vector(None) is vectors
    with pytest.raises(TypeError):
        state.get_state_vector(np.zeros(4))
    changed = State(w=np.zeros(3), q=[0.0, 1.0, 0.0, 0.0])
    assert not np.array_equal(state.get_state_vector(changed)["r"], state.R)


def test_body_vector_cache_requires_an_initial_state_and_accepts_skip_jacobians(state):
    with pytest.raises(ValueError):
        state.get_state_vector(None)
    state._skip_jacobians = True
    vectors = state.get_state_vector(State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0]))
    assert vectors["dr"].shape == (3, 4)
    assert vectors["ddr"].shape == (3, 4, 4)


def test_sunlit_result_is_cached(state, monkeypatch):
    monkeypatch.setattr(state.sf_pos, "is_sunlit", lambda planets: True)
    assert state.is_sunlit() is True
    monkeypatch.setattr(state.sf_pos, "is_sunlit", lambda planets: False)
    assert state.is_sunlit() is True


def test_j2000_conversion_and_environment_methods_return_physical_values(state):
    assert state.j2000_to_tai() == pytest.approx(2451545.0 + state.J2000 * 36525.0)
    assert np.linalg.norm(state.get_sun_eci()) > 1.0e8
    assert 1.0e-7 < np.linalg.norm(state.get_b_eci()) < 1.0e-3
    assert EarthConstants.R_e < state.geocentric[0]
