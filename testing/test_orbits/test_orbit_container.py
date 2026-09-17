from __future__ import annotations

import numpy as np
import pytest

from ADCS.orbits.orbit import Orbit
from ADCS.orbits.universal_constants import TimeConstants


def build_short_orbit(state, seconds=30.0, dt=10.0, zonal_j=2):
    return Orbit(state, end_time=state.J2000 + seconds * TimeConstants.sec2cent, dt=dt, zonal_J=zonal_j, verbose=False)


@pytest.mark.parametrize("zonal_j", [0, 2, 6])
def test_orbit_propagation_has_requested_grid_and_gravity_setting(state, zonal_j):
    orbit = build_short_orbit(state, zonal_j=zonal_j)
    assert len(orbit.times) == 4
    assert orbit._zonal_J == zonal_j
    assert np.all(np.diff(orbit.times) > 0)


def test_singleton_and_state_list_orbits_copy_their_states(state):
    singleton = Orbit(state, verbose=False)
    assert len(singleton.times) == 1
    assert singleton.states[singleton.times[0]] is not state
    listed = Orbit([state, state.propagate_orbit_rk4(1.0)], verbose=False)
    assert len(listed.times) == 2
    assert listed.states[listed.times[0]] is not state


@pytest.mark.parametrize("sample", [0, 1, -1])
def test_get_os_and_next_state_return_stored_node_data(state, sample):
    orbit = build_short_orbit(state)
    time = float(orbit.times[sample])
    queried = orbit.get_os(time)
    assert np.allclose(queried.R, orbit.states[time].R)
    assert np.allclose(orbit.next_state(time).V, orbit.states[time].V)


def test_get_os_interpolates_by_rk4_and_preserves_the_gravity_model(state):
    orbit = build_short_orbit(state, zonal_j=6)
    midpoint = float((orbit.times[1] + orbit.times[2]) / 2.0)
    expected = orbit.states[orbit.times[1]].propagate_orbit_rk4(5.0, zonal_J=6)
    actual = orbit.get_os(midpoint)
    assert np.allclose(actual.R, expected.R)
    assert np.allclose(actual.V, expected.V)


def test_get_range_supports_nodes_resampling_and_equal_times(state):
    orbit = build_short_orbit(state)
    node_range = orbit.get_range(float(orbit.times[1]), float(orbit.times[-1]))
    assert len(node_range.times) == 3
    resampled = orbit.get_range(float(orbit.times[0]), float(orbit.times[-1]), dt=15.0)
    assert len(resampled.times) == 3
    exact = orbit.get_range(float(orbit.times[1]), float(orbit.times[1]), dt=1.0)
    assert np.allclose(exact.R, orbit.states[orbit.times[1]].R)


@pytest.mark.parametrize("bad_time", [-1.0, 1.0])
def test_orbit_time_bounds_and_invalid_inputs_raise(state, bad_time):
    orbit = build_short_orbit(state)
    outside = orbit.min_time() - 1.0 if bad_time < 0 else orbit.max_time() + 1.0
    with pytest.raises(ValueError):
        orbit.get_os(outside)
    with pytest.raises(ValueError):
        orbit.next_state(outside)
    with pytest.raises(ValueError):
        Orbit("not a state", verbose=False)


def test_orbit_range_and_resampling_validate_inputs(state):
    orbit = build_short_orbit(state)
    with pytest.raises(ValueError):
        orbit.get_range(orbit.times[1], orbit.times[0])
    with pytest.raises(ValueError):
        orbit.get_range((orbit.times[0] + orbit.times[1]) / 2, (orbit.times[0] + orbit.times[1]) / 2)
    with pytest.raises(ValueError):
        orbit.new_orbit_from_times([orbit.min_time() - 1.0])


def test_orbit_vector_and_coordinate_helpers_match_per_state_values(state):
    orbit = build_short_orbit(state)
    r_hist, v_hist, b_hist, s_hist, rho_hist = orbit.get_vecs()
    assert len(r_hist) == len(orbit.times)
    assert np.allclose(np.vstack(b_hist), orbit.get_b_eci_orbit())
    geocentric = np.tile(np.array([1.0, 2.0, 3.0]), (len(orbit.times), 1))
    ecef = orbit.geocentric_to_ecef_orbit(geocentric)
    assert np.allclose(orbit.ecef_to_eci_orbit(ecef)[0], orbit.states[orbit.times[0]].ecef_to_eci(ecef[0]))
    assert all(np.isfinite(rho) for rho in rho_hist)
