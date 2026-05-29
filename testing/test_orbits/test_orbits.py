import numpy as np
import pytest

from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants

from testing.test_orbits._helpers import make_orbit_family, make_reference_orbital_state


def assert_state_matches(left, right):
    assert np.isclose(left.J2000, right.J2000)
    assert np.allclose(left.R, right.R)
    assert np.allclose(left.V, right.V)
    assert np.allclose(left.B, right.B)
    assert np.allclose(left.S, right.S)
    assert np.allclose(left.rho, right.rho)


def test_singleton_orbit_creation_preserves_initial_state():
    os0, orb0, _, _, _, _, _, _ = make_orbit_family()

    assert np.allclose(orb0.times, np.array([os0.J2000]))
    assert_state_matches(orb0.states[os0.J2000], os0)


def test_propagated_orbit_times_match_requested_grid():
    os0, _, orb1, orb2, orb3, dt, n_steps, _ = make_orbit_family()
    expected_times = np.array([os0.J2000 + TimeConstants.sec2cent * dt * j for j in range(n_steps + 1)])

    assert np.allclose(orb1.times, expected_times)
    assert np.allclose(orb2.times, expected_times)
    assert np.allclose(orb3.times, expected_times)


@pytest.mark.parametrize("orbit_index", [2, 3])
def test_propagated_orbits_match_stepwise_rk4(orbit_index):
    _, _, orb1, _, orb3, dt, _, _ = make_orbit_family()
    orbit = {2: orb1, 3: orb3}[orbit_index]

    for i in range(1, len(orbit.times)):
        prev_state = orbit.states[orbit.times[i - 1]]
        current_state = orbit.states[orbit.times[i]]
        propagated = prev_state.propagate_orbit_rk4(dt=dt, J2_perturbation_on=True, fast=True)
        assert_state_matches(current_state, propagated)


def test_orbit_from_state_list_copies_source_states():
    _, _, orb1, orb2, _, _, _, _ = make_orbit_family()

    assert np.allclose(orb2.times, orb1.times)
    for t in orb1.times:
        assert_state_matches(orb2.states[t], orb1.states[t])
        assert orb2.states[t] is not orb1.states[t]


@pytest.mark.parametrize("orbit_index", [1, 2, 3])
def test_orbit_states_recompute_environment_consistently(orbit_index):
    _, _, orb1, orb2, orb3, _, _, _ = make_orbit_family()
    orbit = {1: orb1, 2: orb2, 3: orb3}[orbit_index]

    for t in orbit.times[:: max(1, len(orbit.times) // 4)]:
        state = orbit.states[t]
        rebuilt = Orbital_State(ephem=state.ephem, J2000=t, R=state.R, V=state.V, fast=True)
        assert np.allclose(state.B, rebuilt.get_b_eci())
        assert np.allclose(state.S, rebuilt.get_sun_eci())
        assert np.allclose(state.rho, rebuilt.rho)
        assert np.allclose(state.TAI, rebuilt.TAI)
        assert np.allclose(state.ECEF, rebuilt.ECEF)
        assert np.allclose(state.LLA, rebuilt.LLA)
        assert np.allclose(state.geocentric, rebuilt.geocentric)
        assert np.allclose(state.ECI2ENUmat, rebuilt.ECI2ENUmat)
        assert state.datetime == rebuilt.datetime


@pytest.mark.parametrize("orbit_index", [1, 2, 3])
def test_get_vecs_matches_state_storage(orbit_index):
    _, _, orb1, orb2, orb3, _, _, _ = make_orbit_family()
    orbit = {1: orb1, 2: orb2, 3: orb3}[orbit_index]
    r_hist, v_hist, b_hist, s_hist, rho_hist = orbit.get_vecs()

    assert len(r_hist) == len(orbit.times)
    assert len(v_hist) == len(orbit.times)
    assert len(b_hist) == len(orbit.times)
    assert len(s_hist) == len(orbit.times)
    assert len(rho_hist) == len(orbit.times)

    for i, t in enumerate(orbit.times):
        state = orbit.states[t]
        assert np.allclose(r_hist[i], state.R)
        assert np.allclose(v_hist[i], state.V)
        assert np.allclose(b_hist[i], state.B)
        assert np.allclose(s_hist[i], state.S)
        assert np.allclose(rho_hist[i], state.rho)


def test_get_b_eci_orbit_returns_stored_field_history():
    _, _, _, _, orb3, _, _, _ = make_orbit_family()

    b_hist = orb3.get_b_eci_orbit()

    assert np.allclose(b_hist, np.vstack([orb3.states[t].B for t in orb3.times]))


@pytest.mark.parametrize("orbit_index", [0, 1, 2, 3])
@pytest.mark.parametrize("sample_index", [0, -1, 12])
def test_get_os_and_next_state_match_stored_samples(orbit_index, sample_index):
    os0, orb0, orb1, orb2, orb3, _, _, _ = make_orbit_family()
    orbit = {0: orb0, 1: orb1, 2: orb2, 3: orb3}[orbit_index]

    if len(orbit.times) == 1:
        sample_index = 0
    elif sample_index == 12 and len(orbit.times) <= 12:
        sample_index = len(orbit.times) // 2

    t = orbit.times[sample_index]
    state = orbit.states[t]

    assert_state_matches(orbit.get_os(t), state)
    assert_state_matches(orbit.next_state(float(t)), state)
    assert_state_matches(orbit.next_state(state), state)


def test_get_os_interpolates_between_states():
    _, _, orb1, _, _, dt, _, _ = make_orbit_family()
    t0 = orb1.times[3]
    t1 = orb1.times[4]
    midpoint = 0.5 * (t0 + t1)

    interpolated = orb1.get_os(midpoint)
    expected = orb1.states[t0].average(orb1.states[t1], ratio=0.5)

    assert_state_matches(interpolated, expected)


def test_get_range_with_dt_returns_resampled_orbit():
    _, _, orb1, _, _, dt, _, _ = make_orbit_family()
    t0 = orb1.times[2]
    t1 = orb1.times[5]

    sub_orbit = orb1.get_range(t0, t1, dt=dt)

    assert isinstance(sub_orbit, Orbit)
    assert np.allclose(sub_orbit.times, orb1.times[2:6])


def test_get_range_without_dt_returns_existing_states_only():
    _, _, orb1, _, _, _, _, _ = make_orbit_family()
    t0 = orb1.times[1]
    t1 = orb1.times[4]

    sub_orbit = orb1.get_range(t0, t1)

    assert isinstance(sub_orbit, Orbit)
    assert np.allclose(sub_orbit.times, orb1.times[1:5])


def test_new_orbit_from_times_reuses_get_os_contract():
    _, _, orb1, _, _, _, _, _ = make_orbit_family()
    requested_times = [orb1.times[0], 0.5 * (orb1.times[0] + orb1.times[1]), orb1.times[2]]

    resampled = orb1.new_orbit_from_times(requested_times)

    assert np.allclose(resampled.times, requested_times)
    for t in requested_times:
        assert_state_matches(resampled.states[t], orb1.get_os(t))


def test_orbit_query_bounds_raise_errors():
    reference = make_reference_orbital_state()
    orbit = Orbit(os0=reference, end_time=reference.J2000 + TimeConstants.sec2cent * 3600.0, dt=3600.0, verbose=False)

    with pytest.raises(ValueError):
        orbit.get_os(orbit.min_time() - 1e-6)

    with pytest.raises(ValueError):
        orbit.next_state(orbit.max_time() + 1e-6)
