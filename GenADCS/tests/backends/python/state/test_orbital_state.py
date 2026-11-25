from collections import Counter

import numpy as np
import pytest

from GenADCS.backends.python.state.orbital_state import OrbitalState


JD = 2451545.0


class CountingEphemeris:
    def __init__(self):
        self.calls = Counter()

    def position(self, body, julian_date):
        self.calls[(body, julian_date)] += 1
        return np.array([float(body), julian_date, 0.0])


def test_cartesian_values_are_validated_converted_and_detached():
    source = np.array([1, 2, 3], dtype=np.float32)
    orbit = OrbitalState.from_cartesian(source, [4, 5, 6], JD)
    assert orbit.julian_date == JD
    assert orbit.position.dtype == np.float64
    assert orbit.position.flags.c_contiguous
    source[0] = 99
    np.testing.assert_array_equal(orbit.position, [1, 2, 3])
    np.testing.assert_array_equal(orbit.velocity, [4, 5, 6])


@pytest.mark.parametrize("date", [np.nan, np.inf, -np.inf])
def test_rejects_nonfinite_date(date):
    with pytest.raises(ValueError, match="julian_date must be finite"):
        OrbitalState([0, 0, 0], [0, 0, 0], date)


@pytest.mark.parametrize("bad", [[1, 2], [[1, 2, 3]], [1, np.nan, 3]])
def test_rejects_invalid_position_and_velocity(bad):
    for position, velocity, name in (
        (bad, [0, 0, 0], "position"),
        ([0, 0, 0], bad, "velocity"),
    ):
        with pytest.raises(ValueError, match=f"{name} must be a finite 3-vector"):
            OrbitalState(position, velocity, JD)


def test_keplerian_fields_and_unimplemented_cartesian_conversion():
    orbit = OrbitalState.from_keplerian(7000, 0.01, 0.2, 0.3, 0.4, 0.5, JD)
    assert orbit._keplerian_elements == (7000, 0.01, 0.2, 0.3, 0.4, 0.5)
    assert orbit._position is None and orbit._velocity is None
    for name in ("position", "velocity"):
        with pytest.raises(
            NotImplementedError, match="Keplerian to Cartesian conversion"
        ):
            getattr(orbit, name)


def test_body_positions_are_cached_per_input_and_ephemeris_is_called_once_per_key():
    ephemeris = CountingEphemeris()
    orbit = OrbitalState.from_cartesian([0, 0, 0], [0, 0, 0], JD, ephemeris)
    first = orbit.sun_position
    assert orbit.body_position("sun") is first
    assert orbit.body_position("SUN") is not first
    assert ephemeris.calls[(0, JD)] == 2
    assert orbit.earth_position[0] == 1
    assert orbit.moon_position[0] == 2
    assert orbit.body_position(1)[0] == 1
    assert ephemeris.calls[(1, JD)] == 2
    assert len(orbit._body_positions) == 5


def test_missing_ephemeris_and_unknown_body():
    orbit = OrbitalState.from_cartesian([0, 0, 0], [0, 0, 0], JD)
    with pytest.raises(ValueError, match="ephemeris must be configured"):
        orbit.sun_position
    orbit.ephemeris = CountingEphemeris()
    with pytest.raises(KeyError):
        orbit.body_position("pluto")


@pytest.mark.parametrize(
    "body,body_id",
    [("sun", 0), ("SUN", 0), ("earth", 1), ("moon", 2), (0, 0), (1, 1), (2, 2)],
)
def test_body_identifier_forms(body, body_id):
    ephemeris = CountingEphemeris()
    orbit = OrbitalState.from_cartesian([0, 0, 0], [0, 0, 0], JD, ephemeris)
    np.testing.assert_array_equal(orbit.body_position(body)[0], body_id)


def test_cache_key_tracks_original_input_and_reuses_same_key():
    ephemeris = CountingEphemeris()
    orbit = OrbitalState.from_cartesian([0, 0, 0], [0, 0, 0], JD, ephemeris)
    a = orbit.body_position("earth")
    assert orbit.body_position("earth") is a
    b = orbit.body_position(1)
    assert b is not a
    assert sum(ephemeris.calls.values()) == 2


def test_cache_is_not_invalidated_when_time_or_ephemeris_changes():
    first, second = CountingEphemeris(), CountingEphemeris()
    orbit = OrbitalState.from_cartesian([0, 0, 0], [0, 0, 0], JD, first)
    cached = orbit.sun_position
    orbit.julian_date += 10
    orbit.ephemeris = second
    assert orbit.sun_position is cached
    assert sum(first.calls.values()) == 1 and not second.calls


def test_position_and_velocity_assignments_are_not_revalidated():
    orbit = OrbitalState.from_cartesian([0, 0, 0], [0, 0, 0], JD)
    orbit._position = [1, 2]
    orbit._velocity = [np.nan]
    assert orbit._position == [1, 2]
    assert np.isnan(orbit._velocity[0])


def test_body_ids_accept_int_coercible_values_and_key_errors_are_preserved():
    ephemeris = CountingEphemeris()
    orbit = OrbitalState.from_cartesian([0, 0, 0], [0, 0, 0], JD, ephemeris)
    assert orbit.body_position(1.9)[0] == 1
    with pytest.raises(KeyError):
        orbit.body_position("pluto")


def test_slots_and_constructor_defaults():
    orbit = OrbitalState(None, None, JD)
    assert orbit.ephemeris is None and orbit._keplerian_elements is None
    with pytest.raises(AttributeError):
        orbit.extra = True
