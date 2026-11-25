from typing import Any

import numpy as np
import pytest

JD = 2451545.0


class Ephemeris:
    def __init__(self) -> None:
        self.calls: list[tuple[int, float]] = []

    def position(self, body: int, date: float) -> np.ndarray:
        self.calls.append((body, date))
        return np.array([body, date, 0.0])


def test_cartesian_values_and_date(state_backend: Any) -> None:
    orbit = state_backend.OrbitalState.from_cartesian([7e6, 0, 0], [0, 7500, 0], JD)
    np.testing.assert_array_equal(orbit.position, [7e6, 0, 0])
    np.testing.assert_array_equal(orbit.velocity, [0, 7500, 0])
    assert orbit.julian_date == JD


@pytest.mark.parametrize("date", (np.inf, -np.inf, np.nan))
def test_nonfinite_date_rejected(state_backend: Any, date: float) -> None:
    with pytest.raises(ValueError):
        state_backend.OrbitalState.from_cartesian([0, 0, 0], [0, 0, 0], date)


@pytest.mark.parametrize("vector", ([1, 2], [np.nan, 0, 0], [[1, 2, 3]]))
def test_invalid_vectors_rejected(state_backend: Any, vector: Any) -> None:
    with pytest.raises(ValueError):
        state_backend.OrbitalState.from_cartesian(vector, [0, 0, 0], JD)
    with pytest.raises(ValueError):
        state_backend.OrbitalState.from_cartesian([0, 0, 0], vector, JD)


def test_keplerian_cartesian_access_reports_missing_conversion(state_backend: Any) -> None:
    orbit = state_backend.OrbitalState.from_keplerian(
        7000, 0.01, 0.2, 0.3, 0.4, 0.5, JD
    )
    for field in ("position", "velocity"):
        with pytest.raises(
            Exception, match="Keplerian to Cartesian conversion is not implemented"
        ):
            getattr(orbit, field)


def test_ephemeris_body_properties_and_cache(state_backend: Any) -> None:
    ephemeris = Ephemeris()
    orbit = state_backend.OrbitalState.from_cartesian(
        [0, 0, 0], [0, 0, 0], JD, ephemeris
    )
    first = orbit.sun_position
    np.testing.assert_array_equal(first, [0, JD, 0])
    np.testing.assert_array_equal(orbit.earth_position, [1, JD, 0])
    np.testing.assert_array_equal(orbit.moon_position, [2, JD, 0])
    assert orbit.body_position("sun") is first
    assert len(ephemeris.calls) == 3


def test_missing_ephemeris_and_unknown_body(state_backend: Any) -> None:
    orbit = state_backend.OrbitalState.from_cartesian([0, 0, 0], [0, 0, 0], JD)
    with pytest.raises(ValueError):
        _ = orbit.sun_position
    orbit.ephemeris = Ephemeris()
    with pytest.raises(KeyError):
        orbit.body_position("pluto")
