from typing import Any

import numpy as np
import pytest

JD = 2451545.0


class Ephemeris:
    def __init__(self) -> None:
        self.calls = []

    def position(self, body: Any, date: Any) -> Any:
        self.calls.append((body, date))
        return np.array([body + 1, date, 2.0])


def test_cartesian_and_keplerian_values_agree(state_backends: Any) -> None:
    cartesian = [
        module.OrbitalState.from_cartesian([7e6, 1, 2], [3, 7500, 4], JD)
        for module in state_backends.values()
    ]
    for orbit in cartesian[1:]:
        np.testing.assert_array_equal(orbit.position, cartesian[0].position)
        np.testing.assert_array_equal(orbit.velocity, cartesian[0].velocity)
        assert orbit.julian_date == cartesian[0].julian_date
    keplerian = [
        module.OrbitalState.from_keplerian(7000, 0.01, 0.2, 0.3, 0.4, 0.5, JD)
        for module in state_backends.values()
    ]
    for orbit in keplerian[1:]:
        np.testing.assert_array_equal(
            np.asarray(orbit._keplerian_elements),
            np.asarray(keplerian[0]._keplerian_elements),
        )


def test_ephemeris_results_and_cache_calls_agree(state_backends: Any) -> None:
    outputs = []
    for module in state_backends.values():
        ephemeris = Ephemeris()
        orbit = module.OrbitalState.from_cartesian([0, 0, 0], [0, 0, 0], JD, ephemeris)
        outputs.append(
            (
                orbit.sun_position,
                orbit.earth_position,
                orbit.moon_position,
                orbit.body_position("sun"),
                list(ephemeris.calls),
            )
        )
    for result in outputs[1:]:
        for current, reference in zip(result[:4], outputs[0][:4]):
            np.testing.assert_array_equal(current, reference)
        assert result[4] == outputs[0][4] == [(0, JD), (1, JD), (2, JD)]


@pytest.mark.parametrize("date", [np.nan, np.inf])
def test_invalid_date_error_category_agrees(state_backends: Any, date: Any) -> None:
    for module in state_backends.values():
        with pytest.raises(ValueError):
            module.OrbitalState.from_cartesian([0, 0, 0], [0, 0, 0], date)
