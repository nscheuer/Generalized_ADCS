import numpy as np
import pytest
from skyfield.api import EarthSatellite

from ADCS.environment import Environment
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import EarthConstants


TLE_LINE1 = "1 25544U 98067A   24001.50000000  .00016717  00000+0  30142-3 0  9993"
TLE_LINE2 = "2 25544  51.6400  10.0000 0005000  20.0000  30.0000 15.50000000430000"


def test_from_position_matches_cartesian_constructor():
    environment = Environment()
    kwargs = dict(
        J2000=0.22,
        R=np.array([7000.0, 10.0, -20.0]),
        V=np.array([-0.01, 7.5, 0.2]),
        environment=environment,
    )

    state = Orbital_State.from_position(**kwargs)
    direct = Orbital_State(**kwargs)

    assert np.array_equal(state.R, direct.R)
    assert np.array_equal(state.V, direct.V)
    assert state.J2000 == direct.J2000
    assert state.environment is environment


def test_from_keplerian_builds_circular_equatorial_state():
    radius = 7000.0
    state = Orbital_State.from_keplerian(
        J2000=0.22,
        a=radius,
        e=0.0,
        i=0.0,
        raan=0.0,
        arg_periapsis=0.0,
        true_anomaly=90.0,
    )

    assert np.allclose(state.R, [0.0, radius, 0.0], atol=1e-10)
    assert np.allclose(state.V, [-np.sqrt(EarthConstants.mu_e / radius), 0.0, 0.0], atol=1e-12)


@pytest.mark.parametrize(
    "elements",
    [
        {"a": -1.0, "e": 0.0},
        {"a": 7000.0, "e": 1.0},
    ],
)
def test_from_keplerian_rejects_non_elliptic_or_invalid_axis(elements):
    with pytest.raises(ValueError):
        Orbital_State.from_keplerian(
            J2000=0.22,
            i=0.0,
            raan=0.0,
            arg_periapsis=0.0,
            true_anomaly=0.0,
            **elements,
        )


def test_from_tle_defaults_to_tle_epoch_and_accepts_lines_or_satellite():
    environment = Environment()
    satellite = EarthSatellite(TLE_LINE1, TLE_LINE2, ts=environment.ephemeris.ts)

    from_lines = Orbital_State.from_TLE(TLE_LINE1, TLE_LINE2, environment=environment)
    from_satellite = Orbital_State.from_tle(satellite, environment=environment)
    expected_t = satellite.epoch
    expected_state = satellite.at(expected_t)
    expected_j2000 = (expected_t.tt - 2451545.0) / 36525.0

    assert np.isclose(from_lines.J2000, expected_j2000)
    assert np.isclose(from_satellite.J2000, expected_j2000)
    assert np.allclose(from_lines.R, expected_state.position.km)
    assert np.allclose(from_lines.V, expected_state.velocity.km_per_s)
    assert np.allclose(from_satellite.R, from_lines.R)
    assert np.allclose(from_satellite.V, from_lines.V)
    assert from_lines.environment is environment


def test_from_tle_can_sample_at_requested_epoch():
    environment = Environment()
    satellite = EarthSatellite(TLE_LINE1, TLE_LINE2, ts=environment.ephemeris.ts)
    requested_j2000 = (satellite.epoch.tt - 2451545.0) / 36525.0 + 1.0 / 36525.0

    state = Orbital_State.from_TLE(satellite, J2000=requested_j2000, environment=environment)
    expected = satellite.at(environment.ephemeris.ts.tt_jd(requested_j2000 * 36525.0 + 2451545.0))

    assert state.J2000 == requested_j2000
    assert np.allclose(state.R, expected.position.km)
    assert np.allclose(state.V, expected.velocity.km_per_s)


def test_from_tle_requires_second_line_for_line_input():
    with pytest.raises(TypeError, match="line2 is required"):
        Orbital_State.from_TLE(TLE_LINE1)
