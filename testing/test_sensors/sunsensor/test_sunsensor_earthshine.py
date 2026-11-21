"""SunSensorWithEarthshine = SunSensor + sunlight reflected off the Earth.

The earthshine term is eta * albedo * (R_e/|r|)^2 * max(solar elevation at the
sub-satellite point, 0) * max(axis . nadir, 0). With albedo=0 the subclass is
the base class. The base SunSensor is untouched.
"""

import numpy as np
import pytest

from ADCS.orbits.universal_constants import EarthConstants
from ADCS.satellite_hardware.sensors import SunSensor, SunSensorWithEarthshine

from testing.test_sensors._helpers import make_orbital_state, make_state

R_SAT = np.array([7000.0, 0.0, 0.0])
TILTED_Q = np.array([0.9, 0.2, 0.3, 0.1]) / np.linalg.norm(np.array([0.9, 0.2, 0.3, 0.1]))


def nadir_sun_noon_state():
    """Sun directly over the sub-satellite point: S along +R."""
    os = make_orbital_state(R=R_SAT, S=np.array([1.5e8, 0.0, 0.0]))
    os._sunlit = True
    return os


def fd_basestate_jac(sensor, x, os, eps=1e-6):
    jac = np.zeros((7, 1))
    for i in range(7):
        delta = np.zeros(7)
        delta[i] = eps
        xp = type(x).from_array(x.as_array() + delta)
        xm = type(x).from_array(x.as_array() - delta)
        jac[i, 0] = (sensor.clean_reading(xp, os) - sensor.clean_reading(xm, os)) / (2.0 * eps)
    return jac


def test_zero_albedo_is_the_base_sun_sensor():
    axis = np.array([1.0, 0.4, -0.2])
    base = SunSensor(axis=axis, efficiency=0.7)
    shine = SunSensorWithEarthshine(axis=axis, efficiency=0.7, albedo=0.0)
    os = make_orbital_state()
    x = make_state(q=TILTED_Q)

    assert np.isclose(shine.clean_reading(x, os), base.clean_reading(x, os))
    assert np.allclose(shine.basestate_jac(x, os), base.basestate_jac(x, os))
    assert shine.observation_kind == base.observation_kind
    assert np.allclose(shine.reference_direction(os), base.reference_direction(os))


def test_nadir_sensor_at_noon_reads_the_closed_form_earthshine():
    # Sensor looks at nadir (-x), Sun is behind it (+x): no direct term, full earthshine.
    sensor = SunSensorWithEarthshine(axis=np.array([-1.0, 0.0, 0.0]), efficiency=0.5, albedo=0.3)
    os = nadir_sun_noon_state()
    x = make_state()

    reading = sensor.clean_reading(x, os)

    assert np.isclose(reading, 0.5 * 0.3 * (EarthConstants.R_e / 7000.0) ** 2)
    assert np.isclose(sensor.earthshine_reading(x, os), reading)


def test_earthshine_adds_to_a_direct_sun_reading():
    # Axis half-way between Sun (+x) and nadir (-x) is impossible; use +y with a Sun at 45 deg.
    axis = np.array([0.0, 1.0, 0.0])
    os = make_orbital_state(R=R_SAT, S=np.array([1.0e8, 1.0e8, 0.0]))
    os._sunlit = True
    x = make_state()
    base = SunSensor(axis=axis, efficiency=1.0)
    shine = SunSensorWithEarthshine(axis=axis, efficiency=1.0, albedo=0.3)

    # Axis is orthogonal to nadir, so no earthshine even though the Earth is lit.
    assert np.isclose(shine.clean_reading(x, os), base.clean_reading(x, os))

    # Tip the axis toward nadir and the extra reading is the earthshine term.
    axis = np.array([-1.0, 1.0, 0.0]) / np.sqrt(2.0)
    base = SunSensor(axis=axis, efficiency=1.0)
    shine = SunSensorWithEarthshine(axis=axis, efficiency=1.0, albedo=0.3)
    expected = 0.3 * (EarthConstants.R_e / 7000.0) ** 2 * (1.0 / np.sqrt(2.0)) * (1.0 / np.sqrt(2.0))
    assert np.isclose(shine.clean_reading(x, os) - base.clean_reading(x, os), expected)


def test_night_side_earth_gives_no_earthshine():
    # Sun on the far side of the Earth; force sunlit so only the elevation gate acts.
    sensor = SunSensorWithEarthshine(axis=np.array([-1.0, 0.0, 0.0]), efficiency=0.5, albedo=0.3)
    os = make_orbital_state(R=R_SAT, S=np.array([-1.5e8, 0.0, 0.0]))
    os._sunlit = True

    assert np.isclose(sensor.earthshine_reading(make_state(), os), 0.0)


def test_sensor_facing_away_from_earth_sees_no_earthshine():
    sensor = SunSensorWithEarthshine(axis=np.array([0.0, 0.0, 1.0]), efficiency=0.5, albedo=0.3)
    os = nadir_sun_noon_state()

    assert np.isclose(sensor.earthshine_reading(make_state(), os), 0.0)


def test_earthshine_falls_off_with_altitude():
    sensor = SunSensorWithEarthshine(axis=np.array([-1.0, 0.0, 0.0]), efficiency=1.0, albedo=0.3)
    low = make_orbital_state(R=np.array([6878.0, 0.0, 0.0]), S=np.array([1.5e8, 0.0, 0.0]))
    high = make_orbital_state(R=np.array([13756.0, 0.0, 0.0]), S=np.array([1.5e8, 0.0, 0.0]))
    low._sunlit = high._sunlit = True
    x = make_state()

    assert np.isclose(sensor.earthshine_reading(x, low) / sensor.earthshine_reading(x, high), 4.0)


def test_eclipse_keeps_the_nan_sentinel():
    sensor = SunSensorWithEarthshine(axis=np.array([-1.0, 0.0, 0.0]), efficiency=0.5, albedo=0.3)
    os = nadir_sun_noon_state()
    os._sunlit = False

    assert np.isnan(sensor.clean_reading(make_state(), os))
    assert np.all(np.isnan(sensor.basestate_jac(make_state(), os)))


@pytest.mark.parametrize("axis", [np.array([-1.0, 0.3, 0.2]), np.array([0.2, 1.0, -0.5]), np.array([-0.3, -0.4, 1.0])])
def test_basestate_jacobian_matches_finite_difference(axis):
    sensor = SunSensorWithEarthshine(axis=axis, efficiency=0.6, albedo=0.3)
    os = make_orbital_state(R=R_SAT, S=np.array([1.0e8, 0.8e8, 0.3e8]))
    os._sunlit = True
    x = make_state(q=TILTED_Q)

    jac = sensor.basestate_jac(x, os)

    assert np.allclose(jac, fd_basestate_jac(sensor, x, os), rtol=1e-4, atol=1e-6)


def test_configuration_is_validated():
    with pytest.raises(ValueError):
        SunSensorWithEarthshine(axis=np.array([1.0, 0.0, 0.0]), efficiency=1.0, albedo=-0.1)
    with pytest.raises(ValueError):
        SunSensorWithEarthshine(axis=np.array([1.0, 0.0, 0.0]), efficiency=1.0, earth_radius=0.0)
    sensor = SunSensorWithEarthshine(axis=np.array([1.0, 0.0, 0.0]), efficiency=1.0, albedo=0.25, earth_radius=6400.0)
    assert sensor.albedo == 0.25 and sensor.earth_radius == 6400.0
