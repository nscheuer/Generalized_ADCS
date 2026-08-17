import numpy as np
import pytest

from ADCS.helpers.math_helpers import normalize
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.errors import Bias, ErrorMode, Noise
from ADCS.satellite_hardware.sensors import MTM, SunSensor
from ADCS.satellite_hardware.sensors.sunpair import SunPair
from ADCS.state import State


EPHEM = Ephemeris()
STATE = State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0])


def make_orbital_state(*, sunlit: bool) -> Orbital_State:
    orbital_state = Orbital_State(
        ephem=EPHEM,
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 7.5, 0.0]),
        B=np.array([2e-5, -1e-5, 3e-5]),
        S=np.array([1.5e8, 0.0, 0.0]),
    )
    orbital_state._sunlit = bool(sunlit)
    return orbital_state


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SunSensor(axis=normalize(np.array([1.0, 0.2, -0.3])), efficiency=1.0),
        lambda: SunPair(axis=normalize(np.array([1.0, 0.2, -0.3])), efficiency=(1.0, 1.0)),
    ],
)
def test_sun_sensor_clean_reading_is_finite_when_sunlit(factory):
    sensor = factory()
    reading = sensor.clean_reading(STATE, make_orbital_state(sunlit=True))
    assert np.isfinite(np.asarray(reading)).all()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SunSensor(axis=normalize(np.array([1.0, 0.2, -0.3])), efficiency=1.0),
        lambda: SunPair(axis=normalize(np.array([1.0, 0.2, -0.3])), efficiency=(1.0, 1.0)),
    ],
)
def test_sun_sensor_basestate_jacobian_is_finite_when_sunlit(factory):
    sensor = factory()
    jacobian = sensor.basestate_jac(STATE, make_orbital_state(sunlit=True))
    assert np.isfinite(np.asarray(jacobian)).all()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SunSensor(axis=normalize(np.array([1.0, 0.2, -0.3])), efficiency=1.0),
        lambda: SunPair(axis=normalize(np.array([1.0, 0.2, -0.3])), efficiency=(1.0, 1.0)),
    ],
)
def test_sun_sensor_clean_reading_is_nan_in_eclipse(factory):
    sensor = factory()
    reading = sensor.clean_reading(STATE, make_orbital_state(sunlit=False))
    assert np.isnan(np.asarray(reading)).all()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SunSensor(axis=normalize(np.array([1.0, 0.2, -0.3])), efficiency=1.0),
        lambda: SunPair(axis=normalize(np.array([1.0, 0.2, -0.3])), efficiency=(1.0, 1.0)),
    ],
)
def test_sun_sensor_basestate_jacobian_is_nan_in_eclipse(factory):
    sensor = factory()
    jacobian = np.asarray(sensor.basestate_jac(STATE, make_orbital_state(sunlit=False)))
    assert np.isnan(jacobian).all()
    assert jacobian.shape == (7, 1)


def test_mtm_first_reading_contains_noise():
    dmode = ErrorMode(add_bias=False, add_noise=True, update_bias=False, update_noise=True)
    orbital_state = make_orbital_state(sunlit=True)
    std_noise = 3.0e-6
    first_sample_noise = []

    for seed in range(400):
        np.random.seed(seed)
        sensor = MTM(axis=np.array([1.0, 0.0, 0.0]), noise=Noise(std_noise=std_noise))
        clean = sensor.clean_reading(STATE, orbital_state)
        noisy = sensor.reading(STATE, orbital_state, dmode=dmode)
        first_sample_noise.append(float(np.ravel(noisy - clean)[0]))

    first_sample_noise = np.asarray(first_sample_noise)
    assert np.std(first_sample_noise) > 0.3 * std_noise
    assert np.isclose(np.std(first_sample_noise), std_noise, rtol=0.25)


def test_mtm_bias_diffuses_when_time_advances():
    dmode = ErrorMode(add_bias=True, add_noise=False, update_bias=True, update_noise=False)
    orbital_state = make_orbital_state(sunlit=True)
    np.random.seed(7)
    sensor = MTM(axis=np.array([1.0, 0.0, 0.0]), bias=Bias(bias=0.0, std_bias=1.0e-3))

    values = []
    for _ in range(5):
        values.append(float(np.ravel(sensor.reading(STATE, orbital_state, dmode=dmode))[0]))
        orbital_state.J2000 += 1.0e-7

    values = np.asarray(values)
    assert np.std(values) > 0.0
    assert len(np.unique(values)) == len(values)
