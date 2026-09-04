import numpy as np

from ADCS.satellite_factory import create_rax1_cubesat, create_rax2_cubesat
from ADCS.satellite_factory.sensors import create_adis16405_gyros, create_osram_sfh2430_sun_sensors
from ADCS.satellite_hardware.sensors import Gyro, MTM, SunSensor
from ADCS.orbits.universal_constants import TimeConstants


def test_osram_sfh2430_helper_creates_one_photodiode_by_default():
    suns = create_osram_sfh2430_sun_sensors()

    assert len(suns) == 1
    assert np.allclose(suns[0].axis, np.array([1.0, 0.0, 0.0]))
    assert np.isclose(suns[0].noise.std_noise.item(), 0.05)


def _rax_inertia() -> np.ndarray:
    return 1e-2 * np.diag([2.91058, 2.91058, 0.59261])


def test_create_rax1_cubesat_properties():
    sat = create_rax1_cubesat()

    gyros = [sensor for sensor in sat.sensors if isinstance(sensor, Gyro)]
    mtms = [sensor for sensor in sat.sensors if isinstance(sensor, MTM)]
    suns = [sensor for sensor in sat.sensors if isinstance(sensor, SunSensor)]

    assert np.isclose(sat.mass, 2.8)
    assert np.allclose(sat.J_0, _rax_inertia())
    assert len(gyros) == 3
    assert len(mtms) == 6
    assert len(suns) == 9
    assert len(sat.actuators) == 0
    assert len(sat.disturbances) == 3
    assert all(np.isclose(gyro.noise.std_noise.item(), 0.9 * np.pi / 180.0) for gyro in gyros)
    assert all(np.isclose(gyro.bias.std_bias.item(), 3.14e-5) for gyro in gyros)
    assert all(np.isclose(gyro.bias.bounds[0].item(), gyro.bias.bias.item() - 0.007 * np.pi / 180.0) for gyro in gyros)
    assert all(np.isclose(gyro.bias.bounds[1].item(), gyro.bias.bias.item() + 0.007 * np.pi / 180.0) for gyro in gyros)
    assert all(np.isclose(mtm.noise.std_noise.item(), 100e-9) for mtm in mtms[3:])
    assert all(np.isclose(sun.noise.std_noise.item(), 0.05) for sun in suns)
    assert all(np.isclose(sun.efficiency, 3.0) for sun in suns)


def test_create_rax2_cubesat_properties():
    sat = create_rax2_cubesat()

    gyros = [sensor for sensor in sat.sensors if isinstance(sensor, Gyro)]
    mtms = [sensor for sensor in sat.sensors if isinstance(sensor, MTM)]
    suns = [sensor for sensor in sat.sensors if isinstance(sensor, SunSensor)]

    assert np.isclose(sat.mass, 2.9)
    assert np.allclose(sat.J_0, _rax_inertia())
    assert len(gyros) == 3
    assert len(mtms) == 6
    assert len(suns) == 17
    assert len(sat.actuators) == 0
    assert len(sat.disturbances) == 3
    assert any(np.allclose(sun.axis, np.array([1.0, 0.0, 0.0]), atol=0.35) for sun in suns)
    assert any(np.allclose(sun.axis, np.array([0.0, 0.0, -1.0])) for sun in suns)


def test_create_rax_cubesat_estimated():
    rax1 = create_rax1_cubesat(estimated=True)
    rax2 = create_rax2_cubesat(estimated=True)

    assert all(sensor.estimate_bias for sensor in rax1.sensors)
    assert all(sensor.estimate_bias for sensor in rax2.sensors)


def test_adis16405_bias_bounds_follow_turn_on_bias(monkeypatch):
    turn_on_bias = np.array([3.0, -2.0, 1.0]) * np.pi / 180.0

    def deterministic_normal(loc=0.0, scale=1.0, size=None):
        if size == 3:
            return turn_on_bias.copy()
        return np.asarray(loc)

    monkeypatch.setattr(np.random, "normal", deterministic_normal)
    gyros = create_adis16405_gyros()

    stability = 0.007 * np.pi / 180.0
    for gyro, initial_bias in zip(gyros, turn_on_bias):
        assert np.isclose(gyro.bias.bias.item(), initial_bias)
        assert np.isclose(gyro.bias.bounds[0].item(), initial_bias - stability)
        assert np.isclose(gyro.bias.bounds[1].item(), initial_bias + stability)

        gyro.bias._update_bias(0.0)
        gyro.bias._update_bias(40.0 * 60.0 / TimeConstants.cent2sec)

        assert np.isclose(gyro.bias.bias.item(), initial_bias)
