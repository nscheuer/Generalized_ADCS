import numpy as np

from ADCS.satellite_factory import create_moveii_cubesat
from ADCS.satellite_factory.sensors import (
    create_bmx055_gyros,
    create_bmx055_magnetometers,
    create_nano_iss60_sun_sensors,
)
from ADCS.satellite_hardware.actuators import MTQ
from ADCS.satellite_hardware.disturbances import Dipole_Disturbance
from ADCS.satellite_hardware.sensors import Gyro, MTM, SunSensor


def test_bmx055_helpers_create_one_triad():
    gyros = create_bmx055_gyros()
    mtms = create_bmx055_magnetometers()

    assert len(gyros) == 3
    assert len(mtms) == 3
    assert np.allclose([gyro.bias.bias.item() for gyro in gyros], [1.75e-3, 3.49e-3, -1.75e-3])
    assert np.allclose([mtm.bias.bias.item() for mtm in mtms], [2.0e-6, -3.0e-6, 3.0e-6])


def test_nano_iss60_helper_creates_one_sensor_proxy():
    suns = create_nano_iss60_sun_sensors()

    assert len(suns) == 1
    assert np.allclose(suns[0].axis, np.array([1.0, 0.0, 0.0]))
    assert np.isclose(suns[0].noise.std_noise.item(), np.sin(0.06 * np.pi / 180.0))


def test_create_moveii_cubesat_properties():
    sat = create_moveii_cubesat()

    gyros = [sensor for sensor in sat.sensors if isinstance(sensor, Gyro)]
    mtms = [sensor for sensor in sat.sensors if isinstance(sensor, MTM)]
    suns = [sensor for sensor in sat.sensors if isinstance(sensor, SunSensor)]
    mtqs = [act for act in sat.actuators if isinstance(act, MTQ)]
    dipoles = [dist for dist in sat.disturbances if isinstance(dist, Dipole_Disturbance)]

    assert np.isclose(sat.mass, 1.2)
    assert np.allclose(sat.J_0, np.diag([0.00297, 0.00330, 0.00320]))
    assert np.allclose(sat.COM, np.zeros(3))
    assert np.allclose(sat.get_boresight(None), np.array([0.0, 0.0, 1.0]))

    assert len(mtqs) == 3
    assert len(mtms) == 18
    assert len(gyros) == 18
    assert len(suns) == 5
    assert len(sat.disturbances) == 4
    assert len(dipoles) == 1

    assert all(np.isclose(mtq.u_max, 0.10) for mtq in mtqs)
    assert np.allclose(dipoles[0].main_param, np.array([-0.001, 0.012, -0.045]))

    gyro_biases = np.array([gyro.bias.bias.item() for gyro in gyros[:3]])
    mtm_biases = np.array([mtm.bias.bias.item() for mtm in mtms[:3]])
    assert np.allclose(gyro_biases, np.array([1.75e-3, 3.49e-3, -1.75e-3]))
    assert np.allclose(mtm_biases, np.array([2.0e-6, -3.0e-6, 3.0e-6]))
    assert all(np.isclose(gyro.noise.std_noise.item(), 8.727e-4) for gyro in gyros)
    assert all(np.isclose(mtm.noise.std_noise.item(), 5.0e-7) for mtm in mtms)
    assert all(np.isclose(sun.noise.std_noise.item(), np.sin(0.06 * np.pi / 180.0)) for sun in suns)
    assert all(not sun.bias for sun in suns)


def test_create_moveii_cubesat_estimated():
    sat = create_moveii_cubesat(estimated=True)

    assert all(act.estimate_bias for act in sat.actuators)
    assert all(sensor.estimate_bias for sensor in sat.sensors)
