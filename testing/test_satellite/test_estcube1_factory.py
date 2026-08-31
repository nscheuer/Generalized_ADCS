import numpy as np

from ADCS.satellite_factory import create_estcube1_cubesat
from ADCS.satellite_hardware.actuators import MTQ
from ADCS.satellite_hardware.sensors import Gyro, MTM, SunSensor


def test_create_estcube1_cubesat_properties():
    sat = create_estcube1_cubesat()

    expected_J = 1e-3 * np.array([
        [1.813, 0.024, 0.042],
        [0.024, 1.963, 0.029],
        [0.042, 0.029, 1.796],
    ])

    assert np.isclose(sat.mass, 1.048)
    assert np.allclose(sat.J_0, expected_J)
    assert np.allclose(sat.COM, np.zeros(3))
    assert np.allclose(sat.get_boresight(None), np.array([0.0, 0.0, 1.0]))

    assert len([act for act in sat.actuators if isinstance(act, MTQ)]) == 3
    mtms = [sensor for sensor in sat.sensors if isinstance(sensor, MTM)]
    gyros = [sensor for sensor in sat.sensors if isinstance(sensor, Gyro)]
    suns = [sensor for sensor in sat.sensors if isinstance(sensor, SunSensor)]

    assert len(mtms) == 6
    assert len(gyros) == 12
    assert len(suns) == 12
    assert len(sat.disturbances) == 3

    assert all(np.isclose(abs(gyro.bias.bounds[0].item()), 0.5 * np.pi / 180.0) for gyro in gyros)
    assert all(np.isclose(gyro.noise.std_noise.item(), 1.8 * np.pi / 180.0) for gyro in gyros)
    assert all(np.isclose(abs(mtm.bias.bounds[0].item()), 2400e-9) for mtm in mtms)
    assert all(np.isclose(mtm.noise.std_noise.item(), 50e-6 * np.sin(1.6 * np.pi / 180.0)) for mtm in mtms)
    assert all(np.isclose(abs(sun.bias.bounds[0].item()), np.sin(1.0 * np.pi / 180.0)) for sun in suns)
    assert all(np.isclose(sun.noise.std_noise.item(), np.sin(1.25 * np.pi / 180.0)) for sun in suns)


def test_create_estcube1_cubesat_estimated():
    sat = create_estcube1_cubesat(estimated=True)

    assert all(act.estimate_bias for act in sat.actuators)
    assert all(sensor.estimate_bias for sensor in sat.sensors)
