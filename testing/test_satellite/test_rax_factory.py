import numpy as np

from ADCS.satellite_factory import create_rax1_cubesat, create_rax2_cubesat
from ADCS.satellite_hardware.sensors import Gyro, MTM, SunSensor


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
