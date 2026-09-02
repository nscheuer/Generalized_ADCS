import numpy as np

from ADCS.satellite_factory import create_brite_austria
from ADCS.satellite_factory.sensors import (
    create_aeroastro_mst,
    create_gnb_magnetometer,
    create_gnb_rate_sensors,
    create_gnb_sun_sensors,
)
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.disturbances import Drag_Disturbance, GG_Disturbance, SRP_Disturbance
from ADCS.satellite_hardware.sensors import Gyro, MTM, StarTracker, SunSensor


def test_brite_component_helpers_create_expected_channels():
    gyros = create_gnb_rate_sensors()
    mtms = create_gnb_magnetometer()
    suns = create_gnb_sun_sensors()
    tracker = create_aeroastro_mst()

    assert len(gyros) == 3
    assert len(mtms) == 3
    assert len(suns) == 6
    assert tracker.output_length == 3

    assert all(np.isclose(gyro.noise.std_noise.item(), 0.05 * np.pi / 180.0) for gyro in gyros)
    assert all(np.isclose(abs(gyro.bias.bounds[0].item()), 0.2 * np.pi / 180.0) for gyro in gyros)
    assert all(np.isclose(gyro.bias.std_bias.item(), 0.0004 * np.pi / 180.0) for gyro in gyros)
    assert all(np.isclose(mtm.noise.std_noise.item(), 2.0e-7) for mtm in mtms)
    assert all(np.isclose(abs(mtm.bias.bounds[0].item()), 4.0e-6) for mtm in mtms)
    assert all(np.isclose(mtm.bias.std_bias.item(), 1.0e-9) for mtm in mtms)
    assert all(np.isclose(sun.noise.std_noise.item(), np.sin(1.0 * np.pi / 180.0)) for sun in suns)
    assert np.all(np.diag(tracker.noise.cov()) > 0.0)


def test_create_brite_austria_properties():
    sat = create_brite_austria()

    expected_J = np.array([
        [0.0465, -0.0007, 0.0004],
        [-0.0007, 0.0486, -0.0021],
        [0.0004, -0.0021, 0.0482],
    ])

    assert np.isclose(sat.mass, 6.9)
    assert np.allclose(sat.J_0, expected_J)
    assert np.allclose(sat.COM, np.zeros(3))
    assert np.allclose(sat.get_boresight(None), np.array([0.0, 0.0, 1.0]))

    rws = [act for act in sat.actuators if isinstance(act, RW)]
    mtqs = [act for act in sat.actuators if isinstance(act, MTQ)]
    gyros = [sensor for sensor in sat.sensors if isinstance(sensor, Gyro)]
    mtms = [sensor for sensor in sat.sensors if isinstance(sensor, MTM)]
    suns = [sensor for sensor in sat.sensors if isinstance(sensor, SunSensor)]
    trackers = [sensor for sensor in sat.sensors if isinstance(sensor, StarTracker)]

    assert len(rws) == 3
    assert len(mtqs) == 3
    assert len(gyros) == 3
    assert len(mtms) == 3
    assert len(suns) == 6
    assert len(trackers) == 1
    assert len(sat.disturbances) == 3
    assert len([dist for dist in sat.disturbances if isinstance(dist, GG_Disturbance)]) == 1
    assert len([dist for dist in sat.disturbances if isinstance(dist, Drag_Disturbance)]) == 1
    assert len([dist for dist in sat.disturbances if isinstance(dist, SRP_Disturbance)]) == 1

    assert all(np.isclose(rw.J, 5.12e-5) for rw in rws)
    assert all(np.isclose(rw.h_max, 0.030) for rw in rws)
    assert all(np.isclose(mtq.u_max, 0.12) for mtq in mtqs)
    assert all(np.all(np.diag(sensor.noise.cov()) > 0.0) for sensor in gyros + mtms + suns + trackers)


def test_create_brite_austria_estimated():
    sat = create_brite_austria(estimated=True)

    assert all(act.estimate_bias for act in sat.actuators)
    assert all(sensor.estimate_bias for sensor in sat.sensors)
