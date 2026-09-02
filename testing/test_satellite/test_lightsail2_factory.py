import numpy as np

from ADCS.satellite_factory import create_lightsail2
from ADCS.satellite_factory.sensors import (
    create_analog_devices_pib_gyros,
    create_elmos_sun_sensors,
    create_honeywell_lightsail2_magnetometers,
    create_intrepid_mainboard_gyros,
)
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.disturbances import Drag_Disturbance, GG_Disturbance, SRP_Disturbance
from ADCS.satellite_hardware.sensors import Gyro, MTM, SunSensor


def test_lightsail2_component_helpers_create_expected_channels():
    mtms = create_honeywell_lightsail2_magnetometers()
    primary_gyros = create_analog_devices_pib_gyros()
    secondary_gyros = create_intrepid_mainboard_gyros()
    suns = create_elmos_sun_sensors()

    assert len(mtms) == 6
    assert len(primary_gyros) == 3
    assert len(secondary_gyros) == 3
    assert len(suns) == 5

    assert all(np.isclose(mtm.noise.std_noise.item(), 0.2e-6) for mtm in mtms)
    assert np.allclose(
        [gyro.bias.bias.item() for gyro in primary_gyros],
        np.array([0.47837, 0.14098, 0.043055]) * np.pi / 180.0,
    )
    assert all(np.isclose(gyro.noise.std_noise.item(), 0.27 * np.pi / 180.0) for gyro in primary_gyros)
    assert all(np.isclose(gyro.noise.std_noise.item(), 1.0 * np.pi / 180.0) for gyro in secondary_gyros)
    assert all(np.isclose(sun.noise.std_noise.item(), np.sin(5.0 * np.pi / 180.0)) for sun in suns)


def test_create_lightsail2_properties():
    sat = create_lightsail2()

    expected_J = np.array([
        [3.79, -1.90e-4, -8.18e-4],
        [-1.90e-4, 3.79, 1.47e-3],
        [-8.18e-4, 1.47e-3, 7.33],
    ])

    assert np.isclose(sat.mass, 4.93)
    assert np.allclose(sat.J_0, expected_J)
    assert np.allclose(sat.COM, np.array([0.00046, -0.00003, 0.13746]))
    assert np.allclose(sat.get_boresight(None), np.array([0.0, 0.0, 1.0]))

    rws = [act for act in sat.actuators if isinstance(act, RW)]
    mtqs = [act for act in sat.actuators if isinstance(act, MTQ)]
    mtms = [sensor for sensor in sat.sensors if isinstance(sensor, MTM)]
    gyros = [sensor for sensor in sat.sensors if isinstance(sensor, Gyro)]
    suns = [sensor for sensor in sat.sensors if isinstance(sensor, SunSensor)]

    assert len(rws) == 1
    assert len(mtqs) == 3
    assert len(mtms) == 6
    assert len(gyros) == 6
    assert len(suns) == 5
    assert len(sat.disturbances) == 3
    assert len([dist for dist in sat.disturbances if isinstance(dist, GG_Disturbance)]) == 1
    assert len([dist for dist in sat.disturbances if isinstance(dist, Drag_Disturbance)]) == 1
    assert len([dist for dist in sat.disturbances if isinstance(dist, SRP_Disturbance)]) == 1

    expected_wheel_inertia = 0.06 / (5920.0 * 2.0 * np.pi / 60.0)
    assert np.isclose(rws[0].J, expected_wheel_inertia)
    assert np.isclose(rws[0].h_max, 0.06)
    assert np.isclose(rws[0].u_max, 5.0e-3)
    assert all(np.isclose(mtq.u_max, 1.0) for mtq in mtqs)
    assert all(np.all(np.diag(sensor.noise.cov()) > 0.0) for sensor in mtms + gyros + suns)


def test_create_lightsail2_estimated():
    sat = create_lightsail2(estimated=True)

    assert all(act.estimate_bias for act in sat.actuators)
    assert all(sensor.estimate_bias for sensor in sat.sensors)
