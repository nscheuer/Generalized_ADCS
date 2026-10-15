import numpy as np

from ADCS.satellite_hardware.errors import Bias, ErrorMode, Noise
from ADCS.satellite_hardware.sensors import Gyro

from testing.test_sensors._helpers import make_orbital_state, make_state


def test_gyro_clean_reading_projects_body_rate_onto_axis():
    sensor = Gyro(axis=np.array([0.0, 3.0, 0.0]))
    x = make_state(w=np.array([0.1, -0.2, 0.3]))

    reading = sensor.clean_reading(x, make_orbital_state())

    assert np.isclose(reading, -0.2)


def test_gyro_basestate_jacobian_contains_axis_in_rate_block():
    sensor = Gyro(axis=np.array([1.0, 2.0, -2.0]))

    jac = sensor.basestate_jac(make_state(), make_orbital_state())
    axis = np.array([1.0, 2.0, -2.0]) / 3.0

    assert np.allclose(jac[:3, 0], axis)
    assert np.allclose(jac[3:, 0], np.zeros(4))


def test_gyro_bias_jacobian_matches_bias_presence():
    sensor_with_bias = Gyro(axis=np.array([1.0, 0.0, 0.0]), bias=Bias(bias=0.1, std_bias=0.0))
    sensor_without_bias = Gyro(axis=np.array([1.0, 0.0, 0.0]))

    assert np.allclose(sensor_with_bias.bias_jac(make_state(), make_orbital_state()), np.ones((1, 1)))
    assert np.allclose(sensor_without_bias.bias_jac(make_state(), make_orbital_state()), np.zeros((0, 1)))


def test_gyro_reading_adds_bias_and_noise_deterministically():
    sensor = Gyro(
        axis=np.array([1.0, 0.0, 0.0]),
        bias=Bias(bias=0.25, std_bias=0.0),
        noise=Noise(noise=0.1, std_noise=0.0),
    )
    x = make_state(w=np.array([0.2, 0.0, 0.0]))

    reading = sensor.reading(x, make_orbital_state(), dmode=ErrorMode(add_bias=True, add_noise=True, update_bias=False, update_noise=False))

    assert np.isclose(reading, 0.55)
