import numpy as np

from ADCS.satellite_hardware.errors import Bias, ErrorMode, Noise
from ADCS.satellite_hardware.sensors.sensor import Sensor

from testing.test_sensors._helpers import make_orbital_state, make_state


class DummySensor(Sensor):
    def __init__(self, value, **kwargs):
        self.value = np.asarray(value, dtype=float)
        super().__init__(output_length=self.value.size, **kwargs)

    def clean_reading(self, x, os):
        return self.value.copy()


def test_sensor_defaults_use_zero_bias_and_noise():
    sensor = DummySensor([1.0, -2.0])

    assert sensor.sample_time == 0.1
    assert sensor.output_length == 2


def test_sensor_default_bias_is_zero():
    sensor = DummySensor([1.0, -2.0])
    assert np.allclose(sensor.bias.bias, np.zeros(1))


def test_sensor_default_noise_is_zero():
    sensor = DummySensor([1.0, -2.0])
    assert np.allclose(sensor.noise.noise, np.zeros(1))


def test_sensor_reading_adds_bias_and_noise_when_enabled():
    sensor = DummySensor(
        [1.0, -2.0],
        bias=Bias(bias=np.array([0.5, -0.25]), std_bias=0.0),
        noise=Noise(noise=np.array([0.1, 0.2]), std_noise=0.0),
    )

    out = sensor.reading(make_state(), make_orbital_state(), dmode=ErrorMode(add_bias=True, add_noise=True, update_bias=False, update_noise=False))

    assert np.allclose(out, np.array([1.6, -2.05]))


def test_sensor_reading_can_disable_bias_and_noise():
    sensor = DummySensor(
        [1.0],
        bias=Bias(bias=np.array([0.5]), std_bias=0.0),
        noise=Noise(noise=np.array([0.25]), std_noise=0.0),
    )

    out = sensor.reading(make_state(), make_orbital_state(), dmode=ErrorMode(add_bias=False, add_noise=False, update_bias=False, update_noise=False))

    assert np.allclose(out, np.array([1.0]))


def test_sensor_default_jacobians_are_zero_or_empty():
    sensor = DummySensor([1.0, 2.0, 3.0])

    assert np.allclose(sensor.basestate_jac(make_state(), make_orbital_state()), np.zeros((7, 3)))
    assert sensor.bias_jac(make_state(), make_orbital_state()).shape == (0, 3)
