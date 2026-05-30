import numpy as np

from ADCS.helpers.math_constants import MathConstants
from ADCS.satellite_hardware.errors import Bias, ErrorMode, Noise
from ADCS.satellite_hardware.sensors import GPS

from testing.test_sensors._helpers import make_orbital_state, make_state


def test_gps_clean_reading_returns_ecef_position_and_velocity():
    sensor = GPS()
    os = make_orbital_state()

    reading = sensor.clean_reading(make_state(), os)
    expected = np.concatenate([os.ECEF, os.eci_to_ecef(os.V)])

    assert np.allclose(reading, expected)


def test_gps_clean_reading_has_length_six():
    reading = GPS().clean_reading(make_state(), make_orbital_state())
    assert reading.shape == (6,)


def test_gps_bias_jacobian_is_identity_when_bias_active():
    sensor = GPS(bias=Bias(bias=np.arange(6.0), std_bias=np.zeros(6)))

    assert np.allclose(sensor.bias_jac(make_state(), make_orbital_state()), np.eye(6))


def test_gps_orbitrv_jacobian_is_ecef_block_diagonal_rotation():
    sensor = GPS()
    os = make_orbital_state()

    jac = sensor.orbitRV_jac(make_state(), os)
    rotation = np.vstack([os.eci_to_ecef(axis) for axis in MathConstants.unitvecs]).T
    expected = np.block([[rotation, np.zeros((3, 3))], [np.zeros((3, 3)), rotation]])

    assert np.allclose(jac, expected)


def test_gps_reading_adds_bias_and_noise_deterministically():
    sensor = GPS(
        bias=Bias(bias=np.arange(6.0), std_bias=np.zeros(6)),
        noise=Noise(noise=np.ones(6) * 0.5, std_noise=np.zeros(6)),
    )
    os = make_orbital_state()

    clean = sensor.clean_reading(make_state(), os)
    reading = sensor.reading(make_state(), os, dmode=ErrorMode(add_bias=True, add_noise=True, update_bias=False, update_noise=False))

    assert np.allclose(reading, clean + np.arange(6.0) + 0.5)
