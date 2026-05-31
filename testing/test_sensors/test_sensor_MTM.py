import numpy as np
from ADCS.satellite_hardware.errors import Bias, ErrorMode, Noise
from ADCS.satellite_hardware.sensors import MTM

from testing.test_sensors._helpers import make_orbital_state, make_state


def test_mtm_clean_reading_projects_body_field_onto_axis():
    sensor = MTM(axis=np.array([3.0, 0.0, 0.0]))
    os = make_orbital_state()
    x = make_state()
    vecs = os.get_state_vector(x=x)

    reading = sensor.clean_reading(x=x, os=os)

    assert np.isclose(reading, np.dot(vecs["b"], np.array([1.0, 0.0, 0.0])))


def test_mtm_bias_jacobian_matches_bias_presence():
    sensor_with_bias = MTM(axis=np.array([1.0, 0.0, 0.0]), bias=Bias(bias=0.2, std_bias=0.0))
    sensor_without_bias = MTM(axis=np.array([1.0, 0.0, 0.0]))
    x = make_state()
    os = make_orbital_state()

    assert np.allclose(sensor_with_bias.bias_jac(x, os), np.ones((1, 1)))
    assert np.allclose(sensor_without_bias.bias_jac(x, os), np.zeros((0, 1)))


def test_mtm_basestate_jacobian_matches_finite_difference():
    sensor = MTM(axis=np.array([1.0, 2.0, -1.0]))
    x = make_state(q=np.array([0.9, 0.2, 0.3, 0.1]) / np.linalg.norm(np.array([0.9, 0.2, 0.3, 0.1])))
    os = make_orbital_state()

    eps = 1e-6
    jac_fd = np.zeros((7, 1))
    for i in range(7):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        jac_fd[i, 0] = (sensor.clean_reading(xp, os) - sensor.clean_reading(xm, os)) / (2.0 * eps)
    jac = sensor.basestate_jac(x, os)

    assert np.allclose(jac, jac_fd, rtol=1e-4, atol=1e-8)


def test_mtm_reading_adds_bias_and_noise_deterministically():
    sensor = MTM(
        axis=np.array([1.0, 0.0, 0.0]),
        bias=Bias(bias=0.25, std_bias=0.0),
        noise=Noise(noise=0.1, std_noise=0.0),
    )
    x = make_state()
    os = make_orbital_state()

    clean = sensor.clean_reading(x, os)
    reading = sensor.reading(x, os, dmode=ErrorMode(add_bias=True, add_noise=True, update_bias=False, update_noise=False))

    assert np.isclose(reading, clean + 0.35)
