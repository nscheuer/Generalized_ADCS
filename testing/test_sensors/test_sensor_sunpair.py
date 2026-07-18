import numpy as np
from ADCS.satellite_hardware.errors import Bias, ErrorMode, Noise
from ADCS.satellite_hardware.sensors import SunPair

from testing.test_sensors._helpers import make_orbital_state, make_state


def test_sunpair_clean_reading_uses_front_efficiency_for_positive_projection():
    sensor = SunPair(axis=np.array([1.0, 0.0, 0.0]), efficiency=(0.5, 0.2))
    os = make_orbital_state()
    x = make_state()
    vecs = os.get_state_vector(x=x)
    sun_dir = (vecs["s"] - vecs["r"]) / np.linalg.norm(vecs["s"] - vecs["r"])
    proj = float(np.dot(np.array([1.0, 0.0, 0.0]), sun_dir))

    reading = sensor.clean_reading(x, os)

    expected_eff = 0.5 if proj > 0.0 else 0.2
    assert np.isclose(reading, proj * expected_eff)


def test_sunpair_clean_reading_returns_nan_in_eclipse():
    sensor = SunPair(axis=np.array([1.0, 0.0, 0.0]), efficiency=(0.5, 0.2))
    os = make_orbital_state()
    os._sunlit = False

    assert np.isnan(sensor.clean_reading(make_state(), os))


def test_sunpair_basestate_jacobian_matches_finite_difference():
    sensor = SunPair(axis=np.array([1.0, 1.0, 0.0]), efficiency=(0.4, 0.3))
    x = make_state(q=np.array([0.9, 0.2, 0.3, 0.1]) / np.linalg.norm(np.array([0.9, 0.2, 0.3, 0.1])))
    os = make_orbital_state()

    eps = 1e-6
    jac_fd = np.zeros((7, 1))
    for i in range(7):
        delta = np.zeros(7)
        delta[i] = eps
        xp = type(x).from_array(x.as_array() + delta)
        xm = type(x).from_array(x.as_array() - delta)
        jac_fd[i, 0] = (sensor.clean_reading(xp, os) - sensor.clean_reading(xm, os)) / (2.0 * eps)
    jac = sensor.basestate_jac(x, os)

    assert np.allclose(jac, jac_fd, rtol=1e-4, atol=1e-6)


def test_sunpair_reading_adds_bias_and_noise_deterministically():
    sensor = SunPair(
        axis=np.array([1.0, 0.0, 0.0]),
        efficiency=(0.5, 0.2),
        bias=Bias(bias=0.2, std_bias=0.0),
        noise=Noise(noise=0.05, std_noise=0.0),
    )
    os = make_orbital_state()
    x = make_state()

    clean = sensor.clean_reading(x, os)
    reading = sensor.reading(x, os, dmode=ErrorMode(add_bias=True, add_noise=True, update_bias=False, update_noise=False))

    assert np.isclose(reading, clean + 0.25)
