"""A noise density scales a reading's noise with the sample rate.

``std_noise`` is the standard deviation of one reading whatever the rate. A
``density`` (reading units per root hertz, the angle random walk of a gyro or
the noise density of a magnetometer data sheet) gives a reading averaged over
``sample_time`` seconds a standard deviation of ``density / sqrt(sample_time)``.
The sensor passes its sample time both when it draws and when the estimator
asks for the measurement covariance, so the two sides agree.
"""

from __future__ import annotations

import numpy as np
import pytest

from ADCS.estimators.initialization import vector_observations
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.errors import AnisotropicNoise, Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import MTM, Gyro
from ADCS.state import State

TRUTH = State(w=np.array([0.02, -0.01, 0.015]), q=np.array([1.0, 0.0, 0.0, 0.0]))


def _orbital_state() -> Orbital_State:
    return Orbital_State(ephem=Ephemeris(), J2000=0.22, R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]))


def test_density_scales_with_the_sample_time():
    noise = Noise(std_noise=1.0e-3, density=2.0e-3)
    np.testing.assert_allclose(noise.std(0.1), np.sqrt(1.0e-6 + 4.0e-6 / 0.1))
    np.testing.assert_allclose(noise.std(1.0), np.sqrt(1.0e-6 + 4.0e-6))
    np.testing.assert_allclose(noise.std(4.0), np.sqrt(1.0e-6 + 1.0e-6))
    np.testing.assert_allclose(Noise(std_noise=1.0e-3).std(), 1.0e-3)  # no density: rate-independent
    np.testing.assert_allclose(noise.cov(0.1), np.diag(noise.std(0.1) ** 2))
    np.testing.assert_allclose(noise.srcov(0.1), np.diag(noise.std(0.1)))
    assert bool(Noise(density=1.0e-3)) and not bool(Noise())


def test_drawn_readings_have_the_scaled_spread():
    np.random.seed(1)
    for sample_time in (0.1, 1.0):
        gyro = Gyro(np.array([1.0, 0.0, 0.0]), sample_time=sample_time, noise=Noise(density=1.0e-3))
        os = _orbital_state()
        samples = np.array([gyro.reading(TRUTH, os) - TRUTH.w[0] for _ in range(4000)]).ravel()
        assert np.std(samples) == pytest.approx(1.0e-3 / np.sqrt(sample_time), rel=0.06)


def test_estimator_covariance_matches_what_is_drawn():
    gyro = Gyro(np.array([0.0, 1.0, 0.0]), sample_time=0.5, noise=Noise(std_noise=2.0e-4, density=1.0e-3))
    expected = 4.0e-8 + 1.0e-6 / 0.5
    np.testing.assert_allclose(gyro.measurement_covariance().as_matrix(), [[expected]])
    satellite = EstimatedSatellite(J_0=np.eye(3), sensors=[gyro])
    np.testing.assert_allclose(satellite.measurement_covariance().as_matrix(), [[expected]])


def test_a_density_needs_a_rate():
    with pytest.raises(ValueError, match="sample time"):
        Noise(density=1.0e-3).std()
    with pytest.raises(ValueError, match="Gyro.*sample_time"):
        Gyro(np.array([1.0, 0.0, 0.0]), sample_time=0.0, noise=Noise(density=1.0e-3))
    with pytest.raises(ValueError, match="non-negative"):
        Noise(density=-1.0)
    with pytest.raises(ValueError, match="shape"):
        Noise(std_noise=np.zeros(3), density=np.zeros(2))


def test_density_survives_sizing_and_copies():
    sized = Noise(density=1.0e-3).broadcast_to(3)
    np.testing.assert_allclose(sized.density, np.full(3, 1.0e-3))
    np.testing.assert_allclose(sized.copy().density, np.full(3, 1.0e-3))
    anisotropic = AnisotropicNoise(std_cross=1.0e-3, std_roll=2.0e-3)
    anisotropic._update_noise(sample_time=0.1)  # accepted and ignored: a per-reading model
    np.testing.assert_allclose(anisotropic.std(0.1), [1.0e-3, 1.0e-3, 2.0e-3])


def test_warm_start_uses_the_scaled_standard_deviation():
    satellite = EstimatedSatellite(
        J_0=np.eye(3), sensors=[MTM(axis, sample_time=0.25, noise=Noise(density=1.0e-8)) for axis in np.eye(3)],
    )
    os = _orbital_state()
    observation = vector_observations(satellite, satellite.noiseless_sensor_readings(TRUTH, os), os)[0]
    expected_sigma = (1.0e-8 / np.sqrt(0.25)) / np.linalg.norm(os.B)
    assert observation.sigma == pytest.approx(expected_sigma, rel=1.0e-9)
