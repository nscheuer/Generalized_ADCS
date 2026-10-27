"""DensityNoise: reading noise given as a density, scaled by the sample time.

A density (angle random walk of a gyro, nT/root-hertz of a magnetometer)
gives a reading averaged over ``sample_time`` seconds a standard deviation of
``density / sqrt(sample_time)``, on top of a rate-independent floor. The
subclass folds that into ``std_noise``, so the reading draw, the estimator's
measurement covariance and the warm start all see the same value. The base
``Noise`` is untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from ADCS.estimators.initialization import vector_observations
from ADCS.environment.ephemeris import DE421 as Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.errors import DensityNoise, Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import MTM, Gyro
from ADCS.satellite_hardware.sensors import GPS
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.state import State

TRUTH = State(w=np.array([0.02, -0.01, 0.015]), q=np.array([1.0, 0.0, 0.0, 0.0]))


def _orbital_state() -> Orbital_State:
    return Orbital_State(ephem=Ephemeris(), J2000=0.22, R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]))


def test_standard_deviation_scales_with_the_sample_time():
    for dt in (0.1, 1.0, 4.0):
        noise = DensityNoise(2.0e-3, dt, std_noise=1.0e-3)
        np.testing.assert_allclose(noise.std_noise, [np.sqrt(1.0e-6 + 4.0e-6 / dt)])
        np.testing.assert_allclose(noise.cov(), [[1.0e-6 + 4.0e-6 / dt]])
    np.testing.assert_allclose(DensityNoise(2.0e-3, 0.1).at_sample_time(4.0).std_noise, [1.0e-3])
    assert isinstance(DensityNoise(1.0e-3, 0.1), Noise)
    assert bool(DensityNoise(1.0e-3, 0.1)) and not bool(DensityNoise(0.0, 0.1))


def test_drawn_readings_have_the_scaled_spread():
    np.random.seed(1)
    for sample_time in (0.1, 1.0):
        gyro = Gyro(np.array([1.0, 0.0, 0.0]), sample_time=sample_time, noise=DensityNoise(1.0e-3, sample_time))
        os = _orbital_state()
        samples = np.array([gyro.reading(TRUTH, os) - TRUTH.w[0] for _ in range(4000)]).ravel()
        assert np.std(samples) == pytest.approx(1.0e-3 / np.sqrt(sample_time), rel=0.06)


def test_estimator_covariance_matches_what_is_drawn():
    gyro = Gyro(np.array([0.0, 1.0, 0.0]), sample_time=0.5, noise=DensityNoise(1.0e-3, 0.5, std_noise=2.0e-4))
    expected = 4.0e-8 + 1.0e-6 / 0.5
    np.testing.assert_allclose(gyro.measurement_covariance().as_matrix(), [[expected]])
    satellite = EstimatedSatellite(J_0=np.eye(3), sensors=[gyro])
    np.testing.assert_allclose(satellite.measurement_covariance().as_matrix(), [[expected]])


def test_configuration_is_validated():
    with pytest.raises(ValueError, match="sample_time"):
        DensityNoise(1.0e-3, 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        DensityNoise(-1.0, 0.1)
    with pytest.raises(ValueError, match="broadcastable"):
        DensityNoise(np.zeros(2), 0.1, std_noise=np.zeros(3))


def test_sizing_and_copies_keep_the_subclass_and_its_density():
    sized = DensityNoise(1.0e-3, 0.25).broadcast_to(3)
    assert isinstance(sized, DensityNoise)
    np.testing.assert_allclose(sized.density, np.full(3, 1.0e-3))
    np.testing.assert_allclose(sized.std_noise, np.full(3, 2.0e-3))
    copied = sized.copy()
    assert isinstance(copied, DensityNoise) and copied.sample_time == 0.25
    np.testing.assert_allclose(copied.std_floor, np.zeros(3))
    with pytest.raises(ValueError, match="entries"):
        DensityNoise(np.ones(2), 0.25).broadcast_to(3)
    mtm = MTM(np.array([1.0, 0.0, 0.0]), sample_time=0.25, noise=DensityNoise(1.0e-8, 0.25))
    assert isinstance(mtm.noise, DensityNoise)


def test_warm_start_uses_the_scaled_standard_deviation():
    satellite = EstimatedSatellite(
        J_0=np.eye(3), sensors=[MTM(axis, sample_time=0.25, noise=DensityNoise(1.0e-8, 0.25)) for axis in np.eye(3)],
    )
    os = _orbital_state()
    observation = vector_observations(satellite, satellite.noiseless_sensor_readings(TRUTH, os), os)[0]
    expected_sigma = (1.0e-8 / np.sqrt(0.25)) / np.linalg.norm(os.B)
    assert observation.sigma == pytest.approx(expected_sigma, rel=1.0e-9)


def test_plain_noise_is_unchanged():
    assert not hasattr(Noise(std_noise=1.0e-3), "density")
    np.testing.assert_allclose(Noise(std_noise=1.0e-3).std_noise, [1.0e-3])


def test_noise_interval_is_independent_of_sensor_polling_interval():
    gyro = Gyro(
        np.array([1.0, 0.0, 0.0]), sample_time=0.01,
        noise=DensityNoise(2.0e-3, sample_time=4.0),
    )
    np.testing.assert_allclose(gyro.noise.std_noise, [1.0e-3])
    np.testing.assert_allclose(gyro.measurement_covariance().as_matrix(), [[1.0e-6]])


def test_vector_sensor_broadcast_keeps_density_model():
    gps = GPS(sample_time=0.1, noise=DensityNoise(3.0e-2, sample_time=9.0))
    assert isinstance(gps.noise, DensityNoise)
    np.testing.assert_allclose(gps.noise.std_noise, np.full(6, 1.0e-2))
    np.testing.assert_allclose(gps.measurement_covariance().as_matrix(), np.eye(6) * 1.0e-4)


def test_actuator_and_wheel_measurement_noise_use_density_derived_std():
    mtq = MTQ(np.array([1.0, 0.0, 0.0]), max_torque=1.0, noise=DensityNoise(2.0e-3, sample_time=4.0))
    np.testing.assert_allclose(mtq.control_covariance().as_matrix(), [[1.0e-6]])
    wheel = RW(
        axis=np.array([1.0, 0.0, 0.0]), max_torque=1.0, J=1.0e-3,
        h=0.0, h_max=1.0, h_meas_noise=DensityNoise(3.0e-2, sample_time=9.0),
    )
    assert isinstance(wheel.h_meas_noise, DensityNoise)
    np.testing.assert_allclose(wheel.momentum_measurement_covariance().as_matrix(), [[1.0e-4]])
