"""Fast integration tests for GPS-backed orbit estimators.

These tests used to sit beside orbit-model tests.  They exercise estimator
contracts, so they live with the other estimator tests while retaining the
important navigation regression coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

from ADCS.estimators.orbit_estimators import Orbit_EKF, Orbit_GPS
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.errors import Noise
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import GPS


R0 = np.array([7000.0, 1200.0, -800.0])
V0 = np.array([1.1, 7.4, 2.0])


@pytest.fixture(scope="module")
def ephem():
    return Ephemeris()


def make_truth(ephem, *, j2000=0.22):
    return Orbital_State(ephem=ephem, J2000=j2000, R=R0, V=V0, fast=True)


def make_gps(std=0.01):
    return GPS(noise=Noise(noise=np.zeros(6), std_noise=np.full(6, std)))


def make_gps_estimator(ephem, *, sensors=None):
    sensors = [make_gps(0.0)] if sensors is None else sensors
    estimated = EstimatedSatellite.from_satellite(Satellite(sensors=sensors))
    return Orbit_GPS(est_sat=estimated, J2000=0.22, os_template=make_truth(ephem))


def make_ekf(ephem, *, sensors=None):
    sensors = [make_gps()] if sensors is None else sensors
    estimated = EstimatedSatellite(sensors=sensors)
    prior = Orbital_State(ephem=ephem, J2000=0.22 - TimeConstants.sec2cent, R=[7000.0, 7000.0, 0.0], V=[0.0, 0.0, 8.0], fast=True)
    return Orbit_EKF(
        est_sat=estimated,
        J2000=prior.J2000,
        os_hat=prior,
        P_hat=np.diag([500.0**2] * 3 + [0.5**2] * 3),
        Q_hat=np.diag([1.0, 1.0, 1.0, 10.0, 10.0, 10.0]),
        dt=10.0,
    )


@pytest.mark.parametrize("measurement", ["full", "position"])
def test_orbit_gps_recovers_noiseless_position_and_velocity(ephem, measurement):
    truth = make_truth(ephem)
    gps = make_gps(0.0)
    value = gps.clean_reading(None, truth)
    if measurement == "position":
        value = value[:3]
    estimate = make_gps_estimator(ephem).update([value], truth.J2000)
    np.testing.assert_allclose(estimate.os.R, truth.R, rtol=0.0, atol=1e-6)
    if measurement == "full":
        np.testing.assert_allclose(estimate.os.V, truth.V, rtol=0.0, atol=1e-6)


def test_orbit_gps_empty_measurements_return_the_prior(ephem):
    estimator = make_gps_estimator(ephem)
    assert estimator.update([], 0.22) is estimator.os_hat


def test_orbit_gps_requires_a_gps_sensor(ephem):
    estimated = EstimatedSatellite.from_satellite(Satellite(sensors=[]))
    with pytest.raises(ValueError):
        Orbit_GPS(est_sat=estimated, J2000=0.22, os_template=make_truth(ephem))


def test_ekf_constructs_block_measurement_covariance(ephem):
    ekf = make_ekf(ephem, sensors=[make_gps(0.1), make_gps(0.2)])
    assert ekf.R.shape == (12, 12)
    assert np.allclose(ekf.R, np.diag([0.01] * 6 + [0.04] * 6))


@pytest.mark.parametrize("p_shape, q_shape", [(5, 6), (6, 5)])
def test_ekf_rejects_invalid_covariance_shapes(ephem, p_shape, q_shape):
    estimated = EstimatedSatellite(sensors=[make_gps()])
    prior = make_truth(ephem)
    with pytest.raises(ValueError):
        Orbit_EKF(est_sat=estimated, J2000=prior.J2000, os_hat=prior, P_hat=np.eye(p_shape), Q_hat=np.eye(q_shape), dt=10.0)


def test_ekf_without_measurements_returns_a_symmetric_prediction(ephem):
    ekf = make_ekf(ephem)
    previous = ekf.os_hat.os.copy()
    time = ekf.os_hat.os.J2000 + 10.0 * TimeConstants.sec2cent
    updated = ekf.update([], time)
    expected = previous.propagate_orbit_rk4(10.0, zonal_J=2)
    assert np.allclose(updated.os.R, expected.R)
    assert np.allclose(updated.os.V, expected.V)
    assert np.allclose(updated.P, updated.P.T)


def test_ekf_accepts_position_only_measurements_and_rejects_invalid_lengths(ephem):
    ekf = make_ekf(ephem)
    truth = make_truth(ephem)
    result = ekf.update([truth.ECEF], truth.J2000)
    assert np.all(np.isfinite(result.os.R))
    with pytest.raises(ValueError):
        ekf.update([np.zeros(4)], truth.J2000)


def test_ekf_clean_gps_update_reduces_measurement_error(ephem):
    truth = make_truth(ephem)
    ekf = make_ekf(ephem)
    predicted = ekf.os_hat.os.propagate_orbit_rk4(10.0, zonal_J=2)
    measurement = make_gps(0.0).clean_reading(None, truth)
    updated = ekf.update([measurement], truth.J2000)
    target = np.concatenate([truth.R, truth.V])
    assert np.linalg.norm(np.concatenate([updated.os.R, updated.os.V]) - target) < np.linalg.norm(np.concatenate([predicted.R, predicted.V]) - target)
