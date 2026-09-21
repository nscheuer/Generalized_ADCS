"""GPS noise is specified per ECEF axis; the orbit estimators work in ECI.

``Orbit_GPS`` returned the receiver's diagonal covariance as if it were an ECI
covariance, and ``Orbit_EKF`` applied the receiver's diagonal R to innovations
it had already rotated into ECI. Both are right only for equal noise on every
axis, which is what the existing tests used. With a per-axis specification the
estimate's stated uncertainty was attached to the wrong axes: 19,000 on a
consistency statistic that should be about 6. ``Orbit_GPS`` also kept the
noise of the *last* GPS sensor while consuming the measurement of the first.
"""

from __future__ import annotations

import numpy as np
import pytest

from ADCS.estimators.orbit_estimators import Orbit_EKF, Orbit_GPS
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.errors import Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import GPS
from ADCS.state import State

STD = np.array([0.5, 0.005, 0.005, 1.0e-3, 1.0e-5, 1.0e-5])  # km, km/s: very different per axis
J2000 = 0.22  # a non-trivial Earth rotation angle


def _truth() -> Orbital_State:
    return Orbital_State(ephem=Ephemeris(), J2000=J2000, R=np.array([6900.0, 1200.0, -300.0]), V=np.array([-1.3, 7.2, 0.6]))


def _rotation(os: Orbital_State) -> np.ndarray:
    return np.column_stack([os.ecef_to_eci(axis) for axis in np.eye(3)])


def _satellite(*stds) -> EstimatedSatellite:
    return EstimatedSatellite(sensors=[GPS(noise=Noise(noise=np.zeros(6), std_noise=np.asarray(std, float))) for std in (stds or [STD])])


def _ecef_measurement(os: Orbital_State, rng=None) -> np.ndarray:
    clean = np.concatenate([os.ECEF, os.eci_to_ecef(os.V)])
    return clean if rng is None else clean + STD * rng.standard_normal(6)


def test_ecef_to_eci_rotation_is_orthonormal_and_nontrivial():
    rotation = _rotation(_truth())
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert abs(rotation[0, 1]) > 0.1  # the epoch was chosen so the axes really are rotated


def test_orbit_gps_covariance_is_the_receiver_covariance_rotated_into_eci():
    truth = _truth()
    estimator = Orbit_GPS(est_sat=_satellite(), J2000=J2000, os_template=truth)
    estimate = estimator.update([_ecef_measurement(truth)], J2000)
    rotation = _rotation(truth)
    expected = np.zeros((6, 6))
    for block in (slice(0, 3), slice(3, 6)):
        expected[block, block] = rotation @ np.diag(STD[block] ** 2) @ rotation.T
    np.testing.assert_allclose(estimate.P, expected, rtol=1e-12, atol=1e-18)
    assert abs(estimate.P[0, 1]) > 1e-3  # used to be exactly diagonal (the receiver's axes)


def test_orbit_gps_uncertainty_is_consistent_with_anisotropic_noise():
    truth = _truth()
    estimator = Orbit_GPS(est_sat=_satellite(), J2000=J2000, os_template=truth)
    rng = np.random.default_rng(7)
    nees = []
    for _ in range(300):
        estimate = estimator.update([_ecef_measurement(truth, rng)], J2000)
        error = np.concatenate([estimate.os.R - truth.R, estimate.os.V - truth.V])
        nees.append(error @ np.linalg.solve(estimate.P, error))
    assert 4.5 < np.mean(nees) < 7.5  # 6 degrees of freedom; used to be in the thousands


def test_orbit_gps_uses_the_noise_of_the_sensor_it_reads():
    truth = _truth()
    loud, quiet = STD, STD / 100.0
    estimator = Orbit_GPS(est_sat=_satellite(loud, quiet), J2000=J2000, os_template=truth)
    estimate = estimator.update([_ecef_measurement(truth), _ecef_measurement(truth)], J2000)
    rotation = _rotation(truth)
    np.testing.assert_allclose(estimate.P[:3, :3], rotation @ np.diag(loud[:3] ** 2) @ rotation.T, rtol=1e-12)


def test_orbit_ekf_rotates_the_measurement_noise_before_the_update():
    truth = _truth()
    prior = Orbital_State(ephem=truth.ephem, J2000=J2000, R=truth.R + np.array([2.0, -1.0, 0.5]), V=truth.V + np.array([1e-3, -2e-3, 1e-3]))
    P0 = np.diag([4.0, 4.0, 4.0, 1e-5, 1e-5, 1e-5])
    estimator = Orbit_EKF(est_sat=_satellite(), J2000=J2000, os_hat=prior, P_hat=P0, Q_hat=np.zeros((6, 6)), dt=10.0)
    estimate = estimator.update([_ecef_measurement(truth)], J2000)  # same epoch: no propagation
    rotation = _rotation(truth)
    R_eci = np.zeros((6, 6))
    for block in (slice(0, 3), slice(3, 6)):
        R_eci[block, block] = rotation @ np.diag(STD[block] ** 2) @ rotation.T
    gain = P0 @ np.linalg.inv(P0 + R_eci)
    expected_P = (np.eye(6) - gain) @ P0
    np.testing.assert_allclose(estimate.P, 0.5 * (expected_P + expected_P.T), rtol=1e-9, atol=1e-15)
    unrotated_P = (np.eye(6) - P0 @ np.linalg.inv(P0 + np.diag(STD ** 2))) @ P0
    assert not np.allclose(estimate.P, unrotated_P, rtol=1e-3)  # the test can tell the two apart
