"""The sigma-point spread is kept inside the attitude chart in every chart, not only the default one.

With the default settings the outer sample points sit at 2.45 standard
deviations. In ``quaternion_vector`` the spread was already capped so that no
sample point rotated more than 143.6 degrees; in ``rotation_vector``, ``mrp``,
``two_mrp`` and ``cayley`` it was not, so once a point passed 180 degrees the
retraction folded it back onto the short rotation and a single ``predict``
shrank the attitude covariance (0.75 -> 0.33 in ``mrp`` at a 1-sigma of 0.5)
with nothing to show for it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ADCS.estimators.attitude_estimators import MEKF, UKF
from ADCS.estimators.attitude_estimators.attitude_UKF import _CHART_LIMIT_ANGLE, _CHART_OFFSET_LIMIT
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.errors import Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import Gyro, StarTrackerQuaternion
from ADCS.state import EstimatorState, State

OS0 = Orbital_State(ephem=Ephemeris(), J2000=0.22, R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]), fast=True)
OS1 = Orbital_State(ephem=Ephemeris(), J2000=0.22 + 10.0 * TimeConstants.sec2cent, R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]), fast=True)


def _satellite() -> EstimatedSatellite:
    tracker = StarTrackerQuaternion(noise=Noise(std_noise=np.full(4, 1.0e-3)))
    tracker.clean_reading = lambda *a, **k: np.asarray((k.get("x") or k.get("state") or a[0]).q, float).copy()
    return EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=[Gyro(axis, noise=Noise(std_noise=1.0e-4)) for axis in np.eye(3)] + [tracker])


def _prior(attitude_variance: float) -> EstimatorState:
    return EstimatorState(w=[0.001, 0.002, -0.001], q=[1.0, 0.0, 0.0, 0.0],
                          cov=np.diag([1.0e-6] * 3 + [attitude_variance] * 3), int_cov=np.zeros((6, 6)))


def _attitude_trace(estimator, state) -> float:
    block = state.slice("attitude", coordinates=estimator.covariance_coordinates)
    return float(np.trace(state.covariance.as_matrix()[block, block]))


def _rotation_angle(q) -> float:
    return 2.0 * math.acos(min(1.0, abs(float(q[0]))))


@pytest.mark.parametrize("chart", sorted(_CHART_OFFSET_LIMIT))
def test_every_chart_limit_is_the_same_rotation(chart):
    identity = State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0])
    offset = np.array([0.0, 0.0, 0.0, _CHART_OFFSET_LIMIT[chart], 0.0, 0.0])
    assert _rotation_angle(identity.plus(offset, quaternion_mode=chart).q) == pytest.approx(_CHART_LIMIT_ANGLE, abs=1e-12)
    assert math.degrees(_CHART_LIMIT_ANGLE) == pytest.approx(143.6, abs=0.1)


@pytest.mark.parametrize("chart", ["rotation_vector", "mrp", "two_mrp", "cayley", "quaternion_vector"])
@pytest.mark.parametrize("attitude_variance", [0.25, 1.0, 4.0])
def test_prediction_keeps_the_attitude_covariance_in_every_chart(chart, attitude_variance):
    """A 10 s predict at 1e-3 rad/s should leave the attitude covariance where it was; the MEKF is the reference."""
    prior = _prior(attitude_variance)
    reference = _attitude_trace(*(lambda e: (e, e.predict(np.zeros(0), OS0, OS1)))(MEKF(_satellite(), prior, dt=10.0, quaternion_mode=chart)))
    estimator = UKF(_satellite(), prior, dt=10.0, quaternion_mode=chart)
    predicted = estimator.predict(np.zeros(0), OS0, OS1)
    assert _attitude_trace(estimator, predicted) == pytest.approx(reference, rel=0.02)  # mrp at 0.25 used to give 0.333 for 0.75


@pytest.mark.parametrize("chart", ["rotation_vector", "mrp", "two_mrp", "cayley", "quaternion_vector"])
def test_no_sample_point_rotates_past_the_limit(chart):
    estimator = UKF(_satellite(), _prior(4.0), dt=10.0, quaternion_mode=chart)
    estimator.predict(np.zeros(0), OS0, OS1)
    offsets = np.asarray(estimator.diagnostics["sigma_offsets"], float)
    identity = State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0])
    angles = [_rotation_angle(identity.plus(np.concatenate([np.zeros(3), row[3:6]]), quaternion_mode=chart).q) for row in offsets]
    assert max(angles) <= _CHART_LIMIT_ANGLE + 1e-9


def test_small_priors_are_not_clamped():
    estimator = UKF(_satellite(), _prior(1.0e-4), dt=10.0, quaternion_mode="mrp")
    estimator.predict(np.zeros(0), OS0, OS1)
    gamma = math.sqrt(estimator.alpha**2 * (6 + estimator.kappa))
    offsets = np.asarray(estimator.diagnostics["sigma_offsets"], float)
    assert np.max(np.linalg.norm(offsets[:, 3:6], axis=1)) == pytest.approx(gamma * 1.0e-2, rel=1e-9)


def test_inactive_measurement_cycle_leaves_no_stale_sigma_diagnostics():
    satellite = _satellite()
    estimator = UKF(satellite, _prior(1.0e-4), dt=10.0)
    truth = State(w=np.array([0.001, 0.002, -0.001]), q=[1.0, 0.0, 0.0, 0.0])
    estimator.correct(satellite.noiseless_sensor_readings(truth, OS0), OS0)
    assert estimator.diagnostics["measurement_sigma_deviations"].shape[0] == 13
    estimator.correct(np.full(satellite.measurement_stack.raw_size, np.nan), OS0)  # nothing arrived
    assert estimator.diagnostics["measurement_sigma_deviations"].shape == (0, 0)
    assert estimator.diagnostics["sigma_offsets"].shape[0] == 0
