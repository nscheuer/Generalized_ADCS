"""A sensor that some sigma points cannot see is dropped from that update, not fatal.

Before this change ``correct()`` raised ``ValueError: UKF sigma-point prediction
for <sensor> contains non-finite values`` as soon as one sample point fell
outside a sensor's field of view, even when the estimate itself was well
inside it. That is acquisition with a narrow-field sensor, exactly when the
filter is needed, and the MEKF (which only evaluates the estimate) never had
the problem.
"""

from __future__ import annotations

import numpy as np
import pytest

from ADCS.estimators.attitude_estimators import SRUKF, UKF
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.errors import Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import EarthHorizonSensor, Gyro
from ADCS.state import EstimatorState, State


def _orbital_state() -> Orbital_State:
    # Nadir in inertial axes is [-1, 0, 0].
    return Orbital_State(ephem=Ephemeris(), J2000=0.22, R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]))


def _horizon(fov_deg: float = 10.0, offset_deg: float = 8.0) -> EarthHorizonSensor:
    """Boresight ``offset_deg`` away from nadir at the identity attitude: visible, but near the edge."""
    a = np.deg2rad(offset_deg)
    return EarthHorizonSensor(boresight=np.array([-np.cos(a), np.sin(a), 0.0]), fov=np.deg2rad(fov_deg),
                              noise=Noise(noise=np.zeros(3), std_noise=np.full(3, 1.0e-3)))


def _satellite(with_gyros: bool = True) -> EstimatedSatellite:
    sensors = [Gyro(axis, noise=Noise(std_noise=1.0e-4)) for axis in np.eye(3)] if with_gyros else []
    return EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=sensors + [_horizon()])


def _prior(attitude_std_rad: float) -> EstimatorState:
    cov = np.diag([1.0e-6] * 3 + [attitude_std_rad**2] * 3)
    return EstimatorState(w=[0.001, 0.002, -0.001], q=[1.0, 0.0, 0.0, 0.0], cov=cov, int_cov=np.zeros((6, 6)))


def _measurements(satellite: EstimatedSatellite, truth: State, orbital_state: Orbital_State) -> np.ndarray:
    return satellite.noiseless_sensor_readings(truth, orbital_state)


@pytest.mark.parametrize("filter_type", [UKF, SRUKF])
def test_update_proceeds_when_some_sample_points_cannot_see_the_sensor(filter_type):
    satellite = _satellite()
    orbital_state = _orbital_state()
    truth = State(w=np.array([0.01, 0.002, -0.001]), q=[1.0, 0.0, 0.0, 0.0])
    estimator = filter_type(satellite, _prior(0.05), dt=1.0)  # 2.9 deg 1-sigma: outer points 7 deg away
    stack = satellite.measurement_stack
    horizon = [entry.name for entry in stack.entries if "horizon" in entry.name.lower()][0]
    prior_rate = estimator.state.w.copy()
    corrected = estimator.correct(_measurements(satellite, truth, orbital_state), orbital_state)  # used to raise
    assert estimator.diagnostics["dropped_entries"] == [horizon]
    assert not estimator.diagnostics["active_mask"][[e.name for e in stack.entries].index(horizon)]
    # the gyros still corrected the rate toward the measured value
    assert np.linalg.norm(corrected.w - truth.w) < np.linalg.norm(prior_rate - truth.w)


def test_nothing_is_dropped_when_every_sample_point_sees_the_sensor():
    satellite = _satellite()
    orbital_state = _orbital_state()
    truth = State(w=np.array([0.001, 0.002, -0.001]), q=[1.0, 0.0, 0.0, 0.0])
    estimator = UKF(satellite, _prior(0.005), dt=1.0)  # 0.3 deg: outer points 0.7 deg away
    estimator.correct(_measurements(satellite, truth, orbital_state), orbital_state)
    assert estimator.diagnostics["dropped_entries"] == []
    assert bool(np.all(estimator.diagnostics["active_mask"]))


def test_an_only_sensor_that_is_unavailable_skips_the_update_gracefully():
    satellite = _satellite(with_gyros=False)
    orbital_state = _orbital_state()
    truth = State(w=np.array([0.001, 0.002, -0.001]), q=[1.0, 0.0, 0.0, 0.0])
    estimator = UKF(satellite, _prior(0.05), dt=1.0)
    before = estimator.state
    after = estimator.correct(_measurements(satellite, truth, orbital_state), orbital_state)
    np.testing.assert_allclose(after.as_array(), before.as_array())
    np.testing.assert_allclose(after.cov, before.cov)
    assert len(estimator.diagnostics["dropped_entries"]) == 1
    assert estimator.diagnostics["innovation"].size == 0


def test_sensor_is_used_again_once_the_uncertainty_shrinks():
    satellite = _satellite()
    orbital_state = _orbital_state()
    truth = State(w=np.array([0.001, 0.002, -0.001]), q=[1.0, 0.0, 0.0, 0.0])
    estimator = UKF(satellite, _prior(0.05), dt=1.0)
    measurements = _measurements(satellite, truth, orbital_state)
    estimator.correct(measurements, orbital_state)
    assert len(estimator.diagnostics["dropped_entries"]) == 1
    # a confident prior (the estimate is right, the spread is small) brings it back
    estimator.reset(_prior(0.005)) if hasattr(estimator, "reset") else None
    estimator = UKF(satellite, _prior(0.005), dt=1.0)
    estimator.correct(measurements, orbital_state)
    assert estimator.diagnostics["dropped_entries"] == []
