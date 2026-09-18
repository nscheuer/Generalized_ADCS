"""Shared staged predict/step/update contract for all attitude estimators."""

from __future__ import annotations

import numpy as np
import pytest

from ADCS.estimators.attitude_estimators import (
    EKF,
    MEKF,
    UKF,
    SRUKF,
    AugmentedEKF,
    AugmentedMEKF,
    AugmentedUKF,
    AugmentedSRUKF,
)
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.errors import Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import StarTrackerQuaternion
from ADCS.state import EstimatorState


FILTERS = [
    (EKF, 7),
    (MEKF, 6),
    (UKF, 6),
    (SRUKF, 6),
    (AugmentedEKF, 7),
    (AugmentedMEKF, 6),
    (AugmentedUKF, 6),
    (AugmentedSRUKF, 6),
]


def _orbital_state() -> Orbital_State:
    return Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 7.5, 0.0]),
        fast=True,
    )


def _satellite() -> EstimatedSatellite:
    tracker = StarTrackerQuaternion(noise=Noise(std_noise=np.full(4, 1.0e-3)))
    tracker.clean_reading = lambda state, orbital_state: state.q.copy()
    return EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=[tracker])


@pytest.mark.parametrize("filter_type, covariance_size", FILTERS)
def test_all_attitude_estimators_support_staged_contract(filter_type, covariance_size):
    satellite = _satellite()
    orbital_state = _orbital_state()
    state = EstimatorState(
        w=[0.01, -0.02, 0.015],
        q=[1.0, 0.0, 0.0, 0.0],
        cov=np.eye(covariance_size) * 0.1,
        int_cov=np.zeros((covariance_size, covariance_size)),
    )
    estimator = filter_type(satellite, state, dt=0.1)
    measurements = satellite.measurement_stack.predict(estimator.state, orbital_state)

    predicted = estimator.predict(
        np.empty(0), orbital_state, orbital_state,
        midpoint_orbital_state=orbital_state,
    )
    corrected = estimator.step(measurements, orbital_state)
    committed = estimator.update()

    assert predicted.covariance.shape == (covariance_size, covariance_size)
    assert corrected.covariance.shape == (covariance_size, covariance_size)
    np.testing.assert_allclose(
        committed.as_estimator_array(), corrected.as_estimator_array()
    )
