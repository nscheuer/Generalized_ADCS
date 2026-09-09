"""Fast, shared integration matrix for all supported attitude estimators.

Each test is deliberately deterministic: it validates a public estimator
contract or a physically meaningful correction without Monte-Carlo sampling.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest

from ADCS.estimators.attitude_estimators import (
    AugmentedEKF,
    AugmentedMEKF,
    AugmentedSRUKF,
    AugmentedUKF,
    EKF,
    MEKF,
    SRUKF,
    UKF,
)
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import RW
from ADCS.satellite_hardware.errors import AnisotropicNoise, Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import (
    EarthHorizonSensor,
    Gyro,
    MTM,
    StarTracker,
    StarTrackerQuaternion,
    SunPair,
    SunSensor,
)
from ADCS.state import EstimatorState


FILTERS = [
    pytest.param(EKF, 7, id="ekf"),
    pytest.param(MEKF, 6, id="mekf"),
    pytest.param(UKF, 6, id="ukf"),
    pytest.param(SRUKF, 6, id="srukf"),
    pytest.param(AugmentedEKF, 7, id="augmented_ekf"),
    pytest.param(AugmentedMEKF, 6, id="augmented_mekf"),
    pytest.param(AugmentedUKF, 6, id="augmented_ukf"),
    pytest.param(AugmentedSRUKF, 6, id="augmented_srukf"),
]


def _orbital_state() -> Orbital_State:
    result = Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 7.5, 0.0]),
        B=np.array([2.0e-5, -1.0e-5, 3.0e-5]),
        S=np.array([1.5e8, 1.0e7, -2.0e7]),
        rho=0.0,
        fast=True,
    )
    result._sunlit = True
    return result


def _state(covariance_size: int, *, wheel_momentum=()) -> EstimatorState:
    return EstimatorState(
        w=[0.01, -0.02, 0.015],
        q=[1.0, 0.0, 0.0, 0.0],
        h=wheel_momentum,
        cov=np.eye(covariance_size + len(wheel_momentum)) * 0.1,
        int_cov=np.zeros(
            (covariance_size + len(wheel_momentum), covariance_size + len(wheel_momentum))
        ),
    )


def _tracker_vector() -> StarTracker:
    sensor = StarTracker(
        fov=np.pi,
        anisotropic_noise=AnisotropicNoise(std_cross=1.0e-3, std_roll=1.0e-3),
    )
    sensor._select_star = lambda q, os: SimpleNamespace(
        s_eci=np.array([0.3, -0.4, np.sqrt(0.75)]), vmag=1.0
    )
    return sensor


def _tracker_quaternion() -> StarTrackerQuaternion:
    sensor = StarTrackerQuaternion(
        fov=np.pi, min_stars=1, noise=Noise(std_noise=np.full(4, 1.0e-3))
    )
    sensor._select_stars = lambda q, os: [
        SimpleNamespace(s_eci=np.array([0.3, -0.4, np.sqrt(0.75)]), vmag=1.0)
    ]
    return sensor


SensorFactory = Callable[[], list]


SENSOR_SETS: list[pytest.ParamSpecArgs] = [
    pytest.param(
        lambda: [Gyro(axis, noise=Noise(std_noise=1.0e-3)) for axis in np.eye(3)],
        id="gyro",
    ),
    pytest.param(
        lambda: [MTM(axis, noise=Noise(std_noise=1.0e-8)) for axis in np.eye(3)],
        id="magnetometer",
    ),
    pytest.param(
        lambda: [
            SunSensor(axis, efficiency=0.8, noise=Noise(std_noise=1.0e-3))
            for axis in np.eye(3)
        ],
        id="sun_sensor",
    ),
    pytest.param(
        lambda: [
            SunPair(axis, efficiency=(0.8, 0.8), noise=Noise(std_noise=1.0e-3))
            for axis in np.eye(3)
        ],
        id="sun_pair",
    ),
    pytest.param(lambda: [_tracker_vector()], id="star_tracker"),
    pytest.param(lambda: [_tracker_quaternion()], id="star_tracker_quaternion"),
    pytest.param(
        lambda: [
            EarthHorizonSensor(
                boresight=[-1.0, 0.0, 0.0],
                fov=np.pi,
                noise=Noise(std_noise=np.full(3, 1.0e-3)),
            )
        ],
        id="earth_horizon",
    ),
]


@pytest.mark.parametrize("filter_type, covariance_size", FILTERS)
def test_estimator_public_contract_and_reset_are_owned(filter_type, covariance_size):
    satellite = EstimatedSatellite(sensors=[_tracker_quaternion()])
    estimator = filter_type(satellite, _state(covariance_size), dt=0.1)
    orbital_state = _orbital_state()
    measurement = satellite.measurement_stack.predict(estimator.state, orbital_state)

    initial = estimator.state
    initial.w[0] = 99.0
    assert estimator.state.w[0] != 99.0
    assert estimator.covariance_coordinates in {"full", "tangent"}
    assert estimator.correction_mode
    assert estimator.measurement_quaternion_mode

    predicted = estimator.predict(
        np.empty(0), orbital_state, orbital_state, midpoint_orbital_state=orbital_state
    )
    corrected = estimator.step(measurement, orbital_state)
    committed = estimator.update()
    reset = estimator.reset(initial)

    assert predicted.covariance.shape == (covariance_size, covariance_size)
    assert corrected.covariance.shape == (covariance_size, covariance_size)
    np.testing.assert_allclose(committed.as_estimator_array(), corrected.as_estimator_array())
    np.testing.assert_allclose(reset.as_estimator_array(), initial.as_estimator_array())
    assert estimator.diagnostics == {}


@pytest.mark.parametrize("filter_type, covariance_size", FILTERS)
@pytest.mark.parametrize("sensor_factory", SENSOR_SETS)
def test_each_sensor_type_contributes_a_nonzero_correction(
    filter_type, covariance_size, sensor_factory: SensorFactory
):
    satellite = EstimatedSatellite(sensors=sensor_factory())
    estimator = filter_type(satellite, _state(covariance_size), dt=0.1)
    orbital_state = _orbital_state()
    truth = estimator.state.plus(
        [0.03, -0.02, 0.01, 0.04, -0.03, 0.02],
        quaternion_mode="quaternion_vector",
    )
    measurements = satellite.measurement_stack.predict(truth, orbital_state)
    before = estimator.state
    corrected = estimator.step(measurements, orbital_state)

    diagnostics = estimator.diagnostics
    assert diagnostics["innovation"].size > 0
    assert np.linalg.norm(diagnostics["correction"]) > 0.0
    assert np.linalg.norm(truth.minus(corrected)) < np.linalg.norm(truth.minus(before))


@pytest.mark.parametrize("filter_type, covariance_size", FILTERS)
def test_estimator_tracks_a_slewing_reaction_wheel(filter_type, covariance_size):
    wheel = RW(
        axis=np.array([1.0, 0.0, 0.0]),
        max_torque=0.02,
        J=0.001,
        h=0.01,
        h_max=1.0,
        h_meas_noise=Noise(std_noise=1.0e-5),
    )
    tracker = _tracker_quaternion()
    satellite = EstimatedSatellite(
        J_0=np.diag([0.5, 0.8, 1.2]), actuators=[wheel], sensors=[tracker]
    )
    estimator = filter_type(satellite, _state(covariance_size, wheel_momentum=[0.01]), dt=0.1)
    orbital_state = _orbital_state()
    control = np.array([1.0e-5])
    truth = estimator.state.plus(
        [0.0, 0.0, 0.0, 0.03, -0.02, 0.01, 0.0],
        quaternion_mode="quaternion_vector",
    )
    measurement = satellite.measurement_stack.predict(truth, orbital_state)
    before = estimator.state

    predicted = estimator.predict(
        control, orbital_state, orbital_state, midpoint_orbital_state=orbital_state
    )
    corrected = estimator.step(measurement, orbital_state)

    assert not np.isclose(predicted.h[0], before.h[0])
    assert corrected.covariance.dimension == covariance_size + 1
    assert np.linalg.eigvalsh(corrected.cov).min() >= -1.0e-10
    assert estimator.diagnostics["innovation"].size == 4
