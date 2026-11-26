from typing import cast

import numpy as np
import pytest

from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.environment import DE421, IGRF14, Earth, Environment
from GenADCS.backends.python.environment.ephemeris import EphemerisModel
from GenADCS.backends.python.hardware import Satellite, SatelliteGeometry
from GenADCS.backends.python.hardware.ADCS_subsystem import ADCSSubsystem
from GenADCS.backends.python.hardware.ADCS_subsystem.estimator import (
    AttitudeSequentialSRUKF,
)
from GenADCS.backends.python.hardware.ADCS_subsystem.estimator.attitude_estimator.attitude_sequential_srukf import (
    predict,
    step,
)
from GenADCS.backends.python.hardware.ADCS_subsystem.estimator.attitude_estimator.attitude_sequential_ukf import (
    predict as ukf_predict,
    step as ukf_step,
)
from GenADCS.backends.python.hardware.ADCS_subsystem.estimator.attitude_estimator.direction_updates import (
    measurement,
)
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import (
    AutonomousStarTracker,
    CoarseSunSensor,
    FineEarthSensor,
    FineSunSensor,
    Gyroscope,
    Magnetometer,
    Sensor,
)
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors.sensor_sampling import (
    prepare_sensors,
)
from GenADCS.backends.python.helpers.math.rotations.quaternion.conversions import (
    quaternion_to_dcm,
)
from GenADCS.backends.python.helpers.math.rotations.quaternion.interpolation import (
    rotation_vector_to_quaternion,
)
from GenADCS.backends.python.propagation import CoupledRK4Propagator
from GenADCS.backends.python.simulate import ResultsFormat, simulate
from GenADCS.backends.python.solver import CoupledSolver
from GenADCS.backends.python.state import AttitudeState, OrbitalState, State

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def test_srukf_constructor_validates_state_and_hyperparameters() -> None:
    q_unnormalized = np.array([2.0, 0.0, 0.0, 0.0])
    w = np.array([0.1, -0.2, 0.3])
    p = np.eye(6) * 0.1
    estimator = AttitudeSequentialSRUKF(
        q_unnormalized,
        w,
        p,
        1e-5,
        alpha=0.5,
        beta=2.0,
        kappa=0.0,
        record=("estimated_rate",),
    )

    assert estimator.COVARIANCE_SIZE == 6
    np.testing.assert_allclose(estimator.quaternion, IDENTITY, atol=1e-15)
    np.testing.assert_allclose(estimator.rate, w)
    np.testing.assert_allclose(estimator.covariance, p)
    np.testing.assert_allclose(estimator.acceleration_psd, np.eye(3) * 1e-5)
    assert estimator.alpha == 0.5
    assert estimator.beta == 2.0
    assert estimator.kappa == 0.0
    assert estimator.record == frozenset({"estimated_rate"})

    with pytest.raises(ValueError):
        AttitudeSequentialSRUKF(IDENTITY, w, np.eye(7), 1e-5)
    with pytest.raises(ValueError):
        AttitudeSequentialSRUKF(IDENTITY, w, p, 1e-5, alpha=0.0)
    with pytest.raises(ValueError):
        AttitudeSequentialSRUKF(IDENTITY, w, p, 1e-5, beta=np.nan)
    with pytest.raises(ValueError):
        AttitudeSequentialSRUKF(IDENTITY, w, p, 1e-5, kappa=-6.0)


def test_srukf_prediction_matches_ukf_with_positive_and_negative_w0() -> None:
    q0 = rotation_vector_to_quaternion(np.array([0.05, -0.03, 0.02]))
    w = np.array([0.2, -0.1, 0.05])
    p0 = np.eye(6) * 1e-2
    psd = np.diag([1e-5, 2e-5, 1.5e-5])

    for alpha in (1.0, 0.5):
        q_sr, w_sr, p_sr = predict(q0, w, p0, psd, 0.5, alpha=alpha, beta=2.0, kappa=0.0)
        q_ukf, w_ukf, p_ukf = ukf_predict(q0, w, p0, psd, 0.5, alpha=alpha, beta=2.0, kappa=0.0)
        np.testing.assert_allclose(q_sr, q_ukf, atol=1e-12)
        np.testing.assert_allclose(w_sr, w_ukf, atol=1e-12)
        np.testing.assert_allclose(p_sr, p_ukf, atol=1e-12)
        assert np.linalg.eigvalsh(p_sr).min() >= -1e-12

    with pytest.raises(ValueError):
        predict(IDENTITY, w, p0, psd, -0.1)


def test_srukf_step_consumes_all_attitude_sensors_and_matches_ukf() -> None:
    q_true = rotation_vector_to_quaternion(np.array([0.05, -0.03, 0.04]))
    w_true = np.array([0.02, -0.015, 0.01])
    refs = np.array([[0.3, 0.2, 1.0], [0.4, 0.3, 1.0], [0.0, 0.0, 1.0]])

    sensors: list[Sensor] = [
        Gyroscope(np.zeros(3), IDENTITY, IDENTITY, 10.0, np.eye(3) * 1e-6, np.zeros(3)),
        Magnetometer(np.zeros(3), IDENTITY, IDENTITY, 10.0, np.eye(3) * 1e-5, np.zeros(3)),
        AutonomousStarTracker(np.zeros(3), IDENTITY, IDENTITY, 10.0, np.eye(3) * 1e-6, np.zeros(3)),
        CoarseSunSensor(
            np.zeros(3), IDENTITY, IDENTITY, np.array([0.0, 0.0, 1.0]),
            1.0, 1.0, 1e-5, 0.0, np.deg2rad(85.0),
        ),
        FineSunSensor(
            np.zeros(3), IDENTITY, IDENTITY, np.eye(2) * 1e-5, np.zeros(2), np.deg2rad([60.0, 60.0])
        ),
        FineEarthSensor(
            np.zeros(3), IDENTITY, IDENTITY, np.eye(2) * 1e-5, np.zeros(2), np.deg2rad([60.0, 60.0])
        ),
    ]
    config = prepare_sensors(
        sensors,
        Environment(ephemeris=cast(EphemerisModel, object()), magnetic_field=IGRF14()),
    )

    readings = np.empty(15)
    readings[0:3] = w_true
    readings[3:6] = measurement(q_true, 1, config, refs)
    readings[6:10] = q_true
    readings[10:11] = measurement(q_true, 3, config, refs)
    readings[11:13] = measurement(q_true, 4, config, refs)
    readings[13:15] = measurement(q_true, 5, config, refs)

    q_est = rotation_vector_to_quaternion(np.array([-0.04, 0.05, -0.02]))
    w_est = np.array([-0.05, 0.04, -0.03])
    p_est = np.eye(6) * 0.05
    diag_sr = np.full(53, np.nan)
    diag_ukf = np.full(53, np.nan)
    flags = 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512

    q_sr, w_sr, p_sr = step(
        q_est, w_est, p_est, np.eye(3) * 1e-7, 0.0, readings, config,
        flags=flags, truth_q=q_true, truth_w=w_true, diagnostics=diag_sr, references=refs,
    )
    q_ukf, w_ukf, p_ukf = ukf_step(
        q_est, w_est, p_est, np.eye(3) * 1e-7, 0.0, readings, config,
        flags=flags, truth_q=q_true, truth_w=w_true, diagnostics=diag_ukf, references=refs,
    )

    err_before = np.linalg.norm(quaternion_to_dcm(q_est) - quaternion_to_dcm(q_true))
    err_after = np.linalg.norm(quaternion_to_dcm(q_sr) - quaternion_to_dcm(q_true))
    assert err_after < err_before * 0.1
    np.testing.assert_allclose(q_sr, q_ukf, atol=1e-10)
    np.testing.assert_allclose(w_sr, w_ukf, atol=1e-10)
    np.testing.assert_allclose(p_sr, p_ukf, atol=1e-10)
    assert np.isfinite(diag_sr[12])
    assert np.isfinite(diag_sr[13])
    assert np.all(np.isfinite(diag_sr[35:53]))


def test_srukf_runs_in_coupled_simulation() -> None:
    sensors: list[Sensor] = [
        Gyroscope(np.zeros(3), IDENTITY, IDENTITY, 10.0, np.eye(3) * 1e-8, np.zeros(3)),
        AutonomousStarTracker(np.zeros(3), IDENTITY, IDENTITY, 10.0, np.eye(3) * 1e-6, np.zeros(3)),
    ]
    estimator = AttitudeSequentialSRUKF(
        rotation_vector_to_quaternion(np.array([0.05, -0.04, 0.03])),
        np.array([0.01, -0.02, 0.015]),
        np.eye(6) * 1e-2,
        1e-8,
    )
    satellite = Satellite(
        SatelliteGeometry(12.0, np.diag([0.13, 0.10, 0.05]), []),
        ADCSSubsystem(sensors=sensors, estimator=estimator),
    )
    state = State(
        AttitudeState(IDENTITY, np.array([0.01, -0.015, 0.02])),
        OrbitalState.from_cartesian(
            (6_878_137.0, 0.0, 0.0), (0.0, 7_612.6, 0.0), 2_461_000.5
        ),
    )
    solver = CoupledSolver(CoupledRK4Propagator(dt=0.25), tf=2.0)
    result = simulate(
        satellite,
        Environment(planetary_body=Earth(), ephemeris=DE421()),
        DynamicsConfig(forces=[], torques=[]),
        state,
        solver,
        backend="Python",
        resultsformat=ResultsFormat(
            estimated_attitude=True,
            estimated_rate=True,
            estimated_covariance=True,
            normalized_estimation_error_squared=True,
        ),
        seed=7,
    )
    rows = result.satellites["sat_000"]
    assert np.all(np.isfinite(rows.fields["estimated_attitude"]))
    assert np.all(np.isfinite(rows.fields["estimated_rate"]))
    assert rows.fields["estimated_covariance"].shape[1:] == (36,)
    assert np.all(np.isfinite(rows.fields["normalized_estimation_error_squared"]))
