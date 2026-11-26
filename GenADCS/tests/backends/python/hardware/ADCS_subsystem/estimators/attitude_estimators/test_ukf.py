from typing import cast

import numpy as np
import pytest

from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.environment import DE421, IGRF14, Earth, Environment
from GenADCS.backends.python.environment.ephemeris import EphemerisModel
from GenADCS.backends.python.hardware import Satellite, SatelliteGeometry
from GenADCS.backends.python.hardware.ADCS_subsystem import ADCSSubsystem
from GenADCS.backends.python.hardware.ADCS_subsystem.estimator import (
    AttitudeSequentialUKF,
)
from GenADCS.backends.python.hardware.ADCS_subsystem.estimator.attitude_estimator.attitude_sequential_ukf import (
    predict,
    step,
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
    quaternion_mean,
    rotation_vector_to_quaternion,
)
from GenADCS.backends.python.propagation import CoupledRK4Propagator
from GenADCS.backends.python.simulate import ResultsFormat, simulate
from GenADCS.backends.python.solver import CoupledSolver
from GenADCS.backends.python.state import AttitudeState, OrbitalState, State

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def test_ukf_constructor_validates_state_and_hyperparameters() -> None:
    q_unnormalized = np.array([2.0, 0.0, 0.0, 0.0])
    w = np.array([0.1, -0.2, 0.3])
    p = np.eye(6) * 0.1
    estimator = AttitudeSequentialUKF(
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

    q_out, w_out, p_out, a_out = estimator.prepare()
    q_out[0] = 99.0
    assert estimator.quaternion[0] == pytest.approx(1.0)
    assert w_out.shape == (3,)
    assert p_out.shape == (6, 6)
    assert a_out.shape == (3, 3)

    with pytest.raises(ValueError):
        AttitudeSequentialUKF(IDENTITY, w, np.eye(7), 1e-5)
    with pytest.raises(ValueError):
        AttitudeSequentialUKF(IDENTITY, w, p, 1e-5, alpha=0.0)
    with pytest.raises(ValueError):
        AttitudeSequentialUKF(IDENTITY, w, p, 1e-5, alpha=-0.5)
    with pytest.raises(ValueError):
        AttitudeSequentialUKF(IDENTITY, w, p, 1e-5, beta=np.nan)
    with pytest.raises(ValueError):
        AttitudeSequentialUKF(IDENTITY, w, p, 1e-5, kappa=-6.0)
    with pytest.raises(ValueError):
        AttitudeSequentialUKF(IDENTITY, w, p, 1e-5, record=("unknown_field",))


def test_quaternion_mean_recovers_symmetric_weights() -> None:
    dq = rotation_vector_to_quaternion(np.array([0.1, -0.05, 0.02]))
    quaternions = np.vstack([IDENTITY, dq, np.array([dq[0], -dq[1], -dq[2], -dq[3]])])
    weights = np.array([0.5, 0.25, 0.25])
    mean = quaternion_mean(quaternions, weights)
    np.testing.assert_allclose(mean, IDENTITY, atol=1e-12)


def test_ukf_prediction_propagates_quaternion_and_covariance() -> None:
    p0 = np.eye(6) * 0.01
    q0, w0, p_zero = predict(IDENTITY, np.zeros(3), p0, np.zeros((3, 3)), 0.0)
    np.testing.assert_allclose(q0, IDENTITY, atol=1e-13)
    np.testing.assert_allclose(w0, np.zeros(3), atol=1e-14)
    np.testing.assert_allclose(p_zero, p0, atol=1e-12)

    w = np.array([0.2, -0.1, 0.05])
    psd = np.eye(3) * 2e-5
    q1, w1, p1 = predict(IDENTITY, w, np.eye(6) * 1e-8, psd, 1.0)
    np.testing.assert_allclose(q1, rotation_vector_to_quaternion(-w), atol=1e-7)
    np.testing.assert_allclose(w1, w, atol=1e-13)
    np.testing.assert_allclose(np.linalg.norm(q1), 1.0, atol=1e-14)
    np.testing.assert_allclose(p1, p1.T, atol=1e-14)
    assert np.linalg.eigvalsh(p1).min() >= -1e-12

    with pytest.raises(ValueError):
        predict(IDENTITY, w, p0, psd, -0.1)


def test_ukf_step_consumes_all_attitude_sensors() -> None:
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
        Environment(
            ephemeris=cast(EphemerisModel, object()),
            magnetic_field=IGRF14(),
        ),
    )

    # Build clean readings matching q_true, w_true across all 6 sensors.
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
    diagnostics = np.full(53, np.nan)
    flags = 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512

    q_out, w_out, p_out = step(
        q_est,
        w_est,
        p_est,
        np.eye(3) * 1e-7,
        0.0,
        readings,
        config,
        flags=flags,
        truth_q=q_true,
        truth_w=w_true,
        diagnostics=diagnostics,
        references=refs,
    )

    err_before = np.linalg.norm(quaternion_to_dcm(q_est) - quaternion_to_dcm(q_true))
    err_after = np.linalg.norm(quaternion_to_dcm(q_out) - quaternion_to_dcm(q_true))
    assert err_after < err_before * 0.1
    assert np.linalg.norm(w_out - w_true) < 1e-2
    assert np.linalg.eigvalsh(p_out).min() > -1e-12
    assert np.isfinite(diagnostics[12])  # NIS
    assert np.isfinite(diagnostics[13])  # NEES
    assert np.all(np.isfinite(diagnostics[35:53]))  # att & rate covariances


def test_ukf_direction_sensors_individually_and_nan_skipped() -> None:
    q_true = IDENTITY.copy()
    refs = np.array([[0.3, 0.2, 1.0], [0.4, 0.3, 1.0], [-0.3, 0.4, 1.0]])
    for kind in (1, 3, 4, 5):
        config = (
            np.array([kind]),
            q_true[None, :],
            q_true[None, :],
            np.zeros((1, 6)),
            np.eye(6)[None, :, :] * 0.001,
            np.array([[0.0, 0.0, 1.0]]),
            np.ones((1, 2)),
            np.array([[100.0, 1.0, 0.001]]),
            np.ones(1, dtype=np.bool_),
            np.zeros(1, dtype=np.int64),
            np.zeros(8, dtype=np.int64),
            np.zeros(8, dtype=np.int64),
        )
        readings = measurement(q_true, 0, config, refs)
        initial = rotation_vector_to_quaternion(np.array([0.04, -0.03, 0.01]))
        out, _, p = step(
            initial,
            np.zeros(3),
            np.eye(6) * 1e-2,
            np.zeros((3, 3)),
            0.0,
            readings,
            config,
            references=refs,
        )
        assert np.linalg.norm(measurement(out, 0, config, refs) - readings) < np.linalg.norm(
            measurement(initial, 0, config, refs) - readings
        )
        assert np.linalg.eigvalsh(p).min() > -1e-10

        # NaN reading is skipped without altering predicted state.
        nan_out, _, _ = step(
            initial,
            np.zeros(3),
            np.eye(6) * 0.1,
            np.zeros((3, 3)),
            0.0,
            np.full_like(readings, np.nan),
            config,
            references=refs,
        )
        np.testing.assert_allclose(nan_out, initial, atol=1e-12)


def test_ukf_runs_in_coupled_simulation() -> None:
    sensors: list[Sensor] = [
        Gyroscope(np.zeros(3), IDENTITY, IDENTITY, 10.0, np.eye(3) * 1e-8, np.zeros(3)),
        AutonomousStarTracker(np.zeros(3), IDENTITY, IDENTITY, 10.0, np.eye(3) * 1e-6, np.zeros(3)),
    ]
    estimator = AttitudeSequentialUKF(
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
