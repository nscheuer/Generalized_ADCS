from typing import cast

import numpy as np
import pytest

from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.environment import Earth, Environment
from GenADCS.backends.python.environment.ephemeris import EphemerisModel
from GenADCS.backends.python.hardware import Satellite, SatelliteGeometry
from GenADCS.backends.python.hardware.ADCS_subsystem import ADCSSubsystem
from GenADCS.backends.python.hardware.ADCS_subsystem.estimator import AttitudeEKF
from GenADCS.backends.python.hardware.ADCS_subsystem.estimator.attitude_estimator.attitude_ekf import (
    correct,
    predict,
    step,
)
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import (
    AutonomousStarTracker,
    Gyroscope,
    Sensor,
)
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors.sensor_sampling import (
    prepare_sensors,
)
from GenADCS.backends.python.helpers.math.rotations.quaternion.interpolation import (
    rotation_vector_to_quaternion,
)
from GenADCS.backends.python.propagation import CoupledRK4Propagator
from GenADCS.backends.python.simulate import ResultsFormat, simulate
from GenADCS.backends.python.solver import CoupledSolver
from GenADCS.backends.python.state import AttitudeState, OrbitalState, State

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def _tangent_eye7(q: np.ndarray, scale: float) -> np.ndarray:
    p = np.eye(7) * scale
    p[:4, :4] -= scale * np.outer(q, q)
    return p


def test_ekf_constructor_validates_and_normalizes_state() -> None:
    q_unnormalized = np.array([2.0, 0.0, 0.0, 0.0])
    w = np.array([0.1, -0.2, 0.3])
    p = np.eye(7) * 0.1
    estimator = AttitudeEKF(q_unnormalized, w, p, 1e-5, record=("estimated_rate",))

    assert estimator.COVARIANCE_SIZE == 7
    np.testing.assert_allclose(estimator.quaternion, IDENTITY, atol=1e-15)
    np.testing.assert_allclose(estimator.rate, w)
    np.testing.assert_allclose(estimator.covariance, p)
    np.testing.assert_allclose(estimator.acceleration_psd, np.eye(3) * 1e-5)
    assert estimator.record == frozenset({"estimated_rate"})

    q_out, w_out, p_out, a_out = estimator.prepare()
    q_out[0] = 99.0
    w_out[0] = 99.0
    p_out[0, 0] = 99.0
    a_out[0, 0] = 99.0
    np.testing.assert_allclose(estimator.quaternion, IDENTITY)
    np.testing.assert_allclose(estimator.rate, w)
    np.testing.assert_allclose(estimator.covariance, p)
    np.testing.assert_allclose(estimator.acceleration_psd, np.eye(3) * 1e-5)

    with pytest.raises(ValueError):
        AttitudeEKF(IDENTITY, w, np.eye(6), 1e-5)
    bad_asym = np.eye(7)
    bad_asym[0, 1] = 0.5
    with pytest.raises(ValueError):
        AttitudeEKF(IDENTITY, w, bad_asym, 1e-5)
    bad_neg = np.eye(7)
    bad_neg[0, 0] = -1.0
    with pytest.raises(ValueError):
        AttitudeEKF(IDENTITY, w, bad_neg, 1e-5)
    with pytest.raises(ValueError):
        AttitudeEKF(np.zeros(4), w, p, 1e-5)
    with pytest.raises(ValueError):
        AttitudeEKF(np.array([1.0, np.nan, 0.0, 0.0]), w, p, 1e-5)
    with pytest.raises(ValueError):
        AttitudeEKF(IDENTITY, np.array([0.0, np.inf, 0.0]), p, 1e-5)
    with pytest.raises(ValueError):
        AttitudeEKF(IDENTITY, w, p, -1e-3)
    with pytest.raises(ValueError):
        AttitudeEKF(IDENTITY, w, p, np.eye(3), record=("unknown_field",))


def test_ekf_prediction_propagates_quaternion_and_projects_tangent_covariance() -> None:
    p0 = np.eye(7)
    q0, w0, p_zero = predict(IDENTITY, np.zeros(3), p0, np.zeros((3, 3)), 0.0)
    np.testing.assert_allclose(q0, IDENTITY, atol=1e-14)
    np.testing.assert_allclose(w0, np.zeros(3), atol=1e-14)
    np.testing.assert_allclose(p_zero[:4, :4] @ q0, np.zeros(4), atol=1e-12)
    np.testing.assert_allclose(p_zero, p_zero.T, atol=1e-14)

    w = np.array([0.2, -0.1, 0.05])
    psd = np.eye(3) * 2e-5
    q1, w1, p1 = predict(IDENTITY, w, _tangent_eye7(IDENTITY, 0.01), psd, 1.0)
    np.testing.assert_allclose(q1, rotation_vector_to_quaternion(-w), atol=1e-14)
    np.testing.assert_allclose(w1, w, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(q1), 1.0, atol=1e-14)
    np.testing.assert_allclose(p1[:4, :4] @ q1, np.zeros(4), atol=1e-12)
    np.testing.assert_allclose(p1, p1.T, atol=1e-14)
    assert np.linalg.eigvalsh(p1).min() >= -1e-12

    with pytest.raises(ValueError):
        predict(IDENTITY, w, p0, psd, -0.1)


def test_ekf_correct_reduces_error_and_projects_tangent_covariance() -> None:
    q = rotation_vector_to_quaternion(np.array([0.1, -0.05, 0.02]))
    w = np.array([0.5, -0.4, 0.3])
    p = _tangent_eye7(q, 0.1)
    r = np.eye(3) * 1e-5

    h_gyro = np.zeros((3, 7))
    h_gyro[:, 4:] = np.eye(3)
    q_g, w_g, p_g = correct(q, w, p, -w, h_gyro, r)
    assert np.linalg.norm(w_g) < 1e-3
    assert p_g[4, 4] < p[4, 4]
    np.testing.assert_allclose(np.linalg.norm(q_g), 1.0, atol=1e-14)
    np.testing.assert_allclose(p_g, p_g.T, atol=1e-14)
    np.testing.assert_allclose(p_g[:4, :4] @ q_g, np.zeros(4), atol=1e-12)
    assert np.linalg.eigvalsh(p_g).min() >= -1e-12


def test_ekf_step_fuses_gyro_and_star_tracker_and_skips_nan() -> None:
    gyro = Gyroscope(
        np.zeros(3), IDENTITY, IDENTITY, 10.0, np.eye(3) * 1e-6, np.zeros(3)
    )
    tracker = AutonomousStarTracker(
        np.zeros(3), IDENTITY, IDENTITY, 10.0, np.eye(3) * 1e-6, np.zeros(3)
    )
    config = prepare_sensors(
        cast(list[Sensor], [gyro, tracker]),
        Environment(ephemeris=cast(EphemerisModel, object())),
    )

    q_init = rotation_vector_to_quaternion(np.array([0.1, -0.05, 0.02]))
    w_init = np.array([0.5, -0.3, 0.2])
    p_init = np.eye(7) * 0.1
    psd = np.eye(3) * 1e-8
    readings = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    diagnostics = np.full(53, np.nan)
    flags = 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512
    q, w, p = step(
        q_init,
        w_init,
        p_init,
        psd,
        0.1,
        readings,
        config,
        flags=flags,
        truth_q=IDENTITY,
        truth_w=np.zeros(3),
        diagnostics=diagnostics,
    )
    assert np.linalg.norm(w) < 0.01
    assert np.linalg.norm(q[1:]) < 0.01
    np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-14)
    np.testing.assert_allclose(p, p.T, atol=1e-14)
    np.testing.assert_allclose(p[:4, :4] @ q, np.zeros(4), atol=1e-12)
    assert np.linalg.eigvalsh(p).min() >= -1e-12
    assert np.all(np.isfinite(diagnostics))
    assert diagnostics[12] >= 0.0  # NIS
    assert diagnostics[13] >= 0.0  # NEES

    nan_readings = np.full(7, np.nan)
    q2, w2, p2 = step(q, w, p, np.zeros((3, 3)), 0.1, nan_readings, config)
    np.testing.assert_allclose(w2, w, atol=1e-14)
    assert np.all(np.isfinite(q2))
    assert np.all(np.isfinite(p2))


def test_ekf_simulation_loop_and_record_selection() -> None:
    gyro = Gyroscope(
        np.zeros(3), IDENTITY, IDENTITY, 10.0, np.eye(3) * 1e-8, np.zeros(3)
    )
    estimator = AttitudeEKF(
        IDENTITY, np.array([0.1, 0.1, 0.1]), np.eye(7) * 0.1, 1e-8
    )
    satellite = Satellite(
        SatelliteGeometry(12.0, np.diag([0.13, 0.1, 0.05]), []),
        ADCSSubsystem(sensors=cast(list[Sensor], [gyro]), estimator=estimator),
    )
    state = State(
        AttitudeState(IDENTITY, (0.01, -0.015, 0.02)),
        OrbitalState.from_cartesian(
            (6878137.0, 0.0, 0.0), (0.0, 7612.6, 0.0), 2461000.5
        ),
    )
    rows = simulate(
        satellite,
        Environment(planetary_body=Earth()),
        DynamicsConfig(),
        state,
        CoupledSolver(CoupledRK4Propagator(0.25), tf=0.25),
        resultsformat=ResultsFormat(
            estimated_attitude=True,
            estimated_rate=True,
            estimated_covariance=True,
            estimated_attitude_covariance=True,
            estimated_rate_covariance=True,
            measurement_innovation=True,
            measurement_innovation_covariance=True,
            normalized_innovation_squared=True,
            normalized_estimation_error_squared=True,
            measurement_residual=True,
            kalman_gain=True,
        ),
    ).satellites["sat_000"]

    assert rows.fields["estimated_attitude"].shape == (2, 4)
    assert rows.fields["estimated_covariance"].shape == (2, 49)
    assert rows.fields["kalman_gain"].shape == (2, 18)
    assert np.all(np.isfinite(rows.fields["normalized_innovation_squared"]))
    assert np.all(np.isfinite(rows.fields["normalized_estimation_error_squared"]))
    np.testing.assert_allclose(
        rows.fields["estimated_rate"][-1], np.asarray(state.attitude.rates), atol=0.005
    )

    filtered_estimator = AttitudeEKF(
        IDENTITY,
        np.zeros(3),
        np.eye(7),
        1e-8,
        record=("normalized_innovation_squared",),
    )
    filtered_sat = Satellite(
        SatelliteGeometry(12.0, np.eye(3), []),
        ADCSSubsystem(sensors=cast(list[Sensor], [gyro]), estimator=filtered_estimator),
    )
    filtered_rows = simulate(
        filtered_sat,
        Environment(planetary_body=Earth()),
        DynamicsConfig(),
        state,
        CoupledSolver(CoupledRK4Propagator(0.25), tf=0.0),
    ).satellites["sat_000"]
    assert filtered_rows.fields["normalized_innovation_squared"].shape == (1, 1)
    assert filtered_rows.fields.get("estimated_attitude") is None
