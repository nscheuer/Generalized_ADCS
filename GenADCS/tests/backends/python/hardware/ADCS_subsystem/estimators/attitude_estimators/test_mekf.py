from typing import Any, cast

import numpy as np
import pytest
from numba import njit

from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.environment import Earth, Environment
from GenADCS.backends.python.environment.ephemeris import EphemerisModel
from GenADCS.backends.python.hardware import Satellite, SatelliteGeometry
from GenADCS.backends.python.hardware.ADCS_subsystem import ADCSSubsystem
from GenADCS.backends.python.hardware.ADCS_subsystem.estimator import AttitudeEKF, AttitudeMEKF
from GenADCS.backends.python.hardware.ADCS_subsystem.estimator.attitude_estimator import (
    correct,
    correct_gyro,
    correct_star_tracker,
    predict,
    step,
)
from GenADCS.backends.python.hardware.ADCS_subsystem.estimator.attitude_estimator.attitude_ekf import (
    predict as ekf_predict,
    step as ekf_step,
)
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import (
    AutonomousStarTracker,
    CoarseSunSensor,
    FineSunSensor,
    Gyroscope,
    Magnetometer,
    Sensor,
)
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors.sensor_sampling import (
    prepare_sensors,
)
from GenADCS.backends.python.helpers.math.rotations.quaternion.interpolation import (
    rotation_vector_to_quaternion,
)
from GenADCS.backends.python.propagation import (
    CoupledDOP853Propagator,
    CoupledRK4Propagator,
)
from GenADCS.backends.python.simulate import ResultsFormat, simulate
from GenADCS.backends.python.solver import CoupledSolver
from GenADCS.backends.python.state import AttitudeState, OrbitalState, State
from GenADCS.backends.shared.simulationresult import SimulationResult

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def test_tracker_correction_reduces_attitude_error_and_covariance() -> None:
    initial = rotation_vector_to_quaternion(np.array([0.1, -0.05, 0.02]))
    covariance = np.eye(6) * 0.1
    updated, _, posterior = correct_star_tracker(
        initial, np.zeros(3), covariance, IDENTITY, IDENTITY, np.eye(3) * 1e-5
    )
    assert np.linalg.norm(updated[1:]) < np.linalg.norm(initial[1:]) / 100
    assert np.trace(posterior[:3, :3]) < np.trace(covariance[:3, :3])
    np.testing.assert_allclose(posterior, posterior.T, atol=1e-14)


def test_prediction_noise_and_compiled_sensor_step() -> None:
    covariance = np.zeros((6, 6))
    psd = np.eye(3) * 2e-5
    _, _, propagated = predict(IDENTITY, np.zeros(3), covariance, psd, 2.0)
    np.testing.assert_allclose(propagated[:3, :3], psd * 8.0 / 3.0)
    np.testing.assert_allclose(propagated[3:, 3:], psd * 2.0)
    np.testing.assert_allclose(propagated[:3, 3:], -psd * 2.0)
    turning, _, _ = predict(IDENTITY, np.array([0.0, 0.0, 0.2]), np.eye(6), psd, 1.0)
    np.testing.assert_allclose(
        turning, rotation_vector_to_quaternion(np.array([0.0, 0.0, -0.2])), atol=1e-14
    )

    gyro = Gyroscope(
        np.zeros(3), IDENTITY, IDENTITY, 10.0, np.eye(3) * 1e-6, np.zeros(3)
    )
    config = prepare_sensors(cast(list[Sensor], [gyro]), Environment())
    estimator = AttitudeMEKF(IDENTITY, np.ones(3), np.eye(6), psd)

    @njit
    def run(
        q: np.ndarray,
        w: np.ndarray,
        p: np.ndarray,
        process_noise: np.ndarray,
        readings: np.ndarray,
        sensors: tuple[np.ndarray, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        for _ in range(3):
            q, w, p = step(q, w, p, process_noise, 0.1, readings, sensors)
        return q, w, p

    q, w, p = run(*estimator.prepare(), np.zeros(3), config)
    assert np.linalg.norm(w) < 0.01
    np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-14)
    assert np.linalg.eigvalsh(p).min() >= -1e-12


def test_seven_state_ekf_prediction_and_compiled_sensor_step() -> None:
    psd = np.eye(3) * 2e-5
    initial_covariance = np.eye(7) * 0.1
    q, w, p = ekf_predict(
        IDENTITY, np.array([0.0, 0.0, 0.2]), initial_covariance, psd, 1.0
    )
    np.testing.assert_allclose(
        q, rotation_vector_to_quaternion(np.array([0.0, 0.0, -0.2])), atol=1e-14
    )
    np.testing.assert_allclose(w, np.array([0.0, 0.0, 0.2]))
    np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-14)
    np.testing.assert_allclose(p, p.T, atol=1e-14)
    assert np.linalg.eigvalsh(p).min() >= -1e-12
    np.testing.assert_allclose(p[:4, :4] @ q, np.zeros(4), atol=1e-12)

    gyro = Gyroscope(
        np.zeros(3), IDENTITY, IDENTITY, 10.0, np.eye(3) * 1e-6, np.zeros(3)
    )
    config = prepare_sensors(cast(list[Sensor], [gyro]), Environment())
    estimator = AttitudeEKF(IDENTITY, np.ones(3), np.eye(7), psd)

    @njit
    def run(q, w, p, process_noise, readings, sensors):
        for _ in range(3):
            q, w, p = ekf_step(q, w, p, process_noise, 0.1, readings, sensors)
        return q, w, p

    q, w, p = run(*estimator.prepare(), np.zeros(3), config)
    assert np.linalg.norm(w) < 0.01
    np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-14)
    np.testing.assert_allclose(p, p.T, atol=1e-14)
    assert np.linalg.eigvalsh(p).min() >= -1e-12


def test_plotting_example_converges() -> None:
    from GenADCS.examples.plotting.estimators._setup import run_example

    rows = cast(SimulationResult, run_example(progress=False)).satellites["sat_000"]
    dot = np.sum(rows.fields["satellite_attitude"] * rows.fields["estimated_attitude"], axis=1)
    attitude_error = 2 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0))
    rate_error = np.linalg.norm(rows.fields["satellite_rate"] - rows.fields["estimated_rate"], axis=1)
    assert attitude_error[-1] < 0.001
    assert attitude_error[-1] < attitude_error[0] / 3
    assert rate_error[-1] < rate_error[0]


def test_missing_tracker_and_mixed_sensor_order() -> None:
    gyro = Gyroscope(
        np.zeros(3), IDENTITY, IDENTITY, 10.0, np.eye(3) * 1e-6, np.zeros(3)
    )
    tracker = AutonomousStarTracker(
        np.zeros(3), IDENTITY, IDENTITY, 10.0, np.eye(3) * 1e-6, np.zeros(3)
    )
    for sensors in ([tracker, gyro], [gyro, tracker]):
        config = prepare_sensors(
            cast(list[Sensor], sensors),
            Environment(ephemeris=cast(EphemerisModel, object())),
        )
        readings = np.full(7, np.nan)
        readings[:3] = 0.0
        _q, w, p = step(
            IDENTITY, np.ones(3), np.eye(6), np.eye(3) * 1e-8, 0.1, readings, config
        )
        assert np.linalg.norm(w) < 0.01
        assert np.all(np.isfinite(p))


@pytest.mark.parametrize("propagator", [CoupledRK4Propagator, CoupledDOP853Propagator])
@pytest.mark.parametrize(
    ("estimator_type", "covariance_size"),
    [(AttitudeMEKF, 6), (AttitudeEKF, 7)],
)
def test_estimator_stays_in_simulation_loop(
    propagator: type[CoupledRK4Propagator] | type[CoupledDOP853Propagator],
    estimator_type: type[AttitudeMEKF] | type[AttitudeEKF],
    covariance_size: int,
) -> None:
    gyro = Gyroscope(
        np.zeros(3), IDENTITY, IDENTITY, 10.0, np.eye(3) * 1e-8, np.zeros(3)
    )
    estimator = estimator_type(
        IDENTITY, np.array([0.1, 0.1, 0.1]), np.eye(covariance_size) * 0.1, 1e-8
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
    result = simulate(
        satellite,
        Environment(planetary_body=Earth()),
        DynamicsConfig(),
        state,
        CoupledSolver(propagator(0.25), tf=0.5),
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
    )
    rows = result.satellites["sat_000"]
    assert rows.fields["estimated_attitude"].shape == (3, 4)
    assert rows.fields["estimated_covariance"].shape == (3, covariance_size**2)
    for name, width in (
        ("estimated_attitude_covariance", 9),
        ("estimated_rate_covariance", 9),
        ("measurement_innovation", 3),
        ("measurement_innovation_covariance", 9),
        ("normalized_innovation_squared", 1),
        ("normalized_estimation_error_squared", 1),
        ("measurement_residual", 3),
        ("kalman_gain", 18),
    ):
        value = rows.fields[name]
        assert value.shape == (3, width)
        assert np.all(np.isfinite(value))
    np.testing.assert_allclose(
        rows.fields["estimated_rate"][-1], np.asarray(state.attitude.rates), atol=0.002
    )
    np.testing.assert_allclose(
        np.linalg.norm(rows.fields["estimated_attitude"], axis=1), 1.0, atol=1e-12
    )


def test_estimator_record_selection_and_unrequested_fields() -> None:
    gyro = Gyroscope(
        np.zeros(3), IDENTITY, IDENTITY, 10.0, np.eye(3) * 1e-6, np.zeros(3)
    )
    estimator = AttitudeMEKF(
        IDENTITY,
        np.zeros(3),
        np.eye(6),
        1e-8,
        record=("normalized_innovation_squared",),
    )
    satellite = Satellite(
        SatelliteGeometry(12.0, np.eye(3), []),
        ADCSSubsystem(sensors=cast(list[Sensor], [gyro]), estimator=estimator),
    )
    state = State(
        AttitudeState(IDENTITY, (0.0, 0.0, 0.0)),
        OrbitalState.from_cartesian(
            (6878137.0, 0.0, 0.0), (0.0, 7612.6, 0.0), 2461000.5
        ),
    )
    rows = simulate(
        satellite,
        Environment(planetary_body=Earth()),
        DynamicsConfig(),
        state,
        CoupledSolver(CoupledRK4Propagator(0.25), tf=0.0),
    ).satellites["sat_000"]
    assert rows.fields["normalized_innovation_squared"].shape == (1, 1)
    assert np.isfinite(rows.fields["normalized_innovation_squared"][0, 0])
    assert rows.fields.get("measurement_innovation") is None
    assert rows.fields.get("estimated_attitude") is None


def test_mekf_constructor_validation_and_prepare() -> None:
    est = AttitudeMEKF(
        np.array([2.0, 0.0, 0.0, 0.0]),
        np.array([0.1, -0.2, 0.3]),
        np.eye(6) * 0.25,
        1e-4,
        record=("estimated_attitude",),
    )
    np.testing.assert_allclose(est.quaternion, IDENTITY, atol=1e-14)
    np.testing.assert_allclose(est.acceleration_psd, np.eye(3) * 1e-4, atol=1e-14)
    assert est.record == frozenset({"estimated_attitude"})
    q, w, p, a = est.prepare()
    q[0] = 99.0
    w[0] = 99.0
    p[0, 0] = 99.0
    a[0, 0] = 99.0
    np.testing.assert_allclose(est.quaternion, IDENTITY, atol=1e-14)
    np.testing.assert_allclose(est.rate, [0.1, -0.2, 0.3], atol=1e-14)
    np.testing.assert_allclose(est.covariance, np.eye(6) * 0.25, atol=1e-14)
    np.testing.assert_allclose(est.acceleration_psd, np.eye(3) * 1e-4, atol=1e-14)

    with pytest.raises(ValueError):
        AttitudeMEKF(np.zeros(4), np.zeros(3), np.eye(6), 1e-4)
    with pytest.raises(ValueError):
        AttitudeMEKF(IDENTITY, np.array([np.nan, 0.0, 0.0]), np.eye(6), 1e-4)
    asym = np.eye(6)
    asym[0, 1] = 0.5
    with pytest.raises(ValueError):
        AttitudeMEKF(IDENTITY, np.zeros(3), asym, 1e-4)
    with pytest.raises(ValueError):
        AttitudeMEKF(IDENTITY, np.zeros(3), -np.eye(6), 1e-4)
    with pytest.raises(ValueError):
        AttitudeMEKF(IDENTITY, np.zeros(3), np.eye(7), 1e-4)
    with pytest.raises(ValueError):
        AttitudeMEKF(IDENTITY, np.zeros(3), np.eye(6), -1e-4)
    with pytest.raises(ValueError):
        AttitudeMEKF(IDENTITY, np.zeros(3), np.eye(6), 1e-4, record=("unknown_field",))


def test_predict_zero_and_negative_dt() -> None:
    p0 = np.eye(6) * 0.2
    w0 = np.array([0.1, -0.2, 0.3])
    q1, w1, p1 = predict(IDENTITY, w0, p0, np.eye(3) * 1e-3, 0.0)
    np.testing.assert_allclose(q1, IDENTITY, atol=1e-14)
    np.testing.assert_allclose(w1, w0, atol=1e-14)
    np.testing.assert_allclose(p1, p0, atol=1e-14)
    with pytest.raises(ValueError):
        predict(IDENTITY, w0, p0, np.eye(3) * 1e-3, -0.1)


def test_direct_correct_and_gyro_with_non_identity_mounting() -> None:
    h = np.zeros((3, 6))
    h[:, :3] = np.eye(3)
    q1, w1, p1 = correct(
        IDENTITY,
        np.zeros(3),
        np.eye(6) * 0.1,
        np.array([0.02, -0.01, 0.03]),
        h,
        np.eye(3) * 1e-3,
    )
    assert np.linalg.norm(q1[1:]) > 0.0
    np.testing.assert_allclose(np.linalg.norm(q1), 1.0, atol=1e-14)
    np.testing.assert_allclose(w1, np.zeros(3), atol=1e-14)
    np.testing.assert_allclose(p1, p1.T, atol=1e-14)
    assert np.linalg.eigvalsh(p1).min() >= -1e-12

    mounting = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    q_g, w_g, p_g = correct_gyro(
        IDENTITY,
        np.zeros(3),
        np.eye(6) * 0.5,
        np.array([0.0, 0.4, 0.0]),
        mounting,
        np.eye(3) * 1e-4,
    )
    np.testing.assert_allclose(q_g, IDENTITY, atol=1e-14)
    np.testing.assert_allclose(w_g, np.array([0.4, 0.0, 0.0]), atol=1e-3)
    assert np.trace(p_g[3:, 3:]) < np.trace(np.eye(3) * 0.5)

    with pytest.raises(Exception):
        correct(IDENTITY, np.zeros(3), np.zeros((6, 6)), np.zeros(3), h, np.zeros((3, 3)))


def test_step_disabled_sensor_and_direction_updates() -> None:
    gyro = Gyroscope(
        np.zeros(3), IDENTITY, IDENTITY, 10.0, np.eye(3) * 1e-6, np.zeros(3)
    )
    gyro.enabled = False
    config = prepare_sensors(cast(list[Sensor], [gyro]), Environment())
    q, w, _ = step(
        IDENTITY, np.array([0.2, 0.0, 0.0]), np.eye(6) * 0.1, np.zeros((3, 3)), 0.0, np.zeros(3), config
    )
    np.testing.assert_allclose(q, IDENTITY, atol=1e-14)
    np.testing.assert_allclose(w, [0.2, 0.0, 0.0], atol=1e-14)

    mag = Magnetometer(
        np.zeros(3), IDENTITY, IDENTITY, 10.0, np.eye(3) * 1e-5, np.zeros(3)
    )
    css = CoarseSunSensor(
        np.zeros(3), IDENTITY, IDENTITY, np.array([0.0, 0.0, 1.0]), 1.0, 10.0, 1e-6, 0.0, 1.5
    )
    fss = FineSunSensor(
        np.zeros(3), IDENTITY, IDENTITY, np.eye(2) * 1e-5, np.zeros(2), np.array([1.0, 1.0])
    )
    dir_cfg = prepare_sensors(
        cast(list[Sensor], [mag, css, fss]),
        Environment(
            magnetic_field=cast(Any, object()),
            ephemeris=cast(EphemerisModel, object()),
        ),
    )
    refs = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    initial_q = rotation_vector_to_quaternion(np.array([0.1, -0.08, 0.05]))
    readings = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    q_dir, _, p_dir = step(
        initial_q, np.zeros(3), np.eye(6) * 0.1, np.zeros((3, 3)), 0.0, readings, dir_cfg, refs
    )
    assert np.linalg.norm(q_dir[1:]) < np.linalg.norm(initial_q[1:])
    np.testing.assert_allclose(p_dir, p_dir.T, atol=1e-14)
