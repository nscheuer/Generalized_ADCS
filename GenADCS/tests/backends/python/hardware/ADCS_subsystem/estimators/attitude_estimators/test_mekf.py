import numpy as np
import pytest
from numba import njit

from GenADCS.backends.python.hardware.ADCS_subsystem.estimator import AttitudeMEKF
from GenADCS.backends.python.hardware.ADCS_subsystem.estimator.attitude_estimator import (
    correct_star_tracker, predict, step,
)
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import Gyroscope
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import AutonomousStarTracker
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors.sensor_sampling import prepare_sensors
from GenADCS.backends.python.helpers.math.rotations.quaternion.interpolation import rotation_vector_to_quaternion
from GenADCS.backends.python.hardware import Satellite, SatelliteGeometry
from GenADCS.backends.python.hardware.ADCS_subsystem import ADCSSubsystem
from GenADCS.backends.python.state import AttitudeState, OrbitalState, State
from GenADCS.backends.python.environment import Earth, Environment
from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.propagation import CoupledRK4Propagator, CoupledDOP853Propagator
from GenADCS.backends.python.solver import CoupledSolver
from GenADCS.backends.python.simulate import ResultsFormat, simulate


IDENTITY = np.array([1., 0., 0., 0.])


def test_tracker_correction_reduces_attitude_error_and_covariance():
    initial = rotation_vector_to_quaternion(np.array([0.1, -0.05, 0.02]))
    covariance = np.eye(6) * 0.1
    updated, _, posterior = correct_star_tracker(initial, np.zeros(3), covariance,
                                                  IDENTITY, IDENTITY, np.eye(3) * 1e-5)
    assert np.linalg.norm(updated[1:]) < np.linalg.norm(initial[1:]) / 100
    assert np.trace(posterior[:3, :3]) < np.trace(covariance[:3, :3])
    np.testing.assert_allclose(posterior, posterior.T, atol=1e-14)


def test_prediction_noise_and_compiled_sensor_step():
    covariance = np.zeros((6, 6))
    psd = np.eye(3) * 2e-5
    _, _, propagated = predict(IDENTITY, np.zeros(3), covariance, psd, 2.)
    np.testing.assert_allclose(propagated[:3, :3], psd * 8. / 3.)
    np.testing.assert_allclose(propagated[3:, 3:], psd * 2.)
    np.testing.assert_allclose(propagated[:3, 3:], -psd * 2.)
    turning, _, _ = predict(IDENTITY, np.array([0., 0., .2]),
                           np.eye(6), psd, 1.)
    np.testing.assert_allclose(turning, rotation_vector_to_quaternion(
        np.array([0., 0., -.2])), atol=1e-14)

    gyro = Gyroscope(np.zeros(3), IDENTITY, IDENTITY, 10., np.eye(3) * 1e-6,
                     np.zeros(3))
    config = prepare_sensors([gyro], None)
    estimator = AttitudeMEKF(IDENTITY, np.ones(3), np.eye(6), psd)

    @njit
    def run(q, w, p, process_noise, readings, sensors):
        for _ in range(3):
            q, w, p = step(q, w, p, process_noise, 0.1, readings, sensors)
        return q, w, p

    q, w, p = run(*estimator.prepare(), np.zeros(3), config)
    assert np.linalg.norm(w) < 0.01
    np.testing.assert_allclose(np.linalg.norm(q), 1., atol=1e-14)
    assert np.linalg.eigvalsh(p).min() >= -1e-12


def test_plotting_example_converges():
    from GenADCS.examples.plotting.estimators._setup import run_example

    rows = run_example(progress=False).satellites["sat_000"]
    dot = np.sum(rows.satellite_attitude * rows.estimated_attitude, axis=1)
    attitude_error = 2 * np.arccos(np.clip(np.abs(dot), 0., 1.))
    rate_error = np.linalg.norm(rows.satellite_rate - rows.estimated_rate, axis=1)
    assert attitude_error[-1] < 0.001
    assert attitude_error[-1] < attitude_error[0] / 3
    assert rate_error[-1] < rate_error[0]


def test_missing_tracker_and_mixed_sensor_order():
    gyro = Gyroscope(np.zeros(3), IDENTITY, IDENTITY, 10., np.eye(3) * 1e-6,
                     np.zeros(3))
    tracker = AutonomousStarTracker(np.zeros(3), IDENTITY, IDENTITY, 10.,
                                    np.eye(3) * 1e-6, np.zeros(3))
    for sensors in ([tracker, gyro], [gyro, tracker]):
        config = prepare_sensors(sensors, type("Env", (), {"ephemeris": object()})())
        readings = np.full(7, np.nan)
        readings[:3] = 0.0
        q, w, p = step(IDENTITY, np.ones(3), np.eye(6), np.eye(3) * 1e-8,
                       0.1, readings, config)
        assert np.linalg.norm(w) < 0.01
        assert np.all(np.isfinite(p))


@pytest.mark.parametrize("propagator", [CoupledRK4Propagator, CoupledDOP853Propagator])
def test_estimator_stays_in_simulation_loop(propagator):
    gyro = Gyroscope(np.zeros(3), IDENTITY, IDENTITY, 10., np.eye(3) * 1e-8,
                     np.zeros(3))
    estimator = AttitudeMEKF(IDENTITY, np.array([.1, .1, .1]), np.eye(6) * .1, 1e-8)
    satellite = Satellite(SatelliteGeometry(12., np.diag([.13, .1, .05]), []),
                          ADCSSubsystem(sensors=[gyro], estimator=estimator))
    state = State(AttitudeState(IDENTITY, (.01, -.015, .02)),
                  OrbitalState.from_cartesian((6878137., 0., 0.), (0., 7612.6, 0.), 2461000.5))
    result = simulate(satellite, Environment(planetary_body=Earth()), DynamicsConfig(), state,
                      CoupledSolver(propagator(.25), tf=.5),
                      resultsformat=ResultsFormat(estimated_attitude=True, estimated_rate=True,
                                                  estimated_covariance=True,
                                                  estimated_attitude_covariance=True,
                                                  estimated_rate_covariance=True,
                                                  measurement_innovation=True,
                                                  measurement_innovation_covariance=True,
                                                  normalized_innovation_squared=True,
                                                  normalized_estimation_error_squared=True,
                                                  measurement_residual=True,
                                                  kalman_gain=True))
    rows = result.satellites["sat_000"]
    assert rows.estimated_attitude.shape == (3, 4)
    assert rows.estimated_covariance.shape == (3, 36)
    for name, width in (("estimated_attitude_covariance", 9),
                        ("estimated_rate_covariance", 9),
                        ("measurement_innovation", 3),
                        ("measurement_innovation_covariance", 9),
                        ("normalized_innovation_squared", 1),
                        ("normalized_estimation_error_squared", 1),
                        ("measurement_residual", 3), ("kalman_gain", 18)):
        value = getattr(rows, name)
        assert value.shape == (3, width)
        assert np.all(np.isfinite(value))
    np.testing.assert_allclose(rows.estimated_rate[-1], state.attitude.rates, atol=.002)
    np.testing.assert_allclose(np.linalg.norm(rows.estimated_attitude, axis=1), 1., atol=1e-12)


def test_estimator_record_selection_and_unrequested_fields():
    gyro = Gyroscope(np.zeros(3), IDENTITY, IDENTITY, 10., np.eye(3) * 1e-6,
                     np.zeros(3))
    estimator = AttitudeMEKF(IDENTITY, np.zeros(3), np.eye(6), 1e-8,
                             record=("normalized_innovation_squared",))
    satellite = Satellite(SatelliteGeometry(12., np.eye(3), []),
                          ADCSSubsystem(sensors=[gyro], estimator=estimator))
    state = State(AttitudeState(IDENTITY, (0., 0., 0.)),
                  OrbitalState.from_cartesian((6878137., 0., 0.), (0., 7612.6, 0.), 2461000.5))
    rows = simulate(satellite, Environment(planetary_body=Earth()), DynamicsConfig(), state,
                    CoupledSolver(CoupledRK4Propagator(.25), tf=0.)).satellites["sat_000"]
    assert rows.normalized_innovation_squared.shape == (1, 1)
    assert np.isfinite(rows.normalized_innovation_squared[0, 0])
    assert rows.measurement_innovation is None
    assert rows.estimated_attitude is None
