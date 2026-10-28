"""Invariant attitude-error contracts for the spacecraft IEKF."""

import numpy as np

from ADCS.estimators.attitude_estimators import IEKF
from ADCS.estimators.measurement_stack import MeasurementStack
from ADCS.estimators.process_model import propagate_state
from ADCS.environment.ephemeris import DE421 as Ephemeris
from ADCS.helpers.math_helpers import rot_mat, skewsym
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.errors import Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import EarthHorizonSensor, StarTrackerQuaternion
from ADCS.state import EstimatorState


def _tracker_satellite():
    tracker = StarTrackerQuaternion(noise=Noise(std_noise=np.full(4, 1.0e-3)))
    tracker.clean_reading = lambda state, orbit: state.q.copy()
    return EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=[tracker])


def _state():
    q = np.array([0.8, 0.2, -0.3, 0.45])
    return EstimatorState(w=np.zeros(3), q=q / np.linalg.norm(q), cov=np.eye(6) * 0.1)


def test_quaternion_innovation_is_right_invariant_at_nonidentity_attitude():
    state = _state()
    stack = MeasurementStack(_tracker_satellite())
    delta = np.zeros(6)
    delta[3:6] = [0.07, -0.04, 0.02]
    measured = state.plus(
        delta, quaternion_mode="rotation_vector", quaternion_order="left"
    ).q
    innovation = stack.residual(
        measured, state.q, np.array([True]),
        quaternion_mode="rotation_vector", quaternion_order="left",
    )
    np.testing.assert_allclose(innovation, delta[3:6], atol=1e-13)
    np.testing.assert_allclose(
        stack.jacobian(
            state, None, np.array([True]),
            quaternion_mode="rotation_vector", quaternion_order="left",
        )[:, 3:6],
        np.eye(3),
        atol=1e-13,
    )


def test_iekf_reduces_right_invariant_attitude_error():
    satellite = _tracker_satellite()
    estimate = _state()
    truth = estimate.plus(
        np.array([0.0, 0.0, 0.0, 0.07, -0.04, 0.02]),
        quaternion_mode="rotation_vector", quaternion_order="left",
    )
    estimator = IEKF(satellite, estimate, dt=0.1)
    posterior = estimator.correct(truth.q, None)
    prior_error = truth.minus(
        estimate, quaternion_mode="rotation_vector", quaternion_order="left"
    )
    posterior_error = truth.minus(
        posterior, quaternion_mode="rotation_vector", quaternion_order="left"
    )
    assert np.linalg.norm(posterior_error[3:6]) < np.linalg.norm(prior_error[3:6])
    assert estimator.quaternion_order == "left"
    assert posterior.cov.shape == (6, 6)


def test_invariant_transition_matches_spacecraft_propagation():
    satellite = _tracker_satellite()
    prior = _state()
    prior.w = np.array([0.01, -0.02, 0.015])
    orbit = Orbital_State(
        ephem=Ephemeris(), J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 7.5, 0.0]), fast=True,
    )
    control = np.empty(0)
    dt = 0.001
    estimator = IEKF(satellite, prior, dt=dt)
    nominal = estimator.predict(control, orbit, orbit)
    transition = estimator.diagnostics["transition"]
    epsilon = 1.0e-6
    perturbation = np.zeros(6)
    perturbation[3] = epsilon
    deviations = []
    for sign in (1.0, -1.0):
        perturbed = prior.plus(
            sign * perturbation,
            quaternion_mode="rotation_vector", quaternion_order="left",
        )
        propagated = propagate_state(perturbed, satellite, control, dt, orbit, orbit)
        deviations.append(propagated.minus(
            nominal, quaternion_mode="rotation_vector", quaternion_order="left",
        ))
    numerical_column = (deviations[0] - deviations[1]) / (2.0 * epsilon)
    np.testing.assert_allclose(transition[:, 3], numerical_column, atol=1.0e-5)


def test_nadir_jacobian_in_inertial_error_frame_is_attitude_independent():
    estimate = _state()
    nadir = np.array([-1.0, 0.0, 0.0])
    boresight = rot_mat(estimate.q).T @ nadir
    sensor = EarthHorizonSensor(
        boresight=boresight, fov=np.pi,
        noise=Noise(std_noise=np.full(3, 1.0e-3)),
    )
    satellite = EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=[sensor])
    orbit = Orbital_State(
        ephem=Ephemeris(), J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 7.5, 0.0]), fast=True,
    )
    estimator = IEKF(satellite, estimate, dt=0.1)
    measurement = satellite.measurement_stack.predict(estimate, orbit)
    estimator.correct(measurement, orbit)
    np.testing.assert_allclose(
        estimator.diagnostics["measurement_jacobian"][:, 3:6],
        skewsym(nadir), atol=1e-10,
    )
