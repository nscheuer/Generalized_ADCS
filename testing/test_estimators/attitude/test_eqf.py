"""Equivariant direction-output behavior of the spacecraft EQF."""

import numpy as np

from ADCS import EQF
from ADCS.estimators.process_model import propagate_state
from ADCS.environment.ephemeris import DE421 as Ephemeris
from ADCS.helpers.math_helpers import rot_mat, skewsym
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.actuators import RW
from ADCS.satellite_hardware.satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import EarthHorizonSensor, MTM, StarTrackerQuaternion
from ADCS.state import EstimatorState
from ADCS.satellite_hardware.disturbances import GG_Disturbance


def _orbit():
    return Orbital_State(
        ephem=Ephemeris(), J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 7.5, 0.0]), fast=True,
        B=np.array([2.0e-5, -1.0e-5, 3.0e-5]),
    )


def _direction_case():
    q = np.array([0.8, 0.2, -0.3, 0.45])
    q /= np.linalg.norm(q)
    state = EstimatorState(w=np.zeros(3), q=q, cov=np.eye(6) * 0.1)
    nadir = np.array([-1.0, 0.0, 0.0])
    sensor = EarthHorizonSensor(
        boresight=rot_mat(q).T @ nadir, fov=np.pi,
        noise=Noise(std_noise=np.full(3, 1e-3)),
    )
    satellite = EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=[sensor])
    return satellite, state, _orbit(), nadir


def test_eqf_direction_jacobian_at_identity_and_correction():
    satellite, estimate, orbit, nadir = _direction_case()
    delta = np.array([0.0, 0.0, 0.0, 0.07, -0.04, 0.02])
    truth = estimate.plus(delta, quaternion_mode="rotation_vector", quaternion_order="left")
    measurement = satellite.measurement_stack.predict(truth, orbit)
    estimator = EQF(satellite, estimate, dt=0.1)
    posterior = estimator.correct(measurement, orbit)
    diagnostics = estimator.diagnostics
    residual = diagnostics["innovation"]
    np.testing.assert_allclose(
        diagnostics["measurement_jacobian"][:, 3:6],
        skewsym(nadir), atol=1e-10,
    )
    prior_error = truth.minus(estimate, quaternion_mode="rotation_vector", quaternion_order="left")
    posterior_error = truth.minus(posterior, quaternion_mode="rotation_vector", quaternion_order="left")
    assert np.linalg.norm(posterior_error[3:6]) < np.linalg.norm(prior_error[3:6])
    np.testing.assert_allclose(posterior.cov, posterior.cov.T, atol=1e-12)


def test_eqf_quaternion_measurement_remains_supported():
    sensor = StarTrackerQuaternion(noise=Noise(std_noise=np.full(4, 1e-3)))
    sensor.clean_reading = lambda state, orbit: state.q.copy()
    satellite = EstimatedSatellite(J_0=np.eye(3), sensors=[sensor])
    state = EstimatorState(w=np.zeros(3), q=[1, 0, 0, 0], cov=np.eye(6) * 0.1)
    estimator = EQF(satellite, state, dt=0.1)
    truth = state.plus(
        np.array([0, 0, 0, 0.03, -0.02, 0.01]),
        quaternion_mode="rotation_vector", quaternion_order="left",
    )
    result = estimator.correct(truth.q, None)
    assert np.linalg.norm(result.minus(truth, quaternion_mode="rotation_vector", quaternion_order="left")[3:6]) < 1e-3


def test_three_axis_magnetometer_uses_equivariant_output_update():
    q = np.array([0.8, 0.2, -0.3, 0.45])
    q /= np.linalg.norm(q)
    estimate = EstimatorState(w=np.zeros(3), q=q, cov=np.eye(6) * 0.1)
    sensors = [
        MTM(axis, noise=Noise(std_noise=1.0e-7)) for axis in np.eye(3)
    ]
    satellite = EstimatedSatellite(J_0=np.eye(3), sensors=sensors)
    orbit = _orbit()
    truth = estimate.plus(
        [0, 0, 0, 0.07, -0.04, 0.02],
        quaternion_mode="rotation_vector", quaternion_order="left",
    )
    measurement = satellite.measurement_stack.predict(truth, orbit)
    estimator = EQF(satellite, estimate, dt=0.1)
    estimator.correct(measurement, orbit)
    diagnostics = estimator.diagnostics
    np.testing.assert_allclose(
        diagnostics["measurement_jacobian"][:, 3:6],
        skewsym(orbit.B), atol=1e-10,
    )


def test_full_group_error_generator_includes_wheel_and_gravity_gradient():
    wheel = RW(
        axis=np.array([1.0, 0.0, 0.0]), max_torque=0.02, J=0.001,
        h=0.01, h_max=1.0,
    )
    satellite = EstimatedSatellite(
        J_0=np.diag([0.5, 0.8, 1.2]),
        actuators=[wheel], disturbances=[GG_Disturbance()],
    )
    q = np.array([0.8, 0.2, -0.3, 0.45])
    q /= np.linalg.norm(q)
    state = EstimatorState(
        w=[0.01, -0.02, 0.015], q=q, h=[0.01], cov=np.eye(7) * 0.1,
    )
    orbit = _orbit()
    control = np.array([1.0e-5])
    dt = 1.0e-3
    estimator = EQF(satellite, state, dt=dt)
    nominal = estimator.predict(control, orbit, orbit)
    generator = estimator.diagnostics["error_generator"]
    np.testing.assert_allclose(generator[3:6, :3], rot_mat(q), atol=1e-12)
    assert np.linalg.norm(generator[:3, 6]) > 0.0
    assert np.linalg.norm(generator[6, :3]) > 0.0

    epsilon = 1.0e-6
    for column in (0, 3, 6):
        offset = np.zeros(7)
        offset[column] = epsilon
        deviations = []
        for sign in (1.0, -1.0):
            perturbed = state.plus(
                sign * offset,
                quaternion_mode="rotation_vector", quaternion_order="left",
            )
            propagated = propagate_state(
                perturbed, satellite, control, dt, orbit, orbit,
            )
            deviations.append(
                propagated.minus(
                    nominal,
                    quaternion_mode="rotation_vector", quaternion_order="left",
                )
            )
        numerical = (deviations[0] - deviations[1]) / (2.0 * epsilon)
        expected = np.eye(7)[:, column] + dt * generator[:, column]
        np.testing.assert_allclose(numerical, expected, atol=2.0e-5)


def test_group_generator_captures_estimated_actuator_bias_coupling():
    wheel = RW(
        axis=np.array([1.0, 0.0, 0.0]), max_torque=0.02, J=0.001,
        h=0.01, h_max=1.0, bias=Bias(std_bias=1.0e-5),
        estimate_bias=True,
    )
    satellite = EstimatedSatellite(
        J_0=np.diag([0.5, 0.8, 1.2]), actuators=[wheel],
    )
    state = EstimatorState(
        w=[0.01, -0.02, 0.015], q=[1.0, 0.0, 0.0, 0.0],
        h=[0.01], act_bias=[1.0e-4], cov=np.eye(8) * 0.1,
    )
    orbit = _orbit()
    control = np.array([1.0e-5])
    dt = 1.0e-3
    estimator = EQF(satellite, state, dt=dt)
    nominal = estimator.predict(control, orbit, orbit)
    generator = estimator.diagnostics["error_generator"]
    assert np.linalg.norm(generator[:7, 7]) > 0.0
    np.testing.assert_allclose(generator[7, :], 0.0, atol=1e-12)

    epsilon = 1.0e-6
    offset = np.zeros(8)
    offset[7] = epsilon
    deviations = []
    for sign in (1.0, -1.0):
        perturbed = state.plus(
            sign * offset,
            quaternion_mode="rotation_vector", quaternion_order="left",
        )
        satellite.match_estimate(perturbed, dt)
        propagated = propagate_state(perturbed, satellite, control, dt, orbit, orbit)
        deviations.append(
            propagated.minus(
                nominal,
                quaternion_mode="rotation_vector", quaternion_order="left",
            )
        )
    satellite.match_estimate(state, dt)
    numerical = (deviations[0] - deviations[1]) / (2.0 * epsilon)
    expected = np.eye(8)[:, 7] + dt * generator[:, 7]
    np.testing.assert_allclose(numerical, expected, atol=2.0e-5)
