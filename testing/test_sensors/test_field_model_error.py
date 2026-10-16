"""The plant's magnetometers see a field the model does not know about.

``FieldModelError`` is a slowly varying error field in the inertial frame,
shared by a spacecraft's magnetometers, standing in for the difference between
the real geomagnetic field and the IGRF model. The estimator's clone of the
satellite does not carry it: that difference is exactly what the filter lacks.
"""

from __future__ import annotations

import numpy as np
import pytest

from ADCS.helpers.math_helpers import normalize, rot_mat
from ADCS.environment.ephemeris import DE421 as Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.errors import FieldModelError
from ADCS.satellite_hardware.satellite import EstimatedSatellite, Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.state import State

TRUTH = State(w=np.array([0.02, -0.01, 0.015]), q=normalize(np.array([0.9, 0.2, -0.3, 0.1])))
T0 = 0.22
STEP = 10.0 * TimeConstants.sec2cent


def _orbital_state(j2000: float = T0) -> Orbital_State:
    return Orbital_State(ephem=Ephemeris(), J2000=j2000, R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]))


def test_the_error_field_is_stationary_and_correlated():
    np.random.seed(4)
    error = FieldModelError(std=2.0e-7, correlation_time=600.0)
    history = []
    for k in range(20000):
        error._update(T0 + k * STEP)
        history.append(error.eci.copy())
    history = np.array(history)
    assert np.std(history, axis=0) == pytest.approx(np.full(3, 2.0e-7), rel=0.15)
    lag = np.array([np.corrcoef(history[:-1, i], history[1:, i])[0, 1] for i in range(3)])
    np.testing.assert_allclose(lag, np.exp(-10.0 / 600.0), atol=0.02)
    assert abs(np.corrcoef(history[:, 0], history[:, 1])[0, 1]) < 0.1  # axes independent


def test_one_error_field_is_shared_and_steps_once_per_epoch():
    np.random.seed(6)
    error = FieldModelError(std=2.0e-7, correlation_time=600.0)
    sensors = [MTM(axis, field_error=error) for axis in np.eye(3)]
    os = _orbital_state()
    readings = np.array([sensor.reading(TRUTH, os) for sensor in sensors]).ravel()
    body_field = rot_mat(TRUTH.q).T @ (os.B + error.eci)
    np.testing.assert_allclose(readings, body_field)  # all three saw the same error field at the same epoch
    first = error.eci.copy()
    later = _orbital_state(T0 + STEP)
    sensors[0].reading(TRUTH, later)
    assert not np.array_equal(error.eci, first)
    sensors[1].reading(TRUTH, later)
    sensors[2].reading(TRUTH, later)
    np.testing.assert_allclose(error.eci, error.eci)  # unchanged by the second and third reads at the same epoch
    noiseless = np.array([sensor.reading(TRUTH, later, dmode=None) for sensor in sensors]).ravel()
    np.testing.assert_allclose(noiseless, rot_mat(TRUTH.q).T @ (later.B + error.eci))


def test_the_jacobian_includes_the_error_field():
    np.random.seed(8)
    error = FieldModelError(std=5.0e-6, correlation_time=600.0)  # large, so the term is not lost in the field
    os = _orbital_state()
    sensor = MTM(np.array([0.6, 0.8, 0.0]), field_error=error, misalignment=np.array([0.0, 1.0e-3, 0.0]))
    sensor.reading(TRUTH, os)
    analytic = sensor.basestate_jac(TRUTH, os).ravel()
    numeric = np.zeros(7)
    for i in range(7):
        plus = TRUTH.as_array().copy(); plus[i] += 1.0e-6
        minus = TRUTH.as_array().copy(); minus[i] -= 1.0e-6
        numeric[i] = (sensor.clean_reading(State.from_array(plus), os) - sensor.clean_reading(State.from_array(minus), os)) / 2.0e-6
    np.testing.assert_allclose(analytic, numeric, rtol=1.0e-5, atol=1.0e-12)


def test_the_estimator_clone_has_no_error_field():
    error = FieldModelError(std=2.0e-7, correlation_time=600.0)
    satellite = Satellite(J_0=np.eye(3), sensors=[MTM(axis, field_error=error) for axis in np.eye(3)])
    estimated = EstimatedSatellite.from_satellite(satellite)
    assert all(sensor.field_error is None for sensor in estimated.sensors)
    assert all(sensor.field_error is error for sensor in satellite.sensors)


def test_configuration_is_validated():
    with pytest.raises(ValueError, match="correlation_time"):
        FieldModelError(std=1.0e-7, correlation_time=0.0)
    with pytest.raises(ValueError, match="std"):
        FieldModelError(std=-1.0, correlation_time=10.0)
    assert not FieldModelError(std=0.0, correlation_time=10.0)
    silent = MTM(np.array([1.0, 0.0, 0.0]), field_error=FieldModelError(std=0.0, correlation_time=10.0))
    os = _orbital_state()
    np.testing.assert_allclose(silent.reading(TRUTH, os), (rot_mat(TRUTH.q).T @ os.B)[0])


def test_the_satellite_owns_the_error_and_its_torquers_act_on_the_true_field():
    from ADCS.satellite_hardware.actuators import MTQ
    from ADCS.satellite_hardware.disturbances import Dipole_Disturbance

    np.random.seed(12)
    error = FieldModelError(std=5.0e-6, correlation_time=600.0)  # large, to be visible against a 3e-5 T field
    satellite = Satellite(
        J_0=np.eye(3), sensors=[MTM(axis) for axis in np.eye(3)],
        actuators=[MTQ(np.array([1.0, 0.0, 0.0]), max_torque=1.0)],
        disturbances=[Dipole_Disturbance(np.array([0.0, 0.02, 0.0]))],
        field_error=error,
    )
    assert all(sensor.field_error is error for sensor in satellite.sensors)
    os = _orbital_state()
    satellite.update_errors(T0)
    assert np.any(error.eci != 0.0)
    true_body_field = rot_mat(TRUTH.q).T @ (os.B + error.eci)
    model_body_field = rot_mat(TRUTH.q).T @ os.B
    # readings and torques see the same field, the true one
    np.testing.assert_allclose(satellite.noiseless_sensor_readings(TRUTH, os), true_body_field)
    command = np.array([0.3])
    np.testing.assert_allclose(satellite.act_torque(TRUTH, command, os, dmode=None), np.cross(0.3 * np.array([1.0, 0.0, 0.0]), true_body_field))
    np.testing.assert_allclose(satellite.dist_torques(TRUTH, os), np.cross(np.array([0.0, 0.02, 0.0]), true_body_field))
    # the dynamics agree with the torque methods, and the caller's orbital state is untouched
    plant_wdot = satellite.dynamics_core(TRUTH, command, os, dmode=None)[:3]
    expected_torque = np.cross(0.3 * np.array([1.0, 0.0, 0.0]), true_body_field) + np.cross(np.array([0.0, 0.02, 0.0]), true_body_field)
    np.testing.assert_allclose(plant_wdot, np.linalg.inv(np.eye(3)) @ (expected_torque - np.cross(TRUTH.w, TRUTH.w)), atol=1.0e-18)
    np.testing.assert_allclose(rot_mat(TRUTH.q).T @ os.B, model_body_field)
    # the estimator's clone predicts with the model field
    estimated = EstimatedSatellite.from_satellite(satellite)
    assert estimated.field_error is None and all(sensor.field_error is None for sensor in estimated.sensors)
    np.testing.assert_allclose(estimated.act_torque(TRUTH, command, os, dmode=None), np.cross(0.3 * np.array([1.0, 0.0, 0.0]), model_body_field))
    np.testing.assert_allclose(estimated.noiseless_sensor_readings(TRUTH, os), model_body_field)
    # one realisation per step: a second update at the same epoch changes nothing
    before = error.eci.copy()
    satellite.update_errors(T0)
    np.testing.assert_allclose(error.eci, before)
