"""Scale factor and misalignment: the device as built against the device as designed.

A sensor's ``scale_factor`` multiplies what a perfect device would read, and
its ``misalignment`` (a rotation vector, radians) rotates the sensitive axis of
a single-axis sensor or the output frame of a vector sensor. Both belong to the
plant: the estimator's clone of a satellite is reset to the nominal device.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ADCS.helpers.math_helpers import normalize, quat_mult, rot_exp, rot_mat
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.errors import AnisotropicNoise, Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite, Satellite
from ADCS.satellite_hardware.sensors import GPS, MTM, EarthHorizonSensor, Gyro, StarTracker, StarTrackerQuaternion, SunPair, SunSensor
from ADCS.state import State

TRUTH = State(w=np.array([0.02, -0.01, 0.015]), q=normalize(np.array([0.9, 0.2, -0.3, 0.1])))
TILT = np.array([0.0, 0.0, 2.0e-3])  # 2 mrad about the body z axis
AXIS = np.array([1.0, 0.0, 0.0])


def _orbital_state() -> Orbital_State:
    probe = Orbital_State(ephem=Ephemeris(), J2000=0.22, R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]))
    sun = normalize(probe.S)
    os = Orbital_State(ephem=Ephemeris(), J2000=0.22, R=7000.0 * sun, V=7.5 * normalize(np.cross(sun, [0.0, 0.0, 1.0])))
    assert os.is_sunlit()
    return os


def _tilted_axis() -> np.ndarray:
    return rot_mat(rot_exp(TILT)) @ AXIS


def test_single_axis_sensors_read_along_the_axis_as_mounted():
    os = _orbital_state()
    tilted = _tilted_axis()
    assert np.linalg.norm(tilted - AXIS) > 1.0e-3  # the tilt is visible
    gyro = Gyro(AXIS, scale_factor=1.02, misalignment=TILT)
    np.testing.assert_allclose(gyro.sensing_axis, tilted)
    np.testing.assert_allclose(gyro.clean_reading(TRUTH, os), 1.02 * tilted @ TRUTH.w)
    mtm = MTM(AXIS, scale_factor=0.98, misalignment=TILT)
    body_field = rot_mat(TRUTH.q).T @ os.B
    np.testing.assert_allclose(mtm.clean_reading(TRUTH, os), 0.98 * tilted @ body_field)
    sun_body = rot_mat(TRUTH.q).T @ normalize(os.S - os.R)
    sensor = SunSensor(AXIS, efficiency=0.5, scale_factor=1.1, misalignment=TILT)
    np.testing.assert_allclose(sensor.clean_reading(TRUTH, os), 1.1 * 0.5 * max(0.0, tilted @ sun_body))
    pair = SunPair(AXIS, efficiency=(0.9, 0.7), scale_factor=1.1, misalignment=TILT)
    projection = tilted @ sun_body
    np.testing.assert_allclose(pair.clean_reading(TRUTH, os), 1.1 * projection * (0.9 if projection > 0 else 0.7))


def test_vector_sensors_report_in_their_frame_as_mounted():
    os = _orbital_state()
    rotation = rot_mat(rot_exp(TILT))
    horizon = EarthHorizonSensor(boresight=rot_mat(TRUTH.q).T @ (-normalize(os.R)), fov=np.deg2rad(60.0), scale_factor=1.05, misalignment=TILT)
    nominal = EarthHorizonSensor(boresight=rot_mat(TRUTH.q).T @ (-normalize(os.R)), fov=np.deg2rad(60.0))
    np.testing.assert_allclose(horizon.clean_reading(TRUTH, os), 1.05 * rotation.T @ nominal.clean_reading(TRUTH, os))

    star = SimpleNamespace(s_eci=np.array([0.3, -0.4, np.sqrt(0.75)]), vmag=1.0)
    tracker = StarTracker(fov=np.pi, anisotropic_noise=AnisotropicNoise(std_cross=1.0e-3, std_roll=1.0e-3), misalignment=TILT)
    tracker._select_star = lambda q, os: star
    np.testing.assert_allclose(tracker.clean_reading(TRUTH, os), rotation.T @ (rot_mat(TRUTH.q).T @ star.s_eci))

    quaternion_tracker = StarTrackerQuaternion(fov=np.pi, min_stars=1, noise=Noise(std_noise=np.full(4, 1.0e-3)), misalignment=TILT)
    quaternion_tracker._select_stars = lambda q, os: [star]
    measured = quaternion_tracker.clean_reading(TRUTH, os)
    # the reported frame is the body frame rotated by the mounting error
    np.testing.assert_allclose(rot_mat(measured), rot_mat(TRUTH.q) @ rotation, atol=1.0e-12)


@pytest.mark.parametrize("make", [
    lambda os: Gyro(AXIS, scale_factor=1.02, misalignment=TILT),
    lambda os: MTM(AXIS, scale_factor=0.98, misalignment=TILT),
    lambda os: SunSensor(AXIS, efficiency=0.5, scale_factor=1.1, misalignment=TILT),
    lambda os: SunPair(AXIS, efficiency=(0.9, 0.7), scale_factor=1.1, misalignment=TILT),
    lambda os: EarthHorizonSensor(boresight=rot_mat(TRUTH.q).T @ (-normalize(os.R)), fov=np.deg2rad(60.0), scale_factor=1.05, misalignment=TILT),
], ids=["gyro", "mtm", "sunsensor", "sunpair", "horizon"])
def test_jacobians_follow_the_device_as_built(make):
    os = _orbital_state()
    sensor = make(os)
    analytic = np.atleast_2d(sensor.basestate_jac(TRUTH, os))
    numeric = np.zeros_like(analytic)
    step = 1.0e-6
    for i in range(7):
        plus = TRUTH.as_array().copy(); plus[i] += step
        minus = TRUTH.as_array().copy(); minus[i] -= step
        numeric[i, :] = (np.atleast_1d(sensor.clean_reading(State.from_array(plus), os)) - np.atleast_1d(sensor.clean_reading(State.from_array(minus), os))) / (2.0 * step)
    np.testing.assert_allclose(analytic, numeric, rtol=1.0e-5, atol=1.0e-9)


def test_quaternion_tracker_jacobian_follows_the_mounting():
    os = _orbital_state()
    star = SimpleNamespace(s_eci=np.array([0.3, -0.4, np.sqrt(0.75)]), vmag=1.0)
    tracker = StarTrackerQuaternion(fov=np.pi, min_stars=1, misalignment=TILT)
    tracker._select_stars = lambda q, os: [star]
    tracker.clean_reading(TRUTH, os)  # the Jacobian is defined at the last reading's star set
    analytic = tracker.basestate_jac(TRUTH, os)
    delta = rot_exp(TILT)
    for i in range(4):
        unit = np.zeros(4); unit[i] = 1.0
        np.testing.assert_allclose(analytic[3 + i, :], quat_mult(unit, delta))


def test_the_estimator_clone_is_the_nominal_device():
    os = _orbital_state()
    satellite = Satellite(J_0=np.eye(3), sensors=[Gyro(AXIS, scale_factor=1.02, misalignment=TILT), MTM(np.array([0.0, 1.0, 0.0]), scale_factor=0.98)])
    estimated = EstimatedSatellite.from_satellite(satellite)
    for truth_sensor, model in zip(satellite.sensors, estimated.sensors):
        assert model.scale_factor == 1.0 and model.misalignment is None
        np.testing.assert_allclose(model.mounting_rotation, np.eye(3))
        assert not np.allclose(truth_sensor.clean_reading(TRUTH, os), model.clean_reading(TRUTH, os))
    np.testing.assert_allclose(estimated.sensors[0].clean_reading(TRUTH, os), AXIS @ TRUTH.w)


def test_mounting_errors_are_validated():
    with pytest.raises(ValueError, match="misalignment"):
        Gyro(AXIS, misalignment=[1.0e-3, 2.0e-3])
    with pytest.raises(ValueError, match="scale_factor"):
        Gyro(AXIS, scale_factor=0.0)
    with pytest.raises(ValueError, match="no scale factor"):
        StarTrackerQuaternion(scale_factor=1.01)
    with pytest.raises(ValueError, match="GPS"):
        GPS(misalignment=TILT)
    nominal = MTM(AXIS)
    assert nominal.scale_factor == 1.0 and nominal.misalignment is None
    np.testing.assert_allclose(nominal.sensing_axis, AXIS)
