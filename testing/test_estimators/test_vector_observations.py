"""Readings -> direction pairs, checked against the sensor models' own clean readings."""

from __future__ import annotations

import numpy as np
import pytest

from ADCS.estimators.attitude_determination import attitude_error, quest, triad
from ADCS.estimators.initialization import angular_rate_from_readings, vector_observations, wheel_momentum_from_readings
from ADCS.helpers.math_helpers import normalize, rot_mat
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import RW
from ADCS.satellite_hardware.errors import Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import MTM, EarthHorizonSensor, Gyro, StarTracker, SunPair, SunSensor
from ADCS.state import State

EPHEM = Ephemeris()
SKEW_AXES = np.array([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.3, 0.3, 0.9055]])
TRUTH = State(w=np.array([0.02, -0.01, 0.015]), q=np.array([0.9, 0.2, -0.3, 0.1]) / np.linalg.norm([0.9, 0.2, -0.3, 0.1]))


def _orbital_state(*, sunlit: bool) -> Orbital_State:
    probe = Orbital_State(ephem=EPHEM, J2000=0.22, R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]))
    sun = normalize(probe.S)
    position = 7000.0 * (sun if sunlit else -sun)
    velocity = 7.5 * normalize(np.cross(sun, np.array([0.0, 0.0, 1.0])))
    os = Orbital_State(ephem=EPHEM, J2000=0.22, R=position, V=velocity)
    assert os.is_sunlit() == sunlit
    return os


def _by_kind(observations):
    return {observation.kind: observation for observation in observations}


def _expected_body(os: Orbital_State, reference: np.ndarray) -> np.ndarray:
    return rot_mat(TRUTH.q).T @ normalize(reference)


def test_magnetometer_triads_give_the_field_direction():
    for axes in (np.eye(3), SKEW_AXES):
        satellite = EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=[MTM(axis=axis) for axis in axes])
        os = _orbital_state(sunlit=True)
        readings = satellite.noiseless_sensor_readings(TRUTH, os)
        observation = _by_kind(vector_observations(satellite, readings, os))["magnetic_field"]
        np.testing.assert_allclose(observation.body, _expected_body(os, os.B), atol=1e-12)
        np.testing.assert_allclose(observation.reference, normalize(os.B), atol=1e-12)
        assert len(observation.sources) == 3


def test_sun_pairs_and_sun_sensors_give_the_sun_direction_when_lit():
    os = _orbital_state(sunlit=True)
    pairs = EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=[SunPair(axis, efficiency=(0.9, 0.7)) for axis in np.eye(3)])
    diodes = EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]),
                                sensors=[SunSensor(sign * axis, efficiency=0.8) for axis in np.eye(3) for sign in (1.0, -1.0)])
    for satellite in (pairs, diodes):
        readings = satellite.noiseless_sensor_readings(TRUTH, os)
        observation = _by_kind(vector_observations(satellite, readings, os))["sun"]
        np.testing.assert_allclose(observation.body, _expected_body(os, os.S - os.R), atol=1e-12)
        np.testing.assert_allclose(observation.reference, normalize(os.S - os.R), atol=1e-12)


def test_eclipse_yields_no_sun_direction_and_dark_diodes_are_ignored():
    satellite = EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]),
                                   sensors=[MTM(axis=axis) for axis in np.eye(3)] + [SunPair(axis, efficiency=(0.9, 0.7)) for axis in np.eye(3)])
    os = _orbital_state(sunlit=False)
    kinds = _by_kind(vector_observations(satellite, satellite.noiseless_sensor_readings(TRUTH, os), os))
    assert "magnetic_field" in kinds and "sun" not in kinds
    # only two lit diodes: the sun direction is not determined and must not be invented
    lit_os = _orbital_state(sunlit=True)
    sun_body = _expected_body(lit_os, lit_os.S - lit_os.R)
    two = EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=[SunSensor(axis, efficiency=1.0) for axis in (sun_body, normalize(np.cross(sun_body, [0.3, 0.4, 0.5])))])
    assert "sun" not in _by_kind(vector_observations(two, two.noiseless_sensor_readings(TRUTH, lit_os), lit_os))


def test_horizon_sensor_gives_the_nadir_direction():
    satellite = EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]),
                                   sensors=[EarthHorizonSensor(boresight=rot_mat(TRUTH.q).T @ np.array([-1.0, 0.0, 0.0]), fov=np.deg2rad(60.0), noise=Noise(noise=np.zeros(3), std_noise=np.full(3, 1e-3)))])
    os = _orbital_state(sunlit=True)
    nadir = -normalize(os.R)
    satellite.attitude_sensors[0].boresight = rot_mat(TRUTH.q).T @ nadir  # look straight at nadir at the truth attitude
    observation = _by_kind(vector_observations(satellite, satellite.noiseless_sensor_readings(TRUTH, os), os))["nadir"]
    np.testing.assert_allclose(observation.body, _expected_body(os, -os.R), atol=1e-12)
    np.testing.assert_allclose(observation.reference, nadir, atol=1e-12)
    assert observation.sigma == pytest.approx(1e-3, rel=1e-6)


def test_star_tracker_needs_a_supplied_star_direction():
    tracker = StarTracker(fov=np.deg2rad(170.0))
    satellite = EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=[tracker])
    os = _orbital_state(sunlit=True)
    readings = satellite.noiseless_sensor_readings(TRUTH, os)
    if not np.all(np.isfinite(readings)):
        pytest.skip("no navigation star visible at the test attitude")
    # The object that took the reading knows its star; an estimator's own copy does not.
    star = tracker.current_star.s_eci
    assert vector_observations(satellite, readings, os)[0].kind == "star"
    estimator_side = EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=[StarTracker(fov=np.deg2rad(170.0))])
    assert vector_observations(estimator_side, readings, os) == []
    entry = estimator_side.measurement_stack.entries[0]
    observation = vector_observations(estimator_side, readings, os, star_references={entry.name: star})[0]
    np.testing.assert_allclose(observation.body, rot_mat(TRUTH.q).T @ star, atol=1e-12)


def test_reported_angular_sigma_matches_the_scatter_of_noisy_readings():
    satellite = EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=[MTM(axis=axis, noise=Noise(noise=0.0, std_noise=2e-7)) for axis in np.eye(3)])
    os = _orbital_state(sunlit=True)
    np.random.seed(4)
    truth_direction = _expected_body(os, os.B)
    angles, sigmas = [], []
    for _ in range(400):
        observation = _by_kind(vector_observations(satellite, satellite.sensor_readings(TRUTH, os), os))["magnetic_field"]
        angles.append(np.arccos(np.clip(observation.body @ truth_direction, -1.0, 1.0)))
        sigmas.append(observation.sigma)
    rms_angle = float(np.sqrt(np.mean(np.square(angles))))
    assert rms_angle / np.sqrt(2.0) == pytest.approx(np.mean(sigmas), rel=0.15)  # two tangent axes


def test_base_case_magnetometers_and_sun_pairs_determine_the_attitude():
    satellite = EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]),
                                   sensors=[MTM(axis=axis) for axis in np.eye(3)] + [SunPair(axis, efficiency=(0.9, 0.7)) for axis in np.eye(3)])
    os = _orbital_state(sunlit=True)
    observations = vector_observations(satellite, satellite.noiseless_sensor_readings(TRUTH, os), os)
    assert sorted(o.kind for o in observations) == ["magnetic_field", "sun"]
    body = np.vstack([o.body for o in observations]); reference = np.vstack([o.reference for o in observations])
    solution = quest(body, reference, [o.sigma for o in observations])
    assert np.linalg.norm(attitude_error(solution.quaternion, TRUTH.q)) < 1e-8
    direct = triad(body[0], body[1], reference[0], reference[1])
    assert np.linalg.norm(attitude_error(direct.quaternion, TRUTH.q)) < 1e-8


def test_angular_rate_from_gyros_and_wheel_momentum():
    wheels = [RW(axis=axis, max_torque=0.02, J=1e-3, h=h, h_max=1.0) for axis, h in zip(np.eye(3), (0.1, -0.2, 0.05))]
    for axes in (np.eye(3), SKEW_AXES):
        satellite = EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=[Gyro(axis) for axis in axes], actuators=wheels)
        os = _orbital_state(sunlit=True)
        readings = satellite.noiseless_sensor_readings(TRUTH, os)
        rate, covariance = angular_rate_from_readings(satellite, readings)
        np.testing.assert_allclose(rate, TRUTH.w, atol=1e-12)
        assert covariance.shape == (3, 3)
        np.testing.assert_allclose(wheel_momentum_from_readings(satellite, readings), [0.1, -0.2, 0.05], atol=1e-12)
    two = EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=[Gyro(axis) for axis in np.eye(3)[:2]])
    with pytest.raises(ValueError, match="not observable"):
        angular_rate_from_readings(two, two.noiseless_sensor_readings(TRUTH, _orbital_state(sunlit=True)))
