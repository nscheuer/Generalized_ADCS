"""Estimators start from their first readings: the WarmStart recipe and the state it builds."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import ADCS
from ADCS.estimators.attitude_determination import attitude_error
from ADCS.estimators.attitude_estimators import (
    AugmentedEKF,
    AugmentedMEKF,
    AugmentedSRUKF,
    AugmentedUKF,
    EKF,
    MEKF,
    SRUKF,
    UKF,
)
from ADCS.estimators.initialization import WarmStart, attitude_from_readings, initial_state_from_readings
from ADCS.helpers.math_helpers import normalize
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.actuators import RW
from ADCS.satellite_hardware.disturbances import Torque_Disturbance
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite, Satellite
from ADCS.satellite_hardware.sensors import MTM, Gyro, StarTrackerQuaternion, SunPair
from ADCS.state import EstimatorState, State, _quaternion_tangent_scale

EPHEM = Ephemeris()
INERTIA = np.diag([0.5, 0.8, 1.2])
WHEEL_MOMENTUM = 0.05
TRUTH = EstimatorState(w=[0.02, -0.01, 0.015], q=normalize(np.array([0.9, 0.2, -0.3, 0.1])), h=[WHEEL_MOMENTUM])
MTM_STD, SUN_STD, GYRO_STD, WHEEL_STD = 5.0e-8, 1.0e-3, 1.0e-4, 1.0e-4
CHARTS = ("quaternion_vector", "rotation_vector", "mrp", "two_mrp", "cayley")


def _orbital_state(*, sunlit: bool = True, seconds: float = 0.0) -> Orbital_State:
    probe = Orbital_State(ephem=EPHEM, J2000=0.22, R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]))
    sun = normalize(probe.S)
    position = 7000.0 * (sun if sunlit else -sun)
    velocity = 7.5 * normalize(np.cross(sun, np.array([0.0, 0.0, 1.0])))
    os = Orbital_State(ephem=EPHEM, J2000=0.22 + seconds * TimeConstants.sec2cent, R=position, V=velocity)
    assert os.is_sunlit() == sunlit
    return os


def _tracker() -> StarTrackerQuaternion:
    sensor = StarTrackerQuaternion(fov=np.pi, min_stars=1, noise=Noise(std_noise=np.full(4, 1.0e-3)))
    sensor._select_stars = lambda q, os: [SimpleNamespace(s_eci=np.array([0.3, -0.4, np.sqrt(0.75)]), vmag=1.0)]
    return sensor


def _sensors(*, noisy: bool = True, gyro_bias: bool = False) -> list:
    scale = 1.0 if noisy else 0.0
    return (
        [MTM(axis, noise=Noise(std_noise=scale * MTM_STD)) for axis in np.eye(3)]
        + [SunPair(axis, efficiency=(0.9, 0.7), noise=Noise(std_noise=scale * SUN_STD)) for axis in np.eye(3)]
        + [
            Gyro(
                axis,
                noise=Noise(std_noise=scale * GYRO_STD),
                bias=Bias(bias=0.0, std_bias=1.0e-10 if gyro_bias else 0.0),
                estimate_bias=gyro_bias,
            )
            for axis in np.eye(3)
        ]
    )


def _wheels(*, bias: bool = False) -> list:
    return [
        RW(
            axis=np.array([0.0, 0.0, 1.0]), max_torque=0.02, J=1.0e-3, h=WHEEL_MOMENTUM, h_max=1.0,
            h_meas_noise=Noise(std_noise=WHEEL_STD), bias=Bias(bias=0.0, std_bias=1.0e-10 if bias else 0.0),
            estimate_bias=bias,
        )
    ]


def _satellite(*, noisy=True, gyro_bias=False, wheel_bias=False, disturbance=False, tracker=False) -> EstimatedSatellite:
    sensors = _sensors(noisy=noisy, gyro_bias=gyro_bias) + ([_tracker()] if tracker else [])
    disturbances = [Torque_Disturbance(np.zeros(3), estimate_dist=True)] if disturbance else []
    return EstimatedSatellite(J_0=INERTIA, sensors=sensors, actuators=_wheels(bias=wheel_bias), disturbances=disturbances)


AUGMENTED = dict(gyro_bias=True, wheel_bias=True, disturbance=True)
AUGMENTED_STDS = dict(act_bias_std=1.0e-3, sens_bias_std=1.0e-4, dist_param_std=1.0e-6)


def _block(state: EstimatorState, name: str) -> np.ndarray:
    index = state.slice(name, coordinates="tangent")
    return state.cov[index, index]


# ----------------------------------------------------------------------------
# the state builder


def test_state_from_noiseless_readings_matches_the_truth():
    satellite = _satellite(noisy=False)
    os = _orbital_state()
    state = initial_state_from_readings(satellite, satellite.noiseless_sensor_readings(TRUTH, os), os)
    assert np.linalg.norm(attitude_error(state.q, TRUTH.q)) < 1.0e-8
    np.testing.assert_allclose(state.w, TRUTH.w, atol=1.0e-12)
    np.testing.assert_allclose(state.h, [WHEEL_MOMENTUM], atol=1.0e-12)
    assert state.act_bias.size == state.sens_bias.size == state.dist_param.size == 0
    assert state.cov.shape == (7, 7)
    assert np.all(np.linalg.eigvalsh(state.cov) >= 0.0)
    np.testing.assert_allclose(_block(state, "angular_velocity"), np.zeros((3, 3)))  # perfect gyros
    np.testing.assert_allclose(_block(state, "wheel_momentum"), [[WHEEL_STD**2]])
    np.testing.assert_allclose(state.process_noise.as_matrix(), np.zeros((7, 7)))


def test_covariance_blocks_follow_the_sensor_noise():
    satellite = _satellite()
    os = _orbital_state()
    readings = satellite.noiseless_sensor_readings(TRUTH, os)
    state = initial_state_from_readings(satellite, readings, os)
    solution = attitude_from_readings(satellite, readings, os)
    assert solution.method == "quest"
    np.testing.assert_allclose(_block(state, "angular_velocity"), GYRO_STD**2 * np.eye(3), rtol=1.0e-12)
    scale = 2.0 * _quaternion_tangent_scale(State.DEFAULT_QUATERNION_MODE)
    np.testing.assert_allclose(_block(state, "attitude"), solution.covariance / scale**2, rtol=1.0e-12)
    np.testing.assert_allclose(_block(state, "wheel_momentum"), [[WHEEL_STD**2]])


def test_attitude_block_follows_the_chart_and_the_full_covariance_does_not():
    satellite = _satellite()
    os = _orbital_state()
    readings = satellite.noiseless_sensor_readings(TRUTH, os)
    rotation_vector_covariance = attitude_from_readings(satellite, readings, os).covariance
    full_covariances = []
    for chart in CHARTS:
        tangent = initial_state_from_readings(satellite, readings, os, quaternion_mode=chart)
        scale = 2.0 * _quaternion_tangent_scale(chart)
        np.testing.assert_allclose(_block(tangent, "attitude"), rotation_vector_covariance / scale**2, rtol=1.0e-12)
        full = initial_state_from_readings(satellite, readings, os, covariance_coordinates="full", quaternion_mode=chart)
        assert full.cov.shape == (8, 8)
        np.testing.assert_allclose(full.cov, tangent.covariance_to_full(tangent.cov, quaternion_mode=chart), atol=1.0e-15)
        full_covariances.append(full.cov)
    for covariance in full_covariances[1:]:
        np.testing.assert_allclose(covariance, full_covariances[0], atol=1.0e-12)


def test_values_can_be_supplied_and_then_need_their_standard_deviations():
    satellite = _satellite()
    os = _orbital_state()
    readings = satellite.noiseless_sensor_readings(TRUTH, os)
    chart_scale = 2.0 * _quaternion_tangent_scale(State.DEFAULT_QUATERNION_MODE)

    state = initial_state_from_readings(
        satellite, readings, os,
        WarmStart(rate=[0.1, 0.0, 0.0], rate_std=0.01, wheel_momentum=[0.1], wheel_momentum_std=1.0e-3,
                  attitude_std=[0.1, 0.2, 0.3]),
    )
    np.testing.assert_allclose(state.w, [0.1, 0.0, 0.0])
    np.testing.assert_allclose(_block(state, "angular_velocity"), 1.0e-4 * np.eye(3))
    np.testing.assert_allclose(state.h, [0.1])
    np.testing.assert_allclose(_block(state, "wheel_momentum"), [[1.0e-6]])
    np.testing.assert_allclose(_block(state, "attitude"), np.diag([0.01, 0.04, 0.09]) / chart_scale**2)

    with pytest.raises(ValueError, match="rate_std"):
        initial_state_from_readings(satellite, readings, os, WarmStart(rate=[0.1, 0.0, 0.0]))
    with pytest.raises(ValueError, match="wheel_momentum_std"):
        initial_state_from_readings(satellite, readings, os, WarmStart(wheel_momentum=[0.1]))
    with pytest.raises(ValueError, match="estimates no actuator biases"):
        initial_state_from_readings(satellite, readings, os, WarmStart(act_bias=[1.0e-3], act_bias_std=1.0e-3))
    with pytest.raises(ValueError, match="rate_std must be"):
        initial_state_from_readings(satellite, readings, os, WarmStart(rate_std=[0.1, 0.2]))
    with pytest.raises(ValueError, match="method"):
        WarmStart(method="bogus")
    with pytest.raises(ValueError, match="rate must be"):
        WarmStart(rate="gyros")


def test_augmented_blocks_start_at_zero_or_the_given_values():
    satellite = _satellite(**AUGMENTED)
    os = _orbital_state()
    readings = satellite.noiseless_sensor_readings(TRUTH, os)
    with pytest.raises(ValueError, match="act_bias_std is required"):
        initial_state_from_readings(satellite, readings, os)
    with pytest.raises(ValueError, match="sens_bias_std is required"):
        initial_state_from_readings(satellite, readings, os, WarmStart(act_bias_std=1.0e-3))
    with pytest.raises(ValueError, match="dist_param_std is required"):
        initial_state_from_readings(satellite, readings, os, WarmStart(act_bias_std=1.0e-3, sens_bias_std=1.0e-4))

    zeros = initial_state_from_readings(satellite, readings, os, WarmStart(**AUGMENTED_STDS))
    np.testing.assert_allclose(zeros.act_bias, np.zeros(1))
    np.testing.assert_allclose(zeros.sens_bias, np.zeros(3))
    np.testing.assert_allclose(zeros.dist_param, np.zeros(3))
    assert zeros.cov.shape == (14, 14)
    np.testing.assert_allclose(_block(zeros, "actuator_bias"), [[1.0e-6]])
    np.testing.assert_allclose(_block(zeros, "sensor_bias"), 1.0e-8 * np.eye(3))
    np.testing.assert_allclose(_block(zeros, "disturbance_parameter"), 1.0e-12 * np.eye(3))

    given = initial_state_from_readings(
        satellite, readings, os,
        WarmStart(sens_bias=[1.0e-4, -2.0e-4, 3.0e-4], dist_param=[1.0e-5, 0.0, 0.0],
                  act_bias_std=1.0e-3, sens_bias_std=[1.0e-4, 2.0e-4, 3.0e-4], dist_param_std=1.0e-6),
    )
    np.testing.assert_allclose(given.sens_bias, [1.0e-4, -2.0e-4, 3.0e-4])
    np.testing.assert_allclose(given.dist_param, [1.0e-5, 0.0, 0.0])
    np.testing.assert_allclose(_block(given, "sensor_bias"), np.diag([1.0e-8, 4.0e-8, 9.0e-8]))
    for name in ("angular_velocity", "attitude", "wheel_momentum"):  # unaffected by the augmentation
        np.testing.assert_allclose(_block(given, name), _block(zeros, name))


def test_block_diagonal_covariance_has_no_cross_terms():
    satellite = _satellite(**AUGMENTED)
    os = _orbital_state()
    state = initial_state_from_readings(
        satellite, satellite.noiseless_sensor_readings(TRUTH, os), os, WarmStart(**AUGMENTED_STDS),
    )
    names = ("angular_velocity", "attitude", "wheel_momentum", "actuator_bias", "sensor_bias", "disturbance_parameter")
    expected = np.zeros_like(state.cov)
    for name in names:
        index = state.slice(name, coordinates="tangent")
        expected[index, index] = _block(state, name)
    np.testing.assert_allclose(state.cov, expected)


def test_process_noise_is_taken_in_the_filter_coordinates():
    satellite = _satellite()
    os = _orbital_state()
    readings = satellite.noiseless_sensor_readings(TRUTH, os)
    tangent_noise = np.diag([1.0e-8] * 3 + [1.0e-6] * 3 + [1.0e-10])
    state = initial_state_from_readings(satellite, readings, os, WarmStart(process_noise=tangent_noise))
    np.testing.assert_allclose(state.process_noise.as_matrix(), tangent_noise)
    full = initial_state_from_readings(
        satellite, readings, os, WarmStart(process_noise=tangent_noise), covariance_coordinates="full",
    )
    np.testing.assert_allclose(full.process_noise.as_matrix(), state.covariance_to_full(tangent_noise), atol=1.0e-18)
    with pytest.raises(ValueError, match="process_noise must be a 7x7"):
        initial_state_from_readings(satellite, readings, os, WarmStart(process_noise=np.eye(6)))


# ----------------------------------------------------------------------------
# the attitude methods


def test_attitude_methods_agree_on_noiseless_readings():
    satellite = _satellite(noisy=False)
    os = _orbital_state()
    readings = satellite.noiseless_sensor_readings(TRUTH, os)
    for method in ("auto", "quest", "triad"):
        solution = attitude_from_readings(satellite, readings, os, method=method)
        assert solution.method == ("quest" if method == "auto" else method)
        assert np.linalg.norm(attitude_error(solution.quaternion, TRUTH.q)) < 1.0e-8
    with pytest.raises(ValueError, match="no quaternion star tracker"):
        attitude_from_readings(satellite, readings, os, method="tracker")


def test_a_quaternion_star_tracker_is_preferred_and_can_be_bypassed():
    satellite = _satellite(tracker=True)
    os = _orbital_state()
    readings = satellite.noiseless_sensor_readings(TRUTH, os)
    tracker = attitude_from_readings(satellite, readings, os)
    assert tracker.method == "tracker"
    assert np.linalg.norm(attitude_error(tracker.quaternion, TRUTH.q)) < 1.0e-8
    np.testing.assert_allclose(tracker.covariance, (2.0e-3) ** 2 * np.eye(3))  # 1e-3 per quaternion component
    assert attitude_from_readings(satellite, readings, os, method="quest").method == "quest"
    state = initial_state_from_readings(satellite, readings, os, WarmStart(method="tracker"))
    np.testing.assert_allclose(_block(state, "attitude"), (2.0e-3) ** 2 * np.eye(3))


def test_eclipse_leaves_only_the_magnetometers_and_is_refused():
    satellite = _satellite()
    os = _orbital_state(sunlit=False)
    with pytest.raises(ValueError, match="pass the initial attitude explicitly"):
        initial_state_from_readings(satellite, satellite.noiseless_sensor_readings(TRUTH, os), os)


def test_first_cycle_errors_are_consistent_with_the_covariance():
    # Monte Carlo: normalised squared error of rate and attitude ~ chi^2 with 6 dof.
    satellite = _satellite()
    os = _orbital_state()
    np.random.seed(7)
    scores = []
    for _ in range(300):
        state = initial_state_from_readings(
            satellite, satellite.sensor_readings(TRUTH, os), os, quaternion_mode="rotation_vector",
        )
        error = np.concatenate([state.w - TRUTH.w, attitude_error(state.q, TRUTH.q)])  # rotation vector: chart = angle
        covariance = state.cov[:6, :6]
        scores.append(error @ np.linalg.solve(covariance, error))
    assert 5.2 < np.mean(scores) < 6.8


# ----------------------------------------------------------------------------
# the estimators

FILTERS = [
    pytest.param(EKF, {}, id="ekf"),
    pytest.param(MEKF, {}, id="mekf"),
    pytest.param(UKF, {}, id="ukf"),
    pytest.param(SRUKF, {}, id="srukf"),
    pytest.param(AugmentedEKF, AUGMENTED, id="augmented_ekf"),
    pytest.param(AugmentedMEKF, AUGMENTED, id="augmented_mekf"),
    pytest.param(AugmentedUKF, AUGMENTED, id="augmented_ukf"),
    pytest.param(AugmentedSRUKF, AUGMENTED, id="augmented_srukf"),
]


@pytest.mark.parametrize("filter_class, augmentation", FILTERS)
def test_every_filter_starts_from_its_first_readings(filter_class, augmentation):
    # Noise models on (so the filter has a measurement covariance to work with), readings noiseless.
    satellite = _satellite(**augmentation)
    os = _orbital_state()
    later = _orbital_state(seconds=10.0)
    readings = satellite.noiseless_sensor_readings(TRUTH, os)
    control = np.zeros(satellite.control_len)
    recipe = WarmStart(**(AUGMENTED_STDS if augmentation else {}))

    estimator = filter_class(satellite, recipe, dt=10.0)
    with pytest.raises(RuntimeError, match="first readings"):
        estimator.state
    with pytest.raises(RuntimeError, match="first readings"):
        estimator.predict(control, os, later)
    with pytest.raises(RuntimeError, match="first readings"):
        estimator.update()
    assert estimator.warm_start_attitude is None

    started = estimator.step(readings, os)
    assert estimator.warm_start_attitude.method == "quest"
    assert np.linalg.norm(attitude_error(started.q, TRUTH.q)) < 1.0e-8
    np.testing.assert_allclose(started.w, TRUTH.w, atol=1.0e-12)
    np.testing.assert_allclose(started.h, [WHEEL_MOMENTUM], atol=1.0e-12)
    expected = initial_state_from_readings(
        satellite, readings, os, recipe,
        covariance_coordinates=estimator.covariance_coordinates,
        quaternion_mode=estimator.correction_mode if estimator.covariance_coordinates == "tangent" else "quaternion_vector",
    )
    np.testing.assert_allclose(started.as_estimator_array(), expected.as_estimator_array(), atol=1.0e-15)
    np.testing.assert_allclose(started.cov, expected.cov, atol=1.0e-15)

    # the filter then runs as usual
    estimator.predict(control, os, later)
    corrected = estimator.step(satellite.noiseless_sensor_readings(TRUTH, later), later)
    assert np.all(np.isfinite(corrected.as_estimator_array()))
    assert np.linalg.norm(attitude_error(corrected.q, TRUTH.q)) < 0.3  # the truth did not move, the prediction did


def test_first_readings_are_consumed_not_corrected_on():
    satellite = _satellite()
    os = _orbital_state()
    np.random.seed(3)
    readings = satellite.sensor_readings(TRUTH, os)
    estimator = MEKF(satellite, WarmStart(), dt=10.0, quaternion_mode="mrp")
    started = estimator.step(readings, os)
    built = initial_state_from_readings(satellite, readings, os, quaternion_mode="mrp")
    np.testing.assert_allclose(started.as_estimator_array(), built.as_estimator_array(), atol=1.0e-15)
    np.testing.assert_allclose(started.cov, built.cov, atol=1.0e-15)


def test_from_readings_and_the_one_call_adapter():
    satellite = _satellite()
    os = _orbital_state()
    later = _orbital_state(seconds=10.0)
    readings = satellite.noiseless_sensor_readings(TRUTH, os)
    later_readings = satellite.noiseless_sensor_readings(TRUTH, later)
    control = np.zeros(satellite.control_len)

    built = MEKF.from_readings(satellite, readings, os, dt=10.0, quaternion_mode="mrp")
    assert built.correction_mode == "mrp"
    reference = MEKF(satellite, WarmStart(), dt=10.0, quaternion_mode="mrp")
    reference.step(readings, os)
    np.testing.assert_allclose(built.state.as_estimator_array(), reference.state.as_estimator_array(), atol=1.0e-15)

    adapter = MEKF(satellite, WarmStart(), dt=10.0)
    adapter.update(control, readings, os)
    adapter.update(control, later_readings, later)  # must predict first: the adapter remembers the first orbital state
    staged = MEKF(satellite, WarmStart(), dt=10.0)
    staged.step(readings, os)
    staged.predict(control, os, later)
    staged.step(later_readings, later)
    np.testing.assert_allclose(adapter.state.as_estimator_array(), staged.state.as_estimator_array(), atol=1.0e-15)
    np.testing.assert_allclose(adapter.state.cov, staged.state.cov, atol=1.0e-15)


def test_a_reset_replaces_a_pending_warm_start():
    satellite = _satellite()
    os = _orbital_state()
    estimator = MEKF(satellite, WarmStart(), dt=10.0)
    given = EstimatorState(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0], h=[0.0], cov=np.eye(7) * 0.1, int_cov=np.zeros((7, 7)))
    estimator.reset(given)
    np.testing.assert_allclose(estimator.state.as_estimator_array(), given.as_estimator_array())
    estimator.step(satellite.noiseless_sensor_readings(TRUTH, os), os)  # an ordinary correction now
    assert estimator.warm_start_attitude is None
    assert 0.0 < np.linalg.norm(attitude_error(estimator.state.q, TRUTH.q)) < np.linalg.norm(attitude_error(given.q, TRUTH.q))


def test_bad_state_types_are_rejected():
    with pytest.raises(TypeError, match="EstimatorState or a WarmStart"):
        MEKF(_satellite(), np.zeros(7), dt=10.0)


def test_simulate_runs_a_warm_started_filter_and_it_beats_an_identity_start():
    truth = Satellite(J_0=INERTIA, sensors=_sensors(), actuators=_wheels())
    estimated = _satellite()
    os = _orbital_state()
    dt, tf = 10.0, 60.0
    process_noise = np.diag([1.0e-10] * 3 + [1.0e-8] * 3 + [1.0e-12])

    def run(estimator):
        np.random.seed(11)
        results = ADCS.simulate(
            x=State(w=TRUTH.w, q=TRUTH.q, h=TRUTH.h), satellite=truth, est_satellite=estimated,
            estimator=estimator, os0=os, dt=dt, tf=tf,
        ).first()
        # results.state_hist[k] is the real state after step k's propagation, while
        # results.est_state_hist[k] is the estimate from the readings taken at the
        # start of step k, so the real state it should match is the previous one.
        reals = [State(w=TRUTH.w, q=TRUTH.q, h=TRUTH.h)] + list(results.state_hist[:-1])
        return [
            np.linalg.norm(attitude_error(estimate.q, real.q))
            for estimate, real in zip(results.est_state_hist, reals)
        ]

    warm = run(MEKF(estimated, WarmStart(process_noise=process_noise), dt=dt))
    cold = run(MEKF(
        estimated,
        EstimatorState(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0], h=[0.0],
                       cov=np.diag([1.0e-2] * 3 + [1.0] * 3 + [1.0e-2]), int_cov=process_noise),
        dt=dt,
    ))
    assert len(warm) == len(cold) >= 5
    assert warm[0] < 1.0e-2  # the very first estimate is already the QUEST solution
    assert np.mean(warm) < np.mean(cold)
