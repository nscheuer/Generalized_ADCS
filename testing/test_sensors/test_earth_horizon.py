import numpy as np
import pytest

from ADCS.helpers.math_helpers import random_n_unit_vec, rot_mat
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_factory.sensors import create_generic_earth_horizon, create_irst_horizon_sensor
from ADCS.satellite_hardware.errors import ErrorMode, Noise
from ADCS.satellite_hardware.sensors import EarthHorizonSensor
from ADCS.state import State


def make_sensor(*, fov_deg: float = 90.0, noise_std: float = 0.0, boresight: np.ndarray | None = None, estimate_bias: bool = False):
    noise = None
    if noise_std > 0.0:
        noise = Noise(noise=np.zeros(3), std_noise=np.array([noise_std] * 3))
    return EarthHorizonSensor(
        boresight=np.array([0.0, 0.0, -1.0]) if boresight is None else boresight,
        fov=np.deg2rad(fov_deg),
        noise=noise,
        estimate_bias=estimate_bias,
    )


def make_orbital_state(*, R=None, V=None):
    return Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]) if R is None else R,
        V=np.array([0.0, 7.5, 0.0]) if V is None else V,
    )


def central_difference_jacobian(sensor: EarthHorizonSensor, state: State, orbital_state: Orbital_State, eps: float = 1e-7) -> np.ndarray:
    def measurement(candidate):
        quaternion = candidate[3:7]
        nadir_eci = -orbital_state.R / np.linalg.norm(orbital_state.R)
        return rot_mat(quaternion).T @ nadir_eci

    state_array = state.as_array()
    numeric = np.zeros((state_array.size, 3))
    for index in range(state_array.size):
        delta = np.zeros(state_array.size)
        delta[index] = eps
        numeric[index] = (measurement(state_array + delta) - measurement(state_array - delta)) / (2.0 * eps)
    return numeric


def test_clean_reading_matches_rotated_nadir_direction():
    sensor = make_sensor()
    orbital_state = make_orbital_state()
    angle = np.pi / 4
    quaternion = np.array([np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0])
    state = State(w=np.zeros(3), q=quaternion)
    reading = sensor.clean_reading(state, orbital_state)
    if not np.any(np.isnan(reading)):
        expected = rot_mat(quaternion).T @ np.array([-1.0, 0.0, 0.0])
        np.testing.assert_allclose(reading, expected, atol=1e-10)


def test_clean_reading_output_is_unit_vector_when_visible():
    sensor = make_sensor(fov_deg=180.0)
    orbital_state = make_orbital_state()
    for _ in range(10):
        state = State(w=np.zeros(3), q=random_n_unit_vec(4))
        reading = sensor.clean_reading(state, orbital_state)
        if not np.any(np.isnan(reading)):
            assert abs(np.linalg.norm(reading) - 1.0) < 1e-10


def test_clean_reading_returns_nan_when_nadir_outside_fov():
    sensor = make_sensor(fov_deg=10.0, boresight=np.array([0.0, 0.0, 1.0]))
    state = State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0])
    reading = sensor.clean_reading(state, make_orbital_state())
    assert np.all(np.isnan(reading))


def test_clean_reading_is_always_visible_for_full_hemisphere_fov():
    sensor = make_sensor(fov_deg=180.0)
    orbital_state = make_orbital_state()
    for _ in range(20):
        state = State(w=np.zeros(3), q=random_n_unit_vec(4))
        reading = sensor.clean_reading(state, orbital_state)
        assert not np.any(np.isnan(reading))


def test_clean_reading_is_correct_across_altitudes():
    sensor = make_sensor(fov_deg=180.0)
    state = State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0])
    for altitude in [400, 800, 2000, 10000, 36000]:
        radius = 6378.137 + altitude
        reading = sensor.clean_reading(state, make_orbital_state(R=np.array([radius, 0.0, 0.0])))
        if not np.any(np.isnan(reading)):
            np.testing.assert_allclose(reading, np.array([-1.0, 0.0, 0.0]), atol=1e-10)


def test_clean_reading_sets_expected_earth_angular_radius():
    sensor = make_sensor(fov_deg=180.0)
    orbital_state = make_orbital_state(R=np.array([7000.0, 0.0, 0.0]))
    sensor.clean_reading(State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0]), orbital_state)
    assert sensor.earth_angular_radius == pytest.approx(np.arcsin(sensor._R_earth / 7000.0), rel=1e-10)


def test_basestate_jacobian_matches_finite_difference():
    sensor = make_sensor(fov_deg=180.0)
    orbital_state = make_orbital_state()
    quaternion = np.array([0.9, 0.2, 0.3, 0.1])
    quaternion = quaternion / np.linalg.norm(quaternion)
    state = State(w=np.zeros(3), q=quaternion)
    if not np.any(np.isnan(sensor.clean_reading(state, orbital_state))):
        analytic = sensor.basestate_jac(state, orbital_state)
        numeric = central_difference_jacobian(sensor, state, orbital_state)
        np.testing.assert_allclose(analytic[3:7, :], numeric[3:7, :], rtol=1e-4, atol=1e-8)


def test_basestate_jacobian_has_zero_omega_block():
    sensor = make_sensor(fov_deg=180.0)
    state = State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0])
    jacobian = sensor.basestate_jac(state, make_orbital_state())
    np.testing.assert_allclose(jacobian[0:3, :], np.zeros((3, 3)), atol=1e-15)


def test_basestate_jacobian_is_zero_when_measurement_is_nan():
    sensor = make_sensor(fov_deg=10.0, boresight=np.array([0.0, 0.0, 1.0]))
    state = State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0])
    jacobian = sensor.basestate_jac(state, make_orbital_state())
    np.testing.assert_allclose(jacobian, np.zeros((7, 3)), atol=1e-15)


def test_bias_jacobian_is_empty_without_bias_estimation():
    jacobian = make_sensor().bias_jac(State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0]), make_orbital_state())
    assert jacobian.shape == (0, 3)


def test_bias_jacobian_is_identity_with_bias_estimation():
    jacobian = make_sensor(estimate_bias=True).bias_jac(
        State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0]),
        make_orbital_state(),
    )
    np.testing.assert_allclose(jacobian, np.eye(3))


def test_noisy_reading_remains_unit_vector():
    sensor = make_sensor(fov_deg=180.0, noise_std=1e-3)
    orbital_state = make_orbital_state()
    for _ in range(10):
        state = State(w=np.zeros(3), q=random_n_unit_vec(4))
        reading = sensor.reading(state, orbital_state)
        if not np.any(np.isnan(reading)):
            assert abs(np.linalg.norm(reading) - 1.0) < 1e-10


def test_noise_creates_reasonable_angular_spread():
    noise_std = np.deg2rad(0.5)
    sensor = EarthHorizonSensor(
        boresight=np.array([0.0, 0.0, -1.0]),
        fov=np.deg2rad(180.0),
        noise=Noise(noise=np.zeros(3), std_noise=np.array([noise_std] * 3)),
    )
    orbital_state = make_orbital_state()
    dmode = ErrorMode(add_bias=False, add_noise=True, update_bias=False, update_noise=True)
    state = State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0])
    clean = sensor.clean_reading(state, orbital_state)

    angles = np.zeros(500)
    for index in range(500):
        noisy = sensor.reading(state, orbital_state, dmode=dmode)
        angles[index] = np.arccos(np.clip(np.dot(noisy, clean), -1.0, 1.0))

    rms_angle = np.sqrt(np.mean(angles**2))
    expected = noise_std * np.sqrt(2)
    assert rms_angle < expected * 3
    assert rms_angle > expected * 0.3


def test_generic_earth_horizon_factory_sets_expected_properties():
    sensor = create_generic_earth_horizon(fov_deg=60.0, noise_deg=1.0)
    assert sensor.output_length == 3
    assert sensor.fov == pytest.approx(np.deg2rad(60.0))


def test_irst_horizon_factory_sets_expected_properties():
    sensor = create_irst_horizon_sensor()
    assert sensor.output_length == 3
    assert sensor.fov == pytest.approx(np.deg2rad(60.0))


def test_generic_earth_horizon_factory_normalizes_custom_boresight():
    boresight = np.array([1.0, 0.0, 0.0])
    sensor = create_generic_earth_horizon(boresight=boresight)
    np.testing.assert_allclose(sensor.boresight, boresight / np.linalg.norm(boresight), rtol=1e-10)
