import numpy as np
import pytest

from ADCS.helpers.math_helpers import random_n_unit_vec
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_factory.sensors import create_bct_nst_quaternion, create_generic_star_tracker_quaternion
from ADCS.satellite_hardware.errors import Noise
from ADCS.satellite_hardware.sensors import StarTrackerQuaternion


def make_tracker(*, fov_deg: float = 170.0, noise_std: float = 1e-8, boresight: np.ndarray | None = None, min_stars: int = 2):
    return StarTrackerQuaternion(
        boresight=np.array([0.0, 0.0, 1.0]) if boresight is None else boresight,
        fov=np.deg2rad(fov_deg),
        noise=Noise(noise=np.zeros(4), std_noise=np.array([noise_std] * 4)),
        sun_exclusion=np.deg2rad(25.0),
        min_stars=min_stars,
    )


def make_orbital_state() -> Orbital_State:
    return Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 7.5, 0.0]),
    )


def central_difference_quaternion_jacobian(state: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    def measurement(candidate):
        quaternion = candidate[3:7].copy()
        if quaternion[0] < 0:
            quaternion = -quaternion
        return quaternion

    numeric = np.zeros((state.size, 4))
    for index in range(state.size):
        delta = np.zeros(state.size)
        delta[index] = eps
        numeric[index] = (measurement(state + delta) - measurement(state - delta)) / (2.0 * eps)
    return numeric


def test_star_tracker_quaternion_output_has_expected_shape():
    tracker = make_tracker()
    reading = tracker.clean_reading(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), make_orbital_state())
    if not np.any(np.isnan(reading)):
        assert reading.shape == (4,)


def test_star_tracker_quaternion_output_is_unit_length():
    tracker = make_tracker()
    reading = tracker.clean_reading(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), make_orbital_state())
    if not np.any(np.isnan(reading)):
        assert abs(np.linalg.norm(reading) - 1.0) < 1e-10


def test_star_tracker_quaternion_enforces_nonnegative_scalar_component():
    tracker = make_tracker()
    orbital_state = make_orbital_state()
    for _ in range(10):
        state = np.concatenate([np.zeros(3), random_n_unit_vec(4)])
        reading = tracker.clean_reading(state, orbital_state)
        if not np.any(np.isnan(reading)):
            assert reading[0] >= 0.0


def test_star_tracker_quaternion_matches_true_attitude_when_visible():
    tracker = make_tracker()
    orbital_state = make_orbital_state()
    for _ in range(10):
        quaternion = random_n_unit_vec(4)
        if quaternion[0] < 0:
            quaternion = -quaternion
        state = np.concatenate([np.zeros(3), quaternion])
        reading = tracker.clean_reading(state, orbital_state)
        if not np.any(np.isnan(reading)):
            if np.dot(reading, quaternion) < 0:
                reading = -reading
            np.testing.assert_allclose(reading, quaternion, atol=1e-10)


def test_star_tracker_quaternion_returns_nan_when_too_few_stars_visible():
    tracker = make_tracker(fov_deg=0.001, boresight=np.array([1.0, 0.0, 0.0]), min_stars=2)
    reading = tracker.clean_reading(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), make_orbital_state())
    assert np.all(np.isnan(reading))


def test_star_tracker_quaternion_high_min_stars_makes_solution_unavailable():
    tracker = make_tracker(fov_deg=5.0, min_stars=10)
    reading = tracker.clean_reading(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), make_orbital_state())
    assert np.all(np.isnan(reading))


def test_star_tracker_quaternion_jacobian_matches_finite_difference():
    tracker = make_tracker()
    orbital_state = make_orbital_state()
    quaternion = np.array([0.9, 0.2, 0.3, 0.1])
    quaternion = quaternion / np.linalg.norm(quaternion)
    state = np.concatenate([np.zeros(3), quaternion])
    reading = tracker.clean_reading(state, orbital_state)
    if not np.any(np.isnan(reading)):
        analytic = tracker.basestate_jac(state, orbital_state)
        numeric = central_difference_quaternion_jacobian(state)
        np.testing.assert_allclose(analytic[3:7, :], numeric[3:7, :], rtol=1e-4, atol=1e-8)


def test_star_tracker_quaternion_jacobian_has_zero_omega_block():
    tracker = make_tracker()
    jacobian = tracker.basestate_jac(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), make_orbital_state())
    np.testing.assert_allclose(jacobian[0:3, :], np.zeros((3, 4)), atol=1e-15)


def test_star_tracker_quaternion_bias_jacobian_is_empty():
    jacobian = make_tracker().bias_jac(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), make_orbital_state())
    assert jacobian.shape == (0, 4)


def test_star_tracker_quaternion_noisy_reading_stays_unit_length():
    tracker = make_tracker(noise_std=1e-4)
    orbital_state = make_orbital_state()
    for _ in range(10):
        state = np.concatenate([np.zeros(3), random_n_unit_vec(4)])
        reading = tracker.reading(state, orbital_state)
        if not np.any(np.isnan(reading)):
            assert abs(np.linalg.norm(reading) - 1.0) < 1e-10


def test_star_tracker_quaternion_noisy_reading_keeps_nonnegative_scalar():
    tracker = make_tracker(noise_std=1e-4)
    orbital_state = make_orbital_state()
    for _ in range(10):
        state = np.concatenate([np.zeros(3), random_n_unit_vec(4)])
        reading = tracker.reading(state, orbital_state)
        if not np.any(np.isnan(reading)):
            assert reading[0] >= 0.0


def test_bct_quaternion_factory_sets_expected_properties():
    tracker = create_bct_nst_quaternion()
    assert tracker.output_length == 4
    assert tracker.fov == pytest.approx(np.deg2rad(20.0))
    assert tracker.sun_exclusion == pytest.approx(np.deg2rad(45.0))


def test_generic_quaternion_factory_sets_expected_properties():
    tracker = create_generic_star_tracker_quaternion(noise_arcsec=8.0, fov_deg=25.0, sun_exclusion_deg=40.0, min_stars=3)
    assert tracker.output_length == 4
    assert tracker.fov == pytest.approx(np.deg2rad(25.0))
    assert tracker.sun_exclusion == pytest.approx(np.deg2rad(40.0))
    assert tracker.min_stars == 3


def test_quaternion_factory_normalizes_custom_boresight():
    boresight = np.array([0.0, 1.0, 0.0])
    tracker = create_bct_nst_quaternion(boresight=boresight)
    np.testing.assert_allclose(tracker.boresight, boresight / np.linalg.norm(boresight), rtol=1e-10)
