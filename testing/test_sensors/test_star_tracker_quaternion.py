import numpy as np
import numdifftools as nd
import pytest

from ADCS.satellite_hardware.sensors import StarTrackerQuaternion
from ADCS.satellite_factory.sensors import (
    create_bct_nst_quaternion,
    create_generic_star_tracker_quaternion,
)
from ADCS.satellite_hardware.errors import Noise
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.helpers.math_helpers import random_n_unit_vec, rot_mat


def _make_tracker(fov_deg=170.0, noise_std=1e-8):
    """Helper to create a wide-FOV quaternion tracker for testing."""
    noise = Noise(
        noise=np.zeros(4),
        std_noise=np.array([noise_std] * 4),
    )
    return StarTrackerQuaternion(
        boresight=np.array([0.0, 0.0, 1.0]),
        fov=np.deg2rad(fov_deg),
        noise=noise,
        sun_exclusion=np.deg2rad(25.0),
        min_stars=2,
    )


def _make_os():
    """Helper to create a standard orbital state for testing."""
    ephem = Ephemeris()
    return Orbital_State(
        ephem=ephem,
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 7.5, 0.0]),
    )


# ---- Output shape and properties ----

def test_quaternion_output_shape():
    tracker = _make_tracker()
    os = _make_os()
    x = np.zeros(7)
    x[3:7] = [1.0, 0.0, 0.0, 0.0]

    q_meas = tracker.clean_reading(x, os)

    if not np.any(np.isnan(q_meas)):
        assert q_meas.shape == (4,)


def test_quaternion_is_unit():
    tracker = _make_tracker()
    os = _make_os()
    x = np.zeros(7)
    x[3:7] = [1.0, 0.0, 0.0, 0.0]

    q_meas = tracker.clean_reading(x, os)

    if not np.any(np.isnan(q_meas)):
        assert abs(np.linalg.norm(q_meas) - 1.0) < 1e-10


def test_quaternion_scalar_positive():
    tracker = _make_tracker()
    os = _make_os()

    for _ in range(10):
        q = random_n_unit_vec(4)
        x = np.concatenate([np.zeros(3), q])
        q_meas = tracker.clean_reading(x, os)

        if not np.any(np.isnan(q_meas)):
            assert q_meas[0] >= 0, "Scalar component should be non-negative"


# ---- Accuracy ----

def test_quaternion_matches_true_attitude():
    tracker = _make_tracker()
    os = _make_os()

    for _ in range(10):
        q_true = random_n_unit_vec(4)
        if q_true[0] < 0:
            q_true = -q_true
        x = np.concatenate([np.zeros(3), q_true])

        q_meas = tracker.clean_reading(x, os)

        if not np.any(np.isnan(q_meas)):
            # q and -q represent the same rotation
            if np.dot(q_meas, q_true) < 0:
                q_meas = -q_meas
            np.testing.assert_allclose(q_meas, q_true, atol=1e-10)


# ---- Edge cases ----

def test_insufficient_stars_returns_nan():
    """Tiny FOV should make it impossible to see enough stars."""
    noise = Noise(noise=np.zeros(4), std_noise=np.array([1e-8] * 4))
    tracker = StarTrackerQuaternion(
        boresight=np.array([1.0, 0.0, 0.0]),
        fov=np.deg2rad(0.001),
        noise=noise,
        min_stars=2,
    )
    os = _make_os()
    x = np.zeros(7)
    x[3:7] = [1.0, 0.0, 0.0, 0.0]

    q_meas = tracker.clean_reading(x, os)
    assert np.all(np.isnan(q_meas))


def test_min_stars_parameter():
    """Setting min_stars higher should make it harder to get a solution."""
    noise = Noise(noise=np.zeros(4), std_noise=np.array([1e-8] * 4))
    # Narrow FOV with high min_stars
    tracker = StarTrackerQuaternion(
        boresight=np.array([0.0, 0.0, 1.0]),
        fov=np.deg2rad(5.0),
        noise=noise,
        min_stars=10,
    )
    os = _make_os()
    x = np.zeros(7)
    x[3:7] = [1.0, 0.0, 0.0, 0.0]

    q_meas = tracker.clean_reading(x, os)
    # With a 5-deg FOV and 10 stars required, very likely NaN
    assert np.all(np.isnan(q_meas))


# ---- Jacobians ----

def test_jacobian_finite_difference():
    tracker = _make_tracker()
    os = _make_os()

    q = np.array([0.9, 0.2, 0.3, 0.1])
    q = q / np.linalg.norm(q)
    x = np.concatenate([np.zeros(3), q])

    q_meas = tracker.clean_reading(x, os)
    if np.any(np.isnan(q_meas)):
        return

    J_ana = tracker.basestate_jac(x, os)

    # clean_reading returns the quaternion directly (identity mapping).
    # The FD Jacobian of q -> q (with scalar-positive enforcement) is I
    # in the neighborhood of a scalar-positive quaternion.
    def measurement_func(state):
        q_test = state[3:7].copy()
        if q_test[0] < 0:
            q_test = -q_test
        return q_test

    J_fd = nd.Jacobian(measurement_func)(x)
    # Compare quaternion block only (omega block is zero)
    np.testing.assert_allclose(
        J_ana[3:7, :], J_fd[:, 3:7].T, rtol=1e-4, atol=1e-8
    )


def test_omega_jacobian_is_zero():
    tracker = _make_tracker()
    os = _make_os()

    x = np.zeros(7)
    x[3:7] = [1.0, 0.0, 0.0, 0.0]

    tracker.clean_reading(x, os)
    J = tracker.basestate_jac(x, os)
    np.testing.assert_allclose(J[0:3, :], np.zeros((3, 4)), atol=1e-15)


def test_bias_jacobian_is_empty():
    tracker = _make_tracker()
    os = _make_os()

    x = np.zeros(7)
    x[3:7] = [1.0, 0.0, 0.0, 0.0]

    J_bias = tracker.bias_jac(x, os)
    assert J_bias.shape == (0, 4)


# ---- Noisy reading ----

def test_noisy_reading_is_unit_quaternion():
    tracker = _make_tracker(noise_std=1e-4)
    os = _make_os()

    for _ in range(10):
        q = random_n_unit_vec(4)
        x = np.concatenate([np.zeros(3), q])

        q_meas = tracker.reading(x, os)

        if not np.any(np.isnan(q_meas)):
            assert abs(np.linalg.norm(q_meas) - 1.0) < 1e-10


def test_noisy_reading_scalar_positive():
    tracker = _make_tracker(noise_std=1e-4)
    os = _make_os()

    for _ in range(10):
        q = random_n_unit_vec(4)
        x = np.concatenate([np.zeros(3), q])

        q_meas = tracker.reading(x, os)

        if not np.any(np.isnan(q_meas)):
            assert q_meas[0] >= 0


# ---- Factory functions ----

def test_create_bct_nst_quaternion():
    tracker = create_bct_nst_quaternion()
    assert tracker.output_length == 4
    assert tracker.fov == pytest.approx(np.deg2rad(20.0))
    assert tracker.sun_exclusion == pytest.approx(np.deg2rad(45.0))


def test_create_generic_star_tracker_quaternion():
    tracker = create_generic_star_tracker_quaternion(
        noise_arcsec=8.0,
        fov_deg=25.0,
        sun_exclusion_deg=40.0,
        min_stars=3,
    )
    assert tracker.output_length == 4
    assert tracker.fov == pytest.approx(np.deg2rad(25.0))
    assert tracker.sun_exclusion == pytest.approx(np.deg2rad(40.0))
    assert tracker.min_stars == 3


def test_custom_boresight():
    boresight = np.array([0.0, 1.0, 0.0])
    tracker = create_bct_nst_quaternion(boresight=boresight)
    np.testing.assert_allclose(
        tracker.boresight,
        boresight / np.linalg.norm(boresight),
        rtol=1e-10,
    )


if __name__ == "__main__":
    test_quaternion_output_shape()
    test_quaternion_is_unit()
    test_quaternion_scalar_positive()
    test_quaternion_matches_true_attitude()
    test_insufficient_stars_returns_nan()
    test_min_stars_parameter()
    test_jacobian_finite_difference()
    test_omega_jacobian_is_zero()
    test_bias_jacobian_is_empty()
    test_noisy_reading_is_unit_quaternion()
    test_noisy_reading_scalar_positive()
    test_create_bct_nst_quaternion()
    test_create_generic_star_tracker_quaternion()
    test_custom_boresight()
    print("All tests passed.")
