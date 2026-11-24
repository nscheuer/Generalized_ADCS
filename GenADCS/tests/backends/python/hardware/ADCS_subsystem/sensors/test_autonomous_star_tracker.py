"""Detailed tests for the Numba autonomous star-tracker model."""

import numpy as np

from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import AutonomousStarTracker
from GenADCS.backends.python.helpers.math.rotations.quaternion.algebra import _multiply
from GenADCS.backends.python.helpers.math.rotations.quaternion.conversions import _axis_angle_to_quaternion


def make_tracker(noise=None, orientation=None, misalignment=None):
    identity = np.array([1., 0., 0., 0.])
    return AutonomousStarTracker(
        np.zeros(3), identity if orientation is None else orientation,
        identity if misalignment is None else misalignment, 1.,
        np.eye(3) * 1e-8 if noise is None else noise, np.zeros(3),
    )


def test_noiseless_quaternion_measurement_applies_mounting_in_declared_order():
    q_mount = _axis_angle_to_quaternion(np.array([0., 0., 1.]), .3)
    q_misalign = _axis_angle_to_quaternion(np.array([1., 0., 0.]), -.2)
    q_att = _axis_angle_to_quaternion(np.array([0., 1., 0.]), .5)
    tracker = make_tracker(orientation=q_mount, misalignment=q_misalign)
    actual = tracker.noiseless_biasfree_measurement(q_att)
    expected = _multiply(q_misalign, _multiply(q_mount, q_att))
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(actual), 1.)
    np.testing.assert_allclose(tracker.noiseless_biased_measurement(q_att), expected)


def test_noisy_measurement_is_normalized_and_close_at_small_covariance():
    tracker = make_tracker(noise=np.eye(3) * 1e-12)
    q = _axis_angle_to_quaternion(np.array([1., 2., -1.]), .7)
    measured = tracker.noisy_biasfree_measurement(q)
    assert measured.shape == (4,)
    assert np.all(np.isfinite(measured))
    np.testing.assert_allclose(np.linalg.norm(measured), 1., atol=1e-14)
    assert min(np.linalg.norm(measured - q), np.linalg.norm(measured + q)) < 1e-5


def test_noisy_biased_and_biased_measurement_follow_attitude_only_model():
    tracker = make_tracker()
    q = np.array([1., 0., 0., 0.])
    np.testing.assert_allclose(tracker.noiseless_biased_measurement(q), q)
    result = tracker.noisy_biased_measurement(q)
    assert result.shape == (4,)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(np.linalg.norm(result), 1., atol=1e-14)


def test_residual_is_zero_for_equal_quaternions_and_sign_invariant():
    tracker = make_tracker()
    q = _axis_angle_to_quaternion(np.array([1., -2., 3.]), .4)
    np.testing.assert_allclose(tracker.residual(q, q), np.zeros(3), atol=1e-14)
    np.testing.assert_allclose(tracker.residual(-q, q), np.zeros(3), atol=1e-14)


def test_residual_recovers_small_rotation_vector_to_first_order():
    tracker = make_tracker()
    estimate = np.array([1., 0., 0., 0.])
    delta = np.array([2e-4, -3e-4, 1e-4])
    measured = _axis_angle_to_quaternion(delta / np.linalg.norm(delta), np.linalg.norm(delta))
    np.testing.assert_allclose(tracker.residual(measured, estimate), delta, atol=1e-11)


def test_quaternion_and_mrp_jacobians_match_spec_at_zero():
    tracker = make_tracker()
    np.testing.assert_array_equal(tracker.jacobian_relative_quaternion(), 2. * np.eye(3))
    np.testing.assert_array_equal(tracker.jacobian_relative_mrp(np.zeros(3)), 4. * np.eye(3))


def test_angle_jacobian_is_identity_at_zero_and_finite_away_from_zero():
    tracker = make_tracker()
    np.testing.assert_allclose(tracker.jacobian_relative_angle_vector(np.zeros(3)), np.eye(3))
    result = tracker.jacobian_relative_angle_vector(np.array([.1, -.2, .05]))
    assert result.shape == (3, 3)
    assert np.all(np.isfinite(result))


def test_mrp_jacobian_matches_residual_finite_difference_away_from_zero():
    tracker = make_tracker()
    parameters = np.array([.12, -.08, .04])
    step = 1e-6
    finite_difference = np.empty((3, 3))
    for column in range(3):
        perturbation = np.zeros(3)
        perturbation[column] = step
        plus = 4 * (parameters + perturbation) / (1 + np.dot(parameters + perturbation, parameters + perturbation))
        minus = 4 * (parameters - perturbation) / (1 + np.dot(parameters - perturbation, parameters - perturbation))
        finite_difference[:, column] = (plus - minus) / (2 * step)
    np.testing.assert_allclose(tracker.jacobian_relative_mrp(parameters), finite_difference, rtol=1e-8, atol=1e-9)


def test_availability_and_mounting_adjusted_fisher_information():
    covariance = np.diag([.2, .3, .5])
    orientation = _axis_angle_to_quaternion(np.array([0., 0., 1.]), .4)
    misalignment = _axis_angle_to_quaternion(np.array([1., 0., 0.]), -.2)
    tracker = make_tracker(noise=covariance, orientation=orientation, misalignment=misalignment)
    assert tracker.availability()
    from GenADCS.backends.python.helpers.math.rotations.sensor_kernels import mounting_dcm
    rotation = mounting_dcm(orientation, misalignment)
    expected = rotation.T @ np.linalg.solve(covariance, rotation)
    np.testing.assert_allclose(tracker.fisher_information_matrix(), expected)
