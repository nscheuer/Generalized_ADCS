"""Contracts for geometry now shared by estimators, goals, and plots."""

import numpy as np

from ADCS.helpers.math_helpers import (
    angle_between_vectors,
    quaternion_angle,
    rot_mat,
    rot_mat_batch,
    rotation_between_vectors,
    rotate_covariance,
    skewsym,
    skewsym_batch,
)


def test_batched_quaternion_rotation_and_skew_match_scalar_operations():
    quaternions = np.array([[1.0, 0.0, 0.0, 0.0], [0.8, 0.2, -0.1, 0.3]])
    vectors = np.array([[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0]])
    np.testing.assert_allclose(rot_mat_batch(quaternions), np.stack([rot_mat(q) for q in quaternions]))
    np.testing.assert_allclose(skewsym_batch(vectors), np.stack([skewsym(v) for v in vectors]))


def test_opposite_vector_rotation_is_proper_and_maps_the_vector():
    first = np.array([1.0, 0.0, 0.0])
    second = -first
    rotation = rotation_between_vectors(first, second)
    np.testing.assert_allclose(rotation @ first, second)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3))
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0)


def test_shared_angles_and_covariance_transform():
    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(angle_between_vectors(x, y), np.pi / 2.0)
    np.testing.assert_allclose(quaternion_angle(np.array([1.0, 0.0, 0.0, 0.0]),
                                                np.array([-1.0, 0.0, 0.0, 0.0])), 0.0)
    transform = rotation_between_vectors(x, y)
    covariance = np.diag([4.0, 1.0, 2.0])
    np.testing.assert_allclose(rotate_covariance(covariance, transform),
                               transform @ covariance @ transform.T)
