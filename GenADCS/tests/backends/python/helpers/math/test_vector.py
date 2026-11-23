import numpy as np
import pytest

from GenADCS.backends.python.helpers.math.vector import cross, norm, normalize, skew, unit


@pytest.mark.parametrize(
    "a,b",
    [([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), ([1.0, 2.0, 3.0], [-4.0, 5.0, 2.0])],
)
def test_cross_matches_numpy_and_is_orthogonal(a, b):
    actual = cross(np.asarray(a), np.asarray(b))
    np.testing.assert_allclose(actual, np.cross(a, b))
    assert np.dot(actual, a) == pytest.approx(0.0, abs=1e-14)
    assert np.dot(actual, b) == pytest.approx(0.0, abs=1e-14)


def test_cross_is_antisymmetric_and_zero_for_parallel_vectors():
    a = np.array([2.0, -3.0, 4.0])
    b = np.array([-1.0, 5.0, 2.0])
    np.testing.assert_allclose(cross(a, b), -cross(b, a))
    np.testing.assert_array_equal(cross(a, 3.5 * a), np.zeros(3))


@pytest.mark.parametrize("vector", [np.array([3.0, 4.0, 0.0]), np.array([1.0, -2.0, 2.0]), np.zeros(3)])
def test_norm_matches_euclidean_norm(vector):
    assert norm(vector) == pytest.approx(np.linalg.norm(vector))


def test_normalize_returns_unit_vector_and_preserves_zero_vector():
    vector = np.array([2.0, -3.0, 6.0])
    actual = normalize(vector)
    np.testing.assert_allclose(actual, vector / np.linalg.norm(vector))
    assert norm(actual) == pytest.approx(1.0)
    np.testing.assert_array_equal(normalize(np.zeros(3)), np.zeros(3))


def test_unit_returns_unit_vector_and_rejects_zero_vector():
    actual = unit(np.array([0.0, 3.0, 4.0]))
    np.testing.assert_allclose(actual, [0.0, 0.6, 0.8])
    with pytest.raises(ValueError, match="zero vector"):
        unit(np.zeros(3))


def test_skew_matrix_encodes_cross_product():
    vector = np.array([1.5, -2.0, 4.0])
    other = np.array([-3.0, 0.5, 2.0])
    matrix = skew(vector)
    np.testing.assert_allclose(matrix, -matrix.T)
    np.testing.assert_allclose(matrix @ other, np.cross(vector, other))
    np.testing.assert_allclose(skew(vector) @ vector, np.zeros(3), atol=1e-14)


def test_vector_kernels_accept_noncontiguous_inputs():
    backing = np.array([1.0, 9.0, -2.0, 8.0, 3.0, 7.0])
    vector = backing[::2]
    assert not vector.flags.c_contiguous
    other = np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(cross(vector, other), np.cross(vector, other))
    assert norm(vector) == pytest.approx(np.linalg.norm(vector))
