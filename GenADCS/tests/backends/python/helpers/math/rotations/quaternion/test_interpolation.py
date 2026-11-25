import numpy as np
import pytest
from numpy.typing import ArrayLike

from GenADCS.backends.python.helpers.math.rotations.quaternion.algebra import (
    _inverse,
    _multiply,
    _normalize_quaternion,
)
from GenADCS.backends.python.helpers.math.rotations.quaternion.interpolation import (
    quaternion_to_rotation_vector,
    rotation_vector_to_quaternion,
    slerp,
)


@pytest.mark.parametrize("fraction", [-0.5, 0.0, 0.25, 0.5, 1.0, 1.5])
def test_slerp_follows_constant_angle_geodesic_even_when_extrapolating(
    fraction: float,
) -> None:
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    end = rotation_vector_to_quaternion(np.array([0.0, 0.0, np.pi / 2]))
    actual = slerp(identity, end, fraction)
    expected = rotation_vector_to_quaternion(np.array([0.0, 0.0, fraction * np.pi / 2]))
    np.testing.assert_allclose(actual, expected, atol=2e-14)


def test_slerp_shortest_path_and_near_equal_branch() -> None:
    a = rotation_vector_to_quaternion(np.array([0.1, -0.2, 0.3]))
    b = rotation_vector_to_quaternion(np.array([0.10001, -0.2, 0.3]))
    np.testing.assert_allclose(slerp(a, b, 0.5), slerp(a, -b, 0.5), atol=1e-14)
    assert np.linalg.norm(slerp(a, b, 0.5)) == pytest.approx(1)


def test_slerp_and_inverse_reject_zero_quaternion() -> None:
    with pytest.raises(ValueError):
        slerp(np.zeros(4), np.array([1.0, 0, 0, 0]), 0.5)
    with pytest.raises(ValueError):
        _inverse(np.zeros(4))
    with pytest.raises(ValueError):
        _normalize_quaternion(np.zeros(4))


@pytest.mark.parametrize(
    "vector", [np.zeros(3), np.array([1e-12, 0, 0]), np.array([0.2, -0.3, 0.1])]
)
def test_rotation_vector_roundtrip_at_zero_small_and_general_angles(
    vector: ArrayLike,
) -> None:
    recovered = quaternion_to_rotation_vector(rotation_vector_to_quaternion(vector))
    np.testing.assert_allclose(recovered, vector, atol=2e-14)


def test_hamilton_product_matches_sequential_rotation() -> None:
    x = rotation_vector_to_quaternion(np.array([0.4, 0.0, 0.0]))
    y = rotation_vector_to_quaternion(np.array([0.0, 0.3, 0.0]))
    product = _multiply(x, y)
    assert np.linalg.norm(product) == pytest.approx(1)
    np.testing.assert_allclose(
        _multiply(product, _inverse(product)), [1.0, 0.0, 0.0, 0.0], atol=1e-14
    )
