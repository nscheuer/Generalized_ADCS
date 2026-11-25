"""Public vector helper identities and zero-vector behavior."""

from typing import Any

import numpy as np
import pytest


def test_cross_product_and_skew_matrix_contract(math_api: Any) -> None:
    left = np.array([1.0, 2.0, 3.0])
    right = np.array([-4.0, 5.0, 2.0])
    result = math_api.cross(left, right)
    np.testing.assert_allclose(result, np.cross(left, right), atol=1e-14)
    np.testing.assert_allclose(np.dot(result, left), 0.0, atol=1e-14)
    matrix = math_api.skew(left)
    np.testing.assert_allclose(matrix, -matrix.T, atol=1e-14)
    np.testing.assert_allclose(matrix @ right, result, atol=1e-14)


def test_norm_normalize_and_unit_contract(math_api: Any) -> None:
    vector = np.array([3.0, 4.0, 0.0])
    assert math_api.norm(vector) == pytest.approx(5.0)
    np.testing.assert_allclose(math_api.normalize(vector), vector / 5.0)
    np.testing.assert_allclose(math_api.unit(vector), vector / 5.0)
    np.testing.assert_array_equal(math_api.normalize(np.zeros(3)), np.zeros(3))
    with pytest.raises(ValueError, match="zero vector"):
        math_api.unit(np.zeros(3))
