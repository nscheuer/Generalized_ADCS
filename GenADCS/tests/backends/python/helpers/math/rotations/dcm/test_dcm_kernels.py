import numpy as np
import pytest
from numpy.typing import ArrayLike

from GenADCS.backends.python.helpers.math.rotations.dcm.kernels import (
    axis_angle_to_dcm,
    compose_dcm,
    dcm_orthogonality_error,
    inverse_dcm,
    normalize_dcm,
    relative_dcm,
    rotate_vector,
)


def test_axis_angle_and_kernel_composition() -> None:
    x = axis_angle_to_dcm(np.array([1.0, 0, 0]), 0.3)
    y = axis_angle_to_dcm(np.array([0.0, 1, 0]), -0.2)
    np.testing.assert_allclose(compose_dcm(x, y), x @ y)
    np.testing.assert_allclose(relative_dcm(x, y), x @ y.T)
    np.testing.assert_allclose(inverse_dcm(x), x.T)
    np.testing.assert_allclose(rotate_vector(x, np.array([1.0, 2, 3])), x @ [1.0, 2, 3])


def test_normalize_dcm_repairs_small_drift_and_preserves_handedness() -> None:
    matrix = axis_angle_to_dcm(np.array([0.0, 0, 1]), 0.4)
    matrix[0, 0] += 1e-4
    repaired = normalize_dcm(matrix)
    assert dcm_orthogonality_error(repaired) < 1e-14
    assert np.linalg.det(repaired) == pytest.approx(1, abs=1e-14)


@pytest.mark.parametrize(
    "matrix", [np.zeros((3, 3)), np.array([[1.0, 2, 0], [0, 0, 0], [0, 0, 1]])]
)
def test_normalize_dcm_rejects_degenerate_columns(matrix: ArrayLike) -> None:
    with pytest.raises(ValueError):
        normalize_dcm(matrix)


def test_zero_axis_is_rejected_by_kernel() -> None:
    with pytest.raises(ValueError):
        axis_angle_to_dcm(np.zeros(3), 0.5)
