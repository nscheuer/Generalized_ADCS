import numpy as np
import pytest

from GenADCS.backends.python.helpers.math.rotations.mrp.kernels import (
    mrp_shadow_set,
    mrp_to_quaternion,
    quaternion_to_mrp,
)


@pytest.mark.parametrize(
    "parameters", [np.zeros(3), np.array([0.1, -0.2, 0.3]), np.array([0.0, 0, 1.0])]
)
def test_mrp_quaternion_roundtrip(parameters: np.ndarray) -> None:
    np.testing.assert_allclose(
        quaternion_to_mrp(mrp_to_quaternion(parameters)), parameters, atol=1e-14
    )


def test_shadow_set_involution_and_same_rotation() -> None:
    parameters = np.array([0.2, -0.4, 0.1])
    shadow = mrp_shadow_set(parameters)
    np.testing.assert_allclose(mrp_shadow_set(shadow), parameters, atol=1e-14)
    q = mrp_to_quaternion(parameters)
    shadow_q = mrp_to_quaternion(shadow)
    np.testing.assert_allclose(q, -shadow_q, atol=1e-14)


def test_identity_has_no_finite_shadow() -> None:
    with pytest.raises(ValueError):
        mrp_shadow_set(np.zeros(3))
