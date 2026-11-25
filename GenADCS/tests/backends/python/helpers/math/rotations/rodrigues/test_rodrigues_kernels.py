import numpy as np
import pytest

from GenADCS.backends.python.helpers.math.rotations.rodrigues.kernels import (
    quaternion_to_rodrigues,
    rodrigues_to_quaternion,
)


@pytest.mark.parametrize(
    "parameters", [np.zeros(3), np.array([0.2, -0.1, 0.3]), np.array([0.0, 0, 10.0])]
)
def test_rodrigues_quaternion_roundtrip(parameters):
    np.testing.assert_allclose(
        quaternion_to_rodrigues(rodrigues_to_quaternion(parameters)),
        parameters,
        atol=1e-13,
    )


def test_antipodal_quaternion_represents_same_parameters():
    q = rodrigues_to_quaternion(np.array([0.2, 0.1, -0.3]))
    np.testing.assert_allclose(quaternion_to_rodrigues(q), quaternion_to_rodrigues(-q))


def test_half_turn_is_singular():
    with pytest.raises(ValueError, match="singular"):
        quaternion_to_rodrigues(np.array([0.0, 1.0, 0.0, 0.0]))
