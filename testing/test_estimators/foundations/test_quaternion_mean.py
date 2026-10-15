"""Small algebra-only tests: no orbit, satellite integration, or simulation."""

import numpy as np
import pytest

from ADCS.estimators.quaternion_mean import _deviations, quaternion_mean
from ADCS.helpers.math_helpers import quat_diff, quat_mult
from ADCS.state import State


@pytest.mark.parametrize(
    "mode", ["quaternion_vector", "rotation_vector", "mrp", "two_mrp", "cayley"]
)
def test_chart_mean_and_jacobian_with_signed_weights(mode):
    vectors = np.random.default_rng(24).normal(size=(13, 3)) * 0.15
    values = np.array(
        [
            State.quaternion_delta_from_vector(vector, mode="rotation_vector")
            for vector in vectors
        ]
    )
    weights = np.array([-2.0] + [0.25] * 12)
    mean = quaternion_mean(values, weights, mode=mode)
    errors = np.array(
        [
            State.quaternion_delta_to_vector(quat_diff(mean, value), mode=mode)
            for value in values
        ]
    )
    np.testing.assert_allclose(weights @ errors, 0.0, atol=1e-12)
    # Quaternion signs must not affect either the mean or the derivatives.
    flipped = values.copy()
    flipped[1::2] *= -1
    np.testing.assert_allclose(
        quaternion_mean(flipped, weights, mode=mode), mean, atol=1e-12
    )

    _, jacobian = _deviations(mean, values, mode)
    step = 1e-6
    for axis in range(3):
        delta = np.eye(3)[axis] * step
        plus = quat_mult(
            mean, State.quaternion_delta_from_vector(delta, mode="rotation_vector")
        )
        minus = quat_mult(
            mean, State.quaternion_delta_from_vector(-delta, mode="rotation_vector")
        )
        numeric = (
            _deviations(plus, values, mode)[0] - _deviations(minus, values, mode)[0]
        ) / (2 * step)
        np.testing.assert_allclose(jacobian[:, :, axis], numeric, atol=2e-9)


def test_cayley_mean_reports_exact_singularity():
    with pytest.raises(ValueError, match="Cayley chart singularity.*rotation_vector"):
        quaternion_mean(
            np.array([[1.0, 0, 0, 0], [0.0, 1, 0, 0]]),
            np.array([0.5, 0.5]),
            mode="cayley",
        )
