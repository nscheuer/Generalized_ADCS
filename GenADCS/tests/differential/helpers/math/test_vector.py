"""Pairwise comparisons for vector helper results."""

from typing import Any

import numpy as np


def test_vector_operations_match_pairwise(math_backends: Any) -> None:
    left = np.array([1.0, -2.0, 3.0])
    right = np.array([-4.0, 0.5, 2.0])
    results = {
        name: (
            api.cross(left, right),
            api.norm(left),
            api.normalize(left),
            api.unit(left),
            api.skew(left),
        )
        for name, api in math_backends.items()
    }
    reference = results["python"]
    for backend, values in results.items():
        for actual, expected in zip(values, reference):
            np.testing.assert_allclose(
                actual, expected, rtol=2e-14, atol=2e-14, err_msg=backend
            )
    for values in results.values():
        np.testing.assert_allclose(values[4] @ right, values[0], atol=2e-14)
