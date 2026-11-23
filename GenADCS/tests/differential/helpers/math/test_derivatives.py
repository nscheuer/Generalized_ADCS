"""Pairwise comparisons for derivative helper results."""

import numpy as np


def test_quaternion_derivative_family_matches_pairwise(math_backends):
    q = np.array([0.7, -0.2, 0.3, 0.6])
    q /= np.linalg.norm(q)
    omega = np.array([0.4, -0.7, 0.2])
    vector = np.array([1.3, -0.7, 2.1])
    results = {
        name: (
            api.quaternion_derivative(q, omega),
            api.quaternion_rate_matrix(q),
            api.angular_velocity_from_quaternion_derivative(q, api.quaternion_derivative(q, omega)),
            api.quaternion_normalization_jacobian(q),
            api.rotated_vector_quaternion_jacobian(q, vector),
            api.quaternion_to_euler321(q),
        )
        for name, api in math_backends.items()
    }
    reference = results["python"]
    for backend, values in results.items():
        for actual, expected in zip(values, reference):
            np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-14, err_msg=backend)
