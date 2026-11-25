from typing import Any

import numpy as np


def test_gravity_gradient_torque_matches_across_backends(dynamics_backends: Any) -> None:
    cases = (
        (
            np.array([7e6, 7e6, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.diag([2.0, 3.0, 4.0]),
            3.986004418e14,
        ),
        (
            np.array([-2e6, 3e6, 6e6]),
            np.array([0.9238795325112867, 0.0, 0.3826834323650898, 0.0]),
            np.array([[4.0, 0.2, -0.1], [0.2, 3.0, 0.3], [-0.1, 0.3, 2.0]]),
            3.986004418e14,
        ),
        (
            np.array([7e6, -1e6, 2e6]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            4.0 * np.eye(3),
            3.986004418e14,
        ),
    )

    for position, quaternion, inertia, mu in cases:
        results = {
            name: backend.static_gravity_gradient_torque(
                position, quaternion, inertia, mu
            )
            for name, backend in dynamics_backends.items()
        }
        reference = results["python"]
        for torque in results.values():
            np.testing.assert_allclose(torque, reference, rtol=2e-12, atol=1e-16)
