from typing import Any

import numpy as np


def test_drag_force_and_torque_match_across_backends(dynamics_backends: Any) -> None:
    angle = 0.63
    cases = (
        (
            np.array([0.0, 0.0, 7500.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            1.1e-11,
            np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
            np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0]]),
            np.array([2.0, 1.5]),
            np.array([2.2, 1.8]),
        ),
        (
            np.array([500.0, -120.0, 30.0]),
            np.array([np.cos(angle / 2.0), 0.0, np.sin(angle / 2.0), 0.0]),
            0.0,
            np.array([[1.0, 0.0, 0.0]]),
            np.array([[0.2, -0.4, 0.1]]),
            np.array([3.0]),
            np.array([2.0]),
        ),
    )

    for velocity, quaternion, density, normals, arms, areas, coefficients in cases:
        outputs: dict[str, tuple[Any, Any]] = {}
        for name, backend in dynamics_backends.items():
            outputs[name] = (
                backend.projected_area_drag_force(
                    velocity, quaternion, density, normals, areas, coefficients
                ),
                backend.projected_area_drag_torque(
                    velocity, quaternion, density, normals, arms, areas, coefficients
                ),
            )

        reference = outputs["python"]
        for force, torque in outputs.values():
            np.testing.assert_allclose(force, reference[0], rtol=2e-12, atol=1e-18)
            np.testing.assert_allclose(torque, reference[1], rtol=2e-12, atol=1e-18)
