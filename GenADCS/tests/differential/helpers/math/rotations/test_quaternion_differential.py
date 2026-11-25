from typing import Any

import numpy as np


def test_quaternion_outputs_match_pairwise(rotation_backends: Any) -> None:
    results = {}
    for name, m in rotation_backends.items():
        q = m.Quaternion.from_axis_angle([1.0, -2.0, 3.0], 1.2)
        results[name] = (
            q.hamilton_components,
            q.to_dcm(),
            q.rotate([0.2, -0.4, 0.9]),
            q.inverse().hamilton_components,
            m.slerp(np.array([1.0, 0, 0, 0]), q.hamilton_components, 0.37),
            m.quaternion_to_rotation_vector(q.hamilton_components),
        )
    ref = results["python"]
    for result in results.values():
        for actual, expected in zip(result, ref):
            np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
