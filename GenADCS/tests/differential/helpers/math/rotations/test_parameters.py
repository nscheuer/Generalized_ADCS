from typing import Any

import numpy as np


def test_mrp_and_rodrigues_represent_same_rotations_across_backends(rotation_backends: Any) -> None:
    results = []
    for m in rotation_backends.values():
        mrp = m.Modified_Rodriguez_Parameters.from_axis_angle([1.0, -2.0, 3.0], 1.1)
        rod = m.Rodriguez_Parameters.from_axis_angle([1.0, -2.0, 3.0], 1.1)
        results.append(
            (
                mrp.components,
                mrp.to_dcm(),
                mrp.switch_shadow_set().to_dcm(),
                rod.components,
                rod.to_dcm(),
                m.mrp.mrp_to_quaternion(mrp.components),
                m.rodrigues.rodrigues_to_quaternion(rod.components),
            )
        )
    for result in results[1:]:
        for a, b in zip(result, results[0]):
            np.testing.assert_allclose(a, b, rtol=3e-12, atol=3e-12)
