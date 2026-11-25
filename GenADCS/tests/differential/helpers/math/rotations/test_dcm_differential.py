from typing import Any

import numpy as np


def test_dcm_kernels_and_classes_match(rotation_backends: Any) -> None:
    results = []
    for m in rotation_backends.values():
        d = m.DCM.from_axis_angle([1.0, 2.0, -1.0], 0.83)
        results.append(
            (
                d.to_dcm(),
                d.inverse().to_dcm(),
                d.rotate([0.3, -0.7, 1.1]),
                m.dcm.normalize_dcm(d.to_dcm()),
                m.dcm.dcm_orthogonality_error(d.to_dcm()),
            )
        )
    for result in results[1:]:
        for a, b in zip(result, results[0]):
            np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-12)
