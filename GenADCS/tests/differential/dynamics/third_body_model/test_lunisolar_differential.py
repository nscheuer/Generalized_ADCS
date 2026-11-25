from typing import Any

import numpy as np


def test_lunisolar_acceleration_matches_across_backends(dynamics_backends: Any) -> None:
    cases = (
        (
            np.array([7e6, 0.0, 0.0]),
            np.array([1.5e11, 0.0, 0.0]),
            np.array([0.0, 3.8e8, 0.0]),
            1.3271244e20,
            4.9048695e12,
        ),
        (
            np.array([-3e6, 4e6, 2e6]),
            np.array([1.0e11, -8.0e10, 2.0e10]),
            np.array([-1.0e8, 3.0e8, 2.0e8]),
            1.32712440018e20,
            4.9048695e12,
        ),
    )

    for args in cases:
        results = {
            name: backend.lunisolar_acceleration(*args)
            for name, backend in dynamics_backends.items()
        }
        reference = results["python"]
        for acceleration in results.values():
            np.testing.assert_allclose(acceleration, reference, rtol=5e-11, atol=1e-20)
