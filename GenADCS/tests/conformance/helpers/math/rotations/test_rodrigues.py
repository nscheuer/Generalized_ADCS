from typing import Any

import numpy as np
import pytest


def test_rodrigues_roundtrip_conversion_and_rotation(rotations: Any) -> None:
    C = rotations.Rodriguez_Parameters
    p = C.from_axis_angle([0.0, 0.0, 1.0], 0.5)
    np.testing.assert_allclose(
        C.from_quaternion(p.to_quaternion()).to_dcm(), p.to_dcm(), atol=2e-12
    )
    native = rotations.rodrigues
    np.testing.assert_allclose(
        native.quaternion_to_rodrigues(
            native.rodrigues_to_quaternion(np.array([0.1, 0.2, 0.3]))
        ),
        [0.1, 0.2, 0.3],
        atol=2e-12,
    )


def test_rodrigues_half_turn_singularity(rotations: Any) -> None:
    with pytest.raises((ValueError, TypeError)):
        rotations.Rodriguez_Parameters.from_axis_angle([1.0, 0, 0], np.pi)
