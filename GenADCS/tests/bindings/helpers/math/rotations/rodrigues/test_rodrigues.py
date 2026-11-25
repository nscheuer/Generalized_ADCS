from importlib import import_module

import numpy as np
import pytest


def cls(r):
    return r.Rodriguez_Parameters


def test_rodrigues_aliases_numpy_roundtrip_and_ownership(rotations):
    C = cls(rotations)
    assert C is rotations.RodriguesParameters
    source = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    p = C(source)
    source[:] = 0
    np.testing.assert_allclose(np.asarray(p), [0.1, -0.2, 0.3], atol=1e-7)
    rodrigues = import_module(rotations.__name__ + ".rotations.rodrigues")
    np.testing.assert_allclose(
        rodrigues.quaternion_to_rodrigues(
            rodrigues.rodrigues_to_quaternion([0.1, -0.2, 0.3])
        ),
        [0.1, -0.2, 0.3],
    )


def test_rodrigues_singular_and_invalid_binding_inputs(rotations):
    with pytest.raises(ValueError):
        cls(rotations).from_axis_angle([1.0, 0, 0], np.pi)
    with pytest.raises(ValueError):
        cls(rotations)([np.inf, 0, 0])
