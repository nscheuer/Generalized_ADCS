import numpy as np
import pytest


def test_mrp_roundtrip_shadow_and_rotation(rotations):
    C = rotations.Modified_Rodriguez_Parameters
    p = C.from_axis_angle([0.0, 0.0, 1.0], 0.6)
    np.testing.assert_allclose(
        C.from_quaternion(p.to_quaternion()).to_dcm(), p.to_dcm(), atol=2e-12
    )
    np.testing.assert_allclose(p.switch_shadow_set().to_dcm(), p.to_dcm(), atol=2e-12)
    np.testing.assert_allclose(
        rotations.mrp.quaternion_to_mrp(
            rotations.mrp.mrp_to_quaternion(np.array([0.1, 0.2, 0.3]))
        ),
        [0.1, 0.2, 0.3],
        atol=2e-12,
    )


def test_mrp_invalid_and_identity_shadow(rotations):
    with pytest.raises((ValueError, TypeError)):
        rotations.Modified_Rodriguez_Parameters([np.nan, 0, 0])
    with pytest.raises((ValueError, TypeError)):
        rotations.mrp.mrp_shadow_set(np.zeros(3))
