import numpy as np
import pytest


def test_quaternion_construction_conventions_and_rotation(rotations):
    q = rotations.Quaternion.from_axis_angle([0.0, 0.0, 2.0], np.pi / 2)
    np.testing.assert_allclose(q.rotate([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=2e-12)
    np.testing.assert_allclose(
        q.body_to_inertial([0.0, 1.0, 0.0]), [1.0, 0.0, 0.0], atol=2e-12
    )
    j = q.to_convention("JPL")
    np.testing.assert_allclose(
        j.to_convention("Hamilton").hamilton_components,
        q.hamilton_components,
        atol=2e-12,
    )
    assert q.norm == pytest.approx(1.0, abs=2e-12)


def test_quaternion_algebra_conversion_and_slerp(rotations):
    q = rotations.Quaternion.from_rotation_vector([0.0, 0.0, 0.7])
    inv = q.inverse()
    np.testing.assert_allclose((q * inv).to_dcm(), np.eye(3), atol=2e-12)
    np.testing.assert_allclose(
        q.to_dcm(), rotations.DCM.from_quaternion(q).to_dcm(), atol=2e-12
    )
    result = rotations.quaternion.slerp(
        np.array([1.0, 0, 0, 0]), q.hamilton_components, 0.5
    )
    assert np.linalg.norm(result) == pytest.approx(1.0, abs=2e-12)
    np.testing.assert_allclose(
        rotations.quaternion.quaternion_to_rotation_vector(
            rotations.quaternion.rotation_vector_to_quaternion(
                np.array([0.0, 0.0, 0.2])
            )
        ),
        [0.0, 0.0, 0.2],
        atol=2e-12,
    )


@pytest.mark.parametrize(
    "bad", [[0.0, 0.0, 0.0, 0.0], [1.0, 2.0], [np.nan, 0.0, 0.0, 0.0]]
)
def test_quaternion_invalid_inputs_rejected(rotations, bad):
    with pytest.raises((ValueError, TypeError)):
        rotations.Quaternion(bad)
