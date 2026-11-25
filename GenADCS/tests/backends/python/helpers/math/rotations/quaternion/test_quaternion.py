import numpy as np
import pytest

from GenADCS.backends.python.helpers.math.rotations.quaternion.quaternion import (
    Quaternion,
)
from GenADCS.backends.python.helpers.math.rotations.quaternion.interpolation import (
    slerp,
    rotation_vector_to_quaternion,
    quaternion_to_rotation_vector,
)


def test_identity_normalization_and_copy_semantics():
    q = Quaternion([2, 0, 0, 0])
    np.testing.assert_array_equal(q.hamilton_components, [1, 0, 0, 0])
    assert q.norm == pytest.approx(1)
    values = q.components
    values[0] = 99
    assert q.components[0] == 1
    np.testing.assert_array_equal(np.asarray(Quaternion.identity()), [1, 0, 0, 0])


def test_jpl_hamilton_convention_roundtrip():
    q = Quaternion([0, 0, -np.sqrt(0.5), np.sqrt(0.5)], convention="JPL")
    np.testing.assert_allclose(
        q.hamilton_components, [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    )
    np.testing.assert_allclose(
        q.as_convention("JPL"), [0, 0, -np.sqrt(0.5), np.sqrt(0.5)]
    )
    assert q.to_convention("Hamilton").convention == "Hamilton"


def test_rotation_inverse_multiply_and_dcm():
    q = Quaternion.from_axis_angle([0, 0, 1], np.pi / 2)
    np.testing.assert_allclose(q.rotate([1, 0, 0]), [0, 1, 0], atol=1e-14)
    np.testing.assert_allclose(q.body_to_inertial([0, 1, 0]), [1, 0, 0], atol=1e-14)
    np.testing.assert_allclose((q * q.inverse()).to_dcm(), np.eye(3), atol=1e-14)
    np.testing.assert_allclose(
        Quaternion.from_dcm(q.to_dcm()).to_dcm(), q.to_dcm(), atol=1e-14
    )
    np.testing.assert_allclose(
        q.relative_to(Quaternion.identity()).to_dcm(), q.to_dcm(), atol=1e-14
    )


@pytest.mark.parametrize("bad", [[0, 0, 0, 0], [1, 2, 3], [np.nan, 0, 0, 0]])
def test_rejects_bad_quaternions(bad):
    with pytest.raises(ValueError):
        Quaternion(bad)


def test_rejects_bad_convention_axis_and_vectors():
    with pytest.raises(ValueError):
        Quaternion(convention="bad")
    with pytest.raises(ValueError):
        Quaternion.from_axis_angle([1, 0], 0.1)
    with pytest.raises(ValueError):
        Quaternion.identity().rotate([np.inf, 0, 0])


def test_slerp_endpoints_midpoint_and_antipodal_sign():
    q0 = Quaternion.identity().hamilton_components
    q1 = Quaternion.from_axis_angle([0, 0, 1], np.pi / 2).hamilton_components
    np.testing.assert_allclose(slerp(q0, q1, 0), q0)
    np.testing.assert_allclose(slerp(q0, q1, 1), q1)
    midpoint = Quaternion(slerp(q0, q1, 0.5))
    np.testing.assert_allclose(
        midpoint.rotate([1, 0, 0]), [np.sqrt(0.5), np.sqrt(0.5), 0], atol=1e-14
    )
    np.testing.assert_allclose(
        Quaternion(slerp(q0, -q1, 0.5)).to_dcm(), midpoint.to_dcm(), atol=1e-14
    )


def test_rotation_vector_conversion_roundtrip():
    vector = np.array([0.2, -0.1, 0.3])
    recovered = quaternion_to_rotation_vector(rotation_vector_to_quaternion(vector))
    np.testing.assert_allclose(recovered, vector, atol=1e-14)


def test_conjugate_and_normalized_preserve_rotation():
    q = Quaternion.from_axis_angle([1, 0, 0], 0.7)
    np.testing.assert_allclose(q.conjugate().to_dcm(), q.inverse().to_dcm(), atol=1e-14)
    np.testing.assert_allclose(q.normalized().to_dcm(), q.to_dcm(), atol=1e-14)


@pytest.mark.parametrize(
    "bad", [np.zeros((2, 2)), np.diag([1, 1, -1]), np.full((3, 3), np.inf)]
)
def test_from_dcm_rejects_invalid_matrix(bad):
    with pytest.raises(ValueError):
        Quaternion.from_dcm(bad)
