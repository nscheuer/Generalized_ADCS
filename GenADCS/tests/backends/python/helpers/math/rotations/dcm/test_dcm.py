import numpy as np
import pytest

from GenADCS.backends.python.helpers.math.rotations.dcm.dcm import DCM


def test_identity_and_owned_arrays():
    rotation = DCM.identity()
    np.testing.assert_array_equal(rotation.components, np.eye(3))
    assert rotation.norm == pytest.approx(0)
    copy = rotation.as_matrix()
    copy[0, 0] = 99
    assert rotation.as_matrix()[0, 0] == 1


def test_axis_angle_rotation_inverse_and_composition():
    r = DCM.from_axis_angle([0, 0, 1], np.pi / 2)
    np.testing.assert_allclose(r.rotate([1, 0, 0]), [0, 1, 0], atol=1e-14)
    np.testing.assert_allclose(r.body_to_inertial([0, 1, 0]), [1, 0, 0], atol=1e-14)
    np.testing.assert_allclose((r * r.inverse()).as_matrix(), np.eye(3), atol=1e-14)
    np.testing.assert_allclose(r.relative_to(DCM.identity()).as_matrix(), r.as_matrix())
    np.testing.assert_allclose(r.to_quaternion().to_dcm(), r.as_matrix(), atol=1e-14)


def test_rotation_vector_and_array_protocol():
    r = DCM.from_rotation_vector([0, 0, np.pi / 2])
    assert np.asarray(r).shape == (3, 3)
    np.testing.assert_allclose(r.to_dcm(), r.as_matrix())


@pytest.mark.parametrize("bad", [np.zeros((2, 2)), np.zeros((3, 3)), np.diag([1, 1, -1]), np.full((3, 3), np.nan)])
def test_rejects_bad_matrices(bad):
    with pytest.raises(ValueError):
        DCM(bad)


@pytest.mark.parametrize("axis,angle", [([0, 0], 1), ([np.inf, 0, 0], 1), ([1, 0, 0], np.nan)])
def test_rejects_bad_axis_angle(axis, angle):
    with pytest.raises(ValueError):
        DCM.from_axis_angle(axis, angle)


@pytest.mark.parametrize("vector", [[1, 2], [np.nan, 0, 0]])
def test_rejects_bad_vectors(vector):
    with pytest.raises(ValueError):
        DCM.identity().rotate(vector)


def test_normalized_conjugate_and_relative_rotation():
    rotation = DCM.from_axis_angle([1, 0, 0], .4)
    np.testing.assert_allclose(rotation.normalized().as_matrix(), rotation.as_matrix(), atol=1e-14)
    np.testing.assert_allclose(rotation.conjugate().as_matrix(), rotation.inverse().as_matrix(), atol=1e-14)
    np.testing.assert_allclose(rotation.relative_to(rotation).as_matrix(), np.eye(3), atol=1e-14)


def test_input_matrix_and_array_protocol_are_owned():
    original = np.eye(3)
    rotation = DCM(original)
    original[0, 0] = 99
    assert rotation.as_matrix()[0, 0] == 1
    array = np.asarray(rotation)
    array[0, 0] = 88
    assert rotation.as_matrix()[0, 0] == 1
