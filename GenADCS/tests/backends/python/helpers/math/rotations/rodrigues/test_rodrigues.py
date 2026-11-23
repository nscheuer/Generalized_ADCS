import numpy as np
import pytest

from GenADCS.backends.python.helpers.math.rotations.rodrigues.rodrigues_parameters import Rodriguez_Parameters as RP


def test_identity_quaternion_and_inverse():
    np.testing.assert_array_equal(RP.identity().components, [0, 0, 0])
    r = RP.from_axis_angle([0, 0, 1], np.pi / 3)
    np.testing.assert_allclose(RP.from_quaternion(r.to_quaternion()).to_dcm(), r.to_dcm(), atol=1e-14)
    np.testing.assert_allclose((r * r.inverse()).to_dcm(), np.eye(3), atol=1e-14)


def test_vector_rotation_and_relative_rotation():
    r = RP.from_rotation_vector([0, 0, np.pi / 4])
    v = np.array([1., 2., 3.])
    np.testing.assert_allclose(r.body_to_inertial(r.rotate(v)), v, atol=1e-14)
    np.testing.assert_allclose(r.relative_to(RP.identity()).to_dcm(), r.to_dcm(), atol=1e-14)


def test_singularity_and_invalid_parameters():
    with pytest.raises(ValueError, match="singular"):
        RP.from_axis_angle([1, 0, 0], np.pi)
    with pytest.raises(ValueError):
        RP([1, 2])
    with pytest.raises(ValueError):
        RP([np.nan, 0, 0])


def test_components_are_copies_and_bad_relative_input_is_rejected():
    rotation = RP([.1, .2, .3])
    values = rotation.components
    values[0] = 99
    assert rotation.components[0] == pytest.approx(.1)
    with pytest.raises(TypeError):
        rotation.relative_to([0, 0, 0])


def test_dcm_and_quaternion_constructors_agree():
    original = RP([.2, -.1, .3])
    np.testing.assert_allclose(RP.from_dcm(original.to_dcm()).to_dcm(), original.to_dcm(), atol=1e-14)
    np.testing.assert_allclose(RP.from_quaternion_array(original.hamilton_components).to_dcm(), original.to_dcm(), atol=1e-14)
