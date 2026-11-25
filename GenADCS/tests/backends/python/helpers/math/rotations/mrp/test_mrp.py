import numpy as np
import pytest

from GenADCS.backends.python.helpers.math.rotations.mrp.modified_rodrigues_parameters import (
    Modified_Rodriguez_Parameters as MRP,
)


def test_identity_shadow_and_canonicalization() -> None:
    np.testing.assert_array_equal(MRP.identity().components, [0, 0, 0])
    np.testing.assert_allclose(MRP([2, 0, 0]).components, [-0.5, 0, 0])
    with pytest.raises(ValueError):
        MRP.identity().switch_shadow_set()


def test_quaternion_dcm_and_vector_roundtrips() -> None:
    r = MRP.from_axis_angle([0, 0, 1], np.pi / 3)
    np.testing.assert_allclose(
        MRP.from_quaternion(r.to_quaternion()).to_dcm(), r.to_dcm(), atol=1e-14
    )
    np.testing.assert_allclose(
        MRP.from_dcm(r.to_dcm()).to_dcm(), r.to_dcm(), atol=1e-14
    )
    v = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(r.body_to_inertial(r.inertial_to_body(v)), v, atol=1e-14)
    np.testing.assert_allclose((r * r.inverse()).to_dcm(), np.eye(3), atol=1e-14)


@pytest.mark.parametrize("bad", [[1, 2], [np.nan, 0, 0], [[1, 2, 3]]])
def test_rejects_invalid_parameters(bad: object) -> None:
    with pytest.raises(ValueError):
        MRP(bad)


def test_shadow_set_represents_same_rotation() -> None:
    rotation = MRP([0.25, -0.2, 0.1])
    shadow = rotation.switch_shadow_set()
    np.testing.assert_allclose(shadow.to_dcm(), rotation.to_dcm(), atol=1e-14)
    assert shadow.norm > 1


def test_components_are_copies_and_bad_multiplication_is_rejected() -> None:
    rotation = MRP([0.1, 0.2, 0.3])
    components = rotation.components
    components[0] = 99
    assert rotation.components[0] == pytest.approx(0.1)
    with pytest.raises(TypeError):
        rotation.multiply([1, 0, 0])
