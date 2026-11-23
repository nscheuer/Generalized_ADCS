import numpy as np
import pytest

def test_dcm_construction_composition_inverse_and_rotation(rotations):
    a=rotations.DCM.from_axis_angle([0,0,1],.4); b=rotations.DCM.from_rotation_vector([0,0,.3])
    np.testing.assert_allclose(a.inverse().to_dcm(),a.to_dcm().T,atol=1e-12)
    np.testing.assert_allclose((a*b).to_dcm(),a.to_dcm()@b.to_dcm(),atol=1e-12)
    np.testing.assert_allclose(a.to_quaternion().to_dcm(),a.to_dcm(),atol=1e-12)
    np.testing.assert_allclose(a.to_dcm()@a.to_dcm().T,np.eye(3),atol=1e-12)

def test_dcm_validation_and_normalization(rotations):
    with pytest.raises((ValueError,TypeError)): rotations.DCM(np.diag([-1.,1.,1.]))
    with pytest.raises((ValueError,TypeError)): rotations.DCM(np.zeros((3,3)))
    with pytest.raises((ValueError,TypeError)): rotations.dcm.normalize_dcm(np.zeros((3,3)))
