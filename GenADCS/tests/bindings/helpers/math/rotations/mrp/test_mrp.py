import gc
import numpy as np
import pytest

def cls(r): return getattr(r,"Modified_Rodriguez_Parameters")

def test_mrp_aliases_numpy_conversion_and_ownership(rotations):
    C=cls(rotations); assert C is rotations.ModifiedRodriguesParameters
    source=np.array([2.,0.,0.],dtype=np.float32); p=C(source); source[0]=4
    np.testing.assert_allclose(np.asarray(p),[-.5,0,0]); assert np.asarray(p).dtype==np.float64
    arr=np.asarray(p); del p;gc.collect();np.testing.assert_allclose(arr,[-.5,0,0])

def test_mrp_shadow_and_quaternion_binding(rotations):
    p=cls(rotations)([.2,.3,.4]); q=p.to_quaternion(); shadow=p.switch_shadow_set()
    np.testing.assert_allclose(shadow.to_dcm(),p.to_dcm(),atol=1e-12)
    np.testing.assert_allclose(rotations.quaternion_to_mrp(rotations.mrp_to_quaternion([.2,.3,.4])),[.2,.3,.4])

def test_mrp_illegal_inputs_and_identity_shadow_error(rotations):
    with pytest.raises(ValueError): rotations.mrp_shadow_set([0.,0.,0.])
    with pytest.raises(ValueError): cls(rotations)([np.nan,0,0])
