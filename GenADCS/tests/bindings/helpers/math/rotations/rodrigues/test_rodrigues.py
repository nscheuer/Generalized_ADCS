import numpy as np
import pytest
from importlib import import_module

def cls(r): return getattr(r,"Rodriguez_Parameters")

def test_rodrigues_aliases_numpy_roundtrip_and_ownership(rotations):
    C=cls(rotations); assert C is rotations.RodriguesParameters
    source=np.array([.1,-.2,.3],dtype=np.float32); p=C(source); source[:]=0
    np.testing.assert_allclose(np.asarray(p),[.1,-.2,.3],atol=1e-7)
    rodrigues=import_module(rotations.__name__ + ".rotations.rodrigues")
    np.testing.assert_allclose(rodrigues.quaternion_to_rodrigues(rodrigues.rodrigues_to_quaternion([.1,-.2,.3])),[.1,-.2,.3])

def test_rodrigues_singular_and_invalid_binding_inputs(rotations):
    with pytest.raises(ValueError): cls(rotations).from_axis_angle([1.,0,0],np.pi)
    with pytest.raises(ValueError): cls(rotations)([np.inf,0,0])
