from types import SimpleNamespace
import numpy as np
import pytest
from GenADCS.backends.python.dynamics.gravity_gradient_model import StaticGravityGradientModel, static_gravity_gradient_torque


def test_gravity_gradient_torque_known_orientation_and_zero_sphere():
    inertia=np.diag([2.,3.,4.]); q=np.array([1.,0.,0.,0.]); pos=np.array([1.,1.,0.])*7e6
    tau=static_gravity_gradient_torque(pos,q,inertia,3.986e14)
    assert np.isfinite(tau).all() and np.linalg.norm(tau)>0
    assert np.linalg.norm(static_gravity_gradient_torque(pos,q,np.eye(3)*4,3.986e14))==0
    with pytest.raises(ValueError,match="body's centre"):
        static_gravity_gradient_torque(np.zeros(3),q,inertia,3.986e14)


@pytest.mark.parametrize('inertia', [np.ones((2,2)),np.full((3,3),np.nan),np.array([[1,1,0],[0,1,0],[0,0,1]])])
def test_model_rejects_bad_inertia(inertia):
    with pytest.raises(ValueError): StaticGravityGradientModel(SimpleNamespace(inertia=inertia))
