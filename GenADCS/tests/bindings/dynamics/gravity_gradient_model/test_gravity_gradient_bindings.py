from types import SimpleNamespace

import numpy as np
import pytest


def test_native_gravity_gradient_accepts_numpy_inputs(dynamics_backend):
    position = np.array([1.0, 1.0, 0.0], dtype=np.float32) * 7e6
    quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    inertia = np.diag(np.array([2.0, 3.0, 4.0], dtype=np.float32))

    torque = dynamics_backend.static_gravity_gradient_torque(
        position, quaternion, inertia, 3.986e14
    )

    assert isinstance(torque, np.ndarray)
    assert torque.shape == (3,)
    assert torque.dtype == np.float64
    assert np.isfinite(torque).all()
    assert np.linalg.norm(torque) > 0.0


def test_gravity_gradient_model_binding_reads_geometry_and_environment(
    dynamics_backend,
):
    geometry = SimpleNamespace(inertia=np.diag([2.0, 3.0, 4.0]))
    environment = SimpleNamespace(
        planetary_body=SimpleNamespace(gravitational_parameter=3.986e14)
    )
    model = dynamics_backend.StaticGravityGradientModel(geometry)

    torque = model.evaluate([7e6, 7e6, 0.0], [1.0, 0.0, 0.0, 0.0], environment)
    assert torque.shape == (3,)
    assert np.isfinite(torque).all()


@pytest.mark.parametrize("backend", ("cpp", "rust"))
def test_gravity_gradient_binding_maps_centre_singularity_to_value_error(backend):
    module = __import__(
        f"GenADCS.backends.{backend}.dynamics.gravity_gradient_model",
        fromlist=["static_gravity_gradient_torque"],
    )
    with pytest.raises(ValueError, match="body's centre"):
        module.static_gravity_gradient_torque(
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], np.eye(3), 3.986e14
        )
