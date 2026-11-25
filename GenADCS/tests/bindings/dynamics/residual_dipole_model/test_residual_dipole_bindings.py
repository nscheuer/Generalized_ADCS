from types import SimpleNamespace
from typing import Any

import numpy as np


def test_native_residual_dipole_accepts_numpy_inputs_and_returns_torque(
    dynamics_backend: Any,
) -> None:
    dipole = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    field = np.array([0.0, 0.0, 2e-5], dtype=np.float32)
    quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    torque = dynamics_backend.static_residual_dipole_torque(dipole, field, quaternion)

    assert isinstance(torque, np.ndarray)
    assert torque.shape == (3,)
    assert torque.dtype == np.float64
    np.testing.assert_allclose(torque, np.cross(dipole, field))


def test_residual_dipole_model_binding_uses_environment_field(dynamics_backend: Any) -> None:
    dipole = np.array([1.0, 0.0, 0.0])
    field = np.array([0.0, 0.0, 2e-5])
    environment = SimpleNamespace(
        magnetic_field=SimpleNamespace(
            field_inertial_tesla=lambda position, epoch: field
        )
    )
    model = dynamics_backend.StaticResidualDipoleModel(dipole)

    torque = model.evaluate(np.ones(3), [1.0, 0.0, 0.0, 0.0], 0.0, environment)

    np.testing.assert_allclose(torque, np.cross(dipole, field))
