from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import NDArray


def test_native_residual_dipole_accepts_numpy_inputs_and_returns_torque(
    dynamics_backend: Any,
) -> None:
    dipole = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    field = np.array([0.0, 0.0, 2e-5], dtype=np.float32)
    quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    torque: NDArray[np.float64] = dynamics_backend.static_residual_dipole_torque(
        dipole, field, quaternion
    )

    assert isinstance(torque, np.ndarray)
    assert torque.shape == (3,)
    assert torque.dtype == np.float64
    np.testing.assert_allclose(torque, np.cross(dipole, field))


def test_residual_dipole_model_binding_uses_environment_field(dynamics_backend: Any) -> None:
    dipole = np.array([1.0, 0.0, 0.0])
    field = np.array([0.0, 0.0, 2e-5])

    def field_inertial_tesla(
        position: NDArray[np.float64], epoch: float
    ) -> NDArray[np.float64]:
        return field

    environment = SimpleNamespace(
        magnetic_field=SimpleNamespace(
            field_inertial_tesla=field_inertial_tesla
        )
    )
    model = dynamics_backend.StaticResidualDipoleModel(dipole)

    torque: NDArray[np.float64] = model.evaluate(
        np.ones(3), [1.0, 0.0, 0.0, 0.0], 0.0, environment
    )

    np.testing.assert_allclose(torque, np.cross(dipole, field))
