from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


def test_native_zonal_binding_accepts_numpy_inputs(dynamics_backend: Any) -> None:
    position = np.array([7e6, 0.0, 0.0], dtype=np.float32)
    mass = 10.0
    mu = 3.986004418e14
    radius = 6_378_137.0
    coefficients = np.zeros(5, dtype=np.float32)

    force = dynamics_backend.zonal_gravity_force(
        position, mass, 0.0, mu, radius, coefficients
    )

    position_f64 = position.astype(np.float64)
    expected = -mu * mass * position_f64 / np.linalg.norm(position_f64) ** 3
    assert isinstance(force, np.ndarray)
    assert force.shape == (3,)
    assert force.dtype == np.float64
    np.testing.assert_allclose(force, expected, rtol=1e-12)


def test_zonal_force_model_binding_uses_body_parameters(dynamics_backend: Any) -> None:
    body = SimpleNamespace(
        gravitational_parameter=3.986004418e14,
        reference_radius=6_378_137.0,
    )
    environment = SimpleNamespace(planetary_body=body, j_gravity=None)
    model = dynamics_backend.ZonalForceModel(
        SimpleNamespace(mass=10.0), environment, degrees=()
    )

    force = model.evaluate(np.array([7e6, 0.0, 0.0], dtype=np.float32))
    assert force.shape == (3,)
    assert force.dtype == np.float64
    assert np.isfinite(force).all()
    assert force[0] < 0.0


@pytest.mark.parametrize("backend", ("cpp", "rust"))
def test_zonal_model_binding_rejects_malformed_position(backend: Any) -> None:
    module = __import__(
        f"GenADCS.backends.{backend}.dynamics.zonal_model",
        fromlist=["ZonalForceModel"],
    )
    environment = SimpleNamespace(
        planetary_body=SimpleNamespace(
            gravitational_parameter=3.986004418e14,
            reference_radius=6_378_137.0,
        ),
        j_gravity=None,
    )
    model = module.ZonalForceModel(SimpleNamespace(mass=10.0), environment)
    with pytest.raises(ValueError, match="finite inertial 3-vector"):
        model.evaluate([1.0, 2.0])
