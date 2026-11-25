from types import SimpleNamespace
import numpy as np
import pytest
from GenADCS.backends.python.dynamics.residual_dipole_model import (
    StaticResidualDipoleModel,
    static_residual_dipole_torque,
)


def test_dipole_torque_is_cross_product_and_model_uses_field():
    dipole = np.array([1.0, 0.0, 0.0])
    field = np.array([0.0, 0.0, 2e-5])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        static_residual_dipole_torque(dipole, field, q), np.cross(dipole, field)
    )
    model = StaticResidualDipoleModel(dipole)
    env = SimpleNamespace(
        magnetic_field=SimpleNamespace(field_inertial_tesla=lambda p, t: field)
    )
    np.testing.assert_allclose(
        model.evaluate(np.ones(3), q, 0.0, env), np.cross(dipole, field)
    )
    with pytest.raises(ValueError, match="requires environment.magnetic_field"):
        model.evaluate(np.ones(3), q, 0.0, SimpleNamespace(magnetic_field=None))


@pytest.mark.parametrize("dipole", [np.ones(2), np.array([1.0, np.nan, 0.0])])
def test_dipole_model_rejects_malformed_vector(dipole):
    with pytest.raises(ValueError):
        StaticResidualDipoleModel(dipole)
