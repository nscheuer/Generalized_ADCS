from types import SimpleNamespace

import numpy as np
import pytest

from GenADCS.backends.python.dynamics.zonal_model import ZonalForceModel


def setup_model(degrees=(2, 3, 4, 5, 6), gravity=None):
    body = SimpleNamespace(
        gravitational_parameter=3.986004418e14, reference_radius=6378137.0
    )
    env = SimpleNamespace(planetary_body=body, j_gravity=gravity)
    return ZonalForceModel(SimpleNamespace(mass=10.0), env, degrees)


def test_zonal_force_is_mass_times_acceleration_and_decreases_with_altitude():
    model = setup_model()
    low = np.array([7e6, 0.0, 0.0])
    high = np.array([8e6, 0.0, 0.0])
    assert np.isfinite(model.evaluate(low)).all()
    assert model.evaluate(low)[0] < 0 and np.linalg.norm(
        model.evaluate(low)
    ) > np.linalg.norm(model.evaluate(high))


@pytest.mark.parametrize("degrees", [(1,), (7,), (2, 2.5)])
def test_zonal_model_rejects_invalid_degrees(degrees):
    with pytest.raises(ValueError, match="degrees must be selected"):
        setup_model(degrees)


@pytest.mark.parametrize("mass", [0.0, -1.0, np.nan])
def test_zonal_model_rejects_invalid_mass(mass):
    with pytest.raises(ValueError, match="mass must be finite and positive"):
        ZonalForceModel(
            SimpleNamespace(mass=mass),
            SimpleNamespace(
                planetary_body=SimpleNamespace(
                    gravitational_parameter=1.0, reference_radius=1.0
                ),
                j_gravity=None,
            ),
        )


def test_zonal_evaluation_rejects_malformed_position():
    with pytest.raises(ValueError, match="finite inertial 3-vector"):
        setup_model().evaluate([1.0, 2.0])
