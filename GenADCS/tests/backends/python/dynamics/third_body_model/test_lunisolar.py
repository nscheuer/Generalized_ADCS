from types import SimpleNamespace

import numpy as np
import pytest

from GenADCS.backends.python.dynamics.third_body_model import (
    LunisolarForceModel,
    lunisolar_acceleration,
)


def test_lunisolar_differential_acceleration_is_finite_and_small():
    a = lunisolar_acceleration(
        np.array([7e6, 0.0, 0.0]),
        np.array([1.5e11, 0.0, 0.0]),
        np.array([0.0, 3.8e8, 0.0]),
        1.3271244e20,
        4.9048695e12,
    )
    assert np.isfinite(a).all() and 1e-8 < np.linalg.norm(a) < 1e-5


@pytest.mark.parametrize(
    "sun,moon", [(np.zeros(3), np.ones(3)), (np.ones(3), np.zeros(3))]
)
def test_lunisolar_rejects_zero_separation(sun, moon):
    with pytest.raises(ValueError, match="distance must be nonzero"):
        lunisolar_acceleration(np.zeros(3), sun, moon, 1.0, 1.0)


@pytest.mark.parametrize("mass", [0.0, -1.0, np.nan])
def test_lunisolar_model_rejects_invalid_mass(mass):
    with pytest.raises(ValueError, match="mass must be finite and positive"):
        LunisolarForceModel(
            SimpleNamespace(mass=mass),
            SimpleNamespace(_segments=[], _times=[], _data=[]),
        )
