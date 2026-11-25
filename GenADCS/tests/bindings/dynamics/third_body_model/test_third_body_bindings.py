from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray


def test_native_lunisolar_binding_accepts_numpy_vectors(dynamics_backend: Any) -> None:
    acceleration: NDArray[np.float64] = dynamics_backend.lunisolar_acceleration(
        np.array([7e6, 0.0, 0.0], dtype=np.float32),
        np.array([1.5e11, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 3.8e8, 0.0], dtype=np.float32),
        1.3271244e20,
        4.9048695e12,
    )

    assert isinstance(acceleration, np.ndarray)
    assert acceleration.shape == (3,)
    assert acceleration.dtype == np.float64
    assert np.isfinite(acceleration).all()
    assert 1e-8 < np.linalg.norm(acceleration) < 1e-5


@pytest.mark.parametrize(
    "sun,moon",
    [
        ([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 1.0, 0.0], [0.0, 0.0, 0.0]),
    ],
)
def test_native_lunisolar_binding_maps_zero_separation_to_value_error(
    dynamics_backend: Any, sun: Any, moon: Any
) -> None:
    with pytest.raises(ValueError, match="distance must be nonzero"):
        dynamics_backend.lunisolar_acceleration([0.0, 0.0, 0.0], sun, moon, 1.0, 1.0)
