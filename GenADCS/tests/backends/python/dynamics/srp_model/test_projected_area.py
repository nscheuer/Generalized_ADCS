import numpy as np
import pytest

from GenADCS.backends.python.dynamics.srp_model import (
    projected_area_srp_force,
    projected_area_srp_torque,
)


def test_srp_force_is_finite_and_scales_with_illumination() -> None:
    args = (
        np.zeros(3),
        np.array([149597870700.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([[1.0, 0.0, 0.0]]),
        np.array([2.0]),
        np.array([[0.0, 0.0, 1.0]]),
    )
    full = projected_area_srp_force(*args, 1.0)
    half = projected_area_srp_force(*args, 0.5)
    assert np.isfinite(full).all() and np.linalg.norm(full) > 0
    np.testing.assert_allclose(half, 0.5 * full)
    assert projected_area_srp_force(*args, 0.0).tolist() == [0.0, 0.0, 0.0]


def test_srp_torque_and_coincident_sun_rejection() -> None:
    args = (
        np.zeros(3),
        np.array([1.5e11, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([[1.0, 0.0, 0.0]]),
        np.array([[0.0, 1.0, 0.0]]),
        np.array([1.0]),
        np.array([[0.0, 0.0, 1.0]]),
    )
    torque = projected_area_srp_torque(*args)
    assert np.isfinite(torque).all() and np.linalg.norm(torque) > 0
    with pytest.raises(ValueError, match="positions coincide"):
        projected_area_srp_force(
            np.zeros(3), np.zeros(3), args[2], args[3], args[5], args[6]
        )
