"""Tests for the Python backend's noise models and Gaussian kernels."""

import numpy as np
import pytest

from GenADCS.backends.python.noise import (
    FlickerNoise,
    GaussianWhiteNoise,
    GaussMarkovNoise,
    Noise,
    QuantizationNoise,
    UniformNoise,
)
from GenADCS.backends.python.noise.noise import gaussian_sample


def test_covariance_noise_cholesky_and_reproducible_sample() -> None:
    noise = Noise([[4.0, 2.0], [2.0, 3.0]])
    factor = noise.cholesky_factor
    assert factor is not None
    np.testing.assert_allclose(factor, [[2.0, 0.0], [1.0, np.sqrt(2.0)]])

    first = gaussian_sample(factor, np.uint64(12), 3)
    np.testing.assert_array_equal(
        first, gaussian_sample(factor, np.uint64(12), 3)
    )
    assert not np.array_equal(
        first, gaussian_sample(factor, np.uint64(12), 4)
    )

    value = np.array([1.0, -1.0])
    draw = np.array([0.5, -0.25])

    class FixedRng:
        def standard_normal(self, size: int) -> np.ndarray:
            assert size == 2
            return draw

    np.testing.assert_allclose(
        noise.apply(
            value,
            0.1,
            rng=FixedRng(),  # pyright: ignore[reportArgumentType] -- deterministic test double provides the RNG method used by apply.
        ),
        value + factor @ draw,
    )


def test_covariance_noise_rejects_invalid_inputs() -> None:
    with pytest.raises(NotImplementedError):
        Noise().apply([1.0], 0.1)
    with pytest.raises(ValueError):
        Noise([[1.0, 0.0], [0.0]])
    with pytest.raises(np.linalg.LinAlgError):
        Noise([[-1.0]])
    with pytest.raises(ValueError):
        Noise(np.eye(2)).apply([1.0], 0.1)


def test_concrete_noise_models_store_parameters() -> None:
    assert FlickerNoise(0.25).amplitude == 0.25
    assert GaussMarkovNoise(0.5, 12.0).correlation_time == 12.0
    assert GaussianWhiteNoise(0.1).standard_deviation == 0.1
    assert QuantizationNoise(0.01).step_size == 0.01
    uniform = UniformNoise(-1.0, 1.0)
    assert (uniform.lower, uniform.upper) == (-1.0, 1.0)
